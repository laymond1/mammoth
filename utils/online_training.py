# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import copy
import math
import os
import sys
from argparse import Namespace
from time import time
from typing import Iterable, Tuple
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader

from datasets.utils.indexed_dataset import IndexedDataset
from datasets.utils.online_sampler import OnlineTaskSiBlurrySampler, OnlineCILSampler, OnlinePeriodicSampler, OnlineTestSampler, OnlineTrainSampler
from datasets.utils.continual_dataset import ContinualDataset, MammothDatasetWrapper
from models.utils.online_continual_model import OnlineContinualModel
from models.utils.future_model import FutureModel

from utils.stats import track_system_stats
from utils.metrics import online_forgetting

try:
    import wandb
except ImportError:
    wandb = None


def initialize_wandb(args: Namespace) -> None:
    """
    Initializes wandb, if installed.

    Args:
        args: the arguments of the current execution
    """
    assert wandb is not None, "Wandb not installed, please install it or run without wandb"
    run_name = args.wandb_name if args.wandb_name is not None else args.model

    run_id = args.conf_jobnum.split('-')[0]
    name = f'{run_name}_{run_id}'
    mode = 'disabled' if os.getenv('MAMMOTH_TEST', '0') == '1' else os.getenv('WANDB_MODE', 'online')
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), name=name, mode=args.wandb_mode)
    args.wandb_url = wandb.run.get_url()


def get_other_metrics(eval_results, exposed_classes_records):
    """
    Calculate other metrics from the evaluation dictionary.

    Args:
        eval_results: the evaluation dictionary

    Returns:
        the evaluation dictionary with the additional metrics
    """
    metrics = defaultdict(list)
    
    exposed_classes = exposed_classes_records[-2]
    metrics['instant_fgt'] = online_forgetting(eval_results["cls_acc"], exposed_classes, mode='instant')
    metrics['last_fgt'] = online_forgetting(eval_results["cls_acc"], exposed_classes, mode='last')
    
    return metrics


def train(model: OnlineContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.

    Args:
        model: the module to be trained
        dataset: the continual dataset at hand
        args: the arguments of the current execution
    """
    args.test_batch_size = 128
    print(args)
    if not hasattr(args, 'buffer_size'):
        args.buffer_size = 0
    if args.validation:
        log_path = f"{args.log_path}/logs/{args.online_scenario}/{args.dataset}/{args.model}_M{args.buffer_size}"
        os.makedirs(log_path, exist_ok=True)
    else:
        log_path = f"{args.log_path}/logs/{args.online_scenario}/{args.dataset}/{args.model}_M{args.buffer_size}/run-{args.seed}"
        os.makedirs(log_path, exist_ok=True)
    model.log_path = log_path
    
    # dataset setup
    dataset.SETTING = args.online_scenario
    dataset.N_TASKS = args.n_tasks
    dataset.N_CLASSES_PER_TASK = dataset.N_CLASSES // dataset.N_TASKS
    dataset.set_dataset()
    assert hasattr(dataset, 'train_dataset') and hasattr(dataset, 'test_dataset'), "Dataset must have both train_dataset and test_dataset attributes"

    if not args.nowand:
        initialize_wandb(args)
        
    model.net.to(model.device)
    torch.cuda.empty_cache()
    
    # Dataset setup
    total_samples = len(dataset.train_dataset)
    train_dataset = IndexedDataset(dataset.train_dataset)
    test_dataset = dataset.test_dataset
    
    # Scenario setup
    if args.online_scenario == 'si-blurry':
        train_sampler = OnlineTaskSiBlurrySampler(train_dataset, dataset.N_TASKS, args.m, args.n, args.seed, args.rnd_NM) # args.selection_size was used for prompt
    elif args.online_scenario == 'online-stand-cil':
        train_sampler = OnlineCILSampler(train_dataset, dataset.N_TASKS, args.seed, dataset.N_CLASSES_PER_TASK)
    elif args.online_scenario == 'online-cil':
        args.m = 0
        args.n = 100
        train_sampler = OnlineTaskSiBlurrySampler(train_dataset, dataset.N_TASKS, args.m, args.n, args.seed, args.rnd_NM) # args.selection_size was used for prompt
    elif args.online_scenario == 'periodic-gaussian':
        assert args.n_tasks == 1, "Periodic Gaussian is a single task scenario"
        train_sampler = OnlinePeriodicSampler(train_dataset, args.sigma, args.repeat, args.init_cls, args.seed)
    else:
        raise NotImplementedError(f"Scenario {args.online_scenario} not implemented")
    
    # Dataloader setup
    train_dataloader = DataLoader(train_dataset, batch_size=args.minibatch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    full_test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    with track_system_stats() as system_tracker:
        
        print(file=sys.stderr)
        if args.online_scenario == 'si-blurry':
            print(f"Online boundary free training with Stochastic Blurry M:{args.m} and N:{args.n} samples")
        elif args.online_scenario == 'periodic-gaussian':
            print(f"Online boundary free training {args.repeat} times with Periodic Gaussian {args.sigma}")
        elif args.online_scenario == 'online-cil':
            print(f"Online class-incremental training")
        elif args.online_scenario == 'online-stand-cil':
            print(f"Online standard class-incremental training")
        else:
            raise NotImplementedError(f"Scenario {args.online_scenario} not implemented")
        
        end_task = dataset.N_TASKS
        print(f"Incrementally training {end_task} tasks")  
        task_records = defaultdict(list)
        eval_results = defaultdict(list)
        linear_eval_results = defaultdict(list)
        exposed_classes_records = []
        samples_cnt = 0
        num_eval = args.eval_period
        f_eval_dict = None
        f_eval_period = args.f_eval_period
        fgt_first_eval_done = False

        # Clear GPU cache before starting
        torch.cuda.empty_cache()
        # Start Task
        for task_id in range(end_task):
            if args.debug_mode and samples_cnt >= 4000:
                break
            
            print("\n" + "#" * 50)
            print(f"# Task {task_id} Session")
            print("#" * 50 + "\n")
            print("[2-1] Prepare a datalist for the current task")
            
            if args.online_scenario in ['si-blurry', 'online-stand-cil', 'online-cil']:
                train_sampler.set_task(task_id)

            model.online_before_task(task_id)
            model.online_before_train()
            
            # Get future classes
            if args.f_eval:
                future_classes = model.get_future_classes(
                    DataLoader(train_dataset, 
                                batch_size=args.test_batch_size * 8,
                                sampler=train_sampler, 
                                num_workers=args.num_workers, 
                                pin_memory=True)
                )
            
            ## Start Online Training
            for i, (images, labels, not_aug_images, idx) in enumerate(train_dataloader):
                if args.debug_mode and samples_cnt >= 4000:
                    break
                
                samples_cnt += images.size(0) * model.world_size
                if model.NAME == 'online-ovor':
                    loss_dict, ood_loss_dict, acc, ood_acc = model.online_step(images, labels, not_aug_images, idx)
                else:
                    loss_dict, acc = model.online_step(images, labels, not_aug_images, idx)
                
                assert not math.isnan(loss_dict['total_loss']), f"Loss is NaN @ task {task_id} Sample # {samples_cnt}"
                
                # Synchronize CUDA for consistent tracking if optimization is disabled
                if args.code_optimization == 0 and 'cuda' in str(args.device):
                    torch.cuda.synchronize()
            
                system_tracker()
                
                if model.NAME == 'online-ovor':
                    model.report_training(total_samples, samples_cnt, loss_dict, acc, ood_loss_dict, ood_acc)
                else:
                    model.report_training(total_samples, samples_cnt, loss_dict, acc)
            
                ### Anytime evaluation
                if samples_cnt >= num_eval:
                    # Setup test sampler and dataloader for evaluation
                    test_sampler = OnlineTestSampler(test_dataset, model.exposed_classes)
                    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, sampler=test_sampler, num_workers=args.num_workers)
                    # Linear evaluation if enabled
                    if args.linear_eval:
                        linear_train_sampler = OnlineTrainSampler(train_dataset, model.exposed_classes, args.seed)
                        linear_train_loader = DataLoader(train_dataset, batch_size=args.test_batch_size, sampler=linear_train_sampler, num_workers=args.num_workers)
                        linear_eval_dict = model.online_linear_train_eval(linear_train_loader, test_dataloader)
                
                    with torch.no_grad():
                        eval_dict = model.online_evaluate(test_dataloader)

                        # Run online_forgetting_evaluate for the first time
                        if not fgt_first_eval_done and args.f_eval:
                            f_eval_dict = model.online_forgetting_evaluate(full_test_dataloader, future_classes, samples_cnt)
                            fgt_first_eval_done = True
                        
                        # Subsequent forgetting evaluations based on f_eval_period
                        elif samples_cnt >= f_eval_period and args.f_eval:
                            f_eval_dict = model.online_forgetting_evaluate(full_test_dataloader, future_classes, samples_cnt)
                            f_eval_period += args.f_eval_period
                        # TODO: implement distributed evaluation
                        # Aggregate evaluation results across distributed processes if needed
                        if model.distributed:
                            eval_dict =  torch.tensor([eval_dict["avg_loss"], eval_dict["avg_acc"], *eval_dict["cls_acc"]], device=model.device)
                            dist.reduce(eval_dict, dst=0, op=dist.ReduceOp.SUM)
                            eval_dict = eval_dict.cpu().numpy()
                            eval_dict = {
                                "avg_loss": eval_dict[0]/model.world_size,
                                "avg_acc": eval_dict[1]/model.world_size,
                                "cls_acc": eval_dict[2:]/model.world_size
                            }
                        # Update main process results
                        if model.is_main_process():
                            eval_results["test_acc"].append(eval_dict["avg_acc"])
                            eval_results["cls_acc"].append(eval_dict["cls_acc"])
                            eval_results["data_cnt"].append(num_eval)
                            exposed_classes_records.append(model.exposed_classes)
                            if args.linear_eval:
                                # linear eval results
                                linear_eval_results["test_acc"].append(linear_eval_dict['avg_acc'])
                                linear_eval_results["cls_acc"].append(linear_eval_dict['cls_acc'])
                            
                            # Update forgetting metrics if available
                            if f_eval_dict is not None:
                                for k, v in f_eval_dict.items():
                                    eval_results[k].append(v)
                                # Update eval_dict with f_eval_dict to report forgetting metrics
                                eval_dict.update(f_eval_dict)
                                f_eval_dict = None
                        
                            # Additional metrics calculation if available
                            if len(eval_results["cls_acc"]) > 1:
                                metrics_dict = get_other_metrics(eval_results, exposed_classes_records)
                                for k, v in metrics_dict.items():
                                    eval_results[k].append(v)
                                if args.linear_eval:
                                    # Update eval_dict with additional metrics
                                    linear_metrics_dict = get_other_metrics(linear_eval_results, exposed_classes_records)
                                    for k, v in linear_metrics_dict.items():
                                        linear_eval_results[k].append(v)
                                model.report_test(samples_cnt, eval_dict, metrics_dict)
                            else:
                                model.report_test(samples_cnt, eval_dict)

                            # Save model checkpoint, if enabled
                            if args.savecheck == 'eval':
                                save_obj = {
                                    'model': model.state_dict(),
                                    'exposed_classes': model.exposed_classes,
                                    'optimizer': model.opt.state_dict() if hasattr(model, 'opt') else None,
                                    'scheduler': model.scheduler.state_dict() if model.scheduler is not None else None,
                                }

                                # Saving model checkpoint
                                if args.validation:
                                    checkpoint_name = f'{log_path}/checkpoint_seed_{args.seed}_eval_{num_eval}.pt'
                                else:
                                    checkpoint_name = f'{log_path}/checkpoint_eval_{num_eval}.pt'
                                if checkpoint_name is not None:
                                    torch.save(save_obj, checkpoint_name)

                        # Update the next evaluation sample count
                        num_eval += args.eval_period
                    ### End of each anytime evaluation
                ## End of each online training
                sys.stdout.flush()
                model.online_after_train()
            # End of each task
            # model.report_test(samples_cnt, eval_dict, task_id=task_id) # report again the last result after task
            model.online_after_task(task_id)

            # Save model checkpoint, if enabled
            if args.savecheck == 'task':
                save_obj = {
                            'model': model.state_dict(),
                            'exposed_classes': model.exposed_classes,
                            'optimizer': model.opt.state_dict() if hasattr(model, 'opt') else None,
                            'scheduler': model.scheduler.state_dict() if model.scheduler is not None else None,
                    }
                # Saving model checkpoint for the current task
                if args.validation:
                    checkpoint_name = f'{log_path}/checkpoint_seed_{args.seed}_task_{task_id}.pt'
                else:
                    checkpoint_name = f'{log_path}/checkpoint_task_{task_id}.pt'
                torch.save(save_obj, checkpoint_name)
                
            # Evaluate the model after the task
            test_sampler = OnlineTestSampler(test_dataset, model.exposed_classes)
            test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, sampler=test_sampler, num_workers=args.num_workers)
            task_eval_dict = model.online_evaluate(test_dataloader)
            # Report again the last result after task
            model.report_test(samples_cnt, task_eval_dict, task_id=task_id) 
            
            if model.distributed:
                task_eval_dict =  torch.tensor([task_eval_dict["avg_loss"], task_eval_dict["avg_acc"], *task_eval_dict["cls_acc"]], device=model.device)
                dist.reduce(task_eval_dict, dst=0, op=dist.ReduceOp.SUM)
                task_eval_dict = task_eval_dict.cpu().numpy()
                task_eval_dict = {"avg_loss": task_eval_dict[0]/model.world_size, "avg_acc": task_eval_dict[1]/model.world_size, "cls_acc": task_eval_dict[2:]/model.world_size}

            print("[2-4] Update the information for the current task")
            task_records["task_acc"].append(task_eval_dict["avg_acc"])
            task_records["cls_acc"].append(task_eval_dict["cls_acc"])
            
            print("[2-5] Report task result")
            
        if model.is_main_process():
            if args.eval_period is not None:
                if args.validation:     
                    # Task metrics   
                    np.save(f"{log_path}/task_cls_acc_seed_{args.seed}_eval.npy", task_records["cls_acc"])
                    np.save(f"{log_path}/task_acc_seed_{args.seed}_eval.npy", task_records["task_acc"])
                    # Anytime evaluation metrics
                    np.save(f"{log_path}/cls_acc_seed_{args.seed}_eval.npy", eval_results["cls_acc"])
                    np.save(f"{log_path}/test_acc_seed_{args.seed}_eval.npy", eval_results["test_acc"])
                    np.save(f"{log_path}/data_cnt_seed_{args.seed}_eval_time.npy", eval_results["data_cnt"])
                    np.save(f"{log_path}/instant_fgt_seed_{args.seed}_eval.npy", eval_results["instant_fgt"])
                    np.save(f"{log_path}/last_fgt_seed_{args.seed}_eval.npy", eval_results["last_fgt"])
                    if args.f_eval:
                        np.save(f"{log_path}/klr_seed_{args.seed}_eval.npy", eval_results["klr"][1:])
                        np.save(f"{log_path}/kgr_seed_{args.seed}_eval.npy", eval_results["kgr"])
                    if args.linear_eval:
                        # Linear evaluation metrics
                        np.save(f"{log_path}/linear_cls_acc_seed_{args.seed}_eval.npy", linear_eval_results["cls_acc"])
                        np.save(f"{log_path}/linear_test_acc_seed_{args.seed}_eval.npy", linear_eval_results["test_acc"])
                        np.save(f"{log_path}/linear_instant_fgt_seed_{args.seed}_eval.npy", linear_eval_results["instant_fgt"])
                        np.save(f"{log_path}/linear_last_fgt_seed_{args.seed}_eval.npy", linear_eval_results["last_fgt"])
                else:   
                    # Task metrics   
                    np.save(f"{log_path}/task_cls_acc_eval.npy", task_records["cls_acc"])
                    np.save(f"{log_path}/task_acc_eval.npy", task_records["task_acc"])
                    # Anytime evaluation metrics
                    np.save(f"{log_path}/cls_acc_eval.npy", eval_results["cls_acc"])
                    np.save(f"{log_path}/test_acc_eval.npy", eval_results["test_acc"])
                    np.save(f"{log_path}/data_cnt_eval_time.npy", eval_results["data_cnt"])
                    np.save(f"{log_path}/instant_fgt_eval.npy", eval_results["instant_fgt"])
                    np.save(f"{log_path}/last_fgt_eval.npy", eval_results["last_fgt"])
                    if args.f_eval:
                        np.save(f"{log_path}/klr_eval.npy", eval_results["klr"][1:])
                        np.save(f"{log_path}/kgr_eval.npy", eval_results["kgr"])
                    if args.linear_eval:
                        # Linear evaluation metrics
                        np.save(f"{log_path}/linear_cls_acc_eval.npy", linear_eval_results["cls_acc"])
                        np.save(f"{log_path}/linear_test_acc_eval.npy", linear_eval_results["test_acc"])
                        np.save(f"{log_path}/linear_instant_fgt_eval.npy", linear_eval_results["instant_fgt"])
                        np.save(f"{log_path}/linear_last_fgt_eval.npy", linear_eval_results["last_fgt"])

            # Calculate final summary metrics
            A_task_avg = np.mean(task_records["task_acc"])
            if len(task_records["task_acc"]) > 0:
                A_task_last = task_records["task_acc"][-1]
            A_auc = np.mean(eval_results["test_acc"])
            if len(eval_results["test_acc"]) > 0:
                A_last = eval_results["test_acc"][-1]
            F_auc = np.mean(eval_results["instant_fgt"])
            if len(eval_results["last_fgt"]) > 0:
                F_last = eval_results["last_fgt"][-1]
            F_last_auc = np.mean(eval_results["last_fgt"])
            if args.f_eval:
                KLR_avg = np.mean(eval_results['klr'][1:])
                KGR_avg = np.mean(eval_results['kgr'])
            else:
                KLR_avg, KGR_avg = 0.0, 0.0
            print(f"======== Summary =======")
            if len(task_records["task_acc"]) > 0 and len(eval_results["test_acc"]) > 0 and len(eval_results["last_fgt"]) > 0:
                print(f"A_task {A_task_avg} | A_task_last {A_task_last} | A_auc {A_auc} | A_last {A_last} | F_auc {F_auc} | F_last {F_last} | F_last_auc {F_last_auc} | KGR_avg {KGR_avg} | KLR_avg {KLR_avg}")
            print(f"="*24)
            
            # Log results to wandb, if enabled
            if not args.nowand:
                wandb.log({
                    "A_auc": A_auc, "A_last": A_last, 
                    "F_auc": F_auc, "F_last": F_last, "F_last_auc": F_last_auc,
                    "KGR_avg": KGR_avg, "KLR_avg": KLR_avg
                })
            # Save model checkpoint, if enabled
            if args.savecheck == 'last':
                save_obj = {
                    'args': args,
                    'model': model.state_dict(),
                    'optimizer': model.opt.state_dict() if hasattr(model, 'opt') else None,
                    'scheduler': model.scheduler.state_dict() if model.scheduler is not None else None,
                }
                # Saving model checkpoint for the last task
                if args.validation:
                    checkpoint_name = f'{log_path}/checkpoint_seed_{args.seed}.pt'
                else:
                    checkpoint_name = f'{log_path}/checkpoint.pt'
                torch.save(save_obj, checkpoint_name)

        # Display system stats
        system_tracker.print_stats()

    if not args.nowand:
        wandb.finish()
