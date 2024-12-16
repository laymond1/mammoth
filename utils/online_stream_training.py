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

from datasets import get_dataset
from datasets.utils.indexed_dataset import IndexedDataset
from datasets.utils.online_sampler import OnlineSiBlurrySampler, OnlinePeriodicSampler, OnlineTestSampler
from datasets.utils.continual_dataset import ContinualDataset, MammothDatasetWrapper
from datasets.utils.gcl_dataset import GCLDataset
from models.utils.online_continual_model import OnlineContinualModel
from models.utils.future_model import FutureModel

from utils.checkpoints import mammoth_load_checkpoint
from utils.loggers import log_extra_metrics, log_online_accs, log_accs, Logger, OnlineLogger
from utils.schedulers import get_scheduler
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
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), name=name, mode=mode)
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
    
    # exposed_classes = exposed_classes_records[eval_cnt-1]
    exposed_classes = exposed_classes_records[-2]
    metrics['instant_fgt'] = online_forgetting(eval_results["cls_acc"], exposed_classes, mode='instant')
    metrics['last_fgt'] = online_forgetting(eval_results["cls_acc"], exposed_classes, mode='last')
    # TODO: implement FWT and BWT
    # metrics['instant_fwt'] = online_forward_transfer(eval_results["cls_acc"], exposed_classes)
    # metrics['instant_bwt'] = online_backward_transfer(eval_results["cls_acc"], exposed_classes)
    
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
    os.makedirs(f"{args.log_path}/logs/{args.online_scenario}/{args.dataset}/{args.model}", exist_ok=True)
    
    # dataset setup
    # dataset.SETTING = 'online-il'
    dataset.SETTING = args.online_scenario
    dataset.N_TASKS = args.n_tasks
    dataset.N_CLASSES_PER_TASK = dataset.N_CLASSES // dataset.N_TASKS
    dataset.set_dataset()
    assert hasattr(dataset, 'train_dataset') and hasattr(dataset, 'test_dataset'), "Dataset must have both train_dataset and test_dataset attributes"

    if not args.nowand:
        initialize_wandb(args)

    if not args.disable_log:
        logger = OnlineLogger(args, dataset.SETTING, dataset.NAME, model.NAME)
    
    # TODO: implement FWT
    is_fwd_enabled = True
    can_compute_fwd_beforetask = True
    ptm_results = defaultdict(list)
    if args.validation:
        is_fwd_enabled = False
        can_compute_fwd_beforetask = False
        
    model.net.to(model.device)
    torch.cuda.empty_cache()
    
    # Dataset setup
    total_samples = len(dataset.train_dataset)
    train_dataset = IndexedDataset(dataset.train_dataset)
    test_dataset = dataset.test_dataset
    
    # Scenario setup
    if args.online_scenario == 'si-blurry':
        train_sampler = OnlineSiBlurrySampler(train_dataset, dataset.N_TASKS, args.m, args.n, args.seed, args.rnd_NM) # args.selection_size was used for prompt
    elif args.online_scenario == 'periodic-gaussian':
        assert args.n_tasks == 1, "Periodic Gaussian is a single task scenario"
        train_sampler = OnlinePeriodicSampler(train_dataset, args.sigma, args.repeat, args.init_cls, args.seed)
    else:
        raise NotImplementedError(f"Scenario {args.online_scenario} not implemented")
    
    # Dataloader setup
    train_dataloader = DataLoader(train_dataset, batch_size=args.minibatch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    full_test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    with track_system_stats(logger) as system_tracker:
        
        print(file=sys.stderr)
        if args.online_scenario == 'si-blurry':
            print(f"Online boundary free training with Stochastic Blurry M:{args.m} and N:{args.n} samples")
        elif args.online_scenario == 'periodic-gaussian':
            print(f"Online boundary free training {args.repeat} times with Periodic Gaussian {args.sigma}")  
        else:
            raise NotImplementedError(f"Scenario {args.online_scenario} not implemented")
        
        eval_results = defaultdict(list)
        exposed_classes_records = []
        samples_cnt = 0
        num_eval = args.eval_period
        f_eval_period = args.f_eval_period
        fgt_first_eval_done = False

        # Clear GPU cache before starting
        torch.cuda.empty_cache()
        
        print("\n" + "#" * 50)
        print(f"# Start Session")
        print("#" * 50 + "\n")
        
        model.online_before_train()
        
        # Get future classes
        future_classes = model.get_future_classes(
            DataLoader(train_dataset, 
                        batch_size=args.test_batch_size * 8,
                        sampler=train_sampler, 
                        num_workers=args.num_workers, 
                        pin_memory=True)
        )
        # Evaluate before training to get the initial performance
        if is_fwd_enabled and can_compute_fwd_beforetask:
            # Evaluate future performance
            eval_dict = model.future_evaluate(full_test_dataloader, future_classes)
            ptm_results['test_acc'].append(eval_dict['avg_acc'])
            ptm_results['cls_acc'].append(eval_dict['cls_acc'])
            ptm_results['data_cnt'].append(samples_cnt)
            
        ## Start Online Training
        for i, (images, labels, not_aug_images, idx) in enumerate(train_dataloader):
            if args.debug_mode and (i+1) * args.minibatch_size >= 4000:
                break
            
            samples_cnt += images.size(0) * model.world_size
            if model.NAME == 'online-one-prompt':
                loss_dict, ood_loss_dict, acc, ood_acc = model.online_step(images, labels, not_aug_images, idx)
            else:
                loss_dict, acc = model.online_step(images, labels, not_aug_images, idx)
            
            assert not math.isnan(loss_dict['total_loss']), f"Loss is NaN @ Sample # {samples_cnt}"
            
            # Synchronize CUDA for consistent tracking if optimization is disabled
            if args.code_optimization == 0 and 'cuda' in str(args.device):
                torch.cuda.synchronize()
            
            system_tracker()
            if model.NAME == 'online-one-prompt':
                model.report_training(total_samples, samples_cnt, loss_dict, acc, ood_loss_dict, ood_acc)
            else:
                model.report_training(total_samples, samples_cnt, loss_dict, acc)
            
            ### Anytime evaluation
            if samples_cnt >= num_eval:
                with torch.no_grad():
                    # Setup test sampler and dataloader for evaluation
                    test_sampler = OnlineTestSampler(test_dataset, model.exposed_classes)
                    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, sampler=test_sampler, num_workers=args.num_workers)
                    
                    eval_dict = model.online_evaluate(test_dataloader)
                    
                    # Run online_forgetting_evaluate for the first time
                    if not fgt_first_eval_done:
                        f_eval_dict = model.online_forgetting_evaluate(full_test_dataloader, future_classes, samples_cnt)
                        fgt_first_eval_done = True
                    
                    # Subsequent forgetting evaluations based on f_eval_period
                    elif samples_cnt >= f_eval_period:
                        f_eval_dict = model.online_forgetting_evaluate(full_test_dataloader, future_classes, samples_cnt)
                        f_eval_period += args.f_eval_period
                    
                    # TODO: implement distributed evaluation
                    # Aggregate evaluation results across distributed processes if needed
                    if model.distributed:
                        eval_dict_tensor =  torch.tensor([eval_dict["avg_loss"], eval_dict["avg_acc"], *eval_dict["cls_acc"]], device=model.device)
                        dist.reduce(eval_dict_tensor, dst=0, op=dist.ReduceOp.SUM)
                        eval_dict_np = eval_dict_tensor.cpu().numpy()
                        eval_dict = {
                            "avg_loss": eval_dict_np[0] / model.world_size,
                            "avg_acc": eval_dict_np[1] / model.world_size,
                            "cls_acc": eval_dict_np[2:] / model.world_size
                        }
                    
                    # Update main process results
                    if model.is_main_process():
                        eval_results["test_acc"].append(eval_dict['avg_acc'])
                        eval_results["cls_acc"].append(eval_dict['cls_acc'])
                        eval_results["data_cnt"].append(num_eval)
                        exposed_classes_records.append(model.exposed_classes)
                        
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
                            model.report_test(samples_cnt, eval_dict, metrics_dict)
                        else:
                            model.report_test(samples_cnt, eval_dict)

                    # Update the next evaluation sample count
                    num_eval += args.eval_period
                    
                sys.stdout.flush() # Ensure output consistency across evaluations

        ## End of Online Training
        model.online_after_train()
        
        print("Report final result")

    # Save and summarize final metrics in the main process
    if model.is_main_process():     
        if args.eval_period is not None:
            # Anytime evaluation metrics
            log_path = f"{args.log_path}/logs/{args.online_scenario}/{args.dataset}/{args.model}"
            np.save(f"{log_path}/results_seed_{args.seed}_eval.npy", eval_results)
            np.save(f"{log_path}/cls_acc_seed_{args.seed}_eval.npy", eval_results["cls_acc"])
            np.save(f"{log_path}/test_acc_seed_{args.seed}_eval.npy", eval_results["test_acc"])
            np.save(f"{log_path}/data_cnt_seed_{args.seed}_eval_time.npy", eval_results["data_cnt"])

        # Calculate final summary metrics
        A_auc = np.mean(eval_results["test_acc"])
        A_last = eval_results["test_acc"][-1]
        F_auc = np.mean(eval_results["instant_fgt"])
        F_last = eval_results["last_fgt"][-1]
        F_last_auc = np.mean(eval_results["last_fgt"])
        KLR_avg = np.mean(eval_results['klr'][1:])
        KGR_avg = np.mean(eval_results['kgr'])

        # Print summary
        print(f"======== Summary =======")
        print(f"A_auc {A_auc} | A_last {A_last} | F_auc {F_auc} | F_last {F_last} | F_last_auc {F_last_auc} | KGR_avg {KGR_avg} | KLR_avg {KLR_avg}")
        print(f"="*24)
        
        # Log results to wandb, if enabled
        if not args.nowand:
            wandb.log({
                "A_auc": A_auc, "A_last": A_last, 
                "F_auc": F_auc, "F_last": F_last, "F_last_auc": F_last_auc,
                "KGR_avg": KGR_avg, "KLR_avg": KLR_avg
            })
            if hasattr(model, "table"):
                wandb.log({"Prompt_distribution": model.table})

        # Save model checkpoint, if enabled
        if args.savecheck:
            save_obj = {
                'model': model.state_dict(),
                'args': args,
                'results': [A_auc, A_last, F_auc, F_last, F_last_auc, KGR_avg, KLR_avg],
                'optimizer': model.opt.state_dict() if hasattr(model, 'opt') else None,
                'scheduler': model.scheduler.state_dict() if model.scheduler is not None else None,
            }
            # Save buffer if it exists
            if 'buffer_size' in model.args:
                save_obj['buffer'] = copy.deepcopy(model.buffer).to('cpu')

            # Saving model checkpoint
            checkpoint_name = f'checkpoints/{args.ckpt_name}_last.pt'
            if checkpoint_name is not None:
                torch.save(save_obj, checkpoint_name)

        # Display system stats
        system_tracker.print_stats()

    if not args.disable_log:
        logger.write(vars(args)) # fixed error by removing base_path from the logger.write function
        if not args.nowand:
            d = logger.dump()
            d['wandb_url'] = wandb.run.get_url()
            wandb.log(d)

    if not args.nowand:
        wandb.finish()
