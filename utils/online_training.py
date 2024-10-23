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
from datasets.utils.online_sampler import OnlineSampler, OnlineTestSampler
from datasets.utils.continual_dataset import ContinualDataset, MammothDatasetWrapper
from datasets.utils.gcl_dataset import GCLDataset
from models.utils.online_continual_model import OnlineContinualModel
from models.utils.future_model import FutureModel

from utils.checkpoints import mammoth_load_checkpoint
from utils.loggers import log_extra_metrics, log_online_accs, log_accs, Logger, OnlineLogger
from utils.schedulers import get_scheduler
from utils.stats import track_system_stats

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


def train(model: OnlineContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.

    Args:
        model: the module to be trained
        dataset: the continual dataset at hand
        args: the arguments of the current execution
    """
    print(args)
    os.makedirs(f"{args.log_path}/logs/{args.dataset}/{args.notes}", exist_ok=True)
    
    # TODO: improve this implementation
    if args.online_scenario in ['si-blurry']: 
        dataset.SETTING = 'online-il'
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
    ptm_results_class = []

    model.net.to(model.device)
    torch.cuda.empty_cache()
    
    # dataset setup
    total_samples = len(dataset.train_dataset)
    train_dataset = IndexedDataset(dataset.train_dataset)
    test_dataset = dataset.test_dataset
    train_sampler = OnlineSampler(train_dataset, dataset.N_TASKS, args.m, args.n, args.seed, args.rnd_NM, 1) # args.selection_size was used for prompt
    train_dataloader = DataLoader(train_dataset, batch_size=args.minibatch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.minibatch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    with track_system_stats(logger) as system_tracker:
        
        print(file=sys.stderr)
        end_task = dataset.N_TASKS
        
        print(f"Incrementally training {end_task} tasks")  
        task_records = defaultdict(list)
        eval_results = defaultdict(list)
        samples_cnt = 0

        num_eval = args.eval_period

        torch.cuda.empty_cache()
        # Start Task
        for task_id in range(end_task):
            if args.joint and task_id > 0:
                return
            
            print("\n" + "#" * 50)
            print(f"# Task {task_id} Session")
            print("#" * 50 + "\n")
            print("[2-1] Prepare a datalist for the current task")
            
            train_sampler.set_task(task_id)
            model.online_before_task(task_id)
            ## Start Online Training
            for i, (images, labels, not_aug_img, idx) in enumerate(train_dataloader):
                if args.debug_mode and (i+1) * args.minibatch_size//2 >= 500:
                    break
                
                samples_cnt += images.size(0) * model.world_size
                loss_dict, acc = model.online_step(images, labels, idx)
                
                assert not math.isnan(loss_dict['total_loss']), f"Loss is NaN @ task {task_id} Sample # {samples_cnt}"
                
                if args.code_optimization == 0 and 'cuda' in str(args.device):
                    torch.cuda.synchronize()
                system_tracker()
                
                model.report_training(total_samples, samples_cnt, loss_dict, acc)
                ### Anytime evaluation
                if samples_cnt > num_eval:
                    with torch.no_grad():
                        test_sampler = OnlineTestSampler(test_dataset, model.exposed_classes)
                        test_dataloader = DataLoader(test_dataset, batch_size=args.minibatch_size*2, sampler=test_sampler, num_workers=args.num_workers)
                        eval_dict = model.online_evaluate(test_dataloader)
                        if model.distributed:
                            eval_dict =  torch.tensor([eval_dict["avg_loss"], eval_dict["avg_acc"], *eval_dict["cls_acc"]], device=model.device)
                            dist.reduce(eval_dict, dst=0, op=dist.ReduceOp.SUM)
                            eval_dict = eval_dict.cpu().numpy()
                            eval_dict = {"avg_loss": eval_dict[0]/model.world_size, "avg_acc": eval_dict[1]/model.world_size, "cls_acc": eval_dict[2:]/model.world_size}
                        if model.is_main_process():
                            eval_results["test_acc"].append(eval_dict["avg_acc"])
                            eval_results["cls_acc"].append(eval_dict["cls_acc"])
                            eval_results["data_cnt"].append(num_eval)
                            model.report_test(samples_cnt, eval_dict["avg_loss"], eval_dict["avg_acc"])
                        num_eval += args.eval_period
                    ### End of each anytime evaluation
                ## End of each online training
                sys.stdout.flush()
            # End of each task
            model.report_test(samples_cnt, eval_dict["avg_loss"], eval_dict["avg_acc"], task_id=task_id) # report again the last result after task
            model.online_after_task(task_id)
            
            # Evaluate the model after the task
            test_sampler = OnlineTestSampler(test_dataset, model.exposed_classes)
            test_dataloader = DataLoader(test_dataset, batch_size=args.minibatch_size*2, sampler=test_sampler, num_workers=args.num_workers)
            task_eval_dict = model.online_evaluate(test_dataloader)
            log_online_accs(args, logger, task_eval_dict, task_id, dataset.SETTING) # Task metrics
            
            if model.distributed:
                task_eval_dict =  torch.tensor([task_eval_dict["avg_loss"], task_eval_dict["avg_acc"], *task_eval_dict["cls_acc"]], device=model.device)
                dist.reduce(task_eval_dict, dst=0, op=dist.ReduceOp.SUM)
                task_eval_dict = task_eval_dict.cpu().numpy()
                task_eval_dict = {"avg_loss": task_eval_dict[0]/model.world_size, "avg_acc": task_eval_dict[1]/model.world_size, "cls_acc": task_eval_dict[2:]/model.world_size}
            # task_acc = task_eval_dict["avg_acc"] # not used

            print("[2-4] Update the information for the current task")
            task_records["task_acc"].append(task_eval_dict["avg_acc"])
            task_records["cls_acc"].append(task_eval_dict["cls_acc"])
            
            print("[2-5] Report task result")
            
        if model.is_main_process():     
            # Task metrics   
            np.save(f"{args.log_path}/logs/{args.dataset}/{args.notes}/task_acc_seed_{args.seed}.npy", task_records["task_acc"])
            np.save(f"{args.log_path}/logs/{args.dataset}/{args.notes}/cls_acc_seed_{args.seed}.npy", task_records["cls_acc"])

            if args.eval_period is not None:
                # Anytime evaluation metrics
                np.save(f'{args.log_path}/logs/{args.dataset}/{args.notes}/cls_acc_seed_{args.seed}_eval.npy', eval_results["cls_acc"])
                np.save(f'{args.log_path}/logs/{args.dataset}/{args.notes}/test_acc_seed_{args.seed}_eval.npy', eval_results["test_acc"])
                np.save(f'{args.log_path}/logs/{args.dataset}/{args.notes}/data_cnt_seed_{args.seed}_eval_time.npy', eval_results["data_cnt"])
    
            # Accuracy (A)
            A_auc = np.mean(eval_results["test_acc"])
            A_avg = np.mean(task_records["task_acc"])
            A_last = task_records["task_acc"][dataset.N_TASKS - 1]

            # Forgetting (F)
            cls_acc = np.array(task_records["cls_acc"])
            acc_diff = []
            for j in range(dataset.N_CLASSES):
                if np.max(cls_acc[:-1, j]) > 0:
                    acc_diff.append(np.max(cls_acc[:-1, j]) - cls_acc[-1, j])
            F_last = np.mean(acc_diff)
            
            # Forward Transfer (F)
            # li = []
            # for i in range(dataset.N_TASKS):
            #     for j in range(i):
            #         li.append(cls_acc[j, i])
            # F_last = np.mean(li)
            
            # Backward Transfer (B)
            

            print(f"======== Summary =======")
            print(f"A_auc {A_auc} | A_avg {A_avg} | A_last {A_last} | F_last {F_last}")
            # for i in range(len(cls_acc)):
            #     print(f"Task {i}")
            #     print(cls_acc[i])
            print(f"="*24)
            
            if not args.nowand:
                wandb.log({"A_auc": A_auc, "A_avg": A_avg, "A_last": A_last, "F_last": F_last})

            if args.savecheck:
                save_obj = {
                    'model': model.state_dict(),
                    'args': args,
                    'results': [A_auc, A_avg, A_last, F_last, logger.dump()],
                    'optimizer': model.opt.state_dict() if hasattr(model, 'opt') else None,
                    'scheduler': model.scheduler.state_dict() if model.scheduler is not None else None,
                }
                if 'buffer_size' in model.args:
                    save_obj['buffer'] = copy.deepcopy(model.buffer).to('cpu')

                # Saving model checkpoint for the current task
                checkpoint_name = None
                if args.savecheck == 'task':
                    checkpoint_name = f'checkpoints/{args.ckpt_name}_joint.pt' if args.joint else f'checkpoints/{args.ckpt_name}_{task_id}.pt'
                elif args.savecheck == 'last' and task_id == end_task - 1:
                    checkpoint_name = f'checkpoints/{args.ckpt_name}_joint.pt' if args.joint else f'checkpoints/{args.ckpt_name}_last.pt'
                if checkpoint_name is not None:
                    torch.save(save_obj, checkpoint_name)

        system_tracker.print_stats()

    if not args.disable_log:
        logger.write(vars(args)) # fixed error by removing base_path from the logger.write function
        if not args.nowand:
            d = logger.dump()
            d['wandb_url'] = wandb.run.get_url()
            wandb.log(d)

    if not args.nowand:
        wandb.finish()
