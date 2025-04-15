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
from typing import Iterable
import logging
import torch
from tqdm import tqdm

from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset, MammothDatasetWrapper
from datasets.utils.gcl_dataset import GCLDataset
from models.utils.continual_model import ContinualModel
from models.utils.future_model import FutureModel

from utils import disable_logging
from utils.checkpoints import mammoth_load_checkpoint, save_mammoth_checkpoint
from utils.loggers import log_extra_metrics, Logger
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


def _to_device(name: str, x, device):
    if isinstance(x, torch.Tensor):
        if 'label' in name.lower() or 'target' in name.lower():
            return x.to(device, dtype=torch.long)
        return x.to(device)
    return x

# TODO: complete the implementation, warning this for Cars196
def set_task_dataset(args: Namespace, dataset: ContinualDataset) -> None:
    if dataset.N_TASKS != args.n_tasks:
        dataset.N_TASKS = args.n_tasks
        dataset.N_CLASSES_PER_TASK = dataset.N_CLASSES // args.n_tasks


def train_single_batch(model: ContinualModel,
                       batch_data,
                       args: Namespace,
                       system_tracker=None,
                       scheduler=None) -> None:
    """
    Trains the model on a single batch.

    Args:
        model: the model to be trained
        batch_data: the data for the current batch
        args: the arguments from the command line
        system_tracker: the system tracker to monitor the system stats
        scheduler: the scheduler for the current batch
    """
    inputs, labels, not_aug_inputs = batch_data[0], batch_data[1], batch_data[2]
    inputs, labels = inputs.to(model.device), labels.to(model.device, dtype=torch.long)
    not_aug_inputs = not_aug_inputs.to(model.device)

    extra_fields = {
        batch_data.dataset.extra_return_fields[k]: _to_device(batch_data.dataset.extra_return_fields[k], batch_data[3 + k], model.device)
        for k in range(len(batch_data) - 3)
    }

    loss = model.meta_observe(inputs, labels, not_aug_inputs, **extra_fields)

    assert not math.isnan(loss)

    if scheduler is not None and args.scheduler_mode == 'iter':
        scheduler.step()

    if args.code_optimization == 0 and 'cuda' in str(args.device):
        torch.cuda.synchronize()
    system_tracker()

    return loss

# revised code
def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process for online continual learning.

    Args:
        model: the module to be trained
        dataset: the continual dataset at hand
        args: the arguments of the current execution
    """
    print(args)

    is_fwd_enabled = True
    can_compute_fwd_beforetask = True
    random_results_class, random_results_task = [], []

    if not args.nowand:
        initialize_wandb(args)

    if not args.disable_log:
        logger = Logger(args, dataset.SETTING, dataset.NAME, model.NAME)

    # set_task_dataset(args, dataset)

    model.net.to(model.device)
    torch.cuda.empty_cache()

    with track_system_stats(logger) as system_tracker:
        results, results_mask_classes = [], []

        if args.eval_future:
            results_transf, results_mask_classes_transf = [], []

        start_task = 0 if args.start_from is None else args.start_from
        end_task = dataset.N_TASKS if args.stop_after is None else args.stop_after

        for t in range(start_task, end_task):
            model.net.train()
            train_loader, _ = dataset.get_data_loaders()

            if not issubclass(dataset.__class__, GCLDataset):
                assert issubclass(train_loader.dataset.__class__, MammothDatasetWrapper), "Dataset must be an instance of MammothDatasetWrapper (did you forget to call the `store_masked_loaders`?)"

            model.meta_begin_task(dataset)

            scheduler = get_scheduler(model, args, reload_optim=True) if not hasattr(model, 'scheduler') else model.scheduler

            n_iterations = len(train_loader) * model.args.n_iters
            mininterval = 0.2 if n_iterations is not None and n_iterations > 1000 else 0.1
            train_pbar = tqdm(total=n_iterations, disable=args.non_verbose, mininterval=mininterval)

            if args.non_verbose:
                logging.info(f"Task {t + 1}")  # at least print the task number

            for i, batch_data in enumerate(train_loader):
                # Iterate over the current batch for `model.args.n_iters`
                for _iter in range(model.args.n_iters):
                    train_pbar.set_description(f"Task {t + 1} - Batch {i + 1}, Iteration {_iter + 1}")
                    loss = train_single_batch(model, batch_data, args, system_tracker=system_tracker, scheduler=scheduler)
                    train_pbar.set_postfix({'loss': loss, 'lr': model.opt.param_groups[0]['lr']}, refresh=False)
                    train_pbar.update(1)

            train_pbar.close()

            model.meta_end_task(dataset)

            accs = dataset.evaluate(model, dataset)

            logged_accs = dataset.log(args, logger, accs, t, dataset.SETTING)

            if dataset.SETTING != 'biased-class-il':
                results.append(accs[0])
                results_mask_classes.append(accs[1])
            else:
                results.append(logged_accs[0])  # avg
                results_mask_classes.append(logged_accs[1])  # worst

            if args.savecheck:
                save_mammoth_checkpoint(t, end_task, args,
                                        model,
                                        results=[results, results_mask_classes, logger.dump()],
                                        optimizer_st=model.opt.state_dict() if hasattr(model, 'opt') else None,
                                        scheduler_st=scheduler.state_dict() if scheduler is not None else None)

        system_tracker.print_stats()

    if not args.disable_log:
        logger.write(vars(args))
        if not args.nowand:
            d = logger.dump()
            d['wandb_url'] = wandb.run.get_url()
            wandb.log(d)

    if not args.nowand:
        wandb.finish()
