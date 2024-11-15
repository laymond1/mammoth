# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
This module contains the Logger class and related functions for logging accuracy values and other metrics.
"""

from contextlib import suppress
import os
import sys
from typing import Any, Dict

import numpy as np

from utils import create_if_not_exists, smart_joint
from utils.conf import base_path
from utils.metrics import backward_transfer, forward_transfer, forgetting
with suppress(ImportError):
    import wandb


def log_online_accs(args, logger, sample_num, eval_dict, t: int, setting: str, prefix="RESULT", future=False):
    """
    Logs the online loss and accuracy values.

    All metrics are prefixed with `prefix` to be logged on wandb.

    Args:
        args: The arguments for logging.
        logger: The Logger object.
        eval_dict: The evaluation metrics.
        t: The task index.
        setting: The setting of the benchmark (e.g., `si-blurry`).
        prefix: The prefix for the metrics (default="RESULT").
    """

    avg_acc, cls_acc = eval_dict['avg_acc'], eval_dict['cls_acc'] # TODO: munipulate this function

    if not args.disable_log:
        logger.log_avg_acc(avg_acc)
        logger.log_cls_acc(cls_acc)

    if not args.nowand:
        if future:
            prefix += "_transf"
        
        d2 = {f'{prefix}_task_loss': eval_dict['avg_loss'],
              f'{prefix}_task_acc': eval_dict['avg_acc'],
              'Task': t}

        wandb.log(d2, step=sample_num)


def log_accs(args, logger, accs, t: int, setting: str, epoch=None, prefix="RESULT", future=False):
    """
    Logs the accuracy values and other metrics.

    All metrics are prefixed with `prefix` to be logged on wandb.

    Args:
        args: The arguments for logging.
        logger: The Logger object.
        accs: The accuracy values.
        t: The task index.
        setting: The setting of the benchmark (e.g., `class-il`).
        epoch: The epoch number (optional).
        prefix: The prefix for the metrics (default="RESULT").
    """

    mean_acc = print_mean_accuracy(accs, t + 1 if isinstance(t, (float, int)) else t,
                                   setting, joint=args.joint,
                                   epoch=epoch, future=future)

    if not args.disable_log:
        logger.log(mean_acc)
        logger.log_fullacc(accs)

    if not args.nowand:
        postfix = "" if epoch is None else f"_epoch_{epoch}"
        if future:
            prefix += "_transf"
        if isinstance(mean_acc, float):  # domain or gcl
            d2 = {f'{prefix}_domain_mean_accs{postfix}': mean_acc,
                  **{f'{prefix}_domain_acc_{i}{postfix}': a for i, a in enumerate(accs[0])},
                  'Task': t}
        else:
            d2 = {f'{prefix}_class_mean_accs{postfix}': mean_acc[0], f'{prefix}_task_mean_accs{postfix}': mean_acc[1],
                  **{f'{prefix}_class_acc_{i}{postfix}': a for i, a in enumerate(accs[0])},
                  **{f'{prefix}_task_acc_{i}{postfix}': a for i, a in enumerate(accs[1])},
                  'Task': t}

        wandb.log(d2)


def log_extra_metrics(args, metric: float, metric_mask_class: float, metric_name: str, t: int, prefix="RESULT"):
    """
    Logs the accuracy values and other metrics.

    All metrics are prefixed with `prefix` to be logged on wandb.

    Args:
        args: The arguments for logging.
        metric: Class-IL version of the metric `metric_name`.
        metric_mask_class: Task-IL version of the metric `metric_name`.
        metric_name: The name of the metric.
        t: The task index.
        epoch: The epoch number (optional).
        prefix: The prefix for the metrics (default="RESULT").
    """

    print(f'{metric_name}: [Class-IL]: {metric:.2f} \t [Task-IL]: {metric_mask_class:.2f}', file=sys.stderr)
    print(f'\tRaw {metric_name} values: Class-IL {metric} | Task-IL {metric_mask_class}', file=sys.stderr)

    log_dict = {f'{prefix}_class_{metric_name}': metric, f'{prefix}_task_{metric_name}': metric_mask_class, 'Task': t}

    if not args.nowand:
        wandb.log(log_dict)


def print_mean_accuracy(accs: np.ndarray, task_number: int,
                        setting: str, joint=False, epoch=None, future=False) -> None:
    """
    Prints the mean accuracy on stderr.

    Args:
        accs: accuracy values per task
        task_number: task index
        setting: the setting of the benchmark
        joint: whether it's joint accuracy or not
        epoch: the epoch number (optional)

    Returns:
        The mean accuracy value.
    """
    mean_acc = np.mean(accs, axis=1)

    if joint:
        prefix = "Joint Accuracy" if epoch is None else f"Joint Accuracy (epoch {epoch})"
        if setting == 'domain-il' or setting == 'general-continual':
            mean_acc, _ = mean_acc
            print('{}: \t [Domain-IL]: {} %'.format(prefix, round(mean_acc, 2), file=sys.stderr))
            print('\tRaw accuracy values: Domain-IL {}'.format(accs[0]), file=sys.stderr)
        else:
            mean_acc_class_il, mean_acc_task_il = mean_acc
            print('{}: \t [Class-IL]: {} % \t [Task-IL]: {} %'.format(prefix, round(
                mean_acc_class_il, 2), round(mean_acc_task_il, 2)), file=sys.stderr)
            print('\tRaw accuracy values: Class-IL {} | Task-IL {}'.format(accs[0], accs[1]), file=sys.stderr)
    else:
        prefix = "Accuracy" if epoch is None else f"Accuracy (epoch {epoch})"
        prefix = "Future " + prefix if future else prefix
        if setting == 'domain-il' or setting == 'general-continual':
            mean_acc, _ = mean_acc
            print('{} for {} task(s): [Domain-IL]: {} %'.format(prefix,
                                                                task_number, round(mean_acc, 2)), file=sys.stderr)
            print('\tRaw accuracy values: Domain-IL {}'.format(accs[0]), file=sys.stderr)
        else:
            mean_acc_class_il, mean_acc_task_il = mean_acc
            print('{} for {} task(s): \t [Class-IL]: {} % \t [Task-IL]: {} %'.format(prefix, task_number, round(
                mean_acc_class_il, 2), round(mean_acc_task_il, 2)), file=sys.stderr)
            print('\tRaw accuracy values: Class-IL {} | Task-IL {}'.format(accs[0], accs[1]), file=sys.stderr)
    print('\n', file=sys.stderr)
    return mean_acc


# def print_online_mean_accuracy(eval_dict, task_number: int, setting: str) -> None:
    

class BaseLogger:
    def __init__(self, args, setting_str: str, dataset_str: str, model_str: str) -> None:
        """
        Initializes a BaseLogger object. This will take track and log the accuracy values and other metrics.

        Args:
            args: The args from the command line.
            setting_str: The setting of the benchmark.
            dataset_str: The dataset used.
            model_str: The model used.
        """
        self.args = args
        self.setting = setting_str
        self.dataset = dataset_str
        self.model = model_str
        self.cpu_res = []
        self.gpu_res = []

    def dump(self):
        """
        Dumps the state of the logger in a dictionary.

        Returns:
            A dictionary containing the logged values.
        """
        raise NotImplementedError("This method should be overridden in a subclass")

    def load(self, dic):
        """
        Loads the state of the logger from a dictionary.

        Args:
            dic: The dictionary containing the logged values.
        """
        raise NotImplementedError("This method should be overridden in a subclass")

    def rewind(self, num):
        """
        Rewinds the logger by a given number of values.

        Args:
            num: The number of values to rewind.
        """
        raise NotImplementedError("This method should be overridden in a subclass")

    def log(self, mean_acc):
        """
        Logs a mean accuracy value.

        Args:
            mean_acc: mean accuracy value
        """
        raise NotImplementedError("This method should be overridden in a subclass")

    def log_fullacc(self, accs):
        """
        Logs all the accuracy of the classes from the current and past tasks.

        Args:
            accs: the accuracy values
        """
        raise NotImplementedError("This method should be overridden in a subclass")

    def log_system_stats(self, cpu_res, gpu_res):
        """
        Logs the system stats.
        Supported only if the psutil and torch libraries are installed.

        Args:
            cpu_res: the CPU memory usage
            gpu_res: the GPU memory usage
        """
        raise NotImplementedError("This method should be overridden in a subclass")

    def write(self, args: dict) -> None:
        """
        Writes out the logged value along with its arguments in the default path (data/results).
        The default path can be changed with the --results_path argument.

        Args:
            args: the namespace of the current experiment
        """
        raise NotImplementedError("This method should be overridden in a subclass")


class Logger(BaseLogger):
    def __init__(self, args, setting_str: str, dataset_str: str,
                 model_str: str) -> None:
        """
        Initializes a Logger object. This will take track and log the accuracy values and other metrics in the default path (`data/results`).
        The default path can be changed with the `--results_path` argument.

        Args:
            args: The args from the command line.
            setting_str: The setting of the benchmark.
            dataset_str: The dataset used.
            model_str: The model used.
        """
        super().__init__(args, setting_str, dataset_str, model_str)
        self.accs = []
        self.fullaccs = []
        if setting_str == 'class-il':
            self.accs_mask_classes = []
            self.fullaccs_mask_classes = []
        self.fwt = None
        self.fwt_mask_classes = None
        self.bwt = None
        self.bwt_mask_classes = None
        self.forgetting = None
        self.forgetting_mask_classes = None

    def dump(self):
        """
        Dumps the state of the logger in a dictionary.

        Returns:
            A dictionary containing the logged values.
        """
        dic = {
            'accs': self.accs,
            'fullaccs': self.fullaccs,
            'fwt': self.fwt,
            'bwt': self.bwt,
            'forgetting': self.forgetting,
            'fwt_mask_classes': self.fwt_mask_classes,
            'bwt_mask_classes': self.bwt_mask_classes,
            'forgetting_mask_classes': self.forgetting_mask_classes,
        }
        if self.setting == 'class-il':
            dic['accs_mask_classes'] = self.accs_mask_classes
            dic['fullaccs_mask_classes'] = self.fullaccs_mask_classes

        return dic

    def load(self, dic):
        """
        Loads the state of the logger from a dictionary.

        Args:
            dic: The dictionary containing the logged values.
        """
        self.accs = dic['accs']
        self.fullaccs = dic['fullaccs']
        self.fwt = dic['fwt']
        self.bwt = dic['bwt']
        self.forgetting = dic['forgetting']
        self.fwt_mask_classes = dic['fwt_mask_classes']
        self.bwt_mask_classes = dic['bwt_mask_classes']
        self.forgetting_mask_classes = dic['forgetting_mask_classes']
        if self.setting == 'class-il':
            self.accs_mask_classes = dic['accs_mask_classes']
            self.fullaccs_mask_classes = dic['fullaccs_mask_classes']

    def rewind(self, num):
        """
        Rewinds the logger by a given number of values.

        Args:
            num: The number of values to rewind.
        """
        self.accs = self.accs[:-num]
        self.fullaccs = self.fullaccs[:-num]
        with suppress(BaseException):
            self.fwt = self.fwt[:-num]
            self.bwt = self.bwt[:-num]
            self.forgetting = self.forgetting[:-num]
            self.fwt_mask_classes = self.fwt_mask_classes[:-num]
            self.bwt_mask_classes = self.bwt_mask_classes[:-num]
            self.forgetting_mask_classes = self.forgetting_mask_classes[:-num]

        if self.setting == 'class-il':
            self.accs_mask_classes = self.accs_mask_classes[:-num]
            self.fullaccs_mask_classes = self.fullaccs_mask_classes[:-num]

    def add_fwt(self, results, accs, results_mask_classes, accs_mask_classes):
        """
        Adds forward transfer values.

        Args:
            results: The results.
            accs: The accuracy values.
            results_mask_classes: The results for masked classes.
            accs_mask_classes: The accuracy values for masked classes.
        """
        self.fwt = forward_transfer(results, accs)
        if self.setting == 'class-il':
            self.fwt_mask_classes = forward_transfer(results_mask_classes, accs_mask_classes)

        return self.fwt, self.fwt_mask_classes

    def add_bwt(self, results, results_mask_classes):
        """
        Adds backward transfer values.

        Args:
            results: The results.
            results_mask_classes: The results for masked classes.
        """
        self.bwt = backward_transfer(results)
        if self.setting == 'class-il':
            self.bwt_mask_classes = backward_transfer(results_mask_classes)

        return self.bwt, self.bwt_mask_classes

    def add_forgetting(self, results, results_mask_classes):
        """
        Adds forgetting values.

        Args:
            results: The results.
            results_mask_classes: The results for masked classes.
        """
        self.forgetting = forgetting(results)
        if self.setting == 'class-il':
            self.forgetting_mask_classes = forgetting(results_mask_classes)

        return self.forgetting, self.forgetting_mask_classes

    def log(self, mean_acc: np.ndarray) -> None:
        """
        Logs a mean accuracy value.

        Args:
            mean_acc: mean accuracy value
        """
        if self.setting == 'general-continual':
            self.accs.append(mean_acc)
        elif self.setting == 'domain-il':
            self.accs.append(mean_acc)
        else:
            mean_acc_class_il, mean_acc_task_il = mean_acc
            self.accs.append(mean_acc_class_il)
            self.accs_mask_classes.append(mean_acc_task_il)

    def log_fullacc(self, accs):
        """
        Logs all the accuracy of the classes from the current and past tasks.

        Args:
            accs: the accuracy values
        """
        if self.setting == 'class-il':
            acc_class_il, acc_task_il = accs
            self.fullaccs.append(acc_class_il)
            self.fullaccs_mask_classes.append(acc_task_il)

    def log_system_stats(self, cpu_res, gpu_res):
        """
        Logs the system stats.
        Supported only if the `psutil` and `torch` libraries are installed.

        Args:
            cpu_res: the CPU memory usage
            gpu_res: the GPU memory usage
        """
        if cpu_res is not None:
            self.cpu_res.append(cpu_res)
        if gpu_res is not None:
            self.gpu_res.append(gpu_res)
            gpu_res = {f'GPU_{i}_memory_usage': r for i, r in gpu_res.items()}
        else:
            gpu_res = {}

        if not self.args.nowand:
            wandb.log({'CPU_memory_usage': cpu_res, **gpu_res})

    def write(self, args: Dict[str, Any]) -> None:
        """
        Writes out the logged value along with its arguments in the default path (`data/results`).
        The default path can be changed with the `--results_path` argument.

        Args:
            args: the namespace of the current experiment
        """
        wrargs = args.copy()

        for i, acc in enumerate(self.accs):
            wrargs['accmean_task' + str(i + 1)] = acc

        for i, fa in enumerate(self.fullaccs):
            for j, acc in enumerate(fa):
                wrargs['accuracy_' + str(j + 1) + '_task' + str(i + 1)] = acc

        wrargs['cpu_memory_usage'] = self.cpu_res
        wrargs['gpu_memory_usage'] = self.gpu_res

        wrargs['forward_transfer'] = self.fwt
        wrargs['backward_transfer'] = self.bwt
        wrargs['forgetting'] = self.forgetting

        target_folder = smart_joint(self.args.results_path)

        create_if_not_exists(smart_joint(target_folder, self.setting))
        create_if_not_exists(smart_joint(target_folder, self.setting, self.dataset))
        create_if_not_exists(smart_joint(target_folder, self.setting, self.dataset, self.model))

        path = smart_joint(target_folder, self.setting, self.dataset, self.model, "logs.pyd")
        print("Logging results and arguments in " + path)
        with open(path, 'a') as f:
            f.write(str(wrargs) + '\n')

        if self.setting == 'class-il':
            create_if_not_exists(smart_joint(target_folder, "task-il/", self.dataset))
            create_if_not_exists(smart_joint(target_folder, "task-il", self.dataset, self.model))

            for i, acc in enumerate(self.accs_mask_classes):
                wrargs['accmean_task' + str(i + 1)] = acc

            for i, fa in enumerate(self.fullaccs_mask_classes):
                for j, acc in enumerate(fa):
                    wrargs['accuracy_' + str(j + 1) + '_task' + str(i + 1)] = acc

            wrargs['forward_transfer'] = self.fwt_mask_classes
            wrargs['backward_transfer'] = self.bwt_mask_classes
            wrargs['forgetting'] = self.forgetting_mask_classes

            path = smart_joint(target_folder, "task-il", self.dataset, self.model, "logs.pyd")
            with open(path, 'a') as f:
                f.write(str(wrargs) + '\n')


class OnlineLogger(BaseLogger):
    """
        Online Logger.
    """
    def __init__(self, args, setting_str: str, dataset_str: str,
                 model_str: str, **kwargs):
        super().__init__(args, setting_str, dataset_str, model_str, **kwargs)   
        self.avg_acc = []
        self.cls_acc = []
        self.anytime_fwt =None
        self.anytime_bwt =None
        self.anytime_forgetting =None
        self.fwt = None
        self.bwt = None
        self.forgetting =None
        
    def dump(self):
        dic = {
            'avg_acc': self.avg_acc,
            'cls_acc': self.cls_acc,
            'fwt': self.fwt,
            'bwt': self.bwt,
            'forgetting': self.forgetting
        }
        
        return dic
    
    def load(self, dic):
        self.avg_acc = dic['avg_acc']
        self.cls_acc = dic['cls_acc']
        self.fwt = dic['fwt']
        self.bwt = dic['bwt']
        self.forgetting = dic['forgetting']
    
    def rewind(self, num):
        self.avg_acc = self.avg_acc[:-num]
        self.cls_acc = self.cls_acc[:-num]
        with suppress(BaseException):
            self.fwt = self.fwt[:-num]
            self.bwt = self.bwt[:-num]
            self.forgetting = self.forgetting[:-num]
    
    def _anytime_forward_transfer(self, cls_acc):
        results = self.cls_acc
        
    def _anytime_backward_transfer(self, cls_acc):
        results = self.cls_acc
        
    def _anytime_forgetting(self, seen_classes):
        cls_acc = np.array(self.cls_acc)
        acc_diff = []
        for j in range(seen_classes):
            if np.max(cls_acc[:-1, j]) > 0:
                acc_diff.append(np.max(cls_acc[:-1, j]) - cls_acc[-1, j])
        F_last = np.mean(acc_diff)
        return F_last
    
    def add_anytime_fwt(self, results, accs):
        self.anytime_fwt = self._anytime_forward_transfer(results, accs)
        return self.fwt
    
    def add_anytime_bwt(self, results):
        self.bwt = self._anytime_backward_transfer(results)
        return self.bwt
    
    def add_anytime_forgetting(self, seen_classes):
        self.anytime_forgetting = self._anytime_forgetting(seen_classes)
        return self.anytime_forgetting

    def log_avg_acc(self, avg_acc):
        self.avg_acc.append(avg_acc)
        
    def log_cls_acc(self, cls_acc):
        self.cls_acc.append(cls_acc)
    
    def log_system_stats(self, cpu_res, gpu_res):
        """
        Logs the system stats.
        Supported only if the `psutil` and `torch` libraries are installed.

        Args:
            cpu_res: the CPU memory usage
            gpu_res: the GPU memory usage
        """
        if cpu_res is not None:
            self.cpu_res.append(cpu_res)
        if gpu_res is not None:
            self.gpu_res.append(gpu_res)
            gpu_res = {f'GPU_{i}_memory_usage': r for i, r in gpu_res.items()}
        else:
            gpu_res = {}

        if not self.args.nowand:
            wandb.log({'CPU_memory_usage': cpu_res, **gpu_res})
            
    def write(self, args: Dict[str, Any]) -> None:
        """
        Writes out the logged value along with its arguments in the log path (`./results`).
        The default path can be changed with the `--results_path` argument.

        Args:
            args: the namespace of the current experiment
        """
        wrargs = args.copy()

        for i, acc in enumerate(self.avg_acc):
            wrargs['Task' + str(i + 1)] = acc

        for i, fa in enumerate(self.cls_acc):
            for j, acc in enumerate(fa):
                wrargs['accuracy_' + str(j + 1) + '_class' + str(i + 1)] = acc

        wrargs['cpu_memory_usage'] = self.cpu_res
        wrargs['gpu_memory_usage'] = self.gpu_res

        wrargs['forward_transfer'] = self.fwt
        wrargs['backward_transfer'] = self.bwt
        wrargs['forgetting'] = self.forgetting

        target_folder = smart_joint(self.args.results_path)

        create_if_not_exists(smart_joint(target_folder, self.setting))
        create_if_not_exists(smart_joint(target_folder, self.setting, self.dataset))
        create_if_not_exists(smart_joint(target_folder, self.setting, self.dataset, self.model))

        path = smart_joint(target_folder, self.setting, self.dataset, self.model, "logs.pyd")
        print("Logging results and arguments in " + path)
        with open(path, 'a') as f:
            f.write(str(wrargs) + '\n')
