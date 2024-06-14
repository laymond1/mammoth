# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from datetime import datetime
from time import time
from typing import Union
import shutil


def padded_print(string: str, max_width: int, **kwargs) -> None:
    """
    Prints a string with blank spaces to reach the max_width.

    Args:
        string: the string to print
        max_width: the maximum width of the string
    """
    pad_len = max(0, max_width - len(string))
    print(string + ' ' * pad_len, **kwargs)


class ProgressBar:
    def __init__(self, joint=False, verbose=True, update_every=1):
        """
        Initializes a ProgressBar object.

        Args:
            joint: a boolean indicating whether the progress bar is for a joint task
            verbose: a boolean indicating whether to display the progress bar
            update_every: the number of iterations after which the progress bar is updated
        """
        self.joint = joint
        self.old_time = 0
        self.running_sum = 0
        self.verbose = verbose
        self.update_every = update_every
        self.last_iter = 0

        assert self.update_every > 0

    def prog(self, i: int, max_iter: int, epoch: Union[int, str],
             task_number: int, loss: float) -> None:
        """
        Prints out the progress bar on the stderr file.

        Args:
            i: the current iteration
            max_iter: the maximum number of iteration. If None, the progress bar is not printed.
            epoch: the epoch
            task_number: the task index
            loss: the current value of the loss function
        """
        max_width = shutil.get_terminal_size((80, 20)).columns
        if not self.verbose:
            if i == 0:
                if self.joint:
                    padded_print('[ {} ] Joint | epoch {}\n'.format(
                        datetime.now().strftime("%m-%d | %H:%M"),
                        epoch
                    ), max_width=max_width, file=sys.stderr, end='', flush=True)
                else:
                    padded_print('[ {} ] Task {} | epoch {}\n'.format(
                        datetime.now().strftime("%m-%d | %H:%M"),
                        task_number + 1 if isinstance(task_number, int) else task_number,
                        epoch
                    ), max_width=max_width, file=sys.stderr, end='', flush=True)
            else:
                return

        if i < self.last_iter:
            padded_print('\tLast task took: {} s'.format(round(self.running_sum, 2)), max_width=max_width, file=sys.stderr, flush=True)
        if i == 0:
            self.old_time = time()
            self.running_sum = 0
        else:
            timediff = time() - self.old_time
            self.running_sum = self.running_sum + timediff + 1e-8
            self.old_time = time()
        self.last_iter = i

        # Print the progress bar every update_every iterations
        if (i and i % self.update_every == 0) or i == max_iter - 1:
            progress = min(float((i + 1) / max_iter), 1) if max_iter else 0
            progress_bar = ('█' * int(50 * progress)) + ('┈' * (50 - int(50 * progress))) if max_iter else '~N/A~'
            if self.joint:
                padded_print('\r[ {} ] Joint | epoch {}: |{}| {} ep/h | loss: {} | Time: {} ms/it'.format(
                    datetime.now().strftime("%m-%d | %H:%M"),
                    epoch,
                    progress_bar,
                    round(3600 / (self.running_sum / i * max_iter), 2) if max_iter else 'N/A',
                    round(loss, 8),
                    round(1000 * timediff / self.update_every, 2)
                ), max_width=max_width, file=sys.stderr, end='', flush=True)
            else:
                padded_print('\r[ {} ] Task {} | epoch {}: |{}| {} ep/h | loss: {} | Time: {} ms/it'.format(
                    datetime.now().strftime("%m-%d | %H:%M"),
                    task_number + 1 if isinstance(task_number, int) else task_number,
                    epoch,
                    progress_bar,
                    round(3600 / (self.running_sum / i * max_iter), 2) if max_iter else 'N/A',
                    round(loss, 8),
                    round(1000 * timediff / self.update_every, 2)
                ), max_width=max_width, file=sys.stderr, end='', flush=True)

    def __del__(self):
        max_width = shutil.get_terminal_size((80, 20)).columns
        # if self.verbose:
        #     print('\n', file=sys.stderr, flush=True)
        padded_print('\tLast task took: {} s'.format(round(self.running_sum, 2)), max_width=max_width, file=sys.stderr, flush=True)


def progress_bar(i: int, max_iter: int, epoch: Union[int, str],
                 task_number: int, loss: float) -> None:
    """
    Prints out the progress bar on the stderr file.

    Args:
        i: the current iteration
        max_iter: the maximum number of iteration
        epoch: the epoch
        task_number: the task index
        loss: the current value of the loss function
    """
    global static_bar

    if i == 0:
        static_bar = ProgressBar()
    static_bar.prog(i, max_iter, epoch, task_number, loss)
