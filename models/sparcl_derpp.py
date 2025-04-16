# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import copy
import pickle
import numpy as np
import torch
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from utils import binary_to_boolean_type

from models.sparcl_utils.prune_main import prune_parse_arguments, \
                                           prune_init, prune_apply_masks, prune_print_sparsity, \
                                           prune_update, \
                                           prune_apply_masks_on_grads_efficient, prune_apply_masks_on_grads_mix, prune_apply_masks_on_grads
from models.sparcl_utils.testers import test_sparsity


class SparCLDerpp(ContinualModel):
    """Continual learning via Dark Experience Replay++."""
    NAME = 'sparcl-derpp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight.')
        parser.add_argument('--beta', type=float, required=True,
                            help='Penalty weight.')
        
        # SparCL specific arguments
        parser.add_argument('--warmup', type=binary_to_boolean_type, default=False,
                            help='warm-up scheduler')
        parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='M',
                            help='warmup-lr, smaller than original lr')
        parser.add_argument('--warmup-epochs', type=int, default=0, metavar='M',
                            help='number of epochs for lr warmup')
        parser.add_argument('--gradient_efficient', type=binary_to_boolean_type, default=False,
                            help='Use gradient efficient pruning.')
        parser.add_argument('--gradient_efficient_mix', type=binary_to_boolean_type, default=True,
                            help='add gradient efficiency (mix method)')     
        parser.add_argument('--gradient_remove', type=float, default=0.1, 
                            help="extra removal for gradient efficiency")
        parser.add_argument('--gradient_sparse', type=float, default=0.80,
                            help="total gradient_sparse for training")
        parser.add_argument('--sample_frequency', type=int, default=30, 
                            help="sample frequency for gradient mask")

        prune_parse_arguments(parser)

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)
        # Initialize pruning
        prune_init(args, self.net) 
        prune_apply_masks()  # if wanted to make sure the mask is applied in retrain
        prune_print_sparsity(self.net)
        _, total_sparsity = test_sparsity(self.net, column=False, channel=False, filter=False, kernel=False)
        self.total_sparsity = total_sparsity
        log_filename_dir = os.path.dirname(self.args.log_filename)
        if not os.path.exists(log_filename_dir):
            os.makedirs(log_filename_dir)
            print("New folder {} created...".format(log_filename_dir))

    def begin_task(self, dataset):
        self.example_stats_train = {}
        # optimizer_init_lr = self.args.warmup_lr if self.args.warmup else self.args.lr # Not used
        # initialize optimizer
        self.opt = self.get_optimizer()
        # print current dataset shape
        self.full_dataset = copy.deepcopy(dataset.train_loader.dataset)
        self.train_indx = np.arange(len(dataset.train_loader.dataset.targets))
        print(f'[Task: {self.current_task+1}] Dataset shape:', dataset.train_loader.dataset.data.shape)

    def begin_epoch(self, epoch, dataset):
        self.epoch_total_size = 0
        self.epoch_correct = 0
        prune_update(epoch)
        # Dynamic Data Removal (DDR)
        #########remove data at 25 epoch, update dataset ######
        if epoch > 0 and epoch % self.args.sp_mask_update_freq == 0 and epoch <= self.args.remove_data_epoch:
            unlearned_per_presentation_all, first_learned_all = [], []
            
            _, unlearned_per_presentation, _, first_learned = self.compute_forgetting_statistics()
            print('unlearned_per_presentation', len(unlearned_per_presentation))
            print('first_learned', len(first_learned))

            unlearned_per_presentation_all.append(unlearned_per_presentation)
            first_learned_all.append(first_learned)

            print('unlearned_per_presentation_all', len(unlearned_per_presentation_all))
            print('first_learned_all', len(first_learned_all))

            # Sort examples by forgetting counts in ascending order, over one or more training runs
            ordered_examples, ordered_values, num_unforget = self.sort_examples_by_forgetting(unlearned_per_presentation_all, 
                                                                                              first_learned_all,
                                                                                              int(self.args.n_epochs))
                                                                                            #   int(self.args.n_epochs/self.n_tasks))

            # Save sorted output
            t = self.current_task
            if self.args.output_name.endswith('.pkl'):
                with open(os.path.join(self.args.output_dir, self.args.output_name + "_task_"+ str(t) + "_unforget_"+str(num_unforget)),
                        'wb') as fout:
                    pickle.dump({
                        'indices': ordered_examples,
                        'forgetting counts': ordered_values
                    }, fout)
            else:
                with open(
                        os.path.join(self.args.output_dir, self.args.output_name + "_task_"+ str(t) + "_unforget_"+str(num_unforget) + '.pkl'),
                        'wb') as fout:
                    pickle.dump({
                        'indices': ordered_examples,
                        'forgetting counts': ordered_values
                    }, fout)
            # Get the indices to remove from training
            print('epoch before ordered_examples len', len(ordered_examples))
            print('epoch before len(train_dataset.targets)', len(dataset.train_loader.dataset.targets))

            elements_to_remove = np.array(
                ordered_examples)[self.args.keep_lowest_n:self.args.keep_lowest_n + ( int(self.args.remove_n/( int(self.args.remove_data_epoch)/self.args.sp_mask_update_freq ) ) )]
            # Remove the corresponding elements
            print('elements_to_remove', len(elements_to_remove))

            self.train_indx = np.setdiff1d(self.train_indx, elements_to_remove)
            # np.setdiff1d(range(len(train_dataset.targets)), elements_to_remove)   
            print('removed train_indx', len(self.train_indx))

            # Reassign train data and labels
            dataset.train_loader.dataset.data = self.full_dataset.data[self.train_indx, :, :, :]
            dataset.train_loader.dataset.targets = np.array(self.full_dataset.targets)[self.train_indx].tolist()
            dataset.train_loader.dataset.indexes = np.array(self.full_dataset.indexes)[self.train_indx].tolist()

            print('shape', dataset.train_loader.dataset.data.shape)
            print('len(train_dataset.targets)', len(dataset.train_loader.dataset.targets))

            # print('epoch after random ordered_examples len', len(ordered_examples))
            #####empty example_stats_train!!! Because in original, forget process come before the whole training process
            self.example_stats_train = {}

    def observe(self, inputs, labels, not_aug_inputs, epoch=None, indexes=None):

        self.opt.zero_grad()

        outputs = self.net(inputs)

        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty() and self.current_task > 0:
            buf_inputs, _, buf_logits = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)

            buf_outputs = self.net(buf_inputs)
            loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
            loss += loss_mse

            buf_inputs, buf_labels, _ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)

            buf_outputs = self.net(buf_inputs)
            loss_ce = self.args.beta * self.loss(buf_outputs, buf_labels)
            loss += loss_ce

        # Statistics for Dynamic Data Removal (DDR)
        preds = torch.argmax(outputs[:, :self.n_seen_classes], dim=1)
        acc = preds == labels
        # Compute missclassification margin
        output_correct_class = outputs[torch.arange(outputs.size(0)), labels]
        sorted_outputs, _ = torch.sort(outputs, descending=True)
        
        for i in range(acc.size(0)):
            if acc[i]:
                # Example classified correctly, highest incorrect class is 2nd largest output
                output_highest_incorrect_class = sorted_outputs[i][2]
            else:
                # Example misclassified, highest incorrect class is max output
                output_highest_incorrect_class = sorted_outputs[i][1]
            margin = output_correct_class[i].item() - output_highest_incorrect_class.item()
            # Add the statistics of the current training example to dictionary
            index_stats = self.example_stats_train.get(indexes[i].item(), [[], [], []])
            index_stats[0].append(loss.item()) # TODO: 
            index_stats[1].append(acc[i].sum().item())
            index_stats[2].append(margin)
            self.example_stats_train[indexes[i].item()] = index_stats

        loss.backward()
        if self.args.gradient_efficient:
            prune_apply_masks_on_grads_efficient()
        elif self.args.gradient_efficient_mix:
            if self.epoch_iteration % self.args.sample_frequency == 0:
                prune_apply_masks_on_grads_mix()
            else:
                prune_apply_masks_on_grads_efficient()
        else:
            prune_apply_masks_on_grads()
        self.opt.step()

        prune_apply_masks()
        # Add training accuracy to dict
        self.epoch_total_size += labels.size(0)
        self.epoch_correct += acc.sum().item()
        index_stats = self.example_stats_train.get('train', [[], []])
        index_stats[1].append(100. * self.epoch_correct / float(self.epoch_total_size))
        self.example_stats_train['train'] = index_stats

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data)
        
        return loss.item()

    def compute_forgetting_statistics(self):
        diag_stats = self.example_stats_train
        npresentations = int(self.args.n_epochs)
        # npresentations = int(self.args.n_epochs/self.n_tasks)

        presentations_needed_to_learn = {}
        unlearned_per_presentation = {}
        margins_per_presentation = {}
        first_learned = {}
        print('len(diag_stats.items())',len(diag_stats.items()))

        for example_id, example_stats in diag_stats.items():

            # Skip 'train' and 'test' keys of diag_stats
            if not isinstance(example_id, str):

                # Forgetting event is a transition in accuracy from 1 to 0
                presentation_acc = np.array(example_stats[1][:npresentations])
                transitions = presentation_acc[1:] - presentation_acc[:-1]

                # Find all presentations when forgetting occurs
                if len(np.where(transitions == -1)[0]) > 0:
                    unlearned_per_presentation[example_id] = np.where(
                        transitions == -1)[0] + 2
                else:
                    unlearned_per_presentation[example_id] = []

                # Find number of presentations needed to learn example, 
                # e.g. last presentation when acc is 0
                if len(np.where(presentation_acc == 0)[0]) > 0:
                    presentations_needed_to_learn[example_id] = np.where(
                        presentation_acc == 0)[0][-1] + 1
                else:
                    presentations_needed_to_learn[example_id] = 0

                # Find the misclassication margin for each presentation of the example
                margins_per_presentation = np.array(
                    example_stats[2][:npresentations])

                # Find the presentation at which the example was first learned, 
                # e.g. first presentation when acc is 1
                if len(np.where(presentation_acc == 1)[0]) > 0:
                    first_learned[example_id] = np.where(
                        presentation_acc == 1)[0][0]
                else:
                    first_learned[example_id] = np.nan

        return presentations_needed_to_learn, unlearned_per_presentation, margins_per_presentation, first_learned

    def sort_examples_by_forgetting(self, unlearned_per_presentation_all,
                                    first_learned_all, 
                                    npresentations):
        # Initialize lists
        example_original_order = []
        example_stats = []

        for example_id in unlearned_per_presentation_all[0].keys():

            # Add current example to lists
            example_original_order.append(example_id)
            example_stats.append(0)

            # Iterate over all training runs to calculate the total forgetting count for current example
            for i in range(len(unlearned_per_presentation_all)):

                # Get all presentations when current example was forgotten during current training run
                stats = unlearned_per_presentation_all[i][example_id]

                # If example was never learned during current training run, add max forgetting counts
                if np.isnan(first_learned_all[i][example_id]):
                    example_stats[-1] += npresentations
                else:
                    example_stats[-1] += len(stats)

        num_unforget = len(np.where(np.array(example_stats) == 0)[0])
        print('Number of unforgettable examples: {}'.format(
            len(np.where(np.array(example_stats) == 0)[0])))
        
        return np.array(example_original_order)[np.argsort(
            example_stats)], np.sort(example_stats), num_unforget