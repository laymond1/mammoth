# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


def backward_transfer(results):
    """
    Calculates the backward transfer metric.

    Args:
        results (list): A list of lists representing the results of all classes of all task.

    Returns:
        float: The mean backward transfer value.
    """
    n_tasks = len(results)
    li = []
    for i in range(n_tasks - 1):
        li.append(results[-1][i] - results[i][i])

    return np.mean(li)


def forward_transfer(results, random_results):
    """
    Calculates the forward transfer metric.

    Args:
        results (list): A list of lists representing the results of all classes of all task.
        random_results (list): A list of results from a random baseline.

    Returns:
        float: The mean forward transfer value.
    """
    n_tasks = len(results)
    li = []
    for i in range(1, n_tasks):
        li.append(results[i - 1][i] - random_results[i][0])

    return np.mean(li)


def forgetting(results):
    """
    Calculates the forgetting metric.

    Args:
        results (list): A list of lists representing the results of all classes of all task.

    Returns:
        float: The mean forgetting value.
    """
    n_tasks = len(results)
    li = []
    for i in range(n_tasks - 1):
        results[i] += [0.0] * (n_tasks - len(results[i]))
    np_res = np.array(results)
    maxx = np.max(np_res, axis=0)
    for i in range(n_tasks - 1):
        li.append(maxx[i] - results[-1][i])

    return np.mean(li)


def online_forgetting(cls_accs, exposed_classes, mode='instant'):
    """
    Calculates the online forgetting metric.
    
    Args:
        cls_accs (list): A list of lists representing the accuracy of all classes.
        exposed_classes (list): A list of exposed classes.
        mode (str): The mode of online forgetting to compute. It can be 'instant' or 'last'.
    
    Returns:
        float: The online forgetting value.
    """
    # assert cur_iter > 0, "The current iteration must be greater than 0."

    num_seen_classes = len(exposed_classes)
    
    if mode == 'instant':
        acc_diff = []
        for j in range(num_seen_classes):
            acc_diff.append(cls_accs[-2][j] - cls_accs[-1][j])
        online_fgt = np.mean(acc_diff)
        
    elif mode == 'last':
        acc_diff = []
        for j in range(num_seen_classes):
            maxx = np.max(cls_accs[:-1], axis=0)
            acc_diff.append(maxx[j] - cls_accs[-1][j])
        online_fgt = np.mean(acc_diff)
        
    else:
        raise ValueError("The mode of online forgetting must be 'incremental' or 'instant'.")

    return online_fgt
        

def calculate_online_forgetting(num_classes, y_gt, y_t1, y_t2, n_cls_t1, n_cls_t2):
    """
    Calculates the online forgetting metric.
    
    Args:
        num_classes (int): The number of classes.
        y_gt (list): A list of ground truth labels.
        y_t1 (list): A list of predicted labels from the t1.
        y_t2 (list): A list of predicted labels from the t2.
        n_cls_t1 (int): The number of classes of the t1.
        n_cls_t2 (int): The number of classes of the t2.
        
    Returns:
        float: The online forgetting value: knowledge_loss_rate and knowledge_gain_rate.
    """
    n_classes = num_classes
    total_cnt = len(y_gt)
    cnt_gt = np.zeros(n_classes)
    cnt_y1 = np.zeros(n_cls_t1)
    cnt_y2 = np.zeros(n_cls_t2)
    correct_y1 = np.zeros(n_classes)
    correct_y2 = np.zeros(n_classes)
    correct_both = np.zeros(n_classes)
    for i, gt in enumerate(y_gt):
        y1, y2 = y_t1[i], y_t2[i]
        cnt_gt[gt] += 1
        cnt_y1[y1] += 1
        cnt_y2[y2] += 1
        if y1 == gt:
            correct_y1[gt] += 1
            if y2 == gt:
                correct_y2[gt] += 1
                correct_both[gt] += 1
        elif y2 == gt:
            correct_y2[gt] += 1

    gt_prob = cnt_gt/total_cnt
    y1_prob = cnt_y1/total_cnt
    y2_prob = cnt_y2/total_cnt

    probs = np.zeros([n_classes, n_cls_t1, n_cls_t2])

    for i in range(n_classes):
        cls_prob = gt_prob[i]
        notlearned_prob = 1 - (correct_y1[i] + correct_y2[i] - correct_both[i]) / cnt_gt[i]
        forgotten_prob = (correct_y1[i] - correct_both[i]) / cnt_gt[i]
        newlearned_prob = (correct_y2[i] - correct_both[i]) / cnt_gt[i]
        
        if i < n_cls_t1:
            marginal_y1 = y1_prob/(1-y1_prob[i])
            marginal_y1[i] = forgotten_prob/(notlearned_prob+1e-10)
        else:
            marginal_y1 = y1_prob
        if i < n_cls_t2:
            marginal_y2 = y2_prob/(1-y2_prob[i])
            marginal_y2[i] = newlearned_prob/(notlearned_prob+1e-10)
        else:
            marginal_y2 = y2_prob
        probs[i] = np.expand_dims(marginal_y1, 1) * np.expand_dims(marginal_y2, 0) * notlearned_prob * cls_prob
        if i < n_cls_t1 and i < n_cls_t2:
            probs[i][i][i] = correct_both[i]/total_cnt

    knowledge_loss = np.sum(probs*np.log(np.sum(probs, axis=(0, 1), keepdims=True) * probs / (np.sum(probs, axis=0, keepdims=True)+1e-10) / (np.sum(probs, axis=1, keepdims=True)+1e-10)+1e-10))/np.log(n_classes)
    knowledge_gain = np.sum(probs*np.log(np.sum(probs, axis=(0, 2), keepdims=True) * probs / (np.sum(probs, axis=0, keepdims=True)+1e-10) / (np.sum(probs, axis=2, keepdims=True)+1e-10)+1e-10))/np.log(n_classes)
    prob_gt_y1 = probs.sum(axis=2)
    prev_total_knowledge = np.sum(prob_gt_y1*np.log(prob_gt_y1/(np.sum(prob_gt_y1, axis=0, keepdims=True)+1e-10)/(np.sum(prob_gt_y1, axis=1, keepdims=True)+1e-10)+1e-10))/np.log(n_classes)
    max_knowledge = np.log(n_cls_t2)/np.log(n_classes)

    knowledge_loss_rate = knowledge_loss/prev_total_knowledge
    knowledge_gain_rate = knowledge_gain/(max_knowledge-prev_total_knowledge)
    
    return knowledge_loss_rate, knowledge_gain_rate