# -*- coding: utf-8 -*-
"""
Created on Nov 28 20:00:06 2023
@author: Xukun Luan, Northeastern University, Shenyang, China.
"""

import os
import logging
import math
import random
from matplotlib import pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def mi_softmax(output, T):
    sum = 0
    temp = []
    for x in output:
        sum += math.exp(x.item()/T)
    for x in output:
        temp.append(math.exp(x/T)/sum)
    return temp

def compute_derivatives(s):  
    return np.diff(s)  

def central_diff(s):  
    h = 1.0
    return (s[2:] - s[:-2]) / (2 * h)

def fn_mentr(output, true_label, poison_label, a=0):
    Mmem = None
    for index, p in enumerate(output):
        fy = p[true_label]
        fp = p[poison_label]
        mmem = -(1-a)*(1-fy)*np.log(fy+1e-30)-(1-a)*fp*np.log(1-fp+1e-30)
        Mmem = [mmem] if index == 0 else np.concatenate((Mmem, [mmem]), axis=0)
    return np.array(Mmem)

def metrics(in_plt, out_plt, FL_params):
    y_test = [1] * FL_params.sample_num + [0] * FL_params.sample_num
    num_clust = 2
    y_pred = k_means_clust(in_plt + out_plt, num_clust, 1000, 3)
    acc = accuracy_score(y_test, y_pred)
    if acc < 0.5:
        y_test = [0] * FL_params.sample_num + [1] * FL_params.sample_num
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='binary')
    precision = precision_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')

    print("y_pred:{}".format(y_pred))
    print("accuracy_score:{}".format(acc))
    print("recall_score:{}".format(recall))
    print("precision_score:{}".format(precision))
    print("f1_score:{}".format(f1))
    logging.info("y_pred:{}".format(y_pred))
    logging.info("accuracy_score:{}".format(acc))
    logging.info("recall_score:{}".format(recall))
    logging.info("precision_score:{}".format(precision))
    logging.info("f1_score:{}".format(f1))

def plt_mentr(all_GMs, target_loader_in, target_loader_out, FL_params):

    in_plt = []
    out_plt = []
    
    in_model_output = [[] for i in range(FL_params.sample_num)]
    out_model_output = [[] for i in range(FL_params.sample_num)]

    for index_model, global_model_ in enumerate(all_GMs):
    
        with torch.no_grad(): 
            global_model_ = global_model_.to(FL_params.device_cpu)
            global_model_.eval()
            for index_data, [data, target] in enumerate(target_loader_in):
                in_output = global_model_(data.to(FL_params.device_cpu))[0]
                in_output = mi_softmax(in_output, T=1)
                in_model_output[index_data].append(in_output)

    for i in range(len(in_model_output)):
        a = fn_mentr(output=in_model_output[i], true_label=4, 
                poison_label=7, a=0)
        in_plt.append(a)
        pre_label = []
        for output_ in in_model_output[i]:
            pre_label.append(np.argmax(output_))
        plt.figure() 
        plt.title('in_'+str(FL_params.global_epoch)+'epoch_p_'+str(i))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.plot(range(len(a)), a, color='red', label='IN', alpha=0.3)  
        re_in_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 're', 'in')
        os.makedirs(re_in_dir, exist_ok=True)
        plt.savefig(os.path.join(re_in_dir, 'in_'+str(FL_params.global_epoch)+'epoch_p_'+str(i)+'.png'))
        plt.close()

    for index_model, global_model_ in enumerate(all_GMs):
    
        with torch.no_grad(): 
            global_model_ = global_model_.to(FL_params.device_cpu)
            global_model_.eval()
            for index_data, [data, target] in enumerate(target_loader_out):
                out_output = global_model_(data.to(FL_params.device_cpu))[0]
                out_output = mi_softmax(out_output, T=1)
                out_model_output[index_data].append(out_output)

    for i in range(len(out_model_output)):
        a = fn_mentr(output=out_model_output[i], true_label=4, 
                poison_label=7, a=0)
        out_plt.append(a)
        pre_label = []
        for output_ in out_model_output[i]:
            pre_label.append(np.argmax(output_))

        plt.figure() 
        plt.title('out_'+str(FL_params.global_epoch)+'epoch_p_'+str(i))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.plot(range(len(a)), a, color='blue', label='OUT', alpha=0.3)  
        re_out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 're', 'out')
        os.makedirs(re_out_dir, exist_ok=True)
        plt.savefig(os.path.join(re_out_dir, 'out_'+str(FL_params.global_epoch)+'epoch_p_'+str(i)+'.png'))
        plt.close()

    return in_plt, out_plt

def DTWDistance(s1, s2):
    DTW = {}
    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = (s1[i] - s2[j]) ** 2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])
    return math.sqrt(DTW[len(s1) - 1, len(s2) - 1])

def DTWDistance(s1, s2, w):
    DTW = {}
    w = max(w, abs(len(s1) - len(s2)))
    for i in range(-1, len(s1)):
        for j in range(-1, len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0
    for i in range(len(s1)):
        for j in range(max(0, i - w), min(len(s2), i + w)):
            dist = (s1[i] - s2[j]) ** 2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])
    return math.sqrt(DTW[len(s1) - 1, len(s2) - 1])

def DTWDistanceWithTrendWeight(s1, s2, w, trend_weight=1.0):  
    DTW = {}  
    w = max(w, abs(len(s1) - len(s2)))  
    s1_deriv = central_diff(s1)  
    s1_deriv = np.append(s1_deriv, 0)
    s2_deriv = central_diff(np.array(s2))
    s2_deriv = np.append(s2_deriv, 0)

    for i in range(-1, len(s1)):  
        for j in range(-1, len(s2)):  
            DTW[(i, j)] = float('inf')  
    DTW[(-1, -1)] = 0  
    for i in range(len(s1)):  
        for j in range(max(0, i - w), min(len(s2), i + w + 1)):
            if i == 0 or j == 0:
                dist = (s1[i] - s2[j]) ** 2
            else:
                dist = (s1[i] - s2[j]) ** 2 + trend_weight * (s1_deriv[i-1] - s2_deriv[j-1]) ** 2  
            DTW[(i, j)] = dist + min(  
                DTW.get((i-1, j), float('inf')),  
                DTW.get((i, j-1), float('inf')),  
                DTW.get((i-1, j-1), float('inf'))  
            )  
    return np.sqrt(DTW[len(s1) - 1, len(s2) - 1])  

def LB_Keogh(s1, s2, r):
    LB_sum = 0
    for ind, i in enumerate(s1):
        lower_bound = min(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])
        upper_bound = max(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])
        if i >= upper_bound:
            LB_sum = LB_sum + (i - upper_bound) ** 2
        elif i < lower_bound:
            LB_sum = LB_sum + (i - lower_bound) ** 2
    return math.sqrt(LB_sum)

def k_means_clust(data, num_clust, num_iter=50, w=3):
    best_inertia = float('inf')
    best_re = [0] * len(data)
    
    # Multiple restarts for robust clustering
    for restart in range(10):
        rng = random.Random(restart * 37 + 11)
        centroids = rng.sample(list(data), num_clust)
        assignments = {}
        for n in range(min(num_iter, 40)):
            prev_assignments = {k: list(v) for k, v in assignments.items()}
            assignments = {c: [] for c in range(num_clust)}
            
            for ind, i in enumerate(data):
                min_dist = float('inf')
                closest_clust = 0
                for c_ind, j in enumerate(centroids):
                    cur_dist = DTWDistanceWithTrendWeight(i, j, w)
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        closest_clust = c_ind
                assignments[closest_clust].append(ind)
                
            if assignments == prev_assignments:
                break
                
            for key in assignments:
                if len(assignments[key]) > 0:
                    clust_sum = 0
                    for k in assignments[key]:
                        clust_sum = clust_sum + data[k]
                    centroids[key] = [m / len(assignments[key]) for m in clust_sum]
                    
        # Compute cluster inertia
        inertia = 0.0
        for key in assignments:
            for k in assignments[key]:
                inertia += DTWDistanceWithTrendWeight(data[k], centroids[key], w)
                
        if inertia < best_inertia and all(len(assignments[c]) > 0 for c in range(num_clust)):
            best_inertia = inertia
            re = [0] * len(data)
            for i in assignments.get(1, []):
                re[i] = 1
            best_re = re
            
    return best_re