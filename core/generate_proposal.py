import torch
import torch.nn.functional as F
import random
import pprint
import numpy as np
import os
from torch import nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, det_curve,confusion_matrix,auc,roc_curve
import time
# import librosa
import json

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_resolution(sequence, base_resolution, target_resolution):
    """
    Adjust the resolution of a binary sequence by upsampling or downsampling.

    Args:
        sequence (list): Input binary sequence (e.g., [0, 1, 1]).
        base_resolution (int): Base resolution in milliseconds (e.g., 20ms).
        target_resolution (int): Target resolution in milliseconds (e.g., 10ms, 40ms, 80ms).

    Returns:
        list: Adjusted binary sequence.
    """
    # Calculate the sampling factor
    factor = base_resolution // target_resolution if base_resolution >= target_resolution else target_resolution // base_resolution

    if base_resolution < target_resolution:  # Downsampling
        # Group elements into `factor` chunks and reduce each chunk to 0 or 1
        return [
            0 if 0 in sequence[i:i+factor] else 1
            for i in range(0, len(sequence), factor)
        ]
    elif base_resolution > target_resolution:  # Upsampling
        # Repeat each element `factor` times
        return np.array([
            item
            for elem in sequence
            for item in [elem] * factor
        ])
    else:  # Same resolution
        return np.array(sequence)
    
def truncate_to_min_length(array1, array2):
    """
    Truncate two arrays to their minimum length.

    Args:
        array1 (array-like): The first array (e.g., seg_pred_scores_adj).
        array2 (array-like): The second array (e.g., seg_labels).

    Returns:
        tuple: Truncated versions of array1 and array2.
    """
    min_length = min(len(array1), len(array2))
    truncated_array1 = array1[:min_length]
    truncated_array2 = array2[:min_length]
    return truncated_array1, truncated_array2




def dict2np(item):
    """
    Args:
        item (dict): 
            {
                filename_1:[0.98,0.68,0.1,0.50,0.1,0.0],
                filename_2:[0.0,0.11,0.51,0.60,0.1,0.0],
            }
    Returns:
        dict_np(numpy): flatten array
            array([0.98,0.68,0.1,0.50,0.1,0.0, 0.0,0.11,0.51,0.60,0.1,0.0, ...])
    """
    dict_np=[]
    for key in item.keys():
        dict_np.extend(item[key])
    return np.array(dict_np)

def eval_PFD(seg_score_dict, seg_tar_dict):
    """
        for partial forgery detection evaluation 
    Args:
        seg_score_dict (dict): predicted segmental scores
            {
                filename_1:[0.98,0.68,0.1,0.50,0.1,0.0],
                filename_2:[0.0,0.11,0.51,0.60,0.1,0.0],
                ...
            }
        
        seg_tar_dict (dict): true segmental labels
            {
                filename_1:[0,1,1,1,1,1],
                filename_2:[0,0,0,0,1,0],
                ...
            }
        
    Returns:
        EER, ACC, F1, PRECISION, RECALL, AUC
    """
    label_np, score_np = dict2np(seg_tar_dict), dict2np(seg_score_dict)
    """---------EER----------"""
    frr, far, thresholds = det_curve(label_np, score_np)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    EER = np.mean((frr[min_index], far[min_index]))
    # ACC_threshold=EER_threshold
    # """---------Others----------"""
    # pred_label_dict={}
    # for key in seg_score_dict.keys():
    #     pred_label_dict[key]=pred_label(seg_score_dict[key],ACC_threshold)
    # pred_label_np=dict2np(pred_label_dict)
    # print('1:{}_0:{}'.format(np.sum(pred_label_np)==1, np.sum(pred_label_np)==0))
    # f1=f1_score(label_np, pred_label_np)*100
    # pre=precision_score(label_np, pred_label_np)*100
    # rec=recall_score(label_np, pred_label_np)*100
    # acc=accuracy_score(label_np, pred_label_np)*100
    fpr,tpr,ths=roc_curve(label_np, score_np)
    AUC=auc(fpr,tpr)
    print("EER=%.2f%%"%(EER*100))
    print("AUC=%.2f%%"%(AUC*100))
    # print("F1",f1, "precision", pre, "recall", rec, "ACC", acc, "auc", (AUC))
    return EER*100

import numpy as np
from sklearn.metrics import precision_score
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import f1_score, recall_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def eval_PFD2(seg_score_dict, seg_tar_dict):
    labels, confidence = dict2np(seg_tar_dict), dict2np(seg_score_dict)
    print('mAP',getClassificationMAP(confidence, labels))
    print('mAUC',getClassificationAUC(confidence, labels)) 
    print('mEER',getClassificationEER(confidence, labels))

    
def getAP(conf,labels):
    assert len(conf)==len(labels)
    sortind = np.argsort(-conf)
    tp = labels[sortind]==1; fp = labels[sortind]!=1
    npos = np.sum(labels)

    fp = np.cumsum(fp).astype('float32'); tp = np.cumsum(tp).astype('float32')
    rec = tp/npos; prec=tp/(fp+tp)
    tmp = (labels[sortind]==1).astype('float32')

    return np.sum(tmp*prec)/npos

def getClassificationMAP(confidence, labels):
    ''' confidence and labels are of dimension n_samples x n_label '''
    AP = []
    for i in range(np.shape(labels)[1]):
        ap=getAP(confidence[:, i], labels[:, i])
        AP.append(ap)
        print(f'Label {i} AP: {ap:.4f}')
    return 100 * sum(AP) / len(AP)

def getClassificationAUC(confidence, labels):
    ''' Compute Area Under the Curve (AUC) for multi-label classification '''
    auc_list = []
    for i in range(np.shape(labels)[1]):
        try:
            auc = roc_auc_score(labels[:, i], confidence[:, i])
        except ValueError:
            auc = 0.0  # Handle case where only one class is present in the labels
        auc_list.append(auc)
        print(f'Label {i} AUC: {auc:.4f}')
    return 100 * np.mean(auc_list)

def compute_eer(fpr, tpr):
    ''' Calculate the Equal Error Rate (EER) '''
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

def getClassificationEER(confidence, labels):
    ''' Compute Equal Error Rate (EER) for multi-label classification '''
    from sklearn.metrics import roc_curve

    eer_list = []
    for i in range(np.shape(labels)[1]):
        fpr, tpr, _ = roc_curve(labels[:, i], confidence[:, i])
        eer = compute_eer(fpr, tpr)
        eer_list.append(eer)
        print(f'Label {i} EER: {eer:.4f}')
    return 100 * np.mean(eer_list)








def proposal_func(fake_start, fake_end, segscore_2d, rso=20):
    """function to produce proposals

    Args:
        fake_start (numpy): forgery start frames
        fake_end (numpy):  forgery end frames
        rso (int, optional): time resolution. Defaults to 20.

    Returns:
        regions (list): list of forgery proposals
        [[start_frame, end_frame], ...]
    """
    # regions=[]
    # for i in range(len(fake_start)):
    #     confidence_score=
    #     regions.append([confidence_score ,fake_start[i]/rso, fake_end[i]/rso])


    scores = segscore_2d[:, 1]
    
    # Generate regions
    regions = []
    for i in range(len(fake_start)):
        # Compute average score for the interval
        start_idx = fake_start[i]  
        end_idx = fake_end[i]
        # to frame
        start_time = fake_start[i] / rso
        end_time = fake_end[i] / rso
        # print(start_time, end_time, scores.shape)
        confidence_score = np.mean(scores[int(start_time):int(end_time)])
        regions.append([confidence_score, start_time, end_time])
    
    return regions



def frame2second_proposal(proposal_list, rso=20):
    """
    Args:
        [confidence score, start_frame, end_end]
    output:
        [confidence score, start_second, end_second]
    """
    return [[a, b *rso/ 1000, c * rso/ 1000] for a, b, c in proposal_list]

def segscore2proposal(segscore_2d, fake_label=1, true_label=0, rso=20):
    """ 
        segmental scores to temporal proposals

    Args:
        segscore_2d (numpy): 2D segmental scores produced by FDN
        proposal_func:  function to produce proposals
        rso (int, optional): time resolution. Defaults to 20.

    Returns:
        pred_label (numpy): using segmental scores via the np.argmax function to produce predicted label
        proposals_list (numpy): forgery proposals with the initial confidence score set to 1
        [[1, start_frame, end_frame], ...]
    """
    pred_label=np.array([np.argmax(segscore_2d, axis=1)],dtype=int).reshape(-1)
    _,starts,ends=_seglabel2proposal(rsolabel=pred_label, fake_label=fake_label, true_label=true_label, rso=rso)
    proposals=proposal_func(starts,ends,segscore_2d, rso)
    # print(proposals)
    # proposals_list=[np.insert(pro,0,1) for pro in proposals]
    return pred_label, np.array(proposals).reshape(-1,3)


def _seglabel2proposal(rsolabel, fake_label=1, true_label=0, rso=20):
    """
        segmental labels to temporal proposals
    Args:
        rsolabel (numpy): segmental labels
        rso (int, optional): time resolution. Defaults to 20.

    Returns:
        fake_segments (numpy): forgery segments
        starts (numpy): forgery start frames
        ends (numpy): forgery end frames
    """
    fake_segments = []
    prev_label = None
    current_start = 0
    # print(rsolabel)
    for i, label in enumerate(rsolabel):
        label = int(label)
        # print(label,fake_label,label==fake_label,true_label)
        time = i * rso
        # Detect state changes for fake segments only
        if label == fake_label and prev_label == true_label: # mark fake start
            current_start = time
            if i == len(rsolabel) - 1:  # the end
                fake_segments.append((current_start, time + rso))
        elif label == true_label and prev_label == fake_label:# mark fake end
            fake_segments.append((current_start, time))
        elif label == fake_label and i == len(rsolabel) - 1: # the end
            fake_segments.append((current_start, time + rso))
        prev_label = label
    fake_segments = np.array([[float(start), float(end)] for start, end in fake_segments],dtype=float).reshape(-1,2)
    # print(fake_segments)
    starts=fake_segments[:,0]
    ends=fake_segments[:,1]
    return fake_segments, starts, ends





def writenpy(filepath,content):
    np.save(filepath, content)
    

def readnpy(filepath):
    return np.load(filepath,allow_pickle=True).item()



def savejson(dict_data, filename):
    # Preprocess the dictionary to make it JSON serializable
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()  # Convert numpy array to list
        elif isinstance(value, list):
            # Recursively handle nested lists containing numpy arrays
            return [convert(v) for v in value]
        return value  # Return the value as is if not an ndarray
    
    json_serializable_dict = {key: convert(value) for key, value in dict_data.items()}
    # Save as JSON
    with open(filename, 'w', encoding='utf-8') as json_file:
        json.dump(json_serializable_dict, json_file, indent=4, ensure_ascii=False)