# This code is originally from the official ActivityNet repo
# https://github.com/activitynet/ActivityNet
# Small modification from ActivityNet Code

from concurrent.futures import ProcessPoolExecutor
import json
from typing import List, Union
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from torch import Tensor
import torch

from .utils_eval import get_blocked_videos
from .utils_eval import interpolated_prec_rec
from .utils_eval import segment_iou

import warnings
import os
import torch.nn as nn

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute jaccard score between a box and the anchors."""

    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors + (box_max - box_min) - inter_len
    iou = inter_len / union_len
    return iou

class AP(nn.Module):
    """
    Average Precision

    The mean precision in Precision-Recall curve.
    """

    def __init__(self, iou_thresholds: Union[float, List[float]] = 0.5):
        super().__init__()
        self.iou_thresholds: List[float] = iou_thresholds if type(iou_thresholds) is list else [iou_thresholds]
        self.n_labels = 0
        self.ap: dict = {}
        self.proposals = []
        self.labels = []

    def forward(self, pre_data_dict, gt_list):
        self.proposals = []
        # modifide by xuwb 20241122 start
        # data processing for lavdf   
        '''
        for data in pre_data_dict:
            score = data['scores'].unsqueeze(-1)
            seg = data['segments']
            self.proposals.append(torch.cat((score, seg), dim=-1))

        self.labels = gt_list
        '''
        self.proposals = pre_data_dict
        self.labels = gt_list
        # added by xuwb 20241122 end
        # print('len labels:{}'.format(len(self.labels)))

        for iou_threshold in self.iou_thresholds:
            values = []
            self.n_labels = 0

            for index in range(len(self.proposals)):
                proposals = self.proposals[index]
                # print('proposal size :{}'.format(proposals.size()))
                labels = self.labels[index]

                # with open('summary/result/proposal_show.txt', 'a+') as file:
                #         file.write( str(proposals) + '\n' + str(labels) + '\n' + '\n')

                self.n_labels += len(labels[0])   # Count the total number of ground truth segments
                if len(proposals) == 0:
                    continue
                # print('labels size :{}'.format(labels.size()))
                values.append(AP.get_values(iou_threshold, proposals, labels))

            # sort proposals
            values = torch.cat(values)
            _, ind = torch.sort(values[:, 0], dim=0, descending=True)
            # ind = values[:, 0].sort(stable=True, descending=True).indices
            values = values[ind]

            # accumulate to calculate precision and recall
            curve = self.calculate_curve(values)
            ap = self.calculate_ap(curve)
            self.ap[iou_threshold] = ap

        return self.ap

    def calculate_curve(self, values):
        is_TP = values[:, 1]
        acc_TP = torch.cumsum(is_TP, dim=0)
        precision = acc_TP / (torch.arange(len(is_TP)) + 1)
        recall = acc_TP / self.n_labels
        curve = torch.stack([recall, precision]).T
        curve = torch.cat([torch.tensor([[1., 0.]]), torch.flip(curve, dims=(0,))])
        return curve

    @staticmethod
    def calculate_ap(curve):
        x, y = curve.T
        y_max = y.cummax(dim=0).values
        x_diff = x.diff().abs()
        ap = (x_diff * y_max[:-1]).sum()
        return ap

    # modified by xuwb 20241126 start
    # annotate debug version
    # recovery old version
    ''' 
    @staticmethod
    def get_values(
            iou_threshold: float,
            proposals: Tensor,
            labels: Tensor,
    ) -> Tensor:
        # added by xuwb 20241125 start
        # debug
        proposals = [
            [proposals[0][i], proposals[1][i][0], proposals[1][i][1]] for i in range(len(proposals[0]))
        ]
        proposals = torch.tensor(proposals, dtype=torch.float32)
        proposals = proposals[:10]
        # added by xuwb 20241125 end

        n_labels = len(labels[0])
        n_proposals = len(proposals)

        # Compute IoUs between each proposal and each label
        ious = torch.zeros((len(proposals), n_labels))  # (n_proposals, n_labels)
        for i in range(n_labels):
            # modified by xuwb 20241125 start
            
            # simplify the calculation of iou
            
            # proposals_start = [sublist[0] for sublist in proposals[1]]
            # proposals_start = torch.tensor(proposals_start, dtype=torch.float32)
            # proposals_end = [sublist[1] for sublist in proposals[1]]
            # proposals_end = torch.tensor(proposals_end, dtype=torch.float32)
            # ious[:, i] = iou_with_anchors(proposals_start, proposals_end, labels[0][i][0], labels[0][i][1])
            
            ious[:, i] = iou_with_anchors(proposals[:, 1], proposals[:, 2], labels[0][i][0], labels[0][i][1])
            # modified by xuwb 20241125 end

        # values: (confidence, is_TP) rows
        # n_labels = ious.shape[1]
        # detected = torch.full((n_labels,), False)
        confidence = proposals[:, 0]
        # added by xuwb 20241122 start
        # 
        # confidence = torch.tensor(confidence, dtype=torch.float32)
        # added by xuwb 20241122 end
        potential_TP = ious > iou_threshold

        tp_indexes = []

        for i in range(n_labels):
            potential_TP_index = potential_TP[:, i].nonzero()
            for (j,) in potential_TP_index:
                if j not in tp_indexes:
                    tp_indexes.append(j)
                    break
        is_TP = torch.zeros(n_proposals, dtype=torch.bool)
        if len(tp_indexes) > 0:
            tp_indexes = torch.stack(tp_indexes)
            is_TP[tp_indexes] = True
        values = torch.column_stack([confidence, is_TP])
        return values
    '''
    @staticmethod
    def get_values(
            iou_threshold: float,
            proposals: Tensor,
            labels: Tensor,
    ) -> Tensor:
        n_labels = len(labels[0])
        n_proposals = len(proposals[0])

        # Compute IoUs between each proposal and each label
        ious = torch.zeros((len(proposals[0]), n_labels))  # (n_proposals, n_labels)
        for i in range(n_labels):
            proposals_start = [sublist[0] for sublist in proposals[1]]
            proposals_start = torch.tensor(proposals_start, dtype=torch.float32)
            proposals_end = [sublist[1] for sublist in proposals[1]]
            proposals_end = torch.tensor(proposals_end, dtype=torch.float32)
            ious[:, i] = iou_with_anchors(proposals_start, proposals_end, labels[0][i][0], labels[0][i][1])

        # values: (confidence, is_TP) rows
        # n_labels = ious.shape[1]
        # detected = torch.full((n_labels,), False)
        confidence = proposals[0]
        # added by xuwb 20241122 start
        # 
        confidence = torch.tensor(confidence, dtype=torch.float32)
        # added by xuwb 20241122 end
        potential_TP = ious > iou_threshold

        tp_indexes = []

        for i in range(n_labels):
            potential_TP_index = potential_TP[:, i].nonzero()
            for (j,) in potential_TP_index:
                if j not in tp_indexes:
                    tp_indexes.append(j)
                    break
        is_TP = torch.zeros(n_proposals, dtype=torch.bool)
        if len(tp_indexes) > 0:
            tp_indexes = torch.stack(tp_indexes)
            is_TP[tp_indexes] = True
        values = torch.column_stack([confidence, is_TP])
        return values
    # modified by xuwb 20241126 end
class AR(nn.Module):
    """
    Average Recall

    Args:
        n_proposals_list: Number of proposals. 100 for AR@100.
        iou_thresholds: IOU threshold samples for the curve. Default: [0.5:0.05:0.95]

    """

    def __init__(self, n_proposals_list: Union[List[int], int] = 100, iou_thresholds: List[float] = None,
                 parallel: bool = True
                 ):
        super().__init__()
        if iou_thresholds is None:
            # modified by xuwb 20241127 start
            iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            # iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
            # modified by xuwb 20241127 end
        self.n_proposals_list: List[int] = n_proposals_list if type(n_proposals_list) is list else [n_proposals_list]
        self.iou_thresholds = iou_thresholds
        self.parallel = parallel
        self.ar: dict = {}
        self.proposals = []
        self.labels = []

    def forward(self, pre_data_dict, gt_list):
        self.proposals = []
        # modifide by xuwb 20241122 start
        # data processing for lavdf   
        '''
        for data in pre_data_dict:
            score = data['scores'].unsqueeze(-1)
            seg = data['segments']
            self.proposals.append(torch.cat((score, seg), dim=-1))

        self.labels = gt_list
        '''
        self.proposals = pre_data_dict
        self.labels = gt_list
        # added by xuwb 20241122 end

        for n_proposals in self.n_proposals_list:
            if self.parallel:
                with ProcessPoolExecutor(os.cpu_count() // 2 - 1) as executor:
                    futures = []
                    for meta in metadata:
                        proposals = torch.tensor(proposals_dict[meta.file])
                        labels = torch.tensor(meta.fake_periods)
                        futures.append(executor.submit(AR.get_values, n_proposals, self.iou_thresholds,
                                                       proposals, labels, 25.))

                    values = list(map(lambda x: x.result(), futures))
                    values = torch.stack(values)
            else:
                values = torch.zeros((len(self.proposals), len(self.iou_thresholds), 2))
                for index in range(len(self.proposals)):
                    proposals = self.proposals[index]
                    labels = self.labels[index]
                    values[index] = AR.get_values(n_proposals, self.iou_thresholds, proposals, labels)

            values_sum = values.sum(dim=0)

            TP = values_sum[:, 0]
            FN = values_sum[:, 1]
            recall = TP / (TP + FN)
            self.ar[n_proposals] = recall.mean()

        return self.ar

    @staticmethod
    def get_values(
            n_proposals: int,
            iou_thresholds: List[float],
            proposals: Tensor,
            labels: Tensor,
    ):
        
        # added by xuwb 20241122 start
        # debug
        proposals = [
            [proposals[0][i], proposals[1][i][0], proposals[1][i][1]] for i in range(len(proposals[0]))
        ]
        proposals = torch.tensor(proposals, dtype=torch.float32)
        n_thresholds = len(iou_thresholds)
        values = torch.zeros((n_thresholds, 2))
        if(len(proposals) == 0):
            return values
        # added by xuwb 20241122 end
        proposals = proposals[:n_proposals]
        n_proposals = proposals.shape[0]
        n_labels = len(labels[0])
        ious = torch.zeros((n_proposals, n_labels))
        for i in range(n_labels):
            ious[:, i] = iou_with_anchors(proposals[:, 1], proposals[:, 2], labels[0][i][0], labels[0][i][1])

        

        # values: rows of (TP, FN)
        iou_max = ious.max(dim=0)[0]
        

        for i in range(n_thresholds):
            iou_threshold = iou_thresholds[i]
            TP = (iou_max > iou_threshold).sum()
            FN = n_labels - TP
            values[i] = torch.tensor((TP, FN))

        return values
    
class ANETdetection(object):
    GROUND_TRUTH_FIELDS = ['database']
    # GROUND_TRUTH_FIELDS = ['database', 'taxonomy', 'version']
    # PREDICTION_FIELDS = ['results', 'version', 'external_data']
    PREDICTION_FIELDS = ['method', 'results']

    def __init__(self, ground_truth_filename=None, prediction_filename=None,
                 ground_truth_fields=GROUND_TRUTH_FIELDS,
                 prediction_fields=PREDICTION_FIELDS,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10), 
                 subset='validation', verbose=False, 
                 check_status=False):
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')
        self.subset = subset
        self.tiou_thresholds = tiou_thresholds
        self.verbose = False
        self.gt_fields = ground_truth_fields
        self.pred_fields = prediction_fields
        self.ap = None
        self.check_status = check_status
        # Retrieve blocked videos from server.

        if self.check_status:
            self.blocked_videos = get_blocked_videos()
        else:
            self.blocked_videos = list()

        # modified by xuwb 20241122 start
        '''
        # annotation follow code
        # Import ground truth and predictions.
        self.ground_truth, self.activity_index = self._import_ground_truth(
            ground_truth_filename)
        ## added by xuwb 20241023 start
        self.activity_index['real'] = 1
        ## added by xuwb 20241023 end
        self.prediction = self._import_prediction(prediction_filename)
        '''
        self.ground_truth = self.get_ground_list(ground_truth_filename)
        self.prediction = self.get_pred_list(prediction_filename)

        # modified by xuwb 20241122 end

        if self.verbose:
            print ('[INIT] Loaded annotations from {} subset.'.format(subset))
            nr_gt = len(self.ground_truth)
            print ('\tNumber of ground truth instances: {}'.format(nr_gt))
            nr_pred = len(self.prediction)
            print ('\tNumber of predictions: {}'.format(nr_pred))
            print ('\tFixed threshold for tiou score: {}'.format(self.tiou_thresholds))
    def get_ground_list(self, ground_truth_filename):
        '''
        read the segments of gt, and sort by video_name
        '''
        with open(ground_truth_filename, 'r') as fobj:
            data = json.load(fobj)
        
        ground_list = []
        for value in data:
            videoid = os.path.splitext(os.path.basename(value['file']))[0]
            if self.subset != value['split']:
                continue
            if len(value['fake_periods']) == 0:
                continue
            if (value['modify_audio'] == False or value['modify_video'] == False):
            # if value['modify_video'] == False:
                continue
            if videoid in self.blocked_videos:
                continue

            fps = value['video_frames'] / value['duration']
            segments_tmp = []
            for ann in value['fake_periods']:
                # added by xuwb 20241213 start
                # segments_tmp += [np.array(ann) * fps]
                segments_tmp += [np.array(ann)]
                # added by xuwb 20241213 end

            ground_list += [[videoid, segments_tmp]]

        # ground_list = sorted(ground_list, key=lambda x:x[0])
        ground_list.sort(key=lambda x:x[0])
        return ground_list
    
    def get_pred_list(self, prediction_filename):
        '''
        read the segments of prediction, which contains videoid, scores, segments, then sort by videoid
        '''
        with open(prediction_filename, 'r') as fobj:
            data = json.load(fobj)
        if not all([field in data.keys() for field in self.pred_fields]):
            raise IOError('Please input a valid prediction file.')
        
        pred_list = []
        for videoid, v in data['results'].items():
            segment_list = []
            score_list = []
            for result in v:
                segment_list.append(result['segment'])
                score_list.append(result['score'])
            pred_list += [[videoid, score_list, segment_list]]
        
        pred_list = sorted(pred_list, key=lambda x:x[0])
        return pred_list
        

            

    def _import_ground_truth(self, ground_truth_filename):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.

        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.

        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        with open(ground_truth_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format
        ## modified by xuwb 20241022 start 
        # 注释以下两行
        #if not all([field in data.keys() for field in self.gt_fields]):
            # raise IOError('Please input a valid ground truth file.')
        ## modified by xuwb 20241022 end


        # Read ground truth data.
        activity_index, cidx = {}, 0
        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
        ## added by xuwb 20241022 start
        # 用于替换下方的for循环
        for value in data:
            videoid = os.path.splitext(os.path.basename(value['file']))[0]
            if self.subset != value['split']:
                continue
            if len(value['fake_periods']) == 0:
                continue
            if value['modify_video'] == False:
                continue
            if videoid in self.blocked_videos:
                continue
        
            
            fps = value['video_frames'] / value['duration']
            for ann in value['fake_periods']:
                if 'fake' not in activity_index:
                    activity_index['fake'] = cidx
                    cidx += 1
                video_lst.append(videoid)
                t_start_lst.append(ann[0] * fps)
                t_end_lst.append(ann[1] * fps)
                label_lst.append(activity_index['fake'])

        ## modified by xuwb 20241022 end

        ## annotation by xuwb 20241023 start 
        '''
        for videoid, v in data['database'].items():
            # print(v)
            if self.subset != v['subset']:
                continue
            if videoid in self.blocked_videos:
                continue
            for ann in v['annotations']:
                if ann['label'] not in activity_index:
                    activity_index[ann['label']] = cidx
                    cidx += 1
                video_lst.append(videoid)
                t_start_lst.append(float(ann['segment'][0]))
                t_end_lst.append(float(ann['segment'][1]))
                label_lst.append(activity_index[ann['label']])
        '''
        ## annotation by xuwb 20241023 end
        ground_truth = pd.DataFrame({'video-id': video_lst,
                                     't-start': t_start_lst,
                                     't-end': t_end_lst,
                                     'label': label_lst})
        if self.verbose:
            print(activity_index)
        return ground_truth, activity_index

    def _import_prediction(self, prediction_filename):
        """Reads prediction file, checks if it is well formatted, and returns
           the prediction instances.

        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.

        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        with open(prediction_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format...
        if not all([field in data.keys() for field in self.pred_fields]):
            raise IOError('Please input a valid prediction file.')

        # Read predictions.
        video_lst, t_start_lst, t_end_lst = [], [], []
        label_lst, score_lst = [], []
        for videoid, v in data['results'].items():
            if videoid in self.blocked_videos:
                continue
            for result in v:
                label = self.activity_index[result['label']]
                video_lst.append(videoid)
                t_start_lst.append(float(result['segment'][0]))
                t_end_lst.append(float(result['segment'][1]))
                label_lst.append(label)
                score_lst.append(result['score'])
        prediction = pd.DataFrame({'video-id': video_lst,
                                   't-start': t_start_lst,
                                   't-end': t_end_lst,
                                   'label': label_lst,
                                   'score': score_lst})
        return prediction

    def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label.
        """
        try:
            return prediction_by_label.get_group(cidx).reset_index(drop=True)
        except:
            if self.verbose:
                print ('Warning: No predictions of label \'%s\' were provdied.' % label_name)
            return pd.DataFrame()

    def wrapper_compute_average_precision(self):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby('label')
        prediction_by_label = self.prediction.groupby('label')
        del self.activity_index['real']
        theResults = Parallel(n_jobs=len(self.activity_index))(
                    delayed(compute_average_precision_detection)(
                        ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True),
                        prediction=self._get_predictions_with_label(prediction_by_label, label_name, cidx),
                        tiou_thresholds=self.tiou_thresholds,
                    ) for label_name, cidx in self.activity_index.items())
        results, recall, precision = zip(*theResults)
        for i, cidx in enumerate(self.activity_index.values()):
            ap[:,cidx] = results[i]

        return ap, recall, precision

    def evaluate(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        self.ap, recall, precision = self.wrapper_compute_average_precision()

        self.mAP = self.ap.mean(axis=1)
        self.average_mAP = self.mAP.mean()

        if self.verbose:
            print ('[RESULTS] Performance on ActivityNet detection task.')
            print ('Average-mAP: {}'.format(self.average_mAP))
            
        return self.mAP, self.average_mAP, recall, precision
    
    def evaluate_AP_AR(self):
        
        AP_compute = AP([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.95])
        AR_compute = AR([20, 10, 5, 2], parallel=False)

        pre_data = np.array(self.prediction, dtype=object)
        pre_data = pre_data[:, 1:3]
        gt_list = np.array(self.ground_truth, dtype=object)
        gt_list = gt_list[:, 1:2]

        ap_res = AP_compute(pre_data, gt_list)
        ar_res = AR_compute(pre_data, gt_list)
        
        
        return ap_res, ar_res





def compute_average_precision_detection(ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.

    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    recall = np.zeros(len(tiou_thresholds))
    precision = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap, recall, precision

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds),len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)
    recall_cumsum = tp_cumsum / npos
    recall = np.max(recall_cumsum, axis=1)

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)
    precision = np.max(precision_cumsum, axis=1)
    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])

    ## modified by xuwb 20241113 start
    # add recall and precision to return
    # return ap
    return ap, recall, precision
    ## modified by xuwb 20241113 end
