import os
import time
import copy
import json
import torch
import numpy as np
import warnings
from copy import deepcopy
warnings.filterwarnings("ignore")
from core.model import MDP
from core.loss import TotalLoss_weak
from core.config import cfg
import core.utils as utils
from torch.utils.tensorboard import SummaryWriter
from core.utils import AverageMeter
from core.dataset import NpyFeature
from eval.eval_detection import ANETdetection

from core.generate_proposal import segscore2proposal


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU_ID
    worker_init_fn = None
    if cfg.SEED >= 0:
        utils.set_seed(cfg.SEED)
        worker_init_fn = np.random.seed(cfg.SEED)

    utils.set_path(cfg)
    utils.save_config(cfg)

    model = MDP(cfg)
    model = model.cuda()

    train_loader = torch.utils.data.DataLoader(
        NpyFeature(data_path=cfg.DATA_PATH, audio_path=cfg.AUDIO_PATH, mode='train',
                        modal=cfg.MODAL, feature_fps=cfg.FEATS_FPS,
                        n_segments=cfg.NUM_SEGMENTS, supervision='weak',
                        class_dict=cfg.CLASS_DICT, seed=cfg.SEED, sampling='random'),
            batch_size=cfg.BATCH_SIZE,
            shuffle=True, num_workers=cfg.NUM_WORKERS,
            worker_init_fn=worker_init_fn)

    test_loader = torch.utils.data.DataLoader(
        NpyFeature(data_path=cfg.DATA_PATH, audio_path=cfg.AUDIO_PATH, mode='dev',
                        modal=cfg.MODAL, feature_fps=cfg.FEATS_FPS,
                        n_segments=cfg.NUM_SEGMENTS, supervision='weak',
                        class_dict=cfg.CLASS_DICT, seed=cfg.SEED, sampling='uniform'),
            batch_size=1,
            shuffle=False, num_workers=cfg.NUM_WORKERS,
            worker_init_fn=worker_init_fn)

    test_info = {"step": [], "test_acc": [], 
                "AP@0.1": [], "AP@0.2": [], "AP@0.3": [], "AP@0.4": [], "AP@0.5": [], "AP@0.6": [], "AP@0.7": [], "AP@0.75": [], "AP@0.95": [], 
                "AR@20": [], "AR@10": [], "AR@5": [],
                "AR@2": []}
    best_AP = -1
    criterion = TotalLoss_weak()

    cfg.LR = eval(cfg.LR)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR[0],
        betas=(0.9, 0.999), weight_decay=0.0005)

    writter = SummaryWriter(cfg.LOG_PATH)
        
    print('=> start training pseudo label generator...')
    for step in range(1, cfg.NUM_ITERS + 1):
        if step > 1 and cfg.LR[step - 1] != cfg.LR[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = cfg.LR[step - 1]

        if (step - 1) % len(train_loader) == 0:
            loader_iter = iter(train_loader)

        batch_time = AverageMeter()
        losses = AverageMeter()
        
        end = time.time()
        cost = train_one_step(model, loader_iter, optimizer, criterion, writter, step)
        losses.update(cost.item(), cfg.BATCH_SIZE)
        batch_time.update(time.time() - end)
        end = time.time()
        if step == 1 or step % cfg.PRINT_FREQ == 0:
            print(('Step: [{0:04d}/{1}]\t' \
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    step, cfg.NUM_ITERS, batch_time=batch_time, loss=losses)))
            
        if step > -1 and step % cfg.TEST_FREQ == 0:
            
            
            test_results = test_all(model, cfg, test_loader, test_info, step, writter)

            if test_info["AP@0.75"][-1] > best_AP:
                best_AP = test_info["AP@0.75"][-1]
                best_test_info = copy.deepcopy(test_info)

                utils.save_best_record_thumos(test_info, 
                    os.path.join(cfg.OUTPUT_PATH, "best_results.txt"))

                torch.save(model.state_dict(), os.path.join(cfg.MODEL_PATH, \
                    "model_best.pth.tar"))

            print(('- Test result: \t' \
                   'AP@0.1 {AP_10:.2%}\t' \
                   'AP@0.2 {AP_20:.2%}\t' \
                   'AP@0.3 {AP_30:.2%}\t' \
                   'AP@0.4 {AP_40:.2%}\t' \
                   'AP@0.5 {AP_50:.2%}\t' \
                   'AP@0.6 {AP_60:.2%}\t' \
                   'AP@0.7 {AP_70:.2%}\t' \
                   'AP@0.75 {AP_75:.2%}\t' \
                   'AP@0.95 {AP_95:.2%} (best: {best_AP:.2%})\t' \
                   'AR@20 {AR_20:}\t' \
                   'AR@10 {AR_10:}\t' \
                   'AR@5 {AR_5:}\t' \
                   'AR@2 {AR_2:}'.format(
                       AP_10=test_results['AP@0.1'][-1],
                       AP_20=test_results['AP@0.2'][-1],
                       AP_30=test_results['AP@0.3'][-1],
                       AP_40=test_results['AP@0.4'][-1],
                       AP_50=test_results['AP@0.5'][-1], 
                       AP_60=test_results['AP@0.6'][-1], 
                       AP_70=test_results['AP@0.7'][-1], 
                       AP_75=test_results['AP@0.75'][-1], 
                       AP_95=test_results['AP@0.95'][-1],
                       best_AP=best_AP, 
                       AR_20 = test_results['AR@20'][-1], 
                       AR_10 = test_results['AR@10'][-1],
                       AR_5 = test_results['AR@5'][-1],
                       AR_2 = test_results['AR@2'][-1])))

    print(utils.table_format(best_test_info, cfg.TIOU_THRESH, '[Generator] THUMOS\'14 Performance'))

def train_one_step(model, loader_iter, optimizer, criterion, writter, step):
    model.train()
    '''
    # return rgb_feature and audio_feature, for multimodal fusion
    data, label, _, _, _ = next(loader_iter)
    data = data.cuda()
    label = label.cuda()
    '''

    video_data, audio_data, label, _, _, _, _ = next(loader_iter)
    video_data = video_data.cuda()
    audio_data = audio_data.cuda()
    label = label.cuda()
    optimizer.zero_grad()
    video_scores, _, _, dp_distance = model(video_data, audio_data)
    
    
    cost, loss = criterion(video_scores, label, dp_distance)
    

    cost.backward()
    optimizer.step()

    for key in loss.keys():
        writter.add_scalar(key, loss[key].cpu().item(), step)
    
    return cost

@torch.no_grad()
def test_all(model, cfg, test_loader, test_info, step, writter=None, model_file=None):
    model.eval()

    if model_file:
        print('=> loading model: {}'.format(model_file))
        model.load_state_dict(torch.load(model_file))
        print('=> tesing model...')

    final_res = {'method': '[Generator]', 'results': {}}
    
    acc = AverageMeter()

    '''
    # for seglabel2proposal
    '''
    seg_score_dict = {}
    
    for video_data, audio_data, label, _, vid, vid_num_seg, fps in test_loader:

        video_data, audio_data, label = video_data.cuda(), audio_data.cuda(), label.cuda()
        vid_num_seg = vid_num_seg[0].cpu().item()
        fps = fps.item()
        
        video_scores, _, fas = model(video_data, audio_data)

        label_np = label.cpu().data.numpy()
        score_np = video_scores[0].cpu().data.numpy()
        
        pred_np = np.where(score_np < cfg.CLASS_THRESH, 0, 1)
        correct_pred = np.sum(label_np == pred_np, axis=1)
        acc.update(float(np.sum((correct_pred == cfg.NUM_CLASSES))), correct_pred.shape[0])

        pred = np.where(score_np >= cfg.CLASS_THRESH)[0]
        if len(pred) == 0:
            pred = np.array([np.argmax(score_np)])

        seg_score_dict[vid[0]]=fas[0].cpu().data.numpy()
        proposal_list=segscore2proposal(seg_score_dict[vid[0]], fake_label=1, true_label=0, rso=1)[1]
        final_proposals = []
        final_proposals.append(proposal_list.tolist())
        final_proposals = [[[1.0] + [sublist[0], sublist[1]/fps, sublist[2]/fps] for sublist in inner] for inner in final_proposals]
        
        final_res['results'][vid[0]] = utils.result2json(final_proposals, cfg.CLASS_DICT)

    json_path = os.path.join(cfg.OUTPUT_PATH, 'result.json')
    json.dump(final_res, open(json_path, 'w'))
    
    anet_detection = ANETdetection(cfg.GT_PATH, json_path,
                                subset='dev', tiou_thresholds=cfg.TIOU_THRESH,
                                verbose=False, check_status=False)

    AP, AR = anet_detection.evaluate_AP_AR()

    if writter:
        writter.add_scalar('Test Performance/Accuracy', acc.avg, step)

    test_info["step"].append(step)
    test_info["test_acc"].append(acc.avg)
    test_info["AP@0.1"].append(AP[0.1])
    test_info["AP@0.2"].append(AP[0.2])
    test_info["AP@0.3"].append(AP[0.3])
    test_info["AP@0.4"].append(AP[0.4])
    test_info["AP@0.5"].append(AP[0.5])
    test_info["AP@0.6"].append(AP[0.6])
    test_info["AP@0.7"].append(AP[0.7])
    test_info["AP@0.75"].append(AP[0.75])
    test_info["AP@0.95"].append(AP[0.95])
    test_info["AR@20"].append(AR[20])
    test_info["AR@10"].append(AR[10])
    test_info["AR@5"].append(AR[5])
    test_info["AR@2"].append(AR[2])
    return test_info
    
if __name__ == "__main__":
    main()
