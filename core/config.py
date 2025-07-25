import numpy as np
import os
from easydict import EasyDict as edict

cfg = edict()
# added by xuwb20241010 start
cfg.MODE='train'
cfg.AUDIO_PATH = '/NVME1/xuwb/LAV-DF/'
# added by xuwb20241010 end

cfg.GPU_ID = '2'
cfg.LR = '[0.00001]*200000'
cfg.NUM_ITERS = len(eval(cfg.LR))
cfg.NUM_CLASSES = 2
cfg.MODAL = 'all'
cfg.FEATS_DIM = 2048
# added by xuwb 20241207 start
cfg.VIDEO_DIM = 2048
cfg.AUDIO_DIM = 1024
cfg.ALIGN_DIM = 1024
# added by xuwb 20241207 start
cfg.BATCH_SIZE = 32
cfg.DATA_PATH = '/NVME3/yql/Data/pretrained_feature/Lavdf/'
cfg.NUM_WORKERS = 8
cfg.LAMBDA = 0.01
cfg.R_EASY = 5
cfg.R_HARD = 20
cfg.m = 3
cfg.M = 6
cfg.TEST_FREQ = 1000
cfg.PRINT_FREQ = 200
# cfg.TEST_FREQ = 10
# cfg.PRINT_FREQ = 10
cfg.CLASS_THRESH = 0.5
cfg.NMS_THRESH = 0.5
cfg.CAS_THRESH = np.arange(0.2, 0.8, 0.025)
cfg.ANESS_THRESH = np.arange(0.3, 0.925, 0.025)
# modified by xuwb 20241122 start
# cfg.TIOU_THRESH = np.linspace(0.1, 0.7, 7)
cfg.TIOU_THRESH = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.95])
# modified by xuwb 20241122 end
cfg.UP_SCALE = 24
cfg.GT_PATH = os.path.join(cfg.DATA_PATH, 'annotations/metadata.json')
cfg.SEED = 3407
cfg.FEATS_FPS = 25
cfg.NUM_SEGMENTS = 500
cfg.CLASS_DICT = {'real': 0, 'fake': 1}
