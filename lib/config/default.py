import os
import glob
from pathlib import Path
from yacs.config import CfgNode as CN

_C = CN()

_C.OUT_DIR = 'runs/'
_C.GPUS = (0,)     
_C.WORKERS = 2
_C.PIN_MEMORY = False
_C.PRINT_FREQ = 5
_C.AUTO_RESUME = False
_C.NEED_AUTOANCHOR = True
_C.DEBUG = False
_C.num_seg_class = 2
_C.num_det_class = 6

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# Common params for NETWORK
_C.MODEL = CN(new_allowed=True)
_C.MODEL.NAME = ''
_C.MODEL.STRU_WITHSHARE = False
_C.MODEL.HEADS_NAME = ['']
_C.MODEL.PRETRAINED = ""
_C.MODEL.PRETRAINED_DET = ""
_C.MODEL.IMAGE_SIZE = [320, 192]
_C.MODEL.EXTRA = CN(new_allowed=True)

# Loss params
_C.LOSS = CN(new_allowed=True)
_C.LOSS.LOSS_NAME = ''
_C.LOSS.MULTI_HEAD_LAMBDA = None
_C.LOSS.FL_GAMMA = 0.0
_C.LOSS.CLS_POS_WEIGHT = 1.0
_C.LOSS.OBJ_POS_WEIGHT = 1.0
_C.LOSS.SEG_POS_WEIGHT = 1.0
_C.LOSS.BOX_GAIN = 0.05
_C.LOSS.CLS_GAIN = 0.5
_C.LOSS.OBJ_GAIN = 1.0
_C.LOSS.FL_GAIN = 0.3
_C.LOSS.TK_GAIN = 0.3

# Auto-detect DATASET root
def _detect_dataroot():
    # Priority 1: /kaggle/working/dataset_ucr
    if os.path.exists("/kaggle/working/dataset_ucr/images/train"):
        return "/kaggle/working/dataset_ucr"
    # Priority 2: /kaggle/input search
    k_matches = glob.glob('/kaggle/input/**/images/train', recursive=True)
    if k_matches:
        return str(Path(k_matches[0]).parent.parent).replace('\\', '/')
    # Priority 3: Local workspace
    for p in ['../dataset_ucr', './dataset_ucr', 'dataset_ucr', 'D:/Downloads/ucr2026/dataset_ucr']:
        if os.path.exists(p):
            return p
    return '../dataset_ucr'

_DATA_ROOT = _detect_dataroot()

# DATASET related params
_C.DATASET = CN(new_allowed=True)
_C.DATASET.DATAROOT = f'{_DATA_ROOT}/images'
_C.DATASET.LABELROOT = f'{_DATA_ROOT}/det_annotations'
_C.DATASET.MASKROOT = f'{_DATA_ROOT}/da_seg_annotations'
_C.DATASET.LANEROOT = f'{_DATA_ROOT}/ll_seg_annotations'

_C.DATASET.DATASET = 'BddDataset'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'val'
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.SELECT_DATA = False
_C.DATASET.ORG_IMG_SIZE = [180, 320]

# Training data augmentation
_C.DATASET.FLIP = False
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 10
_C.DATASET.TRANSLATE = 0.1
_C.DATASET.SHEAR = 0.0
_C.DATASET.COLOR_RGB = False
_C.DATASET.HSV_H = 0.015
_C.DATASET.HSV_S = 0.7
_C.DATASET.HSV_V = 0.4

# Train params
_C.TRAIN = CN(new_allowed=True)
_C.TRAIN.LR0 = 0.001
_C.TRAIN.LRF = 0.2
_C.TRAIN.WARMUP_EPOCHS = 1.0
_C.TRAIN.WARMUP_BIASE_LR = 0.1
_C.TRAIN.WARMUP_MOMENTUM = 0.8

_C.TRAIN.OPTIMIZER = 'adamw'
_C.TRAIN.MOMENTUM = 0.937
_C.TRAIN.WD = 0.0005
_C.TRAIN.NESTEROV = True
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 120

_C.TRAIN.VAL_FREQ = 1
_C.TRAIN.BATCH_SIZE_PER_GPU = 4
_C.TRAIN.START_VAL = 0
_C.TRAIN.SHUFFLE = True
_C.TRAIN.IOU_THRESHOLD = 0.2
_C.TRAIN.ANCHOR_THRESHOLD = 4.0

_C.mosaic_rate = 0.0
_C.mixup_rate = 0.0
_C.TRAIN.PLOT = False

# Testing params
_C.TEST = CN(new_allowed=True)
_C.TEST.BATCH_SIZE_PER_GPU = 4
_C.TEST.MODEL_FILE = ''
_C.TEST.SAVE_TXT = False
_C.TEST.PLOTS = False
_C.TEST.NMS_CONF_THRESHOLD = 0.001
_C.TEST.NMS_IOU_THRESHOLD = 0.6

def update_config(cfg, args):
    cfg.defrost()
    cfg.config = args.config
    if args.outDir:
        cfg.OUT_DIR = args.outDir
    cfg.freeze()
