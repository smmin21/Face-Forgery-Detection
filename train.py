import os #, sys
import logging
from omegaconf import OmegaConf
from argparse import ArgumentParser

import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import models
from torchsummary import summary

from model.model import DANN_InceptionV3
from engine.tester import Tester
from engine.trainer import Trainer
from dataloader.ImageDataset import get_image_dataset
from dataloader.VideoDataset import get_video_dataset
from dataloader.DataDomainDataset import get_all_type_video_dataset
from model.i3d import I3D

parser = ArgumentParser('Deepface Training Script')
parser.add_argument('config', type=str, help='config file path')

if __name__ == '__main__':
    args = parser.parse_args()
    opt = OmegaConf.load(args.config)
    
    # Set random seeds
    random_seed = 42
    
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(random_seed)
    
    # Load Dataloader
    print(f"Using {opt.DATA.type} dataset ...")
    if opt.DATA.type == 'image':
        dataloader = get_image_dataset(opt.DATA)
    elif opt.DATA.type == 'video':
        dataloader = get_video_dataset(opt.DATA)
    elif opt.DATA.type == 'all_type_videos':
        dataloader = get_all_type_video_dataset(opt.DATA)
    print("Dataloader loading finished ...")
    
    # Model (Inception, Inception_dann, I3D, I3D_dann)
    if 'I3D' in opt.MODEL.name:
        model = I3D(opt.MODEL, opt.MODEL.name)
    elif opt.MODEL.name == 'Inception_dann':
        model = DANN_InceptionV3()    
    elif opt.MODEL.name == 'Inception':
        model = models.inception_v3(weights='Inception_V3_Weights.IMAGENET1K_V1')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    
    # Logger
    # log train/val loss, acc, etc.
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG) 
    console_logging_format = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_logging_format)
    logger.addHandler(console_handler)
    
    log_save_path = os.path.join('results', opt.EXP_NAME, f'{opt.TRAIN.epochs}_epochs_{opt.DATA.batch_size}_batch_{opt.TRAIN.lr}_lr')
    os.makedirs(log_save_path, exist_ok=False)
    file_handler = logging.FileHandler(os.path.join(log_save_path, 'train.log'))
    file_handler.setFormatter(console_logging_format)
    logger.addHandler(file_handler)

    # BANMo System    
    trainer = Trainer(opt, dataloader, model, logger)

    # train
    if 'dann' in opt.MODEL.name:
        trainer.train_dann()
    else:
        trainer.train()
    
    # test
    tester = Tester(opt, dataloader, model, logger)
    
    if opt.DATA.type == 'image':
        tester.test("test")
    elif isinstance(model, list):
        tester.test_ensemble("test")
    else:
        tester.test_video("test")