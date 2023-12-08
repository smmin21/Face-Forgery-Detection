import os
import h5py
from PIL import Image
import pandas as pd
import numpy as np
import pdb
import torch
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.utils import save_image
import json
from tqdm import tqdm

def get_video_dataset(opt):

    statistics = {
        'celeb': {'mean': [0.3871, 0.2939, 0.2809], 'std': [0.2288, 0.1780, 0.1737]},
        'ff': {'mean': [0.4504, 0.3891, 0.3914], 'std': [0.2964, 0.2137, 0.2182]},
        'dfdc': {'mean': [0.4417, 0.3554, 0.3248], 'std': [0.1946, 0.1899, 0.1742]},
        'vfhq': {'mean': [0.6961, 0.6284, 0.6086], 'std': [0.3482, 0.3722, 0.3813]},
        'dff': {'mean': [0.4027, 0.3584, 0.3549], 'std': [0.2126, 0.1886, 0.1906]},
    }
    
    # For cross evaluation
    # test_data_path에 경로 있으면 cross evaluation
    if opt.test_data_path == 'None':
        test_data_name = opt.train_data_name
        test_data_path = opt.train_data_path
    else:
        test_data_name = opt.test_data_name
        test_data_path = opt.test_data_path
        
    dataset_classes = {'celeb': CelebDF,
                       'ff': FaceForensics,
                       'dfdc': DFDC,
                       'vfhq': VFHQ,
                       'dff': DFF,}
    
    train_augmentation = T.Compose([
        T.Resize((opt.image_size, opt.image_size)),
        T.ToTensor(),
        T.Normalize(mean=statistics[opt.train_data_name]['mean'], std=statistics[opt.train_data_name]['std'])
        ])
    
    test_augmentation = T.Compose([
        T.Resize((opt.image_size, opt.image_size)),
        T.ToTensor(),
        T.Normalize(mean=statistics[test_data_name]['mean'], std=statistics[test_data_name]['std'])
        ])
                
    # train dataset
    train_dataset = dataset_classes[opt.train_data_name](opt.train_data_path, split='train', image_size=128, transform=train_augmentation, num_frames=opt.frame_num)
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=opt.batch_size, 
                                  shuffle=True, 
                                  num_workers=opt.num_workers)
    # val dataset
    val_dataset = dataset_classes[opt.train_data_name](opt.train_data_path, split='val', image_size=128, transform=train_augmentation, num_frames=opt.frame_num)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=opt.batch_size,
                                shuffle=False,
                                num_workers=opt.num_workers) 

    # test dataset
    test_dataset = dataset_classes[test_data_name](test_data_path, split='test', image_size=128, transform=test_augmentation, num_frames=opt.frame_num)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=opt.num_workers)
    
    dataset = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    return dataset

class VideoDataset(Dataset):
    def __init__(self, root_path, split='train', image_size=128, transform=None, num_frames=16, interval=2):
        mode_dict = {'train': 'train',
                     'meta-train': 'train',
                     'val': 'val',
                     'meta-val': 'val',
                     'test': 'test',
                     'meta-test': 'test',
                     }
        self.path = root_path
        self.mode = mode_dict[split]
        
        self.transform = transform
        self.num_frames = num_frames
        self.interval = interval
        
        self.videos = []
        self.labels = []
        self.domain_labels = []
        self.n_classes = 2
        
        self.clips = []
        self.clip_src_idx = []
        
        if self.mode == 'test':
            self._get_clips()
            
    def _get_clips(self):
        for i, video_dir in enumerate(self.videos):
            frame_keys = sorted(os.listdir(video_dir))
            frame_count = len(frame_keys)
            num_samples = self.num_frames
            interval = self.interval # UNIFORM :1,2 / SPREAD: max(total_frames // num_samples, 1)
            max_length = (num_samples - 1) * self.interval + num_samples

            for starting_point in range(0, frame_count, (num_samples-1)*interval + num_samples):
                if (interval == 0) or (frame_count <= max_length):
                    sampled_keys = frame_keys[starting_point:starting_point+num_samples]
                else:
                    sampled_indices = np.arange(starting_point, frame_count, interval)[:num_samples]
                    sampled_keys = [frame_keys[idx] for idx in sampled_indices]

                if len(sampled_keys) < num_samples:
                    break

                self.clips += [sampled_keys]
                self.clip_src_idx.append(i)
    
    def __len__(self):
        if self.mode != 'test':
            return len(self.videos)
        else:
            return len(self.clips)
    
    def __getitem__(self, index):
        if self.mode != 'test':
            video_dir = self.videos[index]
            frame_keys = sorted(os.listdir(video_dir))
            frame_count = len(frame_keys)
            clip_length = (self.num_frames-1)*self.interval + self.num_frames

            if (self.interval == 0) or (frame_count <= clip_length):
                starting_point = random.randint(0, frame_count-self.num_frames)
                sampled_keys = frame_keys[starting_point:starting_point+self.num_frames]
            else:
                starting_point = random.randint(0, frame_count-clip_length)
                sampled_indices = np.arange(starting_point, frame_count, self.interval)[:self.num_frames]
                sampled_keys = [frame_keys[idx] for idx in sampled_indices]
        else:
            src_idx = self.clip_src_idx[index]
            video_dir = self.videos[src_idx]
            sampled_keys = self.clips[index]
        
        frames = []
        for frame_key in sampled_keys:
            frame = Image.open(os.path.join(video_dir, frame_key))
            if self.transform is not None:
                frame = self.transform(frame)
            frames.append(frame)
        frame_data = torch.stack(frames, dim=0).transpose(0, 1)
        
        if self.mode != 'test':
            data = {'frame': frame_data, 'label': self.labels[index], 'domain_label':self.domain_labels[index]}
        else:
            data = {'video': src_idx, 'frame': frame_data, 'label': self.labels[src_idx], 'domain_label':self.domain_labels[src_idx]}
        return data
    
class FaceForensics(VideoDataset):
    def __init__(self, root_path, split='train', image_size=128, transform=None, num_frames=16):
        super().__init__(root_path, split, image_size, transform, num_frames)
        self.split_videos = [[] for _ in range(5)]
        
        # set iteration path
        iter_path = [os.path.join(self.path, 'original_sequences', 'raw', 'crop_jpg'),       # domain_label = 4
                     os.path.join(self.path, 'manipulated_sequences', 'Deepfakes', 'raw', 'crop_jpg'),      # 3
                     os.path.join(self.path, 'manipulated_sequences', 'Face2Face', 'raw', 'crop_jpg'),      # 2
                     os.path.join(self.path, 'manipulated_sequences', 'FaceSwap', 'raw', 'crop_jpg'),       # 1
                     os.path.join(self.path, 'manipulated_sequences', 'NeuralTextures', 'raw', 'crop_jpg')] # 0
        # iteration
        for i, each_path in enumerate(iter_path):
            video_keys = sorted(os.listdir(each_path))
            # test 데이터셋이 정해져있지 않은 경우 (FF++ & DFDC)
            if self.mode == 'train':
                video_keys = video_keys[:int(len(video_keys)*0.8)]
            elif self.mode == 'val':
                video_keys = video_keys[int(len(video_keys)*0.8):int(len(video_keys)*0.9)]
            elif self.mode == 'test':
                video_keys = video_keys[int(len(video_keys)*0.9):]
            
            video_dirs = [os.path.join(each_path, video_key) for video_key in video_keys]
            self.videos += video_dirs
            self.split_videos[4-i] = video_dirs
            self.labels += [0 if each_path.find('original') >= 0 else 1 for _ in range(len(video_keys))]
            for _ in range(len(video_keys)):
                self.domain_labels.append(4-i)
                
        if self.mode == 'test':
            self._get_clips()
                
    def __getitem__(self, index):
        choice = [0, 1, 2, 3, 4]
        probs = [1/8, 1/8, 1/8, 1/8, 1/2]
        domain_choice = np.random.choice(choice, p=probs)
        idx = random.randint(0, len(self.split_videos[domain_choice])-1)
        
        if self.mode != 'test':
            video_dir = self.split_videos[domain_choice][idx]
            frame_keys = sorted(os.listdir(video_dir))

            frame_count = len(frame_keys)
            clip_length = (self.num_frames-1)*self.interval + self.num_frames

            if (self.interval == 0) or (frame_count <= clip_length):
                starting_point = random.randint(0, frame_count-self.num_frames)
                sampled_keys = frame_keys[starting_point:starting_point+self.num_frames]
            else:
                starting_point = random.randint(0, frame_count-clip_length)
                sampled_indices = np.arange(starting_point, frame_count, self.interval)[:self.num_frames]
                sampled_keys = [frame_keys[idx] for idx in sampled_indices]
        else:
            src_idx = self.clip_src_idx[index]
            video_dir = self.videos[src_idx]
            sampled_keys = self.clips[index]
        
        frames = []
        for frame_key in sampled_keys:
            frame = Image.open(os.path.join(video_dir, frame_key))
            if self.transform is not None:
                frame = self.transform(frame)
            frames.append(frame)
        frame_data = torch.stack(frames, dim=0).transpose(0, 1)
        
        if self.mode != 'test':
            data = {'frame': frame_data, 'label': 0 if domain_choice==4 else 1, 'domain_label':domain_choice}
        else:
            data = {'video': src_idx, 'frame': frame_data, 'label': self.labels[src_idx], 'domain_label':self.domain_labels[src_idx]}
        return data

# For Test
class DFDC(VideoDataset):
    def __init__(self, root_path, split='train', image_size=128, transform=None, num_frames=16):
        super().__init__(root_path, split, image_size, transform, num_frames)
        
        self.mtype = [f'dfdc_{i:02}' for i in range(8)]
        
        # set iteration path    
        iter_path = [os.path.join(self.path, set) for set in self.mtype]
        
        # iteration
        for i, each_path in enumerate(iter_path):
            video_keys = sorted(os.listdir(each_path))
            # test 데이터셋이 정해져있지 않은 경우 (FF++ & DFDC)
            if self.mode == 'train':
                video_keys = video_keys[:int(len(video_keys)*0.8)]
            elif self.mode == 'val':
                video_keys = video_keys[int(len(video_keys)*0.8):int(len(video_keys)*0.9)]
            elif self.mode == 'test':
                video_keys = video_keys[int(len(video_keys)*0.9):]
            
            video_dirs = [os.path.join(each_path, video_key) for video_key in video_keys]
            self.videos += video_dirs
            label_path = os.path.join(each_path, 'label.json')
            label_file = open(label_path, encoding='utf-8')
            label_data = json.loads(label_file.read())
            self.labels += [0 if label_data[video_key] == 'REAL' else 1 for video_key in video_keys]
            for _ in range(len(video_keys)):
                self.domain_labels.append(0)
        
        if self.mode == 'test':
            self._get_clips()
           
class CelebDF(VideoDataset):
    def __init__(self, root_path, split='train', image_size=128, transform=None, num_frames=16):
        super().__init__(root_path, split, image_size, transform, num_frames)
        
        self.test_list = None
        
        # set iteration path
        iter_path = [os.path.join(self.path, 'Celeb-real', 'crop_jpg'),
                     os.path.join(self.path, 'Celeb-synthesis', 'crop_jpg'),
                     os.path.join(self.path, 'YouTube-real', 'crop_jpg')]
        with open(self.path + "/List_of_testing_videos.txt", "r") as f:
            self.test_list = f.readlines()
            self.test_list = [x.split("/")[-1].split(".mp4")[0] for x in self.test_list]
        
        # iteration
        for i, each_path in enumerate(iter_path):
            video_keys = sorted(os.listdir(each_path))
            # test 데이터셋이 정해져 있는 경우 (Celeb-DF)
            if self.mode == 'test': 
                video_keys = [x for x in self.test_list if x in video_keys]
            elif self.mode == 'train':
                video_keys = [x for x in video_keys if x not in self.test_list]
                video_keys = video_keys[:int(len(video_keys)*0.8)]
            elif self.mode == 'val':
                video_keys = [x for x in video_keys if x not in self.test_list]
                video_keys = video_keys[int(len(video_keys)*0.8):]
            
            video_dirs = [os.path.join(each_path, video_key) for video_key in video_keys]
            self.videos += video_dirs
            self.labels += [0 if each_path.find('real') >= 0 else 1 for _ in range(len(video_keys))] # 0: real, 1: fake
            for _ in range(len(video_keys)):
                self.domain_labels.append(0)
                
        if self.mode == 'test':
            self._get_clips()

class VFHQ(VideoDataset):
    def __init__(self, root_path, split='train', image_size=128, transform=None, num_frames=16):
        super().__init__(root_path, split, image_size, transform, num_frames)
        
        # set iteration path
        iter_path = [os.path.join(self.path, 'crop_jpg')]
        
        mode_mapping = {'train': 'training',
                        'val': 'validation',
                        'test': 'test'}
        video_key_path = os.path.join(iter_path[0], mode_mapping[self.mode])
        video_keys = sorted(os.listdir(video_key_path))
        video_dirs = [os.path.join(video_key_path, video_key) for video_key in video_keys]
        self.videos += video_dirs
        self.labels += [1 if key.split('_')[2][0] == 'f' else 0 for key in video_keys]
        for _ in range(len(video_keys)):
                self.domain_labels.append(0)
        
        if self.mode == 'test':
            self._get_clips()
        
        
class DFF(VideoDataset):
    def __init__(self, root_path, split='train', image_size=128, transform=None, num_frames=16):
        super().__init__(root_path, split, image_size, transform, num_frames)
        
        # set iteration path
        folders = os.listdir(os.path.join(self.path, 'manipulated_videos'))
        iter_path = [os.path.join(self.path, 'manipulated_videos', folder) for folder in folders]
        iter_path.append(os.path.join(self.path, 'original_sequences/raw/crop_jpg')) 
        
        # iteration
        for i, video_dir in enumerate(iter_path):
            assert os.path.exists(video_dir), f"{video_dir} does not exist"

            all_video_keys = sorted(os.listdir(video_dir))
            final_video_keys = self._get_splits(all_video_keys)

            video_dirs = [os.path.join(video_dir, video_key) for video_key in final_video_keys]
            self.videos += video_dirs
            self.labels += [0 if video_dir.find('original') >= 0 else 1 for _ in range(len(final_video_keys))]
            for _ in range(len(video_dirs)):
                self.domain_labels.append(0)
                
        if self.mode == 'test':
            self._get_clips()
    
    def _get_splits(self, video_keys):
        # Default split logic. Redefine the function if needed
        if self.mode == 'train':
            video_keys = video_keys[:int(len(video_keys)*0.8)]
        elif self.mode == 'val':
            video_keys = video_keys[int(len(video_keys)*0.8):int(len(video_keys)*0.9)]
        elif self.mode == 'test':
            video_keys = video_keys[int(len(video_keys)*0.9):]

        return video_keys