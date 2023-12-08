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

def get_image_dataset(opt):

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

class ImageDataset(Dataset):
    def __init__(self, root_path, split='train', image_size=128, transform=None, num_frames=16):
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
        
        self.videos = []
        self.labels = []
        self.domain_labels = []
        self.n_classes = 2
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, index):
        video_dir = self.videos[index]
        frame_keys = sorted(os.listdir(video_dir))
        frame_key = random.choice(frame_keys)
        frame = Image.open(os.path.join(video_dir, frame_key))
        if self.transform is not None:
            frame = self.transform(frame)
        data = {'frame': frame, 'label': self.labels[index], 'domain_label':self.domain_labels[index]}
        return data
    
class FaceForensics(ImageDataset):
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
                
    def __getitem__(self, index):
        choice = [0, 1, 2, 3, 4]
        probs = [1/8, 1/8, 1/8, 1/8, 1/2]
        domain_choice = np.random.choice(choice, p=probs)
        idx = random.randint(0, len(self.split_videos[domain_choice])-1)
        
        video_dir = self.split_videos[domain_choice][idx]
        frame_keys = sorted(os.listdir(video_dir))
        frame_key = random.choice(frame_keys)
        frame = Image.open(os.path.join(video_dir, frame_key))
        if self.transform is not None:
           frame = self.transform(frame)
        data = {'frame': frame, 'label': 0 if domain_choice==4 else 1, 'domain_label':domain_choice}
        return data

# For Test
class DFDC(ImageDataset):
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
           
class CelebDF(ImageDataset):
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

class VFHQ(ImageDataset):
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
        
        
class DFF(ImageDataset):
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
    
    def _get_splits(self, video_keys):
        # Default split logic. Redefine the function if needed
        if self.mode == 'train':
            video_keys = video_keys[:int(len(video_keys)*0.8)]
        elif self.mode == 'val':
            video_keys = video_keys[int(len(video_keys)*0.8):int(len(video_keys)*0.9)]
        elif self.mode == 'test':
            video_keys = video_keys[int(len(video_keys)*0.9):]

        return video_keys
    
if __name__ == "__main__":
    import torchvision
    
    dataset_classes = {'/root/volume3/dfdc_preprocessed': DFDC,
                       '/root/datasets/celeb': CelebDF,
                       '/root/datasets/ff': FaceForensics,
                       '/root/datasets/vfhq': VFHQ,
                       '/root/volume3/dff_preprocessed': DFF,}
    
    for root, cls in dataset_classes.items():
        if 'dfdc' not in root:
            traindataset = cls(root_path=root, split='train', image_size=128, transform=None, num_frames=16)
        valdataset = cls(root_path=root, split='val', image_size=128, transform=None, num_frames=16)
        testdataset = cls(root_path=root, split='test', image_size=128, transform=None, num_frames=16)
        
        mean = np.zeros(3)
        std = np.zeros(3)
        from tqdm import tqdm
        dataset_zip = traindataset+valdataset+testdataset if "dfdc" not in root else valdataset+testdataset
        for samples in tqdm(dataset_zip):
            frame = np.array(samples["frame"])
            frame = torchvision.transforms.functional.to_tensor(frame)
            mean[0] += frame[0, :, :].mean()
            mean[1] += frame[1, :, :].mean()
            mean[2] += frame[2, :, :].mean()
            
            std[0] += frame[0, :, :].std()
            std[1] += frame[1, :, :].std()
            std[2] += frame[2, :, :].std()
            
        mean /= len(dataset_zip)
        std /= len(dataset_zip)
        
        print("===== Dataspecific mean, std ===== for", root)
        print(mean, std)