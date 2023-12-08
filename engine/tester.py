import time

import torch
import torch.nn as nn
import os.path as osp
import pdb
from sklearn.metrics import roc_auc_score

class Tester():
    def __init__(self, opt, data_loader, model, logger):
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = data_loader['train']
        self.val_loader = data_loader['val']
        self.test_loader = data_loader['test']
        self.test_load_ckpt_dir = opt.get('TEST', {}).get('load_ckpt_dir', 'None')
        self.test_load_ckpt_dir2 = opt.get('TEST', {}).get('load_ckpt_dir2', 'None')
        if isinstance(model, list):
            self.model = []
            for m in model:
                self.model.append(m.to(self.device))
        else:
            self.model = model.to(self.device)
        self.model_name = opt.MODEL.name
        self.loss_function = nn.CrossEntropyLoss().to(self.device)
        self.logger = logger
        
        ## load model
        if self.test_load_ckpt_dir != 'None':
            self.load_model()
            print('load model from ', self.test_load_ckpt_dir)
        else:
            print('no ckpt to load!')
        
    def test(self, split="test"):
        print("Test ...")
        total, correct, loss = 0, 0, 0
        accuracy = []
        self.model.eval()
        if split == "train":
            dataloader = self.train_loader
        elif split == "test":
            dataloader = self.test_loader
        else:
            dataloader = self.val_loader
        with torch.no_grad():
            pred, true = [], []
            for data in dataloader:
                if 'dann' in self.model_name:
                    class_output, _ = self.model(data['frame'].to(self.device), 0)
                else:
                    try:
                        class_output = self.model(data['frame'].to(self.device))
                    except:
                        class_output = self.model(data['frame'].to(self.device)).logits
                _, predicted = torch.max(class_output.data, 1)
                
                pred += class_output.data[:, 1].cpu().tolist()
                true += data['label'].cpu()
                total += data['label'].size(0)
                correct += (predicted == data['label'].to(self.device)).sum().item()
                loss = self.loss_function(class_output, data['label'].to(self.device)).item()
                accuracy.append(100 * correct/total)
            auc_score = roc_auc_score(true, pred) * 100
        self.logger.info('[ %s result ] loss: %.6f, Accuracy: %.2f, AUC: %.2f' %(split, loss, 100*correct/total, auc_score))
        
    def test_video(self, split="test"): # for video dataset
        print("Video Test ...")
        total, correct, loss = 0, 0, 0
        self.model.eval()
        
        if split == "train":
            dataloader = self.train_loader
        elif split == "test":
            dataloader = self.test_loader
        else:
            dataloader = self.val_loader
            
        with torch.no_grad():
            pred = []
            true = []
            total = len(dataloader.dataset.videos)
            
            tmp_pred, tmp_true = [], []
            cur_video = 0
            
            for data in dataloader:
                if 'dann' in self.model_name:
                    class_output, _ = self.model(data['frame'].to(self.device), 0)
                else:
                    try:
                        class_output = self.model(data['frame'].to(self.device))
                    except:
                        class_output = self.model(data['frame'].to(self.device)).logits
                
                pred_results = class_output.data.cpu().tolist()
                for i, batch_inst in enumerate(data['video']):
                    if batch_inst != cur_video:
                        cur_video = batch_inst
                        ensembled_outputs = torch.stack(tmp_pred).mean(dim=0)
                        _, prediction = torch.max(ensembled_outputs, 0)
                        
                        pred.append(ensembled_outputs[1].item())
                        correct += int(prediction == tmp_true)
                        true.append(tmp_true.item())
                        tmp_pred, tmp_true = [], []
                        
                    tmp_pred += [torch.tensor(pred_results[i]).to(self.device)]
                    tmp_true = data['label'][i]
                    
            if len(tmp_pred) != 0:
                ensembled_outputs = torch.stack(tmp_pred).mean(dim=0)
                _, prediction = torch.max(ensembled_outputs, 0)
                pred.append(ensembled_outputs[1].item())
                correct += int(prediction == tmp_true)
                true.append(tmp_true.item())
                
            print(f'{len(true)}/{total} videos evaluated.')
            auc_score = roc_auc_score(true, pred) * 100
        self.logger.info('[ %s result ] loss: %.6f, Accuracy: %.2f, AUC: %.2f' %(split, loss, 100*correct/total, auc_score))
        
        
    def test_ensemble(self, split="test"):
        print("Ensemble Video Test ...")
        total, correct, loss = 0, 0, 0
        self.model[0].eval()
        self.model[1].eval()
        
        if split == "train":
            dataloader = self.train_loader
        elif split == "test":
            dataloader = self.test_loader
        else:
            dataloader = self.val_loader
            
        with torch.no_grad():
            pred, true = [], []
            total = len(dataloader.dataset.videos)
            
            tmp_pred1, tmp_pred2, tmp_true = [], [], []
            cur_video = 0
            
            for data in dataloader:
                # dann model
                class_output1, _ = self.model[0](data['frame'].to(self.device), 0)
               
                # contrastive model
                try:
                    class_output2 = self.model[1](data['frame'].to(self.device))
                except:
                    class_output2 = self.model[1](data['frame'].to(self.device)).logits
                
                pred_results1 = class_output1.data.cpu().tolist()
                pred_results2 = class_output2.data.cpu().tolist()
                for i, batch_inst in enumerate(data['video']):
                    if batch_inst != cur_video:
                        cur_video = batch_inst
                        ensembled_outputs1 = torch.stack(tmp_pred1).mean(dim=0)
                        ensembled_outputs2 = torch.stack(tmp_pred2).mean(dim=0)
                        model_ensemble = (ensembled_outputs1 + ensembled_outputs2) / 2
                        _, prediction = torch.max(model_ensemble, 0)
                        
                        pred.append(model_ensemble[1].item())
                        correct += int(prediction == tmp_true)
                        true.append(tmp_true.item())
                        tmp_pred1, tmp_pred2, tmp_true = [], [], []
                        
                    tmp_pred1 += [torch.tensor(pred_results1[i]).to(self.device)]
                    tmp_pred2 += [torch.tensor(pred_results2[i]).to(self.device)]
                    tmp_true = data['label'][i]
                
            # Evaluate last video
            if len(tmp_pred1) != 0:
                ensembled_outputs1 = torch.stack(tmp_pred1).mean(dim=0)
                ensembled_outputs2 = torch.stack(tmp_pred2).mean(dim=0)
                model_ensemble = (ensembled_outputs1 + ensembled_outputs2) / 2
                _, prediction = torch.max(model_ensemble, 0)
                pred.append(model_ensemble[1].item())
                correct += int(prediction == tmp_true)
                true.append(tmp_true.item())
                
            print(f'{len(true)}/{total} videos evaluated.')
            auc_score = roc_auc_score(true, pred) * 100
        self.logger.info('[ %s result ] loss: %.6f, Accuracy: %.2f, AUC: %.2f' %(split, loss, 100*correct/total, auc_score))
            
        
    def load_model(self):
        if isinstance(self.model, list):
            self.model[0].load_state_dict(torch.load(self.test_load_ckpt_dir))
            self.model[1].load_state_dict(torch.load(self.test_load_ckpt_dir2))
        else:
            self.model.load_state_dict(torch.load(self.test_load_ckpt_dir))
                    