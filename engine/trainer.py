import time
import os
import torch
import torch.nn as nn
import os.path as osp
import pdb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import defaultdict

class Trainer():
    def __init__(self, opt, data_loader, model, logger):
        self.opt = opt
        self.type = opt.DATA.type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.train_set = data_loader['train'][0] if self.type == 'all_type_videos' else None
        self.train_sampler = data_loader['train'][1] if self.type == 'all_type_videos' else None
        self.train_loader = data_loader['train'] if self.type != 'all_type_videos' else None
        self.val_loader = data_loader['val'] 
        
        self.model = model.to(self.device)
        self.model_name = opt.MODEL.name
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.TRAIN.lr)
        self.loss_function = nn.CrossEntropyLoss().to(self.device)
        self.logger = logger

        self.total_epoch = opt.TRAIN.epochs
        self.log_interval = opt.TRAIN.log_interval
        self.save_interval = opt.TRAIN.save_interval
        self.ckpt_dir = os.path.join('..', 'result', opt.EXP_NAME, f'{self.total_epoch}_epochs_{opt.DATA.batch_size}_batch_{opt.TRAIN.lr}_lr')
        self.load_ckpt_dir = opt.TRAIN.load_ckpt_dir
        
        ## load model
        if self.load_ckpt_dir != 'None':
            self.load_model()
            print('load model from ', self.load_ckpt_dir)
        else:
            print('no ckpt to load!')
        
        
    def train(self):  
        ## start train
        total_steps = 0
        err_list = []
        print("Start training,,,")
        print("device: ", self.device)
        
        for epoch in range(self.total_epoch):
            self.model.train()
            steps = 0
            err_sum = 0
            for_loop_length = len(self.train_sampler) if self.type == 'all_type_videos' else len(self.train_loader)
            
            if self.type != 'all_type_videos':
                for data in self.train_loader:
                    steps += 1
                    total_steps += 1
                    
                    # run step
                    err_sum += self.run_step(data)
                    self.mid_record(steps, err_sum, total_steps, epoch, for_loop_length)
            else:
                for _ in range(for_loop_length):
                    steps += 1
                    total_steps += 1
                    data = self.process_data(next(iter(self.train_sampler)))
                
                    # run step
                    err_sum += self.run_step(data)
                    self.mid_record(steps, err_sum, total_steps, epoch, for_loop_length)
            
            err_list.append(err_sum/for_loop_length)
            total, correct, val_loss, val_auc = self.validate()
            self.logger.info('Epoch: %d/%d, Train loss: %.6f, val loss: %.6f, Accuracy: %.2f, AUC: %.2f' %(epoch+1, self.total_epoch, err_sum/for_loop_length, val_loss, 100*correct/total, val_auc))
        self.save_model(total_steps, epoch)
        self.save_train_loss_graph(err_list, 'label')
        self.logger.info('Finished Training : total steps %d' %total_steps)
        
    def train_dann(self):
        ## start train
        total_steps = 0
        err_label_list = []
        err_domain_list = []
        print("Start training,,, with DANN")
        print("device: ", self.device)
        
        for epoch in range(self.total_epoch):
            self.model.train()
            steps = 0
            err_label_sum = 0
            err_domain_sum = 0
            for_loop_length = len(self.train_sampler) if self.type == 'all_type_videos' else len(self.train_loader)
            if self.type != 'all_type_videos':
                for data in self.train_loader:
                    p = float(steps + epoch * for_loop_length) / self.total_epoch / for_loop_length
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1
                    # alpha *= 100
                    steps += 1
                    total_steps += 1
                    
                    weight = 0.005 + 0.995*((epoch/self.total_epoch)**1.3)  
                        
                    train_loss, err_label, err_domain = self.run_step_dann(data, alpha, weight)
                    err_label_sum += err_label
                    err_domain_sum += err_domain
                    self.mid_record_dann(steps, err_label_sum, err_domain_sum, total_steps, epoch, for_loop_length)
            else:
                for _ in range(for_loop_length):
                    p = float(steps + epoch * for_loop_length) / self.total_epoch / for_loop_length
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1
                    steps += 1
                    total_steps += 1
                    data = self.process_data(next(iter(self.train_sampler)))
                    
                    train_loss, err_label, err_domain = self.run_step_dann(data, alpha)
                    err_label_sum += err_label
                    err_domain_sum += err_domain
                    self.mid_record_dann(steps, err_label_sum, err_domain_sum, total_steps, epoch, for_loop_length)
            
            err_label_list.append(err_label_sum/for_loop_length)
            err_domain_list.append(err_domain_sum/for_loop_length)

            total, correct, val_loss, val_auc = self.validate()
            self.logger.info('Epoch: %d/%d, Train loss: %.6f, val loss: %.6f, Accuracy: %.2f, AUC: %.2f' %(epoch+1, self.total_epoch, train_loss/for_loop_length, val_loss, 100*correct/total, val_auc))
        self.save_model(total_steps, epoch)
        self.save_train_loss_graph(err_label_list, 'label')
        self.save_train_loss_graph(err_domain_list, 'domain')
        self.logger.info('Finished DANN Training : total steps %d' %total_steps)

    def run_step(self, data):
        if self.optimizer is not None:
            self.optimizer.zero_grad()
        try:
            class_output = self.model(data['frame'].to(self.device))
            train_loss = self.loss_function(class_output, data['label'].to(self.device))
        except:
            class_output = self.model(data['frame'].to(self.device)).logits
            train_loss = self.loss_function(class_output, data['label'].to(self.device))
        train_loss.backward()
        if self.optimizer is not None:
            self.optimizer.step()
        return train_loss.item()
    
    def run_step_dann(self, data, alpha, weight=1):
        if self.optimizer is not None:
            self.optimizer.zero_grad()
        class_output, domain_output = self.model(data['frame'].to(self.device), alpha)
        if self.type != 'all_type_videos':
            domain_mask = (data["domain_label"] != 4)
            domain_output = domain_output[domain_mask, :]
            domain_label = data["domain_label"][domain_mask]
        else:
            domain_label = data["domain_label"]
        err_label = self.loss_function(class_output, data['label'].to(self.device))
        err_domain = self.loss_function(domain_output, domain_label.to(self.device))
        train_loss = weight*err_label + err_domain
        train_loss.backward()
        if self.optimizer is not None:
            self.optimizer.step()
        return train_loss.item(), err_label.item(), err_domain.item()
    
    def process_data(self, idx_list):
        data = defaultdict(list)
        for idx in idx_list:
            data['frame'] += [self.train_set[idx]['frame']]
            data['label'] += [self.train_set[idx]['label']]
            data['domain_label'] += [self.train_set[idx]['domain_label']]
        data['frame'] = torch.stack(data['frame'])
        data['label'] = torch.LongTensor(data['label'])
        data['domain_label'] = torch.LongTensor(data['domain_label'])
        return data
    
    def mid_record(self, steps, err_sum, total_steps, epoch, for_loop_length):
        if self.logger is not None:
            if steps%self.log_interval == 0:
                self.logger.info(f"err: {err_sum/steps:>7f}  [{steps:>5d}/{for_loop_length:>5d}]")

        if total_steps%self.save_interval == 0:
            self.save_model(total_steps, epoch)
            
    def mid_record_dann(self, steps, err_label_sum, err_domain_sum, total_steps, epoch, for_loop_length):
        if self.logger is not None:
            if steps%self.log_interval == 0:
                self.logger.info(f"err_label: {err_label_sum/steps:>7f}  err_domain: {err_domain_sum/steps:>7f}  [{steps:>5d}/{for_loop_length:>5d}]")

        if total_steps%self.save_interval == 0:
            self.save_model(total_steps, epoch)
    

    def validate(self):
        total, correct, val_loss = 0, 0, 0
        self.model.eval()
        with torch.no_grad():
            pred, true = [], []
            for data in self.val_loader:
                # run step
                if 'dann' in self.model_name:
                    class_output, _ = self.model(data['frame'].to(self.device), 0)
                else:
                    try:
                        class_output = self.model(data['frame'].to(self.device))
                    except:
                        class_output = self.model(data['frame'].to(self.device)).logits
                
                _, predicted = torch.max(class_output.data, 1)
                
                pred += class_output[:, 1].data.cpu().tolist()
                true += data['label'].cpu()
                total += data['label'].size(0)
                correct += (predicted == data['label'].to(self.device)).sum().item()
                val_loss += self.loss_function(class_output.to(self.device), data['label'].to(self.device)).item()
            auc_score = roc_auc_score(true, pred) * 100
            del pred, true
            torch.cuda.empty_cache()
        return total, correct, val_loss/len(self.val_loader), auc_score

    def load_model(self):
        self.model.load_state_dict(torch.load(self.load_ckpt_dir))
            
    def save_model(self, steps, epoch):
        os.makedirs(self.ckpt_dir, exist_ok=True)
        torch.save(self.model.state_dict(), osp.join(self.ckpt_dir, f'step{steps}_ep{epoch+1}.pt'))

    def save_train_loss_graph(self, train_loss_list, type):   
        epochs = [i+1 for i in range(self.total_epoch)]   
        if not isinstance(train_loss_list, list):
            train_loss_list = [train_loss_list]
        train_loss_list = [loss for loss in train_loss_list]
        plt.plot(epochs, train_loss_list, label='Train Loss')
        plt.xlabel('epoch')
        plt.ylabel('train loss')
        plt.savefig(osp.join(self.ckpt_dir, 'train_{}_loss.png'.format(type)))
        plt.close()
