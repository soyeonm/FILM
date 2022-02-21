#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 16:35:46 2021

@author: soyeonmin
"""
import skimage.morphology
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sem_map_model import UNetMulti
from utils.model import Flatten
import math
import argparse
import torch.nn as nn
import copy
import os
import cv2
import random
import time



def print_logs(statements):
    print(statements)
    logs.append(statements)
    

class DataLoadPreprocess(Dataset):
    def __init__(self,  mode='train', grid_size=8, uniform=True, mask=True):
        if mode == 'train':
            lists_path = 'models/semantic_policy/data/maps/train/list_of_dicts_multitask.p'
            self.maps_dir = 'models/semantic_policy/data/maps/train/'
        else:
            lists_path = 'models/semantic_policy/data/maps/tests_unseen/list_of_dicts_tests_multitask.p'
            self.maps_dir = 'models/semantic_policy/data/maps/tests_unseen/'
        self.list_of_dicts = pickle.load(open(lists_path, 'rb'))
        if mode == 'train':
            np.random.seed(0)
            np.random.shuffle(self.list_of_dicts)
        all_objects = ['AlarmClock', 'Apple', 'AppleSliced', 'BaseballBat', 'BasketBall', 'Book', 'Bowl', 'Box', 'Bread', 'BreadSliced', 'ButterKnife', 'CD', 'Candle', 'CellPhone', 'Cloth', 'CreditCard', 'Cup', 'DeskLamp', 'DishSponge', 'Egg', 'Faucet', 'FloorLamp', 'Fork', 'Glassbottle', 'HandTowel', 'HousePlant', 'Kettle', 'KeyChain', 'Knife', 'Ladle', 'Laptop', 'LaundryHamperLid', 'Lettuce', 'LettuceSliced', 'LightSwitch', 'Mug', 'Newspaper', 'Pan', 'PaperTowel', 'PaperTowelRoll', 'Pen', 'Pencil', 'PepperShaker', 'Pillow', 'Plate', 'Plunger', 'Pot', 'Potato', 'PotatoSliced', 'RemoteControl', 'SaltShaker', 'ScrubBrush', 'ShowerDoor', 'SoapBar', 'SoapBottle', 'Spatula', 'Spoon', 'SprayBottle', 'Statue', 'StoveKnob', 'TeddyBear', 'Television', 'TennisRacket', 'TissueBox', 'ToiletPaper', 'ToiletPaperRoll', 'Tomato', 'TomatoSliced', 'Towel', 'Vase', 'Watch', 'WateringCan', 'WineBottle']
        self.all_objects2idx = {o:i for i, o in enumerate(all_objects)}
        map_save_large_objects = ['ArmChair', 'BathtubBasin', 'Bed', 'Cabinet', 'Cart', 'CoffeeMachine', 'CoffeeTable', 'CounterTop', 'Desk', 'DiningTable', 'Drawer', 'Dresser', 'Fridge', 'GarbageCan', 'Microwave', 'Ottoman', 'Safe', 'Shelf', 'SideTable', 'SinkBasin', 'Sofa', 'StoveBurner', 'TVStand', 'Toilet']
        self.large_objects2idx = {obj:i for i, obj in enumerate(map_save_large_objects)}
        self.grid_size = grid_size
        self.uniform= uniform
        self.mask = mask

    def __getitem__(self, idx):
        cur_dict = self.list_of_dicts[idx]
        map_learned = pickle.load(open(self.maps_dir+cur_dict['map_learned'], 'rb')).cpu()
        total_cat2idx_learned = pickle.load(open(self.maps_dir+ '/'.join(cur_dict['map_learned'].split('/')[:-1]) +'/total_cat2idx.p', 'rb'))
        map_gt = pickle.load(open(self.maps_dir+cur_dict['map_gt'], 'rb')).cpu()
        total_cat2idx_gt = pickle.load(open(self.maps_dir+ '/'.join(cur_dict['map_gt'].split('/')[:-1]) +'/total_cat2idx.p', 'rb'))
        map_learned_mask = self.mask_grid(map_gt[0])
        map_learned_reconst = torch.zeros((4+len(self.large_objects2idx),240,240))
        map_learned_reconst[:4] = map_learned[:4]
        for cat, catid in total_cat2idx_learned.items():
            if cat in self.large_objects2idx:
                map_learned_reconst[4+self.large_objects2idx[cat]] = map_learned[4+catid]
                
        #reconst map gt
        map_gt_reconst = torch.zeros((len(self.all_objects2idx),8,8))
        for cat, catid in total_cat2idx_gt.items():
            if cat in self.all_objects2idx:
                map_gt_reconst[self.all_objects2idx[cat]] = self.into_grid(map_gt[4+catid], map_learned_mask)

        return_dict = {'map_learned': map_learned_reconst, 'map_gt': map_gt_reconst, 'mask_grid': map_learned_mask, 'map_gt_obs': map_gt[0]}
        return return_dict

    def into_grid(self, ori_grid, mask):
        one_cell_size = math.ceil(240/self.grid_size)
        return_grid = torch.zeros(self.grid_size,self.grid_size)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if torch.sum(ori_grid[one_cell_size *i: one_cell_size*(i+1),  one_cell_size *j: one_cell_size*(j+1)].bool().float())>0:
                    return_grid[i,j] = 1
        
        if torch.sum(return_grid)!=0:
            return_grid = return_grid/ torch.sum(return_grid)

        return return_grid

    def into_grid_simple(self, ori_grid):
        one_cell_size = math.ceil(240/self.grid_size)
        return_grid = torch.zeros(self.grid_size,self.grid_size)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if torch.sum(ori_grid[one_cell_size *i: one_cell_size*(i+1),  one_cell_size *j: one_cell_size*(j+1)].bool().float()) >0:
                    return_grid[i,j] = 1

        return return_grid.unsqueeze(0)

    def mask_grid(self, ori_grid):
        ori_grid = ori_grid.cpu().numpy()
        ori_grid = skimage.morphology.binary_dilation(ori_grid, skimage.morphology.disk(3))
        center= (120,120)
        connected_regions = skimage.morphology.label(1-ori_grid, connectivity=2)
        connected_lab = connected_regions[120,120]
        mask = np.zeros((240,240))
        mask[np.where(connected_regions==connected_lab)] = 1
        mask[np.where(ori_grid)] = 1
        mask = mask.astype(bool).astype(float) #0 for everywhere else

        #output mask in grid
        mask = torch.tensor(mask)
        mask = self.into_grid_simple(mask)
        return mask        
    #
    def __len__(self):
        return len(self.list_of_dicts)

def KLDivergence(input_prob, target_prob):
    each_class = target_prob * (torch.log(target_prob) - torch.log(input_prob)) #shape is 32 x 64 
    each_class = each_class[torch.where(target_prob!=0)]
    return torch.mean(each_class)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_freq', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--dn', type=str, default='none')
    parser.add_argument('--seed', type=int, default=0)
    
    args = parser.parse_args()

    #Make dirs
    if not os.path.exists('map_pred_models_multi/logs/'):
        os.makedirs('map_pred_models_multi/logs/')




    def torch_seed(seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    N = args.batch_size 
    maps_dir = 'models/semantic_policy/data/maps/train/'
    
    logs = []
    
    
    dataset = DataLoadPreprocess()
    dataloader = DataLoader(dataset, N)
    dataset_eval = DataLoadPreprocess(mode='eval')
    dataloader_eval = DataLoader(dataset_eval, 1)
    device = torch.device('cuda')
    print_logs("Done until device!")
    
    num_epochs = args.num_epochs
    
    torch_seed(args.seed)
    model = UNetMulti((240,240), num_sem_categories=24).to(device=device)
    print("device is ", device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-6)
    loss_fn = KLDivergence
    softmax = nn.Softmax(dim=2)
    min_dist = 100000
    prev_test_loss = 1000
    flatten = Flatten()
    
    all_objects = ['AlarmClock', 'Apple', 'AppleSliced', 'BaseballBat', 'BasketBall', 'Book', 'Bowl', 'Box', 'Bread', 'BreadSliced', 'ButterKnife', 'CD', 'Candle', 'CellPhone', 'Cloth', 'CreditCard', 'Cup', 'DeskLamp', 'DishSponge', 'Egg', 'Faucet', 'FloorLamp', 'Fork', 'Glassbottle', 'HandTowel', 'HousePlant', 'Kettle', 'KeyChain', 'Knife', 'Ladle', 'Laptop', 'LaundryHamperLid', 'Lettuce', 'LettuceSliced', 'LightSwitch', 'Mug', 'Newspaper', 'Pan', 'PaperTowel', 'PaperTowelRoll', 'Pen', 'Pencil', 'PepperShaker', 'Pillow', 'Plate', 'Plunger', 'Pot', 'Potato', 'PotatoSliced', 'RemoteControl', 'SaltShaker', 'ScrubBrush', 'ShowerDoor', 'SoapBar', 'SoapBottle', 'Spatula', 'Spoon', 'SprayBottle', 'Statue', 'StoveKnob', 'TeddyBear', 'Television', 'TennisRacket', 'TissueBox', 'ToiletPaper', 'ToiletPaperRoll', 'Tomato', 'TomatoSliced', 'Towel', 'Vase', 'Watch', 'WateringCan', 'WineBottle']
    all_idx2objects = {i:o for i, o in enumerate(all_objects)}
    for epoch in range(num_epochs):
        training_loss = 0
        print_logs("Training epoch " + str(epoch))
        start = time.time()
        for step, sample_batched in enumerate(dataloader):
            if step == 1:
                print("time took for first step: ",time.time() - start)
            
            if step %100 == 0 and step >0: #and step>0:
                print_logs("Training step "+ str(step))
            optimizer.zero_grad()
            model.train()
            map_learned = sample_batched['map_learned'].to(device=device) #(N x channels x 240 x  240)
            map_gt = sample_batched['map_gt'].to(device=device) #(N x1 x 8x8)
            batch_size_cur = map_gt.shape[0]
                        
            pred_probs = model(map_learned) #shape is Nx 64
            pred_probs = pred_probs.view(batch_size_cur, 73, -1)
            pred_probs = softmax(pred_probs)
            
            loss = loss_fn(pred_probs, map_gt.view(batch_size_cur, 73, -1))
            loss.backward()
            optimizer.step()
            training_loss += loss.detach()
            
            if step % args.eval_freq == 0 and step>0:
                model.eval()
                print_logs("Training loss was " + str (training_loss/args.eval_freq))
                dist = 0
                training_loss = 0
                test_loss = 0
                with torch.no_grad():
                    for e_step, eval_batched in enumerate(dataloader_eval):
                       batch_size_cur =1
                       map_learned =  eval_batched['map_learned'].to(device=device)
                       pred_probs = model(map_learned)
                       pred_probs = pred_probs.view(batch_size_cur, 73, -1)
                       pred_probs = softmax(pred_probs)
                       argmax = torch.argmax(pred_probs, dim = 2)
                       argmax = argmax.view(-1)
                       x= (argmax/8.0).int().float()
                       y = argmax - 8* x
                       map_gt = eval_batched['map_gt'].to(device=device).view(batch_size_cur, 73, -1)
                       argmax_gt = torch.argmax(map_gt, dim = 2)
                       argmax_gt = argmax_gt.view(-1)
                       xg= (argmax_gt/8.0).int().float()
                       yg = argmax_gt - 8* xg
                       dist += torch.sqrt(torch.dot((x-xg), (x-xg).T) + torch.dot((y-yg), (y-yg).T))
                       loss = 0.0
                       loss = loss_fn(pred_probs, map_gt)
                       test_loss += loss
                       
                       #Save
                       if e_step %5 ==0:
                            for k in range(73):
                                if torch.sum(eval_batched['map_gt'].squeeze()[k]) ==0:
                                    pass
                                else:
                                    #save visualization
                                    cat_name = all_idx2objects[k]
                                    cur_dir =  'models/semantic_policy/data/map_vis/vis_seed' + str(args.seed) + '_lr' + str(args.lr) + '/' + cat_name + str(epoch) + '_' + str(step) + '_' + str(e_step) +'/'
                                    if not os.path.exists(cur_dir):
                                        os.makedirs(cur_dir)
                                    save = eval_batched['map_gt_obs'].cpu().view((240,240)).numpy()
                                    save = save/ np.max(save)
                                    save = (255*save).astype(np.uint8)
                                    cv2.imwrite(cur_dir + 'obstruction_gt.png', save)
                                    pickle.dump(eval_batched['map_gt_obs'].cpu().view((240,240)).numpy(), open(cur_dir+ "obstruction_gt.p", "wb"))
                                    
                                    save = eval_batched['map_learned'][0,0].cpu().view((240,240)).numpy()
                                    save = save/ np.max(save)
                                    save = (255*save).astype(np.uint8)
                                    cv2.imwrite(cur_dir + 'obstruction_pred.png', save)
                                    pickle.dump(eval_batched['map_learned'][0,0].cpu().view((240,240)).numpy(), open(cur_dir+ "obstruction_pred.p", "wb"))
                                    
                                    save = cv2.resize(pred_probs[0,k].view(8,8).cpu().numpy(), (240,240))
                                    save = save/ np.max(save)
                                    save = (255*save).astype(np.uint8)
                                    cv2.imwrite(cur_dir + 'pred_probs.png', save)
                                    pickle.dump(pred_probs[0,k].view(8,8).cpu().numpy(), open(cur_dir+ "pred_probs.p", "wb"))
                                    
                                    
                                    save = cv2.resize(map_gt[0,k].view(8,8).cpu().numpy(), (240,240))
                                    save = save/ np.max(save)
                                    save = (255*save).astype(np.uint8)
                                    cv2.imwrite(cur_dir + 'gt_probs.png', save)
                                    pickle.dump(map_gt[0,k].view(8,8).cpu().numpy(), open(cur_dir+ "gt_probs.p", "wb"))
                       
                    dist = torch.sum(dist).detach().cpu().item() / len(dataset_eval)
                    test_loss = test_loss/ len(dataset_eval)
                    print_logs("dist is "+ str(dist))
                    print_logs("test loss is "+ str(test_loss))
                    model_cop = copy.deepcopy(model)
                    model_cop = model_cop.cpu()
                    prev_test_loss = test_loss
                    torch.save(model_cop.state_dict(), 'map_pred_models_multi/'+ str(args.dn) + 'seed_' + str(args.seed) + 'lr_' + str(args.lr) + 'epoch_' + str(epoch)  + '_step_' + str(step) + '.pt')
                    pickle.dump((x,y), open("xy.p", "wb")) 
                    pickle.dump((xg,yg), open("xgyg.p", "wb")) 
                    del model_cop
                    
                    
                        
                    
                #save logs    
                f = open("map_pred_models_multi/logs/" + str(args.lr) + "logs_" + str(args.dn) + 'seed_' + str(args.seed) + "_lr" + str(args.lr) + ".txt", "a")
                for l in logs:
                    f.write(l + "\n")
                logs = []
                f.close()
                    
                    
                    
            
            
            
        
        
        
        
        
        
        

