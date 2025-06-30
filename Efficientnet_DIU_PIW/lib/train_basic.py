# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 12:36:19 2023

@author: Shaik
"""

import tqdm
import torch


#%%function for running training iterations

def train_basic(train_data_loader, model, optimizer, DEVICE, train_loss_hist, train_loss_list, train_itr):
    print('Training')
    
   
    model.train()
     # initialize tqdm progress bar
    prog_bar = tqdm.tqdm(train_data_loader, total=len(train_data_loader), position=0, leave=True)
    
    for i, data in enumerate(prog_bar):
    
        optimizer.zero_grad()
        images, targets = data
      
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)
        train_loss_hist.send(loss_value)
        losses.backward()
        optimizer.step()
        train_itr += 1
        
    
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    
    
    return train_loss_list , train_loss_hist

