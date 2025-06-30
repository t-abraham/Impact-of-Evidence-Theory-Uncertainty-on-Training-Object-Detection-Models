# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:20:29 2023

@author: Shaik
"""

import tqdm
import torch


#%%function for running training iterations


def train_Dynamic(train_data_loader, model, optimizer, DEVICE, train_loss_hist, train_loss_list, train_itr  , Multiplier , epoch , idx_range , loss_type):
    print('Dynamic_Training')
   
    model.train()
     # initialize tqdm progress bar
    prog_bar = tqdm.tqdm(train_data_loader, total=len(train_data_loader), position=0, leave=True)
    
    for i, data in enumerate(prog_bar):
        
        # print(train_itr)
        optimizer.zero_grad()
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
       
        loss_dict = model(images, targets)
         
        
        if epoch <= 0:
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            train_loss_list.append(loss_value)
            train_loss_hist.send(loss_value)
            losses.backward()
            optimizer.step()
         
        else:
            if 'loss_multiplication' == loss_type:
                losses = sum(loss for loss in loss_dict.values())
                prog_bar.set_postfix({"Multiplier": Multiplier[idx_range]})
                losses = losses * Multiplier[idx_range]
                loss_value = losses.item() 
                train_loss_list.append(loss_value)
                train_loss_hist.send(loss_value)
                losses.backward()
                optimizer.step()
            
            elif 'loss_addition' == loss_type:
                losses = sum(loss for loss in loss_dict.values())
                prog_bar.set_postfix({"Multiplier": Multiplier[idx_range]})
                losses = losses + Multiplier[idx_range]
                loss_value = losses.item()
                train_loss_list.append(loss_value)
                train_loss_hist.send(loss_value)
                losses.backward()
                optimizer.step()
                
            elif 'loss_inside_multiplication' == loss_type:
                loss_dict_values = loss_dict.values()
                prog_bar.set_postfix({"Multiplier": Multiplier[idx_range]})
                clasification_NLL_Loss = next(iter(loss_dict_values)) * Multiplier[idx_range]
                final_loss =  [clasification_NLL_Loss if tensor is next(iter(loss_dict_values)) else tensor for tensor in loss_dict_values]
                losses = sum(final_loss)
                loss_value = losses.item()
                train_loss_list.append(loss_value)
                train_loss_hist.send(loss_value)
                losses.backward()
                optimizer.step()
         
        train_itr += 1
        
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    
    
    return train_loss_list , train_loss_hist

