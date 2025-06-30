# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:46:01 2023

@author: Shaik
"""

import tqdm
def warmup_training(train_loader, model, optimizer, warmup_scheduler, DEVICE, epoch ):
    
    model.train()
    # initialize tqdm progress bar
    prog_bar = tqdm.tqdm(train_loader, total=len(train_loader))
    
    for i, data in enumerate(prog_bar):
    
        
        optimizer.zero_grad()
        images, targets = data
             
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
                
        loss_value = losses.item()
        losses.backward()
        optimizer.step()
        
        warmup_scheduler.step(epoch+1)
        
        
        current_lr = optimizer.param_groups[0]['lr']
             
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"epoch: {epoch+1} , Learning Rate: {current_lr:.6f} , Loss : {loss_value:.4f}")
    
    
    return 