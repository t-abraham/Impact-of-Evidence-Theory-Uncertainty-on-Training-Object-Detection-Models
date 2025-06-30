# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 15:17:26 2023

@author: Shaik
"""

import torch
import tqdm
import numpy as np

def get_iou(sample_val_loader,model, optimizer, DEVICE , sample_val_list_iou):
   
    print("calculating IOU")
    
    model.eval()
    
    pbar = tqdm.tqdm(sample_val_loader,total=len(sample_val_loader))
    for i, data in enumerate(pbar):
        
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = list({k: v.to(DEVICE) for k, v in t.items()} for t in targets)
        # tar = targets[0]["boxes"]      
        
        with torch.no_grad():
            preds = model(images)
            
        btch_pred = [pred["boxes"] for pred in preds]
        btch_trgt = [targets["boxes"] for targets in targets]
        btch_pred_score =  [pred["scores"] for pred in preds]  
        
        btch_pred_final = []
        for s, p in zip (btch_pred_score, btch_pred):
            if s.numel() > 0:
                max_idx = torch.argmax(s)
                btch_pred_final.append(p[max_idx])
            else:
                max_idx = None
                # a = torch.tensor(-1)
                a = torch.tensor([0,0,0,0])
                
                btch_pred_final.append(a)
        for tar, pred in zip(btch_trgt,btch_pred_final):
            
            tar =  tar.squeeze()       
            # Coordinates of the area of intersection.
            ix1 = torch.max(tar[0], pred[0])
            iy1 = torch.max(tar[1], pred[1])
            ix2 = torch.min(tar[2], pred[2])
            iy2 = torch.min(tar[3], pred[3])
        
            # Intersection height and width.
            i_height = torch.max(iy2 - iy1 + 1, torch.tensor(0.))
            i_width = torch.max(ix2 - ix1 + 1, torch.tensor(0.))
            
            area_of_intersection = i_height * i_width
            
            # Ground Truth dimensions.
            gt_height = tar[3] - tar[1] + 1
            gt_width = tar[2] - tar[0] + 1
            
            # Prediction dimensions.
            pd_height = pred[3] - pred[1] + 1
            pd_width = pred[2] - pred[0] + 1
         
            
            area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
            
            iou = area_of_intersection / area_of_union
            sample_val_list_iou.append(iou)
    
            
               
        
    
    
    return  sample_val_list_iou
     
if __name__ == "__main__":
    
    
    
    pass

   