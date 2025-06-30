# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 11:05:08 2023

@author: Shaik
"""
import tqdm
import torch


from torchmetrics import F1Score


def F1_score(model, optimizer , sample_val_loader ,DEVICE ,f1_score_list , f1_score_list_ind ):
# def F1_score(model, optimizer , sample_val_loader ,DEVICE):
    
    print("calculating F1_Score")
    
    model.eval()
    
    pbar = tqdm.tqdm(sample_val_loader,total=len(sample_val_loader))
    for i, data in enumerate(pbar):
        # print(i)
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = list({k: v.to(DEVICE) for k, v in t.items()} for t in targets)
        # tar = targets[0]["labels"]      
        
        with torch.no_grad():
            preds = model(images)
        
        btch_pred_lbl = [pred["labels"] for pred in preds]
        btch_trgt_lbl = [targets["labels"] for targets in targets]
        btch_pred_score =  [pred["scores"] for pred in preds]  
        
        btch_pred_lbl_final = []
        for s, p in zip (btch_pred_score, btch_pred_lbl):
            if s.numel() > 0:
                max_idx = torch.argmax(s)
                btch_pred_lbl_final.append(p[max_idx])
            else:
                max_idx = None
                # a = torch.tensor(-1)
                a = torch.tensor(0)
                
                btch_pred_lbl_final.append(a)
                # print(t)
                
            
        btch_pred_lbl_final_stacked = torch.stack(btch_pred_lbl_final)
        btch_trgt_lbl_final_stacked = torch.stack(btch_trgt_lbl)
        metric = F1Score(num_classes=2)
        # f1_score_comb = metric(btch_trgt_lbl_final_stacked,btch_pred_lbl_final_stacked )
        f1_score_comb = metric(btch_pred_lbl_final_stacked, btch_trgt_lbl_final_stacked )
        f1_score_list.append(f1_score_comb)
                                       
    
    return f1_score_list,f1_score_list_ind
            
        



        