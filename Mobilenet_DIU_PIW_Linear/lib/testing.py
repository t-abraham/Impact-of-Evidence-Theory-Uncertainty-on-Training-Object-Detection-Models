# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 13:20:43 2023

@author: Shaik
"""
import torch
import tqdm
from lib.utils import metrics


def testing(test_loader, model, DEVICE, final_classes):
    print("testing")
    
    if "__background__" in final_classes:
        idx = final_classes.index("__background__")
        row = idx
        col = idx
    else:
        row = None
        col = None
        
    model.eval()
    record_metrics= metrics( final_classes, score_thres = 0.25 , iou_thres = 0.50, row=row, col=col )
    pbar = tqdm.tqdm(test_loader,total=len(test_loader), position=0, leave=True)
    for i, data in enumerate(pbar): 
        images, targets = data        
        images = list(image.to(DEVICE) for image in images)        
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]                        
        with torch.no_grad():
            preds = model(images)
        
        record_metrics.update( preds, targets )
    
    return record_metrics
            
            
        


if __name__ == "__main__":
 pass
