# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 18:44:34 2022

@author: Tahasanul
"""

#%% Initialization of Libraries and Directory

import sys, os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0,grandparentdir)

grandgrandparentdir = os.path.dirname(grandparentdir)
sys.path.insert(0,grandgrandparentdir)

import torch, tqdm, gc, typing, collections, torchvision
# from lib.utilities import roc_group_formater, process_roc

from lib.roc import roc
from lib.utils import calculate_mul_factor

#%% eval_forward

def eval_forward( model, images, targets ):
    """
    Args:
        images ( List[torch.Tensor] ): images to be processed
        targets ( List[Dict[str, torch.Tensor]] ): ground-truth boxes present in the image ( optional )
    Returns:
        result ( List[BoxList] or Dict[torch.Tensor] ): the output from the model.
            It returns List[BoxList] contains additional fields
            like `scores`, `labels` and `mask` ( for Mask R-CNN models ).
    """
    model.eval( )

    original_image_sizes: typing.List[typing.Tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        assert len( val ) == 2
        original_image_sizes.append( ( val[0], val[1] ) )

    images, targets = model.transform( images, targets )

    # Check for degenerate boxes
    if targets is not None:
        for target_idx, target in enumerate( targets ):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any( ):
                # print the first degenerate box
                bb_idx = torch.where( degenerate_boxes.any( dim=1 ) )[0][0]
                degen_bb: typing.List[float] = boxes[bb_idx].tolist( )
                raise ValueError( 
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}."
                )

    features = model.backbone( images.tensors )
    if isinstance( features, torch.Tensor ):
        features = collections.OrderedDict( [( "0", features )] )
    model.rpn.training=True
    # model.roi_heads.training=True
    
    #####proposals, proposal_losses = model.rpn( images, features, targets )
    
    features_rpn = list( features.values( ) )
    objectness, pred_bbox_deltas = model.rpn.head( features_rpn )
    anchors = model.rpn.anchor_generator( images, features_rpn )

    num_images = len( anchors )
    num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
    num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
    objectness, pred_bbox_deltas = torchvision.models.detection.rpn.concat_box_prediction_layers( objectness, pred_bbox_deltas )
    
    # apply pred_bbox_deltas to anchors to obtain the decoded proposals
    # note that we detach the deltas because Faster R-CNN do not backprop through
    # the proposals
    
    proposals = model.rpn.box_coder.decode( pred_bbox_deltas.detach( ), anchors )
    proposals = proposals.view( num_images, -1, 4 )
    proposals, scores = model.rpn.filter_proposals( proposals, objectness, images.image_sizes, num_anchors_per_level )

    proposal_losses = {}
    assert targets is not None
    labels, matched_gt_boxes = model.rpn.assign_targets_to_anchors( anchors, targets )
    regression_targets = model.rpn.box_coder.encode( matched_gt_boxes, anchors )
    loss_objectness, loss_rpn_box_reg = model.rpn.compute_loss( 
        objectness, pred_bbox_deltas, labels, regression_targets
    )
    proposal_losses = {
        "loss_objectness": loss_objectness,
        "loss_rpn_box_reg": loss_rpn_box_reg,
    }

    #####detections, detector_losses = model.roi_heads( features, proposals, images.image_sizes, targets )
    
    image_shapes = images.image_sizes
    proposals, matched_idxs, labels, regression_targets = model.roi_heads.select_training_samples( proposals, targets )
    box_features = model.roi_heads.box_roi_pool( features, proposals, image_shapes )
    box_features = model.roi_heads.box_head( box_features )
    class_logits, box_regression = model.roi_heads.box_predictor( box_features )

    result: typing.List[typing.Dict[str, torch.Tensor]] = []
    detector_losses = {}
    loss_classifier, loss_box_reg = torchvision.models.detection.roi_heads.fastrcnn_loss( class_logits, box_regression, labels, regression_targets )
    detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
    boxes, scores, labels = model.roi_heads.postprocess_detections( class_logits, box_regression, proposals, image_shapes )
    num_images = len( boxes )
    for i in range( num_images ):
        result.append( 
            {
                "boxes": boxes[i],
                "labels": labels[i],
                "scores": scores[i],
            }
        )
    detections = result
    detections = model.transform.postprocess( detections, images.image_sizes, original_image_sizes )  # type: ignore[operator]
    model.rpn.training=False
    # model.roi_heads.training=False
    losses = {}
    losses.update( detector_losses )
    losses.update( proposal_losses )
    return losses, detections

#%% validater

def validater_WGT( score_card, valid_data_loader, model, DEVICE, val_loss_hist, val_loss_list, val_itr, avg_mul_factor, avg_unc_k , k1_label_list , k_avg_list ):
    print( "Validating for loss and ROC_Multification factor_WGT"  )
    
           
    data_processed = 0
    with tqdm.tqdm( total=len( valid_data_loader.dataset ), desc = "Validation Progress", position=0, leave=True ) as pbar:
        for data in valid_data_loader:
            
            images, targets = data
            
            images = list( image.to( DEVICE ) for image in images )
            targets = [{k: v.to( DEVICE ) for k, v in t.items( )} for t in targets]
            image_count = len( images )
            
                        
            with torch.no_grad( ):
                loss_dict, preds = eval_forward( model, images, targets )
                       
            
            for i in range(len(targets)):
                # creating target masss for ROC and if duplicate label is present calculate average(which will be 1)
                m_target = {label.item(): 1 for label in targets[i]['labels']}
                            
                group_dict = {}
                for label, score in zip(preds[i]["labels"],preds[i]["scores"]):
                    if label.item() in group_dict:
                        group_dict[label.item()].append(score.item())
                    else:
                        group_dict[label.item()] = [score.item()]
                        
                # calculate the average of all the score for the duplicate labels
                m_preds = {label: sum(scores)/len(scores) for label, scores in group_dict.items()}
                
                if len(m_preds) == 0:
                    m_preds = {0:1}
                
                a = roc(5)
                               
                K_ds, roc_ds = a.perform_ds( m = m_target, m1 = m_preds )
                
                k1_label_list.append(K_ds)
                                        
                # for each epoch graph
                avg_unc_k.send(K_ds)
                k_avg_list.append(avg_unc_k.value)
                 
                mul_factor = calculate_mul_factor(a, K_ds, score_card)
                       
                avg_mul_factor.send(mul_factor)
        
            data_processed += len( images )
            
            del images, targets
            gc.collect( )
            torch.cuda.empty_cache( )
            
            losses = sum( loss for loss in loss_dict.values( ) )
            loss_value = losses.item( )
            
            val_loss_list.append(loss_value)
            val_loss_hist.send(loss_value)
            val_itr += 1
    
            # update the loss value beside the progress bar for each iteration
            pbar.set_description(desc=f"Loss: {loss_value:.4f}, mul_factor: {mul_factor:.4f}, K: {K_ds:.4f}")            
            pbar.update( image_count )
            pbar.refresh( )
            
            del losses, loss_dict, loss_value,# roc_groups, roc_preds
            gc.collect( )
            torch.cuda.empty_cache( )
        
    return val_loss_list, val_loss_hist, avg_mul_factor , avg_unc_k ,k1_label_list, k_avg_list

#%% Standalone Run

if __name__ == "__main__":
    pass
