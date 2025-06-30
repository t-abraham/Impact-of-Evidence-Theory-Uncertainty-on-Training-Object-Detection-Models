# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 14:13:31 2023

@author: Shaik
"""


#%% import libraries 

import numpy as np
import time , copy
import random
import pandas as pd
import torchvision.transforms as T
import os
import torch
import matplotlib.pyplot as plt
import csv
import platform
import warnings
import contextlib
import seaborn as sn
from lib.roc import roc
import pickle
from PIL import ImageFont, ImageDraw
from lib.model import create_model
import xml.etree.ElementTree as ET

from shapely import box
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


#%%

start_time = None
end_time = None

def tic():
    global start_time
    start_time = time.time()

def toc():
    global end_time
    end_time = time.time()

def time_duration():
    duration = end_time - start_time
    rounded_duration = round(duration, 1)  # Round the duration to 1 decimal place
    if rounded_duration >= 3600:
        hours = rounded_duration // 3600
        minutes = (rounded_duration % 3600) // 60
        return f"{hours} hour(s) {minutes} minute(s)"
    elif rounded_duration >= 60:
        minutes = rounded_duration // 60
        return f"{minutes} minute(s)"
    else:
        return f"{rounded_duration} second(s)"

#%% sqlalchemy_db_checker

def sqlalchemy_db_checker( uri ):
    engine = create_engine( uri )
    
    # Create database if it does not exist.
    if not database_exists(engine.url):
        create_database(engine.url)
        
    del engine
    
    return uri
    
#%% for new dataset calculating number of classes present

def number_classes(path, folder):

    # Define the directory containing your XML files
    xml_dir = path

    # Initialize a set to store unique class labels
    unique_classes = set()

    # Iterate over each XML file in the directory
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith('.xml'):
        # Parse the XML file
            tree = ET.parse(os.path.join(xml_dir, xml_file))
            root = tree.getroot()

            # Find all "name" elements (assuming your XML structure is similar to Pascal VOC)
            for obj in root.findall('object'):
                name = obj.find('name').text
                unique_classes.add(name)

    # Print the unique class labels and the total number of classes
    print("Unique Classes:", unique_classes)
    print("Total Number of Classes:", len(unique_classes))
    
    # Convert the set to a list to ensure compatibility with older versions of pickle
    unique_classes_list = list(unique_classes)

    # Save the unique classes to a pickle file in the output folder
    pickle_file_path = os.path.join(folder, 'unique_classes.pkl')
    with open(pickle_file_path, 'wb') as pickle_file:
        pickle.dump(unique_classes_list, pickle_file)
        
    return

    
#%% limiting the number of classes and their samples
# def limited_classes(all_classes, num_class_select):
#     selected_classes = [all_classes[0]]
#     random_indices = random.sample(range(1, len(all_classes)), num_class_select)
#     # selected_indices = [i for i in range(len(all_classes)) if all_classes[i] in selected_classes]
#     for i in random_indices:
#         selected_classes.append(all_classes[i])
#         # selected_indices.append(i)
#     print(f'final selected classes are : {selected_classes}')
#     # print(f'final selected indices are : {selected_indices}')
    
#     # selected_indices_tensor = torch.tensor(selected_indices)
#     return selected_classes 
    
def limited_classes(all_classes, num_class_select):
    if all_classes[0] == '_background_':
        selected_classes = [all_classes[0]]  # Always include the first class (background)
        selected_classes.extend(all_classes[1:num_class_select+1])  # Select additional classes based on the desired order
    else:
        selected_classes = []
        selected_classes.extend(all_classes[0 : num_class_select])
      
    print(f'Final selected classes are: {selected_classes}')
    return selected_classes


# %%

# def create_subset(dataset, selected_classes, all_classes):
    
#     class_indices = {name: [] for name in selected_classes}
#     selected_indices = np.array([i for i in range(len(all_classes)) if all_classes[i] in selected_classes])

#     for idx, (_, target) in enumerate(dataset):
#         labels = np.array(target['labels'])
#         matches = np.isin(labels, selected_indices)
#         for label_idx in np.where(matches)[0]:
#             class_indices[all_classes[labels[label_idx]]].append(idx)
    
#     selected_indices = np.concatenate([class_indices[name] for name in selected_classes])

#     subset = torch.utils.data.Subset(dataset, selected_indices)
    
#     return subset






def create_subset(dataset, selected_classes, all_classes):
    # create a dictionary that maps class names to indices
    class_indices = {name: [] for name in selected_classes}
    selected_indices = [i for i in range(len(all_classes)) if all_classes[i] in selected_classes]
    for idx, (_, target) in enumerate(dataset):
        for label in target['labels']:
            if label in selected_indices:
                class_indices[all_classes[label]].append(idx)

        # create a list of indices for the selected classes
    selected_indices = [class_indices[name] for name in selected_classes]

        # flatten the list of indices
    selected_indices = [idx for indices in selected_indices for idx in indices]
    
    # concatenate the arrays
    # selected_indices = np.concatenate(selected_indices)

    # create the subset based on the mask
    subset = torch.utils.data.Subset(dataset, selected_indices)

    return subset


#%% Averager
""" this class keeps track of the training and validation loss values
    and helps to get the average for each epoch as well """
class Averager:
    
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0
     
    def send(self, value):
        
        self.current_total += value
        self.iterations += 1
        
    @property
    def value(self):
        
        if self.iterations == 0:
             return 0
        else:
            return 1.0 * self.current_total / self.iterations
        
    @property
    def get_iterations(self):
        
        return self.iterations
    
    @property
    def get_current_total(self):
        
        return self.current_total
     
    def reset(self):
        
        self.current_total = 0.0
        self.iterations = 0.0
        
        
# %%
"""  for k average (change) # Initialize variables for the first epoch
previous_average = 0
count = 0"""
# def IKEA(k_average, K_current, count):
#     # def calculate_running_average(previous_average, current_number, count):
#     running_sum = k_average * count  # Multiply previous average by count
#     running_sum += K_current  # Add the current number
#     count += 1  # Increment the count
#     running_average = running_sum / count  # Calculate the running average
#     return running_average, count
#     return
#%% collate_fn
""" handles the images and bounding boxes of varying sizes"""


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))

        

#%% show transformed images

def show_tranformed_image(train_loader, all_classes):
    """
    This function shows the transformed images from the `train_loader`.
    Helps to check whether the tranformed images along with the corresponding
    labels are correct or not.
    Only runs if `visualize_images = True` in config.yaml.
    """
    if len(train_loader) > 0:
        PIL_transform = T.ToPILImage()
        for i in range(5):
            images, targets = next(iter(train_loader))
            images = list(image.to(DEVICE) for image in images)
            targets = list({k: v.to(DEVICE) for k, v in t.items()} for t in targets)
            boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            labels = targets[i]['labels'].cpu().numpy().astype(np.int32)
            sample = PIL_transform(images[i])     
            draw = ImageDraw.Draw(sample)
            for box_num, box in enumerate(boxes):
                color = tuple(np.random.randint(256, size=3))
                draw.rectangle(
                                    (
                                        (box[0], box[1]), 
                                        (box[2], box[3])
                                    ),
                                    outline=color
                                )
                draw.text(
                                (box[0], box[1]-10),
                                all_classes[labels[box_num]], 
                                font=ImageFont.truetype("arial.ttf", 12),
                                fill=color
                            )
            plt.imshow(sample)
            plt.show()
            
    return question_promt_yes_no("Continue with Network Training?",True)


#%% training continuation
""" to continue training : pass Yes
    to stop training  : pass no """
            
def question_promt_yes_no(question,looped=False):
    
    i = 0
    user_input = False
    while i == 0:
        answer = input("{} - Yes(Y/y) or No(N/n)".format(question))
        if any(answer.lower() == f for f in ["yes", 'y', '1', 'ye']):
            user_input = True
            break
        elif any(answer.lower() == f for f in ['no', 'n', '0']):
            user_input = False
            break
        if looped is False:
            i = 1
            
    return user_input


#%% save model and best model
def save_model(epochs, model, optimizer, model_info):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                
                }, os.path.join(model_info, 'final_model.pth'))
    
class Savebestmodel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_score=float('inf')
    ):
        self.best_score = best_score
        
    def __call__(
        self, current_score, 
        epoch, model, optimizer, model_info
    ):
        if current_score < self.best_score:
            self.best_score = current_score
            print(f"\nBest Score: {self.best_score}")
            print(f"\nSaving best model for epoch: {epoch +1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                
                }, os.path.join(model_info, 'best_model.pth'))
        

def load_best_model(parent_dir , all_classes, model_name):
    """
    Function to load the best model from the latest folder and its subfolders.
    Args:
        parent_dir (str): The parent directory where all the model folders are saved.
    Returns:
        The best PyTorch model loaded from the saved state dictionary.
    """
    # Get a list of all the subdirectories (model folders) in the parent directory
    subdirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]

    # Sort the subdirectories by creation time in descending order (latest first)
    subdirs = sorted(subdirs, key=os.path.getctime, reverse=True)

    # Load the best model from the latest folder and its subfolders
    if subdirs:
        latest_folder = subdirs[0]  # Get the latest (first) folder
        for root, dirs, files in os.walk(latest_folder):
            if "best_model.pth" in files:
                model_path = os.path.join(root, "best_model.pth")
                checkpoint = torch.load(model_path)
                best_model_state_dict = checkpoint['model_state_dict']
                # Load the best model using the state dict
                best_model = create_model(len(all_classes), pretrained= False, model_name = model_name)
                best_model.load_state_dict(best_model_state_dict)
                # Now you can use the best model for inference or further training
                return best_model
            else:
                continue
    else:
        print("No model folders found in the parent directory.")
# %%


class SaveBestModel_2:
    def __init__(self, model_name, save_dir='saved_models'):
        self.model_name = model_name
        self.save_dir = os.path.join(save_dir, model_name)

        # Create directories if they don't exist
        os.makedirs(self.save_dir, exist_ok=True)

        # Track the best validation loss and other metrics
        self.best_loss = float('-inf')
        self.best_map = 0.0
        self.best_epoch = 0
        self.best_classes = None
        self.optimizer_name = None
    
    def get_best_score(self):
        
        return self.best_loss
    
    def save_model(self, model, optimizer, epoch, current_loss, map_metric, classes):
        # Save only if the current loss is better than the best loss
        print(f"Stored best score ({self.best_loss}) vs Given best score ({current_loss})")
        if current_loss > self.best_loss:
            # Update the best metrics
            self.best_loss = current_loss
            self.best_map = map_metric
            self.best_epoch = epoch
            self.best_classes = classes
            self.optimizer_name = optimizer.__class__.__name__

            # Create a dictionary to save everything
            save_dict = {
                'epoch': self.best_epoch,
                'optimizer': self.optimizer_name,
                'loss': self.best_loss,
                'map': self.best_map,
                'classes': self.best_classes,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }

            # Save everything in one file
            save_path = os.path.join(self.save_dir, f'{self.model_name}_best.pth')
            torch.save(save_dict, save_path)

            print(f"Best model and additional information saved with Score value: {current_loss}, MAP: {map_metric}, Epoch: {epoch +1}")




#%% save all the plots

def save_training_plots(training_loss, training_plots , epoch):
    # create  subplots for training 
       figure_1, train_ax = plt.subplots()
       train_ax.plot(training_loss, color = 'black')
       train_ax.set_xlabel('iterations')
       train_ax.set_ylabel('training_loss')
       figure_1.savefig(os.path.join(training_plots, f"train_loss_{epoch+1}.png" ))
       
def save_K_plots(k_label_list, k_bnb_list, K_path, epoch):
    # Create subplots for uncertainity comparison
    figure_1, k_ax = plt.subplots()
    k_ax.plot(k_label_list, label='k_label_list')
    k_ax.plot(k_bnb_list, label='k_bnb_list')
    k_ax.set_xlabel('Iterations')
    k_ax.set_ylabel('K_Value')
    k_ax.set_title(f'Epoch {epoch+1} Comparison')
    k_ax.legend()
    # Save the plot to an image file
    filename = os.path.join(K_path, f"comparison_{epoch+1}.png")
    figure_1.savefig(filename)
       
    
def save_validation_plots(validation_loss, validation_plots, epoch): 
    # create  subplots for validation
    figure_1, valid_ax = plt.subplots()
    valid_ax.plot(validation_loss, color='red')
    valid_ax.set_xlabel('iterations')
    valid_ax.set_ylabel('validation loss')  
    figure_1.savefig(os.path.join(validation_plots, f"validation_loss_{epoch+1}.png" ))
    
#%% to store any variable in csv file
def CSV_file(pth, heading, result,  score_card):
    with open(pth, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(heading)
        writer.writerows(result)
        writer.writerow([])  # Add an empty row for separation
        
        writer.writerow(['Score Card'])
        for score_range, score in score_card.items():
            writer.writerow([score_range, score])
    print("File saved.")

    
#%% to load  csv file

def load_csv(parent_dir , all_classes):
    """
    Function to load the csv file from the latest folder and its subfolders.
    Args:
        parent_dir (str): The parent directory where all the  folders are saved.
    Returns:
        The final F1 score in float value.
    """
    # Get a list of all the subdirectories (model folders) in the parent directory
    subdirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]

    # Sort the subdirectories by creation time in descending order (latest first)
    subdirs = sorted(subdirs, key=os.path.getctime, reverse=True)

    # Load the best model from the latest folder and its subfolders
    if subdirs:
        latest_folder = subdirs[0]  # Get the latest (first) folder
        for root, dirs, files in os.walk(latest_folder):
            if "scores.csv" in files:
                csv_file =  os.path.join(root, 'scores.csv')
                df = pd.read_csv(csv_file)
                f1_Score = df.iloc[-1]['F1_score']
                f1_weight = float(f1_Score.strip('tensor()'))
                
                Iou_score = df.iloc[-1]['IOU']
                IOU_weight = float(Iou_score.strip('tensor()'))
                
                return f1_weight ,IOU_weight
            else:
                continue
    else:
        print("No score folders found in the parent directory.")



def visulaize(img, roc_ds, roc_ds_bnb, label_name , gt_box):
    transform = T.ToPILImage()
    img = transform(img)
    draw = ImageDraw.Draw(img)
    
    # Find bounding box with highest score
    best_box = max(roc_ds_bnb, key=roc_ds_bnb.get)
    
    # Draw bounding box on image
    draw.rectangle(best_box, outline="red")
    
    # Get label and score for best box
    label = list(roc_ds.keys())[0]
    score = roc_ds[list(roc_ds.keys())[0]]
    
    # Add label and score to image
    draw.text((best_box[0], best_box[1] -20), f"{label}: {score}", fill="blue")
    
    # visualize ground truth bounding box
    draw.rectangle([(gt_box[0], gt_box[1]), (gt_box[2], gt_box[3])], outline='green')
    draw.text((gt_box[0], gt_box[1] - 10), f"{label_name} (GT)", fill="green")
    
    # Show image
    img.show()
    
    return




#%% calculate_iou

def calculate_Iou(bbox1, bbox2):
    # Calculate the intersection coordinates
    x1_intersection = max(bbox1[0], bbox2[0])
    y1_intersection = max(bbox1[1], bbox2[1])
    x2_intersection = min(bbox1[2], bbox2[2])
    y2_intersection = min(bbox1[3], bbox2[3])

    # Calculate the area of intersection
    intersection_area = max(0, x2_intersection - x1_intersection) * max(0, y2_intersection - y1_intersection)

    # Calculate the area of both bounding boxes
    area_bbox1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area_bbox2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # Calculate the IoU
    iou = intersection_area / (area_bbox1 + area_bbox2 - intersection_area)
    return iou

def calculate_iou(x1_min, y1_min, x1_max, y1_max, x2_min, y2_min, x2_max, y2_max):
    box_1 = box(
                    xmin = x1_min, 
                    ymin = y1_min, 
                    xmax = x1_max, 
                    ymax = y1_max,
                )
    
    box_2 = box(
                    xmin = x2_min, 
                    ymin = y2_min, 
                    xmax = x2_max, 
                    ymax = y2_max,
                )
    
    if box_1.intersects(box_2):
        return box_1.intersection(box_2).area / box_1.union(box_2).area
    else:
        return 0    

#%% roc_group_formater
        
def roc_group_formater(all_preds_list, iou_threshold=0.5, score_threshold=0.5):    
    
    all_detections = [ { } for i in range( len( all_preds_list ) ) ]
    roc_groups = [ { } for i in range( len( all_preds_list ) ) ]
    group_counter = 1
    
    for all_preds_counter, pred in enumerate(all_preds_list):
        for pred_counter, (pred_box, pred_label, pred_score) in enumerate( zip(pred["boxes"], pred["labels"], pred["scores"]) ):
            all_detections[all_preds_counter]["prediction_{}".format(str(pred_counter+1).zfill(3))] = (
                                                                                round( float( pred_label.cpu().numpy() ) ), 
                                                                                [ round(elem) for elem in pred_box.tolist() ],
                                                                                float( pred_score.cpu().numpy() )
                                                                                
                                                                            )
            
    for working_all_detections_counter, working_all_detections in enumerate(all_detections):
        while True:        
            info_left = len(working_all_detections.keys())
            current_roc_group = []
            keys_to_remove = []
            if info_left == 0:
                break
            current_keys = list(working_all_detections.keys())                
            
            bbox1 = copy.deepcopy( working_all_detections[current_keys[0]][1] )
            pre_copy = list( copy.deepcopy( working_all_detections[current_keys[0]] ) )
            pre_copy.append(1)
            pre_copy = tuple( pre_copy )
            current_roc_group.append( copy.deepcopy( pre_copy ) )
            keys_to_remove.append( current_keys[0] )
            
            for current_key in current_keys[1:]:
                bbox2 = copy.deepcopy( working_all_detections[current_key][1] )
                score = copy.deepcopy( working_all_detections[current_key][2] )
                iou = calculate_iou(
                                        x1_min = bbox1[0], 
                                        y1_min = bbox1[1], 
                                        x1_max = bbox1[2], 
                                        y1_max = bbox1[3], 
                                        
                                        x2_min = bbox2[0], 
                                        y2_min = bbox2[1], 
                                        x2_max = bbox2[2], 
                                        y2_max = bbox2[3],
                                    )
                
                if iou >= iou_threshold and score >= score_threshold:
                    pre_copy = list( copy.deepcopy( working_all_detections[current_key] ) )
                    # pre_copy[2] = pre_copy[2] * iou
                    pre_copy.append(iou)
                    pre_copy = tuple( pre_copy )
                    current_roc_group.append( copy.deepcopy( pre_copy ) )
                    keys_to_remove.append( current_key )
                    
            for key_to_remove in keys_to_remove:
                working_all_detections.pop( key_to_remove )
                
            roc_groups[working_all_detections_counter]["Group_{}".format(str(group_counter).zfill(3))] = current_roc_group
            group_counter += 1
    
    # for i in range( len( roc_groups ) ):
    #     for key in list(roc_groups[i].keys()):
    #         if len(roc_groups[i][key]) < 2:
    #             roc_groups[i].pop(key)
                
    return roc_groups

#%% roc_pred

def process_roc(settings, pred_keys, roc_groups, all_preds_list, deci=5):
    
    perform_roc = roc(deci)
    
    roc_preds = [ {j : [] for j in pred_keys} for i in range( len( all_preds_list ) ) ]
    
    for image_number in range( len( roc_groups ) ):
        for current_group in roc_groups[image_number].keys():
            classification_masses_roc = {}
            regression_masses_roc = {}
            regression_masses_roc_total = 0
            for current_mass_id in range( len( roc_groups[image_number][current_group] ) ):
                classification_masses_roc["m{}".format(current_mass_id)] = \
                    {
                        roc_groups[image_number][current_group][current_mass_id][0]     :
                            roc_groups[image_number][current_group][current_mass_id][2],
                    }
                        
                regression_masses_roc["m{}".format(current_mass_id)] = \
                    {
                        str(roc_groups[image_number][current_group][current_mass_id][1])     :
                            roc_groups[image_number][current_group][current_mass_id][2] * roc_groups[image_number][current_group][current_mass_id][3],
                    }
                regression_masses_roc_total += roc_groups[image_number][current_group][current_mass_id][2] * roc_groups[image_number][current_group][current_mass_id][3]
                
            roc_ds_classification_k, roc_ds_classification_matrix = perform_roc.perform_ds(**classification_masses_roc)
            
            for key in regression_masses_roc.keys():
                for bbox in regression_masses_roc[key].keys():
                    regression_masses_roc[key][bbox] /= regression_masses_roc_total
                    
            xmin, ymin, xmax, ymax, bbox_normalizer = 0, 0, 0, 0, 0
            
            for key in regression_masses_roc.keys():
                for bbox in regression_masses_roc[key].keys():
                    bbox_exploded = eval(bbox)
                    xmin += bbox_exploded[0] * regression_masses_roc[key][bbox]
                    ymin += bbox_exploded[1] * regression_masses_roc[key][bbox]
                    xmax += bbox_exploded[2] * regression_masses_roc[key][bbox]
                    ymax += bbox_exploded[3] * regression_masses_roc[key][bbox]
                    bbox_normalizer += regression_masses_roc[key][bbox]
            
            xmin /= bbox_normalizer
            ymin /= bbox_normalizer
            xmax /= bbox_normalizer
            ymax /= bbox_normalizer
            
            roc_preds[image_number]["boxes"].append( [round(xmin), round(ymin), round(xmax), round(ymax)] )
            roc_preds[image_number]["labels"].append( max( roc_ds_classification_matrix, key=roc_ds_classification_matrix.get ) )
            roc_preds[image_number]["scores"].append( 1-roc_ds_classification_k )
        
        for key in list( roc_preds[image_number].keys() ):
            roc_preds[image_number][key] = torch.as_tensor(roc_preds[image_number][key])
            # roc_preds[image_number][key] = roc_preds[image_number][key].to( settings["DEVICE_GBL"] )
            roc_preds[image_number][key] = roc_preds[image_number][key]
    return roc_preds

#%% TryExcept
class TryExcept(contextlib.ContextDecorator):
    
    # YOLOv8 TryExcept class. Usage: @TryExcept() decorator or 'with TryExcept():' context manager
    def __init__(self, msg='', verbose=True):
        self.msg = msg
        self.verbose = verbose
    def __enter__(self):
        pass
    def __exit__(self, exc_type, value, traceback):
        if self.verbose and value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True
        
#%% calculate multification factore
def calculate_mul_factor(a, K_ds, score_card):
    for key in score_card.keys():
        if 'factor' in key:
            return a.factor_generator_v2(score_card, K_ds)
    return a.factor_generator(score_card, K_ds * 100)


class metrics:
    
    def __init__( self, classes, score_thres=None, iou_thres=None, row=None, col=None ):
        if "__background__" in classes:
            self.row = classes.index("__background__")
            self.col = classes.index("__background__")        
        else:
            self.row = row
            self.col = col
            
        self.classes = classes  
        self.score_thres = score_thres
        self.iou_thres = iou_thres        
        self.count_classes = len( self.classes )
        self.MeanAveragePrecision = MeanAveragePrecision( class_metrics=True )
        self.ConfusionMatrix = ConfusionMatrix( self.count_classes, self.score_thres, self.iou_thres, self.row, self.col )
        self.results = {}
    
    def _InputFormatter( self, loads ):
        return_array = []
        for load in loads:
            if "scores" in load:
                for load_box, load_label, load_score in zip( load['boxes'], load['labels'], load['scores'] ):
                    if not torch.is_tensor( load_box ) is True:
                        load_box = torch.tensor( load_box )
                    if not torch.is_tensor( load_label ) is True:
                        load_label = torch.tensor( load_label )
                    if not torch.is_tensor( load_score ) is True:
                        load_score = torch.tensor( load_score )
                        
                    if load_box.dim() == 0:
                        load_box = torch.unsqueeze( load_box, 0 )
                    if load_label.dim() == 0:
                        load_label = torch.unsqueeze( load_label, 0 )
                    if load_score.dim() == 0:
                        load_score = torch.unsqueeze( load_score, 0 )
                        
                    return_array.append( torch.cat( ( load_box, load_score, load_label ) ) )
            else:
                for load_box, load_label in zip( load['boxes'], load['labels'] ):                    
                    if not torch.is_tensor( load_box ) is True:
                        load_box = torch.tensor( load_box )
                    if not torch.is_tensor( load_label ) is True:
                        load_label = torch.tensor( load_label )
                        
                    if load_box.dim() == 0:
                        load_box = torch.unsqueeze( load_box, 0 )
                    if load_label.dim() == 0:
                        load_label = torch.unsqueeze( load_label, 0 )
                        
                    return_array.append( torch.cat( ( load_label, load_box ) ) )
        
        if return_array:
            return torch.stack( return_array )
        else:
            return torch.tensor([[]])       
        
    def update( self, preds, targets ):
        self.MeanAveragePrecision.update( preds, targets )

        formatted_preds = self._InputFormatter( preds )
        formatted_targets = self._InputFormatter( targets )
        
        if not len(formatted_preds[0]) == 0:
            self.ConfusionMatrix.process_batch( formatted_preds, formatted_targets )
        
    def compute( self ):
        self.results = self.MeanAveragePrecision.compute()
        self.results["confusion_matrix"] = self.ConfusionMatrix.get_matrix()
        self.results["tp_fp_fn"] = self.ConfusionMatrix.get_tp_fp_fn()
   
    def GetResults( self ):
        
        return self.results
    
    def plot( self, key="", normalize=True, save_dir='', names=[] ):
        
        self.ConfusionMatrix.cm_plot( key=key, normalize=normalize, save_dir=save_dir, arg_names=names )
    
    
    def print( self, key="" ):
        if not len(key) == 0:
            key = "({}) ".format(key)
            
        print ("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print ("********************************************************")
        
        print("----------------------------------------------------------------------")
        map_global = self.results["map"].numpy().tolist()
        if isinstance(map_global, (int, float)):
            map_global = [map_global]
        print("Model {} MAP Global: {}".format(key, [round(i, 3) for i in map_global]))
        
        print("----------------------------------------------------------------------")
        map_per_class = self.results["map_per_class"].numpy().tolist()
        if isinstance(map_per_class, (int, float)):
            map_per_class = [map_per_class]
        print("Model {} MAP per class: {}".format(key, [round(i, 3) for i in map_per_class]))

        print ( "----------------------------------------------------------------------" )
        self.ConfusionMatrix.print()      
        print ( "----------------------------------------------------------------------" )
        
        print ("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print ("********************************************************")

#%% ConfusionMatrix

class ConfusionMatrix:
    
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.5, row=None, col=None):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres
        self.row = row
        self.col = col
        
    def __box_iou__(self, box1, box2, eps=1e-7):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
            eps

        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
        """

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

        # IoU = inter / (area1 + area2 - inter)
        return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        if detections is None:
            gt_classes = labels.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # background FN
            return

        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = self.__box_iou__(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # true background

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # predicted background

    def get_matrix(self):
        current_matrix = self.matrix
        if not self.row is None:
            current_matrix = np.delete(current_matrix, (self.row), axis=0)
        if not self.col is None:
            current_matrix = np.delete(current_matrix, (self.col), axis=1)
        return current_matrix

    def get_tp_fp_fn(self):
        current_matrix = self.matrix
        if not self.row is None:
            current_matrix = np.delete(current_matrix, (self.row), axis=0)
        if not self.col is None:
            current_matrix = np.delete(current_matrix, (self.col), axis=1)
        tp = current_matrix.diagonal()  # true positives
        fp = current_matrix.sum(1) - tp  # false positives
        fn = current_matrix.sum(0) - tp  # false negatives (missed detections)
        return tp[:-1], fp[:-1], fn[:-1]  # remove background class
        #return tp, fp, fn  # with background class

    @TryExcept('WARNING ⚠️ ConfusionMatrix plot failure')
    def cm_plot(self, key="", normalize=True, save_dir='', arg_names=[]):
        if "__background__" in arg_names:
            names = copy.deepcopy(arg_names)
            names.remove("__background__")
        else:
            names = arg_names
        
        nc, nn = len(names), len(names)  # number of classes, names
            
        if not len(key) == 0:
            key = "_{}".format(key)        
        
        current_matrix = self.matrix
        
        if not self.row is None:
            current_matrix = np.delete(current_matrix, (self.row), axis=0)
        if not self.col is None:
            current_matrix = np.delete(current_matrix, (self.col), axis=1)
        
        array = current_matrix / ((current_matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)        
        sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (names + ['background']) if labels else 'auto'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(array,
                       ax=ax,
                       annot=nc < 30,
                       annot_kws={
                           'size': 8},
                       cmap='Blues',
                       fmt='.2f',
                       square=True,
                       vmin=0.0,
                       xticklabels=ticklabels,
                       yticklabels=ticklabels).set_facecolor((1, 1, 1))
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title('Confusion Matrix')        
        fig.savefig(os.path.join(save_dir, 'confusion_matrix{}.png'.format(key)), dpi=250)
        plt.close(fig)

    def print(self):
        current_matrix = self.matrix
        if not self.row is None:
            current_matrix = np.delete(current_matrix, (self.row), axis=0)
        if not self.col is None:
            current_matrix = np.delete(current_matrix, (self.col), axis=1)
            
        for i in range(len(current_matrix)):
            print('; '.join(map(str, current_matrix[i])))
        tp, fp, fn = self.get_tp_fp_fn()
        precision = tp / ( tp + fp )
        recall = tp / ( tp + fn )
        f1 = 2 / ( ( 1/precision ) + ( 1/recall ) )
        print("Calculated:")
        print("True Positive: {}".format(tp))
        print("False Positive: {}".format(fp))
        print("False Negative: {}".format(fn))
        
        print("Calculated per Classes:")
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1 Score: {}".format(f1))
        
        print("Calculated Mean:")
        print("Precision: {}".format(precision.mean()))
        print("Recall: {}".format(recall.mean()))
        print("F1 Score: {}".format(f1.mean()))
        
        
#%%
class metrics_old:
    
    def __init__( self, classes, score_thres, iou_thres ):
        self.classes = classes
        self.score_thres = score_thres
        self.iou_thres = iou_thres
        self.count_classes = len( self.classes )
        self.MeanAveragePrecision = MeanAveragePrecision( class_metrics=True )
        self.ConfusionMatrix = ConfusionMatrix( self.count_classes, self.score_thres, self.iou_thres )
        self.results = {}
    
    def _InputFormatter( self, loads ):
        return_array = []
        for load in loads:
            if "scores" in load:
                for load_box, load_label, load_score in zip( load['boxes'], load['labels'], load['scores'] ):
                    if not torch.is_tensor( load_box ) is True:
                        load_box = torch.tensor( load_box )
                    if not torch.is_tensor( load_label ) is True:
                        load_label = torch.tensor( load_label )
                    if not torch.is_tensor( load_score ) is True:
                        load_score = torch.tensor( load_score )
                        
                    if load_box.dim() == 0:
                        load_box = torch.unsqueeze( load_box, 0 )
                    if load_label.dim() == 0:
                        load_label = torch.unsqueeze( load_label, 0 )
                    if load_score.dim() == 0:
                        load_score = torch.unsqueeze( load_score, 0 )
                        
                    return_array.append( torch.cat( ( load_box, load_score, load_label ) ) )
            else:
                for load_box, load_label in zip( load['boxes'], load['labels'] ):                    
                    if not torch.is_tensor( load_box ) is True:
                        load_box = torch.tensor( load_box )
                    if not torch.is_tensor( load_label ) is True:
                        load_label = torch.tensor( load_label )
                        
                    if load_box.dim() == 0:
                        load_box = torch.unsqueeze( load_box, 0 )
                    if load_label.dim() == 0:
                        load_label = torch.unsqueeze( load_label, 0 )
                        
                    return_array.append( torch.cat( ( load_label, load_box ) ) )
        
        if return_array:
            return torch.stack( return_array )
        else:
            return torch.tensor([[]])       
        
    def update( self, preds, targets ):
        self.MeanAveragePrecision.update( preds, targets )
        formatted_preds = self._InputFormatter( preds )
        formatted_targets = self._InputFormatter( targets )
        
        if not len(formatted_preds[0]) == 0:
            self.ConfusionMatrix.process_batch( formatted_preds, formatted_targets )
        
    def compute( self ):
        self.results = self.MeanAveragePrecision.compute()
        self.results["confusion_matrix"] = self.ConfusionMatrix.get_matrix()
        self.results["tp_fp"] = self.ConfusionMatrix.get_tp_fp()
   
    def GetResults( self ):
        
        return self.results
    
    def plot( self, key="", normalize=True, save_dir='', names=[] ):
        self.ConfusionMatrix.cm_plot( key=key, normalize=normalize, save_dir=save_dir, names=names )
    
    
    def print( self, key="" ):
        if not len(key) == 0:
            key = "({}) ".format(key)
            
        print ("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print ("********************************************************")
        
        print("----------------------------------------------------------------------")
        map_global = self.results["map"].numpy().tolist()
        if isinstance(map_global, (int, float)):
            map_global = [map_global]
        print("Model {} MAP Global: {}".format(key, [round(i, 3) for i in map_global]))
        
        print("----------------------------------------------------------------------")
        map_per_class = self.results["map_per_class"].numpy().tolist()
        if isinstance(map_per_class, (int, float)):
            map_per_class = [map_per_class]
        print("Model {} MAP per class: {}".format(key, [round(i, 3) for i in map_per_class]))
        print ( "----------------------------------------------------------------------" )
        self.ConfusionMatrix.print()      
        print ( "----------------------------------------------------------------------" )
        
        print ("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print ("********************************************************")
        
#%% emojis
def emojis(string=''):
    
    # Return platform-dependent emoji-safe version of string
    MACOS, LINUX, WINDOWS = (platform.system() == x for x in ['Darwin', 'Linux', 'Windows'])
    return string.encode().decode('ascii', 'ignore') if WINDOWS else string
    
#%% ConfusionMatrix
class ConfusionMatrix_old:
    
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.5):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres
        
    def __box_iou__(self, box1, box2, eps=1e-7):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
            eps
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
        """
        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
        # IoU = inter / (area1 + area2 - inter)
        return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)
    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        if detections is None:
            gt_classes = labels.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # background FN
            return
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = self.__box_iou__(labels[:, 1:], detections[:, :4])
        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))
        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # true background
        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # predicted background
    def get_matrix(self):
        return self.matrix
    def get_tp_fp(self):
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return tp[:-1], fp[:-1]  # remove background class
    @TryExcept('WARNING :warning: ConfusionMatrix plot failure')
    def cm_plot(self, key="", normalize=True, save_dir='', names=[]):
        if not len(key) == 0:
            key = "_{}".format(key)        
        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)
        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)  # number of classes, names
        sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (names + ['background']) if labels else 'auto'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(array,
                       ax=ax,
                       annot=nc < 30,
                       annot_kws={
                           'size': 8},
                       cmap='Blues',
                       fmt='.2f',
                       square=True,
                       vmin=0.0,
                       xticklabels=ticklabels,
                       yticklabels=ticklabels).set_facecolor((1, 1, 1))
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title('Confusion Matrix')        
        fig.savefig(os.path.join(save_dir, 'confusion_matrix{}.png'.format(key)), dpi=250)
        plt.close(fig)
    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))
            

