# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 12:06:04 2023

@author: Tahasanul Abraham
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

import torch, json, copy, tqdm, operator, pathlib, datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image
# from torchmetrics.detection.mean_ap import MeanAveragePrecision

from lib.parse_excel_kx import get_rules
from lib.config import get_config
from lib.utilities import collate_fn, retrieveVariableWP, \
                            roc_group_formater, process_roc, \
                            storeVariableWP, Averager, metrics, \
                            setup_logger, sqlalchemy_db_checker
from lib.timers import tic, toc
from lib.models import get_all_available_models, get_model
from lib.mqtt import MQTTClientWrapper
from lib.dataloaders import ModelAllDataloader
from lib.datasets import ModelAllDataset
import optuna
    
#%% Check for saved trained models and their weights

def getTrainedModels(all_models, settings, dataset_name, optuna_model=False, last_model_mode=False):
    
    trained_models = {}
    checkpoints = {}
    all_trained_classes = {}
    model_scores = {}
    
    for model_name in all_models:        
        
        all_trained_classes[model_name] = {}
        model_scores[model_name] = Averager()
        
        model_path_dir = pathlib.Path(settings["TRAINING_MODEL_DIR_NT"])
        
        if optuna_model is True:            
            run_type = "last_model.optuna"
        else:
            run_type = "optuna"
        
        if last_model_mode is True:
            run_type = "last_model.traditional"
        else:
            run_type = "traditional" 
        
        pre_trained = "pt" if settings["PRETRAINED_IMAGENT_MODELS_NT"] is True else "npt"
        
        model_type_string = "{}_{}_{}".format( pre_trained, model_name, dataset_name )
        
        file_name = "{}.{}*.pth".format( run_type, model_type_string )
        
        model_paths = []
        
        for file in model_path_dir.joinpath(run_type, model_name).glob(file_name):
            model_paths.append(pathlib.Path(file))
            
        for model_path in model_paths:            
            model_name_new = model_name + str(model_path.stem).split(model_name)[-1] 
        
            if model_path.is_file() is True:
                checkpoints[model_name_new] = {}
                
                if settings["DEVICE_GBL"] == torch.device("cpu"):
                    temp_checkpoints = torch.load( 
                                                model_path,
                                                map_location = settings["DEVICE_GBL"],
                                            )
                else:
                    temp_checkpoints = torch.load( 
                                                model_path,
                                            )
               
                trained_models[model_name_new] = get_model( 
                                      num_classes = len( temp_checkpoints["trained_classes"] ), 
                                      pretrained_model = settings["PRETRAINED_IMAGENT_MODELS_NT"], 
                                      model_name = model_name,
                                      masked = settings["MASKED_CNN_GBL"]
                                  )
                
                trained_models[model_name_new].load_state_dict( temp_checkpoints["model_state_dict"] )
                
                checkpoints[model_name_new]["epoch"] = copy.deepcopy( temp_checkpoints["epoch"] )
                checkpoints[model_name_new]["model_map"] = copy.deepcopy( float( temp_checkpoints["model_map"]["map"].cpu().numpy() ) )
                checkpoints[model_name_new]["model_map_per_class"] = copy.deepcopy( temp_checkpoints["model_map"]["map_per_class"].cpu().numpy( ).tolist( ) )
                # checkpoints[model_name_new]["roc_map"] = copy.deepcopy( temp_checkpoints["roc_map"] )
                checkpoints[model_name_new]["score"] = copy.deepcopy( temp_checkpoints["score"] )
                # checkpoints[model_name_new]["stats_list"] = copy.deepcopy( temp_checkpoints["stats_list"] )
                # checkpoints[model_name_new]["punishing_factor"] = copy.deepcopy( temp_checkpoints["punishing_factor"] )
                # checkpoints[model_name_new]["optimizer_name"] = copy.deepcopy( temp_checkpoints["optimizer_name"] )
                # checkpoints[model_name_new]["parameters"] = copy.deepcopy( temp_checkpoints["parameters"] )
                # checkpoints[model_name_new]["model_state_dict"] = copy.deepcopy( temp_checkpoints["model_state_dict"] )
                # checkpoints[model_name_new]["optimizer_state_dict"] = copy.deepcopy( temp_checkpoints["optimizer_state_dict"] )
                checkpoints[model_name_new]["trained_classes"] = copy.deepcopy( temp_checkpoints["trained_classes"] )
                
                model_scores[model_name].send(checkpoints[model_name_new]["score"])
                
                # params = [p for p in trained_models[model_name_new].parameters( ) if p.requires_grad]
                
                # optimizer = getattr( 
                #                         torch.optim, 
                #                         checkpoints[model_name_new]["optimizer_name"] 
                #                     ) ( 
                #                             params, 
                #                             lr = settings["TRAINING_OPTIMIZER_LEARNING_RATE_NT"], 
                #                             weight_decay = settings["TRAINING_OPTIMIZER_WEIGHT_DECAY_NT"]
                #                        )
                
                # optimizer.load_state_dict( checkpoints[model_name_new]["optimizer_state_dict"] )

                for trained_class in checkpoints[model_name_new]["trained_classes"]:
                    index = checkpoints[model_name_new]["trained_classes"].index(trained_class)
                    if not trained_class == "N/A" and not index in all_trained_classes[model_name]:
                        all_trained_classes[model_name][index] = trained_class
                        
        all_trained_classes[model_name] = [v for k, v in sorted(all_trained_classes[model_name].items(), key=lambda item: item[0])]
    
    final_trained_classes = [None] * len(all_trained_classes[max(all_trained_classes, key=all_trained_classes.get)])
    
    for model_name in all_trained_classes.keys():
        for class_label in all_trained_classes[model_name]:
            if not class_label in final_trained_classes:
                final_trained_classes[all_trained_classes[model_name].index(class_label)] = class_label
                
    return trained_models, model_scores, checkpoints, final_trained_classes

#%% Get Dataset

def getDatasets(final_trained_classes, settings, trainer=False, validator=False, tester=True):
        
    all_classes = copy.deepcopy( final_trained_classes )
    targeted_dataloaders = {}
    
    print ("********************************************************")
    
    print ("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    tic( "All Dataset Loading time" )
    all_datasets = ModelAllDataset( settings["AUGMENTED_DIR_GBL"], all_classes )
    toc( "All Dataset Loading time" )
    
    if (trainer is True):
        print ("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print ()
        print ("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        
        tic( "Training Data Loading time" )
        training_dataset = all_datasets.get_training_data( all_classes, settings )    
        print( f"Number of Training images: {len( training_dataset )}" )    
        targeted_dataloaders["training_dataloader"] = ModelAllDataloader( 
                                                        training_dataset,
                                                        batch_size = settings["TRAINING_BATCH_SIZE_NT"],
                                                        shuffle = settings["DATA_SHUFFLE_NT"],
                                                        num_workers = settings["TRAINING_WORKERS_NT"],
                                                        collate_fn = collate_fn
                                                    )    
        toc( "Training Data Loading time" )
    
    if (tester is True):
        print ("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print ()
        print ("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        
        tic( "Testing Data Loading time" )
        testing_dataset = all_datasets.get_testing_data( all_classes, settings )
        print( f"Number of Testing images: {len( testing_dataset )}" ) 
        targeted_dataloaders["testing_dataloader"] = ModelAllDataloader( 
                                                        testing_dataset,
                                                        batch_size = settings["TRAINING_BATCH_SIZE_NT"],
                                                        shuffle = settings["DATA_SHUFFLE_NT"],
                                                        num_workers = settings["TRAINING_WORKERS_NT"],
                                                        collate_fn = collate_fn
                                                   ) 
        toc( "Testing Data Loading time" )
    
    if (validator is True):
        print ("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print ()
        print ("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        
        tic( "Validation Data Loading time" )
        validation_dataset = all_datasets.get_validation_data( all_classes, settings )
        print( f"Number of Validation images: {len( validation_dataset )}" ) 
        targeted_dataloaders["validation_dataloader"] = ModelAllDataloader( 
                                                        validation_dataset,
                                                        batch_size = settings["TRAINING_BATCH_SIZE_NT"],
                                                        shuffle = settings["DATA_SHUFFLE_NT"],
                                                        num_workers = settings["TRAINING_WORKERS_NT"],
                                                        collate_fn = collate_fn
                                                    )
        toc( "Validation Data Loading time" )
    
    print ("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print ("********************************************************")
    
    return targeted_dataloaders

#%% Model sorter 

def modelSelector(all_models, trained_models, model_scores, checkpoints, sorting_key, top_model_count):
    for model_name in all_models:
        model_scores[model_name] = Averager()
        
        for checkpoint_key in checkpoints.keys():
            if model_name in checkpoint_key:
                model_scores[model_name].send(checkpoints[checkpoint_key][sorting_key])

    for model_name in model_scores.keys():
        model_scores[model_name] = model_scores[model_name].avg_value
        
    selected_models =  list ( 
                                dict (
                                        sorted (
                                                    model_scores.items(), 
                                                    key=lambda x:x[1], 
                                                    reverse=True
                                                )[:top_model_count]
                                        ).keys()
                            )
    
    checkpoints = {k: v for k, v in checkpoints.items() for k2 in selected_models if k2 in k}
    
    selected_models = {k: v for k, v in trained_models.items() for k2 in selected_models if k2 in k}
    
    return selected_models

#%% inference

def inference(
                all_models, 
                trained_models, 
                model_scores, 
                checkpoints, 
                sorting_key, 
                top_model_count, 
                settings, 
                final_trained_classes, 
                targeted_dataloaders, 
                iou_threshold, 
                conf_threshold,
                save_test_dir=None,
                roc_regression=True):
    
    selected_models = modelSelector(
                                        all_models = all_models, 
                                        trained_models = trained_models, 
                                        model_scores = model_scores, 
                                        checkpoints = checkpoints, 
                                        sorting_key = sorting_key, 
                                        top_model_count = top_model_count
                                    )
    
    if (save_test_dir is None):
        test_results_dir = pathlib.Path(currentdir).joinpath("test_results")            
        test_results_dir.mkdir(parents=True, exist_ok=True)
        test_counts = len([item for item in test_results_dir.iterdir() if item.is_dir()])
        new_folder_name = "{}".format(str(test_counts+1).zfill(2))
        save_test_dir = test_results_dir.joinpath(new_folder_name)
        
    save_test_dir.mkdir(parents=True, exist_ok=True)    
    
    content = ""
    content += "top_model_count: {}".format(top_model_count)
    content += "\n"
    content += "iou_threshold: {}".format(iou_threshold)
    content += "\n"
    content += "conf_threshold: {}".format(conf_threshold)
                
    print ("********************************************************")                
    print ("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print ()
    print (content)
    print ()
    print ("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print ("********************************************************")
    # Writing to the file
    with open(save_test_dir.joinpath("hyperparameters"), 'w') as file:
        file.write(content)

    record_metrics = {
                    "ROC"   :   metrics( settings, final_trained_classes, conf_threshold, iou_threshold )
                 }
    image_counts = 0
    
    with tqdm.tqdm( total=len( targeted_dataloaders["testing_dataloader"].dataset ), desc = "Testing Progress", position=0, leave=True ) as pbar:
        for data in targeted_dataloaders["testing_dataloader"]:            
            images, targets = data
            images = list( image.to( settings["DEVICE_GBL"] ) for image in images )
            targets = [{k: v.to( settings["DEVICE_GBL"] ) for k, v in t.items( )} for t in targets]
            all_model_results = [ { } for i in range( len( images ) ) ]
            pred_keys = []
            
            for model_name in selected_models.keys():
                if not model_name in record_metrics:
                    record_metrics[model_name] = metrics( settings, final_trained_classes, conf_threshold, iou_threshold )
                
                # initialize the model and move to the computation device
                selected_models[model_name] = selected_models[model_name].to( settings["DEVICE_GBL"] )
                with torch.no_grad( ):
                    selected_models[model_name].eval( )
                    preds = selected_models[model_name]( images )                    
                    selected_models[model_name] = selected_models[model_name].to("cpu")
                    
                record_metrics[model_name].update( preds, targets )
                
                for pred in preds:
                    for key in pred.keys():
                        if not key in pred_keys:
                            pred_keys.append(key)
                            
                for i, pred in enumerate(preds):
                    if len( all_model_results[i] ) == 0:
                        for pred_key in pred.keys():
                            all_model_results[i][pred_key] = pred[pred_key]
                    else:
                        for pred_key in pred.keys():
                            all_model_results[i][pred_key] = torch.cat( (all_model_results[i][pred_key], pred[pred_key]), 0 )
                            
            roc_groups = roc_group_formater( 
                                                settings = settings, 
                                                all_preds_list = all_model_results, 
                                                conf_threshold = conf_threshold, 
                                                iou_threshold = iou_threshold 
                                            )
            
            roc_preds = process_roc( 
                                        settings = settings, 
                                        pred_keys = pred_keys, 
                                        roc_groups = roc_groups, 
                                        all_preds_list = all_model_results, 
                                        deci = 5, 
                                        roc_regression = False
                                    )
            
            record_metrics["ROC"].update( roc_preds, targets )
            
            image_counts += len( images )
                            
            if image_counts == len( targeted_dataloaders["testing_dataloader"].dataset ):
                for key in record_metrics.keys():
                    record_metrics[key].compute( )                    
                    record_metrics[key].print( key )
                    record_metrics[key].plot( key, True, save_test_dir, final_trained_classes )
                    record_metrics[key] = record_metrics[key].GetResults( )
            
            pbar.update( len( images ) )
            pbar.refresh( )
    
    storeVariableWP(
                        record_metrics, 
                        "record_metrics_iou_{}_score_{}".format(
                                                                    iou_threshold*100, 
                                                                    conf_threshold*100
                                                                ),
                        save_test_dir
                    )
    
    return record_metrics, list(selected_models.keys())

#%% inference_optuna

class inference_optuna:
    
    def __init__( self ):
        self.best_sdmo_selected_models = None
        self.best_sdmo_results = None
        self.best_sdmo_optuna_params = None        
        self.best_sdmo_performance_score = None    
        self.best_sdmo_trial_number = None     
        self.best_sdmo_trained_classes = None
        
        self._sdmo_selected_models = None
        self._sdmo_results = None
        self._sdmo_optuna_params = None        
        self._sdmo_performance_score = None    
        self._sdmo_trial_number = None     
        self._sdmo_trained_classes = None
        
    def __call__( 
                    self, 
                    trial, 
                    max_trials, 
                    all_models, 
                    trained_models, 
                    model_scores, 
                    checkpoints, 
                    sorting_key, 
                    # top_model_count, 
                    settings, 
                    final_trained_classes, 
                    targeted_dataloaders, 
                    # iou_threshold,
                    # conf_threshold,
                    optuna_params_arg,
                    roc_regression):
        
        
        optuna_params = {}
        for param_name, param_info in optuna_params_arg.items():
            suggest_method_name = param_info[0].split("=")[1].strip()
            suggest_method_args = {
                arg.split("=")[0].strip(): eval(arg.split("=")[1].strip()) for arg in param_info[1:]
            }
            suggest_method = getattr(trial, suggest_method_name)
            optuna_params[param_name] = suggest_method(param_name, **suggest_method_args)
        
        print ( "**************************** Trial - {} ****************************".format( str( trial.number ) ) )
        # for key in optuna_params.keys():
        #     print ("{} - {}".format(key, optuna_params[key]))
        # print ( )
        # print ( "*************************************************************" )
        
        test_results_dir = pathlib.Path(currentdir).joinpath("test_results")            
        test_results_dir.mkdir(parents=True, exist_ok=True)
        new_folder_name = "trial_{}".format(str(trial.number).zfill(5))
        save_test_dir = test_results_dir.joinpath(new_folder_name)
        
        record_metrics, selected_model_keys = inference(
                                                            all_models = all_models, 
                                                            trained_models = trained_models, 
                                                            model_scores = model_scores, 
                                                            checkpoints = checkpoints, 
                                                            sorting_key = sorting_key, 
                                                            top_model_count = optuna_params["top_model_count"],
                                                            settings = all_configs, 
                                                            final_trained_classes = final_trained_classes, 
                                                            targeted_dataloaders = targeted_dataloaders, 
                                                            iou_threshold = optuna_params["iou_threshold"],
                                                            conf_threshold = optuna_params["conf_threshold"],
                                                            save_test_dir = save_test_dir,
                                                            roc_regression = roc_regression
                                                        )
    
        map_global = record_metrics["ROC"]["map"].numpy().tolist()
        if isinstance(map_global, (int, float)):
            map_global = [map_global]
        
        optuna_performance_score = [round(i, 3) for i in map_global][0]
    
        self._sdmo_selected_models = copy.deepcopy( selected_model_keys )
        self._sdmo_results = copy.deepcopy( record_metrics )
        self._sdmo_optuna_params = copy.deepcopy( optuna_params )        
        self._sdmo_performance_score = copy.deepcopy( optuna_performance_score )    
        self._sdmo_trial_number = copy.deepcopy( trial.number )     
        self._sdmo_trained_classes = copy.deepcopy( final_trained_classes )
        
        return optuna_performance_score
    
    def callback ( self, study, trial ):
        
        if study.best_trial.number == trial.number:
            self.best_sdmo_selected_models = copy.deepcopy( self._sdmo_selected_models )
            self.best_sdmo_results = copy.deepcopy( self._sdmo_results )
            self.best_sdmo_optuna_params = copy.deepcopy( self._sdmo_optuna_params )
            self.best_sdmo_performance_score = copy.deepcopy( self._sdmo_performance_score )
            self.best_sdmo_trial_number = copy.deepcopy( self._sdmo_trial_number )
            self.best_sdmo_trained_classes = copy.deepcopy( self._sdmo_trained_classes )
            
#%% generate_float_list

def generate_float_list(start, end, step):
    """
    Generates a list of floats from 'start' to 'end' with a given 'step'.
    
    :param start: The starting number of the sequence.
    :param end: The ending number of the sequence.
    :param step: The step size between each number in the sequence.
    :return: A list of floats.
    """
    return [start + i * step for i in range(int((end - start) / step) + 1)]

    
#%% Standalone Run

if __name__ == "__main__":

#%% Load Configurations
    
    all_configs = get_config( )
    
    logger_filename = "{}_{}.txt".format( 
                                            pathlib.Path(__file__).stem, 
                                            datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S") 
                                         )
    
    _print = setup_logger( all_configs, logger_filename, currentdir )
    
    all_classes = retrieveVariableWP( "all_labels_training.pkl", all_configs["STATISTICAL_DIR_GBL"] )
    
    all_classes = [
                        "__background__", "person", "bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "bicycle", 
                        "boat", "bus", "car", "motorbike", "train", "bottle", "chair", "diningtable", 
                        "pottedplant", "sofa", "tvmonitor"
                  ]
    
    # dataset_name = "3d_objects_{}_classes".format( len( all_classes ) if not all_classes is None else 2 )
    
    # dataset_name = "pascal_voc_2012_{}_classes".format( len( all_classes ) )
    
    dataset_name = "3d_objects"
    
    dataset_name = "pascal_voc_2012"
    
    dataset_dir = pathlib.Path(all_configs["AUGMENTED_DIR_GBL"]).parent.joinpath("pascal_voc_2012")
    
    all_configs["AUGMENTED_DIR_GBL"] = dataset_dir
    
    last_model_mode = False
    
    optuna_model = False
    
    roc_regression = True
    
    all_models = get_all_available_models( )
    
    top_model_counts = generate_float_list(1, len(all_models), 1)    
    iou_thresholds = generate_float_list(0, 1, 0.05)    
    conf_thresholds = generate_float_list(0, 1, 0.05)
    
    optuna_params_arg = {
                        	"top_model_count": [
                                "name = {}".format("suggest_int"), 
                                "low = {}".format(1), 
                                "high = {}".format(len(all_models)), 
                                "step = {}".format(1), 
                                "log = {}".format(False)],
                        	"iou_threshold": [
                                "name = {}".format("suggest_float"), 
                                "low = {}".format(0.01), 
                                "high = {}".format(1.0), 
                                "step = {}".format(None), 
                                "log = {}".format(True)],
                        	"conf_threshold": [
                                "name = {}".format("suggest_float"), 
                                "low = {}".format(0.01), 
                                "high = {}".format(1.0), 
                                "step = {}".format(None), 
                                "log = {}".format(True)],
                        }
    
    sorting_key = "score" 
    
    use_optuna = False
    
    trained_models, model_scores, checkpoints, final_trained_classes = getTrainedModels(
                                                                                            all_models = all_models, 
                                                                                            settings = all_configs, 
                                                                                            dataset_name = dataset_name, 
                                                                                            optuna_model = optuna_model, 
                                                                                            last_model_mode = last_model_mode
                                                                                        )  
                
    targeted_dataloaders = getDatasets(
                                            final_trained_classes = final_trained_classes, 
                                            settings = all_configs, 
                                            trainer = False, 
                                            validator = False, 
                                            tester = True
                                        )
    
    if (use_optuna is True):    
        objective = inference_optuna( )            
        n_trials = round( ( len(top_model_counts) * len(iou_thresholds) * len(conf_thresholds) ) / 4 )
        print ("Optuna n-Trials: {}".format(n_trials))
        all_configs["OPTUNA_STUDY_NAME_NT"] = "inference_001"
        
        storage_uri = "mysql://{}:{}@{}/{}".format( 
                                                        all_configs["OPTUNA_STUDY_DB_USER_NT"],
                                                        all_configs["OPTUNA_STUDY_DB_PASS_NT"],
                                                        all_configs["OPTUNA_STUDY_DB_ADDRESS_NT"],
                                                        all_configs["OPTUNA_STUDY_NAME_NT"],
                                                    )
        
        func = lambda trial: objective( 
                                            trial = trial, 
                                            max_trials = n_trials,
                                            all_models = all_models, 
                                            trained_models = trained_models, 
                                            model_scores = model_scores, 
                                            checkpoints = checkpoints, 
                                            sorting_key = sorting_key, 
                                            # top_model_count = top_model_count, 
                                            settings = all_configs, 
                                            final_trained_classes = final_trained_classes, 
                                            targeted_dataloaders = targeted_dataloaders, 
                                            # iou_threshold = iou_threshold,
                                            # conf_threshold = conf_threshold,
                                            optuna_params_arg = optuna_params_arg,
                                            roc_regression = roc_regression
                                        )
        
        study = optuna.create_study( 
                                        study_name = all_configs["OPTUNA_STUDY_NAME_NT"], 
                                        storage = sqlalchemy_db_checker( storage_uri ), 
                                        load_if_exists = True,
                                        directions = ["maximize"]
                                    )
        study.optimize( 
                            func, 
                            n_trials=n_trials, 
                            callbacks=[
                                            objective.callback,
                                            optuna.study.MaxTrialsCallback( n_trials ),
                                       ], 
                            gc_after_trial=True, 
                            show_progress_bar=True,
                        )
        
        best_sdmo_selected_models = objective.best_sdmo_selected_models
        best_sdmo_results = objective.best_sdmo_results
        best_sdmo_optuna_params = objective.best_sdmo_optuna_params
        best_sdmo_performance_score = objective.best_sdmo_performance_score
        best_sdmo_trial_number = objective.best_sdmo_trial_number
        best_sdmo_trained_classes = objective.best_sdmo_trained_classes
        
        trials = study.trials_dataframe()
        print("Number of completed trials: {}".format(len(trials[trials.state == "COMPLETE"])))
    
        print("Best trial:")
        trial = study.best_trial
    
        print("  Value: ", trial.value)
    
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))    
        
        print( "Best Score:" )
        print( "Global ROC mAP: {}".format(best_sdmo_performance_score))
        
        print( "Best Selected Models:" )
        print( best_sdmo_selected_models )
        
    else:
        for top_model_count in top_model_counts:
            for iou_threshold in iou_thresholds:
                for conf_threshold in conf_thresholds: 
                    
                    record_metrics, selected_model_keys = inference(
                                                                        all_models = all_models, 
                                                                        trained_models = trained_models, 
                                                                        model_scores = model_scores, 
                                                                        checkpoints = checkpoints, 
                                                                        sorting_key = sorting_key, 
                                                                        top_model_count = top_model_count,
                                                                        settings = all_configs, 
                                                                        final_trained_classes = final_trained_classes, 
                                                                        targeted_dataloaders = targeted_dataloaders, 
                                                                        iou_threshold = iou_threshold,
                                                                        conf_threshold = conf_threshold,
                                                                        roc_regression = roc_regression
                                                                    )