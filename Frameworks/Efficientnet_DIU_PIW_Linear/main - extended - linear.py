# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 14:40:46 2023

@author: Shaik
"""
#%% libraries and directory

import os
import inspect
import torch
import sys
import optuna, copy
import matplotlib.pyplot as plt
from lib.utils import  tic, toc, time_duration, collate_fn, show_tranformed_image, Averager, Savebestmodel, SaveBestModel_2, \
                        save_model, save_training_plots, save_validation_plots, save_K_plots, CSV_file, sqlalchemy_db_checker
        
    
from lib.model import create_model
from lib.train_Dynamic import train_Dynamic
from lib.train_basic import train_basic
from lib.create_folder import create_folder
from lib.testing import testing
from lib.validation import validater
from lib.validation_2 import validater_WGT
from lib.validation_3 import validater_AGT
import yaml, pathlib, datetime
from lib.dataloader_aio import ModelAllDataloader,ModelAllDataset
import warnings
warnings.filterwarnings("ignore")

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0,grandparentdir)

grandgrandparentdir = os.path.dirname(grandparentdir)
sys.path.insert(0,grandgrandparentdir)


#%% loading cofig and pickle file

yml_file_path = os.path.join(currentdir, 'lib/config.yaml')
yml_file = open(yml_file_path, "r")
config = yaml.load(yml_file,  Loader=yaml.FullLoader)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#%% setup_logger

def setup_logger( filename, runningdir ):

    import logging, builtins
        
    logs_path = pathlib.Path(runningdir).joinpath("logs")
    logs_path.mkdir(parents=True, exist_ok=True)
        
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(logs_path.joinpath(filename), 'w'))
        
    _print = print
    # builtins.print = lambda *tup : logger.info(str(" ".join([str(x) for x in tup])))
        
    # Modified lambda function to handle keyword arguments
    def custom_print(*args, **kwargs):
        if 'file' in kwargs and kwargs['file'] is not None:
            _print(*args, **kwargs)  # Calls the original print for specific file outputs
        else:
            logger.info(" ".join([str(x) for x in args]))
                
    builtins.print = custom_print
        
    return _print

#%% start_training

def start_training(score_card, training_type, config, model, optimizer, DEVICE, train_loader, valid_loader, test_loader, loss_type, optuna_trial_num=None):
    
    if not optuna_trial_num is None:
        folder_name = "{}_{}_{}_optuna_trial{}".format( 
                                                            config['model_name'], 
                                                            training_type,
                                                            loss_type,
                                                            str(optuna_trial_num).zfill(3)
                                                        )
    else:    
        folder_name = "{}_{}_{}_manual".format( 
                                                    config['model_name'], 
                                                    training_type,
                                                    loss_type
                                                )
    
    training_plots , validation_plots , model_info, loss_param , K_comp_plot , test_results = create_folder(folder_name)
    train_itr, val_itr, count = 1, 1, 1 
    train_loss_list, val_loss_list, avg_train_loss, avg_val_loss = ([] for _ in range(4))
    k1_label_list, k_avg_list, K_pro_epoc, Multiplier = ([] for _ in range(4))
    idx_range = 0
    csv_itr = 0
    loss_hyper = []
    
    saver = Savebestmodel()
    save_best_model = SaveBestModel_2( model_name=folder_name )
    
    result = {
        'epoch': 0,
        'score': float('-inf'),
        'map_global': None,
        'train_loss': None,
        'val_loss': None,
        'all_results': None,
        }
    
    for epoch in range(config['Num_Epochs']):
        
        print("\nEPOCH {} of {}".format(str( epoch + 1 ).zfill( 3 ), config['Num_Epochs']))
        train_loss_hist, val_loss_hist = (Averager() for _ in range(2))
        avg_mul_factor, avg_unc_k = (Averager() for _ in range(2))
                    
        # reset the training and validation loss histories for the current epoch
        train_loss_hist.reset()
        val_loss_hist.reset()
        avg_mul_factor.reset()
        avg_unc_k.reset()
        
        if training_type == "train_val_basic":
            # start timer for training 
            tic()
            training_loss , train_loss_hist = train_basic(train_loader,model, optimizer, DEVICE, train_loss_hist, train_loss_list, train_itr)
            toc()
            Train_time =  time_duration()
            
            # start timer for validation
            tic()
            validation_loss , val_loss_hist = validater(valid_loader, model, DEVICE, val_loss_hist, val_loss_list, val_itr)
            toc()
            Validation_time =  time_duration()
        
        elif training_type == "train_val_WGT":
            # start timer for training 
            tic()
            training_loss , train_loss_hist = train_Dynamic(train_loader,model, optimizer, DEVICE, train_loss_hist, train_loss_list, train_itr ,Multiplier ,epoch ,idx_range, loss_type)
            if epoch > 0:
                idx_range +=1       
            toc()
            Train_time =  time_duration()
                        
            # start timer for validation
            tic()
            validation_loss , val_loss_hist , avg_mul_factor , avg_unc_k, k1_label_list, k_avg_list = validater_WGT(score_card, valid_loader, model, DEVICE, val_loss_hist, val_loss_list, val_itr, avg_mul_factor, avg_unc_k , k1_label_list , k_avg_list)
            Multiplier.append(avg_mul_factor.value)
            toc()
            Validation_time =  time_duration() 
           
        elif training_type == "train_val_Advance":
            # start timer for validation
            tic()
            training_loss , train_loss_hist = train_Dynamic(train_loader,model, optimizer, DEVICE, train_loss_hist, train_loss_list, train_itr ,Multiplier , epoch , idx_range, loss_type)
            toc()
            Train_time =  time_duration()
            
            # start timer for validation
            tic()         
            validation_loss , val_loss_hist , avg_mul_factor , avg_unc_k, k1_label_list, k_avg_list, K_pro_epoc = validater_AGT(score_card, epoch, valid_loader, model, DEVICE, val_loss_hist, val_loss_list, val_itr,K_pro_epoc, avg_mul_factor, avg_unc_k , k1_label_list , k_avg_list, count )
            Multiplier.append(avg_mul_factor.value)
            if epoch > 0:
                idx_range +=1
                count += 1
            toc()
            Validation_time =  time_duration()
        
        print (f"Total Iterations avg_mul_factor -> {avg_mul_factor.get_iterations}")        
        print (f"Total Value avg_mul_factor -> {avg_mul_factor.get_current_total}")  
        print (f"Total Iterations avg_unc_k -> {avg_unc_k.get_iterations}")        
        print (f"Total Value avg_unc_k -> {avg_unc_k.get_current_total}")
        print( "Epoch #{} train loss: {}\n".format( str( epoch + 1 ).zfill( 3 ), round( train_loss_hist.value, 3 ) ) )
        print( "Epoch #{} validation loss: {}\n".format( str( epoch + 1 ).zfill( 3 ), round( val_loss_hist.value, 3 ) ) )
        print( "final average mul factor : {}\n".format( round( avg_mul_factor.value, 3 ) ) )
        print( "Epoc #{} average uncertainity  K : {}\n".format( str( epoch + 1 ).zfill( 3 ), round( avg_unc_k.value, 3 ) ) )
        print( "Total time taken for  Train_Epoch {}: {}\n".format( str( epoch + 1 ).zfill( 3 ), Train_time ) )
        print( "Total time taken for  Validation_Epoch {}: {}\n".format( str( epoch + 1 ).zfill( 3 ), Validation_time ) )
        
        #final_classes = all_classes[1:]
        final_classes = all_classes
        record_metrics = testing(test_loader, model, DEVICE, final_classes)
        record_metrics.compute()        
        results = record_metrics.GetResults()
        map_global = results["map"].cpu().numpy().tolist()
        if not isinstance(map_global, list):
            map_global = [map_global]  # Wrap the float in a list
        map_per_class = results["map_per_class"].numpy().tolist()
        if not isinstance(map_per_class, list):
            map_per_class = [map_per_class]  # Wrap the float in a list
        record_metrics.print(model_name )
        key = "Epoch_{}".format(str(epoch+1).zfill(3))
        cm_dir = os.path.join(test_results, key)
        os.makedirs(cm_dir)
        record_metrics.plot( key, True, cm_dir, final_classes)
        
        save_model(epoch, model, optimizer, model_info)
        
        if isinstance(map_global, (int, float)):
            map_global = [map_global]
            
        score_now = map_global[0] - train_loss_hist.value - val_loss_hist.value
        
        if score_now > result['score']:
            result['score'] = copy.deepcopy( score_now )
            result['epoch'] = copy.deepcopy( epoch + 1 )
            result['map_global'] = copy.deepcopy( map_global[0] )
            result['train_loss'] = copy.deepcopy( train_loss_hist.value )
            result['val_loss'] = copy.deepcopy( val_loss_hist.value )
            result['all_results'] = copy.deepcopy( results )
            
            
        # Call the instance with the required arguments
        saver(score_now, epoch, model, optimizer, model_info)
        save_training_plots(training_loss, training_plots, epoch)
        save_validation_plots(validation_loss, validation_plots, epoch)
        avg_train_loss.append(train_loss_hist.value)
        avg_val_loss.append(val_loss_hist.value)
        
        if (epoch + 1) % 5 == 0:
            save_training_plots(avg_train_loss, training_plots, epoch)
            save_validation_plots(avg_val_loss, validation_plots, epoch)

        
        save_K_plots(k1_label_list, k_avg_list, K_comp_plot, epoch)
        plt.close('all')
 
        save_best_model.save_model(model, optimizer, epoch, score_now, map_global[0], all_classes)
        loss_fields = ['Model Name']+['Epoch']+ ['Training_loss']+['Train_time' ]+ ['validation_loss']+ ['Validation_time' ]+[ 'MAP_Global'] + ['Score']
        loss_fields.extend([f"mAP_{i+1}" for i in range(len(map_per_class))])
        loss_hyper.append([folder_name, epoch , round(train_loss_hist.value , 3),Train_time , round(val_loss_hist.value,3),Validation_time, map_global[0], score_now ])
        for i in range(len(map_per_class)):
            loss_hyper[csv_itr].append(map_per_class[i])
        
        csv_itr += 1
        pth_loss = os.path.join( loss_param , "loss_hyper.csv")
     
        CSV_file(pth_loss, loss_fields, loss_hyper , score_card)
        
    return result

#%% optuna_run
        
class optuna_run:
    def __init__( self ):
        self.best_results = None
        self._results = None
        self._best_score = float('-inf')
    def __call__( 
                    self, 
                    trial, 
                    max_trials,
                    config, 
                    pretrained,
                    all_classes,
                    DEVICE, 
                    train_loader, 
                    valid_loader, 
                    test_loader,
                    training_type,
                    ):
        
        # optuna_params = {}
        # for param_name, param_info in optuna_params_arg.items():
        #     suggest_method_name = param_info[0].split("=")[1].strip()
        #     suggest_method_args = {
        #         arg.split("=")[0].strip(): eval(arg.split("=")[1].strip()) for arg in param_info[1:]
        #     }
        #     suggest_method = getattr(trial, suggest_method_name)
        #     optuna_params[param_name] = suggest_method(param_name, **suggest_method_args)
        
        # 1) Suggest hyperparameters (our multiplication factors)
        factor_max = trial.suggest_float("factor_max", 0.1, 2.0)
        factor_min  = trial.suggest_float("factor_min", 0.1, 2.0)
        
        # 2) Construct your score_card dict with these trial-suggested values
        score_card = {
            "factor_max": factor_max,
            "factor_min": factor_min,
        }
        
        print ( "**************************** Trial - {} ****************************".format( str( trial.number ) ) )
        for key in score_card.keys():
            print ("{} - {}".format(key, score_card[key]))
        print ( )
        print ( "*************************************************************" )
        
        if trial.should_prune():
            raise optuna.TrialPruned()
            
        model = create_model(len(all_classes), pretrained = pretrained, model_name = config['model_name'])
        model = model.to(DEVICE)
        print ( "*************************************************************" )
        # get the model parameters
        params = [p for p in model.parameters() if p.requires_grad]
        #optimizer 
        optimizer = torch.optim.Adagrad(params, lr = 0.001)
        
        results = start_training(
                                score_card = score_card ,
                                training_type = training_type, 
                                config = config, 
                                model = model, 
                                optimizer = optimizer, 
                                DEVICE = DEVICE, 
                                train_loader = train_loader, 
                                valid_loader = valid_loader, 
                                test_loader = test_loader, 
                                loss_type = 'loss_multiplication',
                                optuna_trial_num = trial.number)
        
        self._results = copy.deepcopy( results )
        
        print(f"Best score for Trial - {trial.number} is {self._best_score}")
        
        if self._results['score'] > self._best_score:
            self._best_score = copy.deepcopy( self._results['score'] )
        
        return self._best_score
        #return self._results['score']
    
    def callback ( self, study, trial ):
        
        if study.best_trial.number == trial.number:
            self.best_results = copy.deepcopy( self._results )

#%% optuna_duplicate_iteration_pruner

class optuna_duplicate_iteration_pruner(optuna.pruners.BasePruner):
    """
    DuplicatePruner

    Pruner to detect duplicate trials based on the parameters.

    This pruner is used to identify and prune trials that have the same set of parameters
    as a previously completed trial.
    """

    def prune( self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial" ) -> bool:
        completed_trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])

        for completed_trial in completed_trials:
            if completed_trial.params == trial.params:
                print ( "********************* TRIAL - {} PRUNED *********************".format( str( trial.number ) ) )
                print ( )
                print ( "*************************************************************" )
                return True

        return False
    
#%% optuna_stop_when_trial_keep_being_pruned

class optuna_stop_when_trial_keep_being_pruned:
    
    def __init__(self, threshold: int):
        self.threshold = threshold
        self._consequtive_pruned_count = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.PRUNED:
            self._consequtive_pruned_count += 1
        else:
            self._consequtive_pruned_count = 0

        if self._consequtive_pruned_count >= self.threshold:
            study.stop()
    
#%% main

if __name__ == "__main__":
    
    print = setup_logger( datetime.datetime.now().strftime("%I_%M%p_%B_%d_%Y"), currentdir )
    
    all_classes = [
                        "__background__", "person" ,"bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "bicycle", "boat",
                       "bus", "car", "motorbike", "train", "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"
                 ]
    
    
    config["MASKED_CNN_GBL"] = False
    config["MULTI_THREADED_DATA_LOADING_NT"] = True
    config["MULTI_THREADED_DATA_LOADING_WORKERS_NT"] = 1000
    config["MULTI_THREADED_DATA_LOADING_QUEUE_NT"] = 10000
    
    trainer = config['trainer']
    validator =  config['validator']
    inference = config['infernce']
    dataset_dir = os.path.join(parentdir, config['data']['main_dir'])
    
    
#%% Dataloader
    
    tic()
    
    all_datasets = ModelAllDataset(dataset_dir, all_classes)
    train_data = all_datasets.get_training_data(all_classes, config)
    train_loader = ModelAllDataloader(
                                        train_data, batch_size= trainer['batch_size'],
                                        shuffle=trainer['shuffle'],num_workers=trainer['num_workers'],
                                        collate_fn = collate_fn
                                        
                                        )
        
    val_data = all_datasets.get_validation_data(all_classes, config)
    valid_loader = ModelAllDataloader(
                                        val_data, batch_size=validator['batch_size'],  
                                        shuffle=validator['shuffle'], num_workers=validator['num_workers'], 
                                        collate_fn=collate_fn 
                                        
                                        )
    
    
    test_data = all_datasets.get_testing_data(all_classes, config)
    test_loader = ModelAllDataloader(
                                        test_data, batch_size=inference['batch_size'],   
                                        shuffle=inference['shuffle'], num_workers= inference['num_workers'],
                                        collate_fn=collate_fn
                                        
                                        )
   
    toc()
    print(f"Number of training samples: {len(train_data)}")    
    print(f"Number of validation samples: {len(val_data)}\n")    
    print(f"Number of testing samples: {len(test_data)}\n")
    print(f"Data loading took : {time_duration()}\n")
    
#%% To check if data is correctly loaded by visualizing the images and there labels and bnb
    
    if config ['visualize_images'] == True:
        show_tranformed_image(train_loader, all_classes)
        
  
#%% model defination and its parameters ,,and optimizer
   
    use_optuna = False
    pretrained = True
    # start training and validation and storing values
    training_types = ["train_val_basic", "train_val_WGT", "train_val_Advance"]
    
    if (use_optuna is True):
        
        for training_type in training_types:
                
            model_name = config['model_name']    
           
            
            objective = optuna_run( )
            #n_trials = 9*2
            n_trials = 50
            
            storage_uri = "mysql://{}:{}@{}/{}".format( 
                                                            "optuna_study",
                                                            "optuna_study_pass",
                                                            "193.174.70.66:3306",
                                                            "tahasanul_optuna",
                                                        )
            func = lambda trial: objective( 
                                                trial = trial, 
                                                max_trials = n_trials,
                                                config = config,  
                                                pretrained = pretrained,
                                                all_classes = all_classes,
                                                DEVICE = DEVICE, 
                                                train_loader = train_loader, 
                                                valid_loader = valid_loader, 
                                                test_loader = test_loader,
                                                training_type = training_type
                                            )
            
            pruner = optuna_duplicate_iteration_pruner()
            stopper = optuna_stop_when_trial_keep_being_pruned(20)
            
            study = optuna.create_study( 
                                            study_name = "Linear_TuneScoreCard_{}_{}".format(model_name, training_type), 
                                            storage = sqlalchemy_db_checker( storage_uri ), 
                                            load_if_exists = True,
                                            directions = ["maximize"],
                                            pruner = pruner
                                        )
            study.optimize( 
                                func, 
                                n_trials=n_trials, 
                                callbacks=[
                                                objective.callback,
                                                optuna.study.MaxTrialsCallback( n_trials ),
                                                stopper,
                                           ], 
                                gc_after_trial=True, 
                                show_progress_bar=True,
                            )
            
            del objective, n_trials, n_trials, study, storage_uri, func, pruner, stopper
        
    else:
        model = create_model(len(all_classes), pretrained = pretrained, model_name = config['model_name'])
        model = model.to(DEVICE)
        model_name = config['model_name']
        print ( "*************************************************************" )
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adagrad(params, lr = 0.001)
        score_card = {
            "factor_max": 1.3149542302719095,
            "factor_min": 0.4816843484470241,
        }
        results = start_training(
                                score_card = score_card ,
                                training_type = "train_val_WGT", 
                                config = config, 
                                model = model, 
                                optimizer = optimizer, 
                                DEVICE = DEVICE, 
                                train_loader = train_loader, 
                                valid_loader = valid_loader, 
                                test_loader = test_loader, 
                                loss_type = 'loss_multiplication')
    
    

