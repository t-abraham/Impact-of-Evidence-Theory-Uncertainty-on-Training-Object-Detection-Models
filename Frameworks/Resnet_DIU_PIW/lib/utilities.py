# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 18:23:43 2022

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

import pickle, torch, yaml, random, requests, copy
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import xml.etree.cElementTree as ET
import xml.dom.minidom as MD

# from shapely import box
from pytimedinput import timedInput
from PIL import ImageFont, ImageDraw, Image
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
# You can use Matplotlib instead of Plotly for visualization by simply replacing `optuna.visualization` with
# `optuna.visualization.matplotlib` in the following examples.
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

from lib.roc import roc

plt.rcParams["axes.grid"] = False

#%% earlystopper

class earlystopper():
    def __init__(  self  ):
        self._bucket = []
        self._bucket_bool = []
        self._windowsize = 5
        self._callstop = False
        
    def set_windowsize( self, size ):
        self._windowsize = size
        
    def send( self, value ):
        self._bucket.append( value )
        self._bucket = self._bucket[ -self._windowsize: ]
        
    def update( self, direction, threshold):        
        self._bucket_bool = []
        
        if "min" in direction.lower():
            # print ( "min" )
            for count, value in enumerate( self._bucket ):
                if count == 0:
                    self._bucket_bool.append( True )
                else:
                    difference = abs( value - self._bucket[ count-1 ] )
                    difference_percent = ( difference / abs( self._bucket[ count-1 ] ) ) * 100
                    if value < self._bucket[ count-1 ] and difference_percent > threshold:
                        self._bucket_bool.append( False )
                    else:
                        self._bucket_bool.append( True )
                        
        elif "max" in direction.lower():
            # print ( "max" )
            for count, value in enumerate( self._bucket ):
                if count == 0:
                    self._bucket_bool.append( True )
                else:
                    difference = abs( value - self._bucket[ count-1 ] )
                    difference_percent = ( difference / abs( self._bucket[ count-1 ] ) ) * 100
                    if value > self._bucket[ count-1 ] and difference_percent > threshold:
                        self._bucket_bool.append( False )
                    else:
                        self._bucket_bool.append( True )
                        
        # self._bucket_bool = self._bucket_bool[ -self._windowsize: ]
        
    @property
    def windowsize( self ):
        return self._windowsize
    
    @property
    def callstop( self ):
        if len( self._bucket_bool ) == self._windowsize:
            self._callstop = all( x is True for x in self._bucket_bool )
            
        return self._callstop
    
    @property
    def bucket( self ):
        return self._bucket
    
    @property
    def bucket_bool( self ):
        return self._bucket_bool
        
#%% punisher

def punisher( x, settings ):
    
    soft_x = round( settings["TRAINING_NUM_EPOCHS_NT"] * settings["PUNISHER_XSOFT_NT"] )
    soft_y = settings["PUNISHER_YSOFT_NT"]
    hard_x = settings["TRAINING_NUM_EPOCHS_NT"]
    
    nomi = np.log( 1 - soft_y)    
    deno = np.log( soft_x ) - np.log( hard_x )
    y = -np.power( ( x / hard_x ), ( nomi / deno ) ) + 1
    
    if y > 1:
        y = 1
    elif y < 0:
        y = 0
    
    return y
    
#%% send_to_telegram

def send_to_telegram(  settings, message  ):   

    Telegram_apiURL = "https://api.telegram.org/bot{}/sendMessage".format( settings["Telegram_apiToken"] )
    
    try:
        response = requests.post( Telegram_apiURL, json={"parse_mode": "HTML", "chat_id": settings["Telegram_chatID"], "text": message} )
        if response.status_code == 200:
            print ( "Successfully Posted to Telegram Channel" )
        else:
            raise Exception( "Response not ok - {}".format( response.status_code ) )
    except Exception as e:
        print( e )
        
#%% filename_correction

def filename_correction(  complete_saving_path, counter=0  ):
    
    if os.path.exists(  complete_saving_path  ) is True:
        counter += 1
        complete_saving_path_list = complete_saving_path.split( "." )
        if complete_saving_path_list[1] == str(  counter-1  ):
            complete_saving_path_list[1] = str(  counter  )
        else:
            complete_saving_path_list.insert(  1, str(  counter  )  )
        
        complete_saving_path = filename_correction(  ".".join(  complete_saving_path_list  ),counter  )
        return complete_saving_path
    else:
        return complete_saving_path

#%% str_to_bool

def str_to_bool( s ):
    try:
        if str( s ).lower( ) == "true":
             return True
        elif str( s ).lower( ) == "false":
             return False
        else:
            return "ERROR"
    except:
        return "BOOL-ERROR"
    
#%% str_to_int

def str_to_int( s ):
    try:
        return int( float( str( s ) ) )
    except:
        return "INT-ERROR"
    
#%% str_to_float
def str_to_float(  s  ):
    try:
        return float(  str(  s  )  )
    except:
        return "FLOAT-ERROR"
    
#%% str_to_tuple

def str_to_tuple( s ):
    try:
        if str( s )[0] == "(" and str( s )[-1] == ")":
            res = s[1:-1].split( "," )
            if not len( res ) == 1:
                return ( str_to_int( res[0].lstrip(  ).rstrip(  ) ), str_to_int( res[1].lstrip(  ).rstrip(  ) ) )
            else:
                return ( str_to_int( res[0].lstrip(  ).rstrip(  ) ) )
        else:
            return ( str_to_int( s ) )
    except:
        return "RES-ERROR"

#%% sqlalchemy_db_checker

def sqlalchemy_db_checker( uri ):
    engine = create_engine( uri )
    
    # Create database if it does not exist.
    if not database_exists(engine.url):
        create_database(engine.url)
        
    del engine
    
    return uri

#%% Averager

class Averager:
    def __init__(  self  ):
        self.list = []
        
    def send(  self, value  ):
        self.list.append( value )
    
    @property
    def avg_value(  self  ):
        if len(  self.list  ) == 0:
            return 0
        else:
            return sum(  self.list  ) / len(  self.list  )
        
    @property    
    def all_values(  self  ):
        return self.list
    
    def reset(  self  ):
        self.list = []
        
#%% check_n_load_existing_model

def check_n_load_existing_model( settings, model_name, file_name, interference=False):
    checkpoint = None
    model_saving_path = os.path.join( settings["TRAINING_MODEL_DIR_NT"], model_name )
    if interference is True:
        model_filename = "{}".format(  file_name  )
    else:
        model_filename = "last_model.{}.pth".format(  file_name  )
    
    last_model_path = os.path.join(  model_saving_path, model_filename  ) 
    
    if ( os.path.exists( last_model_path ) ):
        
            checkpoint = torch.load( 
                                                last_model_path, 
                                                map_location=settings["DEVICE_GBL"]
                                             )
        
    return checkpoint

#%% SaveModel

class SaveModel:
    """
    Class to save the best model while training. If the current epoch"s 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__( self, ):
        self.matric_best_performance_score = None
        self.matric_best_model = None
        self.matric_best_optimizer = None
        self.matric_best_stats_averager = None
        self.matric_best_stats_list = None
        self.matric_best_model_map_results = None
        self.matric_best_roc_map_results = None
        self.matric_best_params = None
        self.matric_best_epoch = None
        self.matric_best_punishing_factor = None        
        self.matric_best_trained_classes = None
        
    def save_best( 
        self, settings, model_name, matric_name, trained_classes, matric_current_score, matric_direction, stats_averager,
        stats_list, model_map_results, roc_map_results, epoch, punishing_factor, optimizer_name, parameters, model, optimizer, file_name, gen_save_plt=False
     ):
        
        trigger = False
        
        if "min" in matric_direction.lower(  ):
            
            if self.matric_best_performance_score is None:
                self.matric_best_performance_score = float( "inf" )
                
            if matric_current_score < self.matric_best_performance_score:
                trigger = True
                
        elif "max" in matric_direction.lower(  ):
            
            if self.matric_best_performance_score is None:
                self.matric_best_performance_score = float( "-inf" )
                
            if matric_current_score > self.matric_best_performance_score:
                trigger = True
                
        if trigger is True:
            self.matric_best_performance_score = copy.deepcopy( matric_current_score )
            self.matric_best_model = copy.deepcopy( model )
            self.matric_best_optimizer = copy.deepcopy( optimizer )
            self.matric_best_stats_averager = copy.deepcopy( stats_averager )
            self.matric_best_stats_list = copy.deepcopy( stats_list )
            self.matric_best_model_map_results = copy.deepcopy( model_map_results )
            self.matric_best_roc_map_results = copy.deepcopy( roc_map_results )
            self.matric_best_params = copy.deepcopy( parameters )
            self.matric_best_epoch = copy.deepcopy( epoch )
            self.matric_best_punishing_factor = copy.deepcopy( punishing_factor )        
            self.matric_best_trained_classes = copy.deepcopy( trained_classes )
            
            print (  "\nBest {} score: {}".format(  matric_name, self.matric_best_performance_score  )  )
            print (  "\nSaving best model for epoch: {}\n".format(  self.matric_best_epoch  )  )
            model_saving_path = os.path.join( settings["TRAINING_MODEL_DIR_NT"], model_name )
            os.makedirs( model_saving_path, exist_ok=True )
            torch_save_filename = "{}.pth".format(  file_name  )
            torch.save( 
                            {
                                "epoch": self.matric_best_epoch,
                                "model_map": self.matric_best_model_map_results,
                                "roc_map": self.matric_best_roc_map_results,
                                "score": self.matric_best_performance_score,
                                "stats_list": self.matric_best_stats_list,
                                "punishing_factor": self.matric_best_punishing_factor,
                                "optimizer_name": optimizer_name,
                                "parameters": self.matric_best_params,
                                "model_state_dict": self.matric_best_model.state_dict(  ),
                                "optimizer_state_dict": self.matric_best_optimizer.state_dict(  ),
                                "trained_classes": self.matric_best_trained_classes,
                            }, 
                            os.path.join(  model_saving_path, torch_save_filename  ) 
                         )
                
            if gen_save_plt is True:
                self._plt_saver(
                                    settings = settings, 
                                    model_name = model_name, 
                                    stats_list = self.matric_best_stats_list, 
                                    file_name = file_name,
                                    caller_type = "best_model"
                                )
        else:
            print (  "\nCurrent Best {} score: {}".format(  matric_name, self.matric_best_performance_score  )  )
            print (  "\nGiven {} score: {}".format(  matric_name, matric_current_score  )  )
            
    def save_last( 
        self, settings, model_name, matric_name, trained_classes, matric_current_score, matric_direction, stats_averager,
        stats_list, model_map_results, roc_map_results, epoch, punishing_factor, optimizer_name, parameters, model, optimizer, file_name, gen_save_plt=False
     ):
        
        print (  "\nSaving last model for epoch: {}\n".format(  epoch  )  )
        model_saving_path = os.path.join( settings["TRAINING_MODEL_DIR_NT"], model_name )
        os.makedirs( model_saving_path, exist_ok=True )            
        torch_save_filename = "last_model.{}.pth".format(  file_name  )
        torch.save( 
                        {
                            "epoch": epoch,
                            "model_map": model_map_results,
                            "roc_map": roc_map_results,
                            "score": matric_current_score,
                            "stats_list": stats_list,
                            "punishing_factor": punishing_factor,
                            "optimizer_name": optimizer_name,
                            "parameters": parameters,
                            "model_state_dict": model.state_dict(  ),
                            "optimizer_state_dict": optimizer.state_dict(  ),
                            "trained_classes": trained_classes,
                        }, 
                        os.path.join(  model_saving_path, torch_save_filename  ) 
                     )
    
        if gen_save_plt is True:
            self._plt_saver(
                                settings = settings, 
                                model_name = model_name, 
                                stats_list = stats_list, 
                                file_name = file_name,
                                caller_type = "last_model"
                            )
        
        
    def _plt_saver(self, settings, model_name, stats_list, file_name, caller_type):  
          
        plt_saving_path = os.path.join(  settings["STATISTICAL_DIR_GBL"], model_name  )
        os.makedirs(  plt_saving_path, exist_ok=True  )
        
        model_name_lower = model_name.lower(  )
        
        for key in stats_list.keys(  ):             
            fig = plt.figure(  )
            x = [ i for i in range(  1, len(  stats_list[key]  )+1  ) ]
            y = stats_list[key]
            plt.plot( x, y, color="green", linestyle="dashed", linewidth = 1, marker="o", markerfacecolor="blue", markersize=5 )
            # xticks = [0] + x
            # plt.xticks( x,xticks )
            plt.xlabel(  "{} Count".format(  str(  str(  key  ).split( "_" )[0]  ).capitalize(  )  )  )
            plt.ylabel(  "{} Time in Seconds".format(  str(  str(  key  ).split( "_" )[0]  ).capitalize(  )  )  )
            plt.title(  "{} Time Elapsed Satistics".format(  model_name_lower.capitalize(  )  )  )
            plt_filename = os.path.join(  plt_saving_path, "{}.{}.{}.jpg".format(  caller_type, file_name, key  )  )
            plt.savefig(  plt_filename  )    
            # plt.show(  )                
            fig.clf(  )
            plt.clf(  )
            plt.close( fig )
            
    @property    
    def get_best_model(  self  ):
        return self.matric_best_performance_score, \
                self.matric_best_model, \
                self.matric_best_optimizer, \
                self.matric_best_stats_averager, \
                self.matric_best_stats_list, \
                self.matric_best_model_map_results, \
                self.matric_best_roc_map_results, \
                self.matric_best_params, \
                self.matric_best_epoch, \
                self.matric_best_punishing_factor

#%% optuna_plt_saver

def optuna_plt_saver( settings, study, model_name, params, file_name ):
    plt_saving_path = os.path.join(  settings["STATISTICAL_DIR_GBL"], model_name  )
    os.makedirs(  plt_saving_path, exist_ok=True  )
    ###################################################################################################
    # Plot functions
    # --------------
    # Visualize the optimization history. See :func:`~optuna.visualization.plot_optimization_history` for the details.
    key = "optimization_history"
    plt_filename = os.path.join(  plt_saving_path, "visuals.{}.{}.jpg".format(  file_name, key  )  )
    plot_optimization_history(study).write_image(plt_filename)
    
    ###################################################################################################
    # Visualize the learning curves of the trials. See :func:`~optuna.visualization.plot_intermediate_values` for the details.
    # key = "learning_curves"
    # plt_filename = os.path.join(  plt_saving_path, "visuals.{}.{}.jpg".format(  file_name, key  )  )
    # plot_intermediate_values(study).write_image(plt_filename)
    
    ###################################################################################################
    # Visualize high-dimensional parameter relationships. See :func:`~optuna.visualization.plot_parallel_coordinate` for the details.
    key = "hd_parameter_relationships"
    plt_filename = os.path.join(  plt_saving_path, "visuals.{}.{}.jpg".format(  file_name, key  )  )
    plot_parallel_coordinate(study).write_image(plt_filename)
    
    ###################################################################################################
    # Select parameters to visualize.
    key = "parameters_coordinate"
    plt_filename = os.path.join(  plt_saving_path, "visuals.{}.{}.jpg".format(  file_name, key  )  )
    plot_parallel_coordinate(study, params=params).write_image(plt_filename)
    
    ###################################################################################################
    # Visualize hyperparameter relationships. See :func:`~optuna.visualization.plot_contour` for the details.
    key = "parameter_relationships"
    plt_filename = os.path.join(  plt_saving_path, "visuals.{}.{}.jpg".format(  file_name, key  )  )
    plot_contour(study).write_image(plt_filename)
    
    ###################################################################################################
    # Select parameters to visualize.
    key = "parameters_contour"
    plt_filename = os.path.join(  plt_saving_path, "visuals.{}.{}.jpg".format(  file_name, key  )  )
    plot_contour(study, params=params).write_image(plt_filename)
    
    ###################################################################################################
    # Visualize individual hyperparameters as slice plot. See :func:`~optuna.visualization.plot_slice` for the details.
    key = "parameters_sliced"
    plt_filename = os.path.join(  plt_saving_path, "visuals.{}.{}.jpg".format(  file_name, key  )  )
    plot_slice(study).write_image(plt_filename)
    
    ###################################################################################################
    # Select parameters to visualize.
    key = "parameters_sliced_solo"
    plt_filename = os.path.join(  plt_saving_path, "visuals.{}.{}.jpg".format(  file_name, key  )  )
    plot_slice(study, params=params).write_image(plt_filename)
    
    ###################################################################################################
    # Visualize parameter importances. See :func:`~optuna.visualization.plot_param_importances` for the details.
    key = "parameter_importances"
    plt_filename = os.path.join(  plt_saving_path, "visuals.{}.{}.jpg".format(  file_name, key  )  )
    plot_param_importances(study).write_image(plt_filename)
    
    ###################################################################################################
    # Learn which hyperparameters are affecting the trial duration with hyperparameter importance.
    key = "parameter_effecting_time"
    plt_filename = os.path.join(  plt_saving_path, "visuals.{}.{}.jpg".format(  file_name, key  )  )
    plot_param_importances(
        study, target=lambda t: t.duration.total_seconds(), target_name="duration"
    ).write_image(plt_filename)
    
    ###################################################################################################
    # Visualize empirical distribution function. See :func:`~optuna.visualization.plot_edf` for the details.
    key = "empirical_distribution"
    plt_filename = os.path.join(  plt_saving_path, "visuals.{}.{}.jpg".format(  file_name, key  )  )
    plot_edf(study).write_image(plt_filename)

#%% collate_fn

def collate_fn( batch ):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple( zip( *batch ) )

#%% storeVariableWP

def storeVariableWP ( variable, filename, dir_path, database_type = None ):
    
    if database_type is None:
        location = os.path.join(  dir_path  )
    else:        
        location = os.path.join(  dir_path, database_type  )
    
    os.makedirs( location, exist_ok=True )
    location = os.path.join(  location, filename  )
    
    variablePkl = open( location,"wb" )
    # print ( location )
    pickle.dump( variable,variablePkl )
    variablePkl.close(  )
    
    return

#%% retrieveVariableWP
    
def retrieveVariableWP ( filename, dir_path, database_types = None ):    
    try:
        if database_types is None:            
            location = os.path.join(  dir_path, filename  )
        else:
            if isinstance( database_types, list ) or isinstance( database_types, tuple ):
                location = os.path.join(  dir_path  )
                for database_type in database_types:
                    location = os.path.join(  location, database_type  )
            else:
                location = os.path.join(  dir_path, database_types  )
            location = os.path.join(  location, filename  )
        variablePkl = open( location, "rb" )    
        variable = pickle.load( variablePkl )
        
    except Exception as e:
        if not len( str( e ) ) == 0:
            print ( "**********************************************************************" )
            print ( "*** Exception cuaght -> {} ***".format( e ) )
            print ( "**********************************************************************" )
        variable = None
    
    return variable

#%% show_tranformed_image

# def show_tranformed_image( train_loader, all_classes ):
#     """
#     This function shows the transformed images from the `train_loader`.
#     Helps to check whether the tranformed images along with the corresponding
#     labels are correct or not.
#     Only runs if `VISUALIZE_TRANSFORMED_IMAGES = True` in config.py.
#     """
    
#     if len( train_loader ) > 0:
#         for i in range( 1 ):
#             images, targets = next( iter( train_loader ) )
#             images = list( image for image in images )
#             targets = list( {k: v for k, v in t.items(  )} for t in targets )
#             boxes = targets[i]["boxes"].cpu(  ).numpy(  ).astype( np.int32 )
#             labels = targets[i]["labels"].cpu(  ).numpy(  ).astype( np.int32 )
#             sample = images[i].permute( 1, 2, 0 ).cpu(  ).numpy(  )
#             for box_num, box in enumerate( boxes ):
#                 cv2.rectangle( sample,
#                             ( box[0], box[1] ),
#                             ( box[2], box[3] ),
#                             ( 0, 0, 255 ), 2 )
#                 cv2.putText( sample, all_classes[labels[box_num]], 
#                             ( box[0], box[1]-10 ), cv2.FONT_HERSHEY_SIMPLEX, 
#                             1.0, ( 0, 0, 255 ), 2 )
#             cv2.imshow( "Transformed image", sample )
#             cv2.waitKey( 0 )
#             cv2.destroyAllWindows(  )

#%% question_promt_yes_no

def question_promt_yes_no( question,looped=False,plt_close=False ):
    
    i = 0
    user_input = False
    while i == 0:
        answer = input( "{} - Yes( Y/y ) or No( N/n )".format( question ) )
        if any( answer.lower(  ) == f for f in ["yes", "y", "1", "ye"] ):
            user_input = True
            break
        elif any( answer.lower(  ) == f for f in ["no", "n", "0"] ):
            user_input = False
            break
        if looped is False:
            i = 1    
    
    if plt_close is True:
        plt.clf(  )
        plt.close( "all" )
    
    return user_input
            
#%% show_tranformed_image

def show_tranformed_image( train_loader, all_classes, sample_size=5 ):
    """
    This function shows the transformed images from the `train_loader`.
    Helps to check whether the tranformed images along with the corresponding
    labels are correct or not.
    Only runs if `VISUALIZE_TRANSFORMED_IMAGES = True` in config.py.
    """
    
    if len( train_loader.dataset ) < sample_size:
        sample_size = len( train_loader.dataset )
    
    selected_images = random.sample(  range(  0, len( train_loader.dataset )  ), sample_size ) 
    PIL_transform = T.ToPILImage(  )  
    
    for i in selected_images:
        image, target = train_loader.dataset[i]        
        boxes = target["boxes"].cpu(  ).numpy(  ).astype( np.int32 )
        labels = target["labels"].cpu(  ).numpy(  ).astype( np.int32 )
        sample = PIL_transform( image ).copy(  )            
        draw = ImageDraw.Draw( sample )
        for box_num, box in enumerate( boxes ):
            color = tuple( np.random.randint( 256, size=3 ) )
            draw.rectangle( 
                                ( 
                                    ( box[0], box[1] ), 
                                    ( box[2], box[3] )
                                 ),
                                outline=color
                             )
            draw.text( 
                            ( box[0], box[1]-10 ),
                            all_classes[labels[box_num]], 
                            font=ImageFont.truetype( "arial.ttf", 12 ),
                            fill=color
                         )

        # sample.show(  )
        plt.imshow( sample )
        plt.show(  )
        
    # if len( train_loader ) > 0:
    #     PIL_transform = T.ToPILImage(  )            
    #     if len( train_loader ) > 5:
    #         display_loader_count = 5
    #     else:
    #         display_loader_count = len( train_loader )
            
    #     for current_loader_count in range( display_loader_count ):
    #         i = random.randint( 0, settings["TRAINING_BATCH_SIZE_NT"] )
    #         plt.figure( "Sample - {}".format( str( i ).zfill( 2 ) ) )
    #         images, targets = next( iter( train_loader ) )
    #         images = list( image for image in images )
    #         targets = list( {k: v for k, v in t.items(  )} for t in targets )
    #         boxes = targets[i]["boxes"].cpu(  ).numpy(  ).astype( np.int32 )
    #         labels = targets[i]["labels"].cpu(  ).numpy(  ).astype( np.int32 )
    #         sample = PIL_transform( images[i] ).copy(  )            
    #         draw = ImageDraw.Draw( sample )
    #         for box_num, box in enumerate( boxes ):
    #             color = tuple( np.random.randint( 256, size=3 ) )
    #             draw.rectangle( 
    #                                 ( 
    #                                     ( box[0], box[1] ), 
    #                                     ( box[2], box[3] )
    #                                  ),
    #                                 outline=color
    #                              )
    #             draw.text( 
    #                             ( box[0], box[1]-10 ),
    #                             all_classes[labels[box_num]], 
    #                             font=ImageFont.truetype( "arial.ttf", 12 ),
    #                             fill=color
    #                          )

    #         # sample.show(  )
    #         plt.imshow( sample )
    #         plt.show(  )
            
    return question_promt_yes_no( "Continue with Network Training?",True,True )
        
#%% user_promt

def user_promt(prompt="", default="", timeout=-1):
    
    if not timeout == -1 and sys.__stdin__.isatty():
        if len(prompt) > 0:
            prompt = str(prompt) + " [{} seconds timeout] >> ".format(timeout)
        else:
            prompt = "[{} seconds timeout] >> ".format(timeout)
            
    if prompt == "":
        prompt = " >> "        
    
    if sys.__stdin__.isatty():       
        
        userText, timedOut = timedInput(prompt=prompt, timeout=timeout)
    
        if timedOut is True:
            print("Timed out when waiting for input.")
    else:
        userText = input( prompt + " >> " )
        
    if len(userText) == 0:
        userText = default
    
    return userText

#%% plot_generated_image

def plot_generated_image( augmented_img ):
    sample = augmented_img["augmented_image"].copy( )
    bbox = augmented_img["bboxes"]
    label = augmented_img["labels"]
    
    draw = ImageDraw.Draw( sample )
    
    plt.clf( )
    
    color = tuple( np.random.randint( 256, size=3 ) )
    draw.rectangle( 
                        ( 
                            ( bbox[0], bbox[1] ), 
                            ( bbox[2], bbox[3] )
                        ),
                        outline=color
                    )
    draw.text( 
                    ( bbox[0], bbox[1]-10 ),
                    label, 
                    font=ImageFont.truetype( "arial.ttf", 12 ),
                    fill=color
                )

    # sample.show( )
    plt.imshow( sample )
    plt.show( )
    
    if not augmented_img["masks"] is None:
        masks = augmented_img["masks"].copy( )
        plt.clf( )
        plt.imshow( masks )
        plt.show( )
    
    return question_promt_yes_no( "Continue with Augmentation?",True,True )

#%% make_and_save_xml

def make_and_save_xml( 
        aug_image_file_name, 
        aug_image_xml_file_path, 
        aug_image_width, 
        aug_image_height, 
        aug_image_depth, 
        aug_image_bboxes, 
        aug_image_labels, 
        database="Tahasanul_Custom_DataSet"
        ):
    
    annotation = ET.Element( "annotation" )
    
    ET.SubElement( annotation, "folder" ).text = " "
    ET.SubElement( annotation, "filename" ).text = aug_image_file_name
    ET.SubElement( annotation, "path" ).text = aug_image_file_name
    
    source = ET.SubElement( annotation, "source" )
    ET.SubElement( source, "database" ).text = str( database )
    
    size = ET.SubElement( annotation, "size" )
    ET.SubElement( size, "width" ).text = str( aug_image_width )
    ET.SubElement( size, "height" ).text = str( aug_image_height )
    ET.SubElement( size, "depth" ).text = str( aug_image_depth )
    
    ET.SubElement( annotation, "segmented" ).text = "0"
    
    if isinstance( aug_image_bboxes, list ) and isinstance( aug_image_labels, list ):
        for bbox, label in zip( aug_image_bboxes, aug_image_labels ):
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[2]
            y2 = bbox[3]
            
            objects = ET.SubElement( annotation, "object" )
            ET.SubElement( objects, "name" ).text = str( label )
            ET.SubElement( objects, "pose" ).text = "Unspecified"
            ET.SubElement( objects, "truncated" ).text = "0"
            ET.SubElement( objects, "difficult" ).text = "0"
            ET.SubElement( objects, "occluded" ).text = "0"
            
            bndbox = ET.SubElement( objects, "bndbox" )
            ET.SubElement( bndbox, "xmin" ).text = str( x1 )
            ET.SubElement( bndbox, "xmax" ).text = str( x2 )
            ET.SubElement( bndbox, "ymin" ).text = str( y1 )
            ET.SubElement( bndbox, "ymax" ).text = str( y2 )
    else:
        bbox = aug_image_bboxes
        label = aug_image_labels
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        
        objects = ET.SubElement( annotation, "object" )
        ET.SubElement( objects, "name" ).text = str( label )
        ET.SubElement( objects, "pose" ).text = "Unspecified"
        ET.SubElement( objects, "truncated" ).text = "0"
        ET.SubElement( objects, "difficult" ).text = "0"
        ET.SubElement( objects, "occluded" ).text = "0"
        
        bndbox = ET.SubElement( objects, "bndbox" )
        ET.SubElement( bndbox, "xmin" ).text = str( x1 )
        ET.SubElement( bndbox, "xmax" ).text = str( x2 )
        ET.SubElement( bndbox, "ymin" ).text = str( y1 )
        ET.SubElement( bndbox, "ymax" ).text = str( y2 )
    
   
    dom = MD.parseString( ET.tostring( annotation ) )
    xml_string = dom.childNodes[0].toprettyxml( )
    
    with open( aug_image_xml_file_path, "w" ) as xfile:
        xfile.write( xml_string )
        xfile.close( )

#%% save_augment_dataset

def save_augment_dataset ( settings, augmented_img, aug_image_file_name, data_type="train" ):
    aug_image_bboxes = augmented_img["bboxes"]
    aug_image_labels = augmented_img["labels"]
    aug_image_xml_file_path = os.path.join( settings["AUGMENTED_DIR_GBL"], str( data_type ).lower( ), aug_image_file_name )
    aug_image_height = augmented_img["augmented_image"].height
    aug_image_width = augmented_img["augmented_image"].width
    aug_image_depth = 24
    aug_image_save_dir = os.path.join( 
                                            settings["AUGMENTED_DIR_GBL"], 
                                            str( data_type ).lower( )
                                     )
    os.makedirs( aug_image_save_dir, exist_ok=True )
    augmented_img["augmented_image"].save( 
                                            os.path.join( 
                                                            aug_image_save_dir,
                                                            "{}.{}".format( aug_image_file_name, settings["AUGMENTED_IMAGE_EXTENSION_DA"] ) 
                                                        ) 
                                         )
    if not augmented_img["masks"] is None:
        # storeVariableWP ( 
        #                     variable = augmented_img["masks"], 
        #                     filename = "{}.mask.pkl".format( aug_image_file_name ), 
        #                     dir_path = aug_image_save_dir, 
        #                     database_type = None 
        #                 )
        if not settings["AUGMENTED_IMAGE_EXTENSION_DA"][0] == ".":
            format_checker = "." + settings["AUGMENTED_IMAGE_EXTENSION_DA"]
        else:
            format_checker = settings["AUGMENTED_IMAGE_EXTENSION_DA"]
            
        if format_checker in Image.EXTENSION.keys():
            augmented_img["masks"].save( 
                                            fp = os.path.join( 
                                                                    aug_image_save_dir,
                                                                    "{}.mask".format( aug_image_file_name ) 
                                                              ),
                                            format = Image.EXTENSION[format_checker]
                                        )
    make_and_save_xml( 
        aug_image_file_name = aug_image_file_name + "." + settings["AUGMENTED_IMAGE_EXTENSION_DA"],
        aug_image_xml_file_path = aug_image_xml_file_path + ".xml", 
        aug_image_width = aug_image_width, 
        aug_image_height = aug_image_height, 
        aug_image_depth = aug_image_depth, 
        aug_image_bboxes = aug_image_bboxes, 
        aug_image_labels = aug_image_labels, 
        database = "Tahasanul_Custom_DataSet"
        )

#%% get_bbox

def get_bbox( image, objectness_score, settings ):
    try:
        
        rows = torch.cat( [( x <= objectness_score ).nonzero( ) for x in image if ( x <= objectness_score ).nonzero( ).nelement( ) != 0] )
        cols = torch.cat( [( x <= objectness_score ).nonzero( ) for x in image.permute( 1,0 ) if ( x <= objectness_score ).nonzero( ).nelement( ) != 0] )
        
        x_min = rows.min( )
        x_max = rows.max( )
        y_min = cols.min( )
        y_max = cols.max( )
        
        if ( x_min >= x_max ) or ( y_min >= y_max ):
            raise
        
        status = True   
        
    except:
        
        x_min = torch.tensor( 0 )
        x_max = torch.tensor( image.shape[1] )
        y_min = torch.tensor( 0 )
        y_max = torch.tensor( image.shape[0] )
        
        status = False
    
        
    if settings["MASKED_CNN_GBL"] is True:
        try:
            mask = torch.where( 
                                    image <= objectness_score, 
                                    1.0, 
                                    0.0
                                )
            
            status_mask = True
            
        except:
            mask = None
            
            status_mask = False
            
        if status is True and status_mask is True:
            status = True
        else:
            status = False
    else:
        mask = None
        
        
    if settings["BBOX_TYPE_TENSOR_DA"] is True:
        return [x_min, y_min, x_max, y_max], status, mask
    else:
        return [int( x_min.item( ) ), int( y_min.item( ) ), int( x_max.item( ) ), int( y_max.item( ) )], status, mask
    
#%% statistic_storer_augmentation

def statistic_storer ( data, data_types, settings, timestampStr=datetime.now( ).strftime( "%d-%b-%Y_%H-%M-%S" ), epoch=None ):
    if epoch is None:
        epoch = ""
    specials = ["completed_list", "all_labels"]    
    new_folder_dir = os.path.join( settings["STATISTICAL_DIR_GBL"], timestampStr )
    os.makedirs( new_folder_dir, exist_ok=True )
    
    for data_type in data_types:
        data_type = data_type.lower( )
        
        for key in data.keys( ):
            
            filename = key + "_" + data_type + ".pkl"
            data[key][data_type]
            storeVariableWP( data[key][data_type], filename, new_folder_dir )
            
            if key in specials:
                storeVariableWP( data[key][data_type], filename, settings["STATISTICAL_DIR_GBL"] )
            
            if "times" in key:                
                fig = plt.figure( )
                x = [ i for i in range( 1, len( data[key][data_type] )+1 ) ]
                y = data[key][data_type]
                plt.plot( x, y, color="green", linestyle="dashed", linewidth = 1, marker="o", markerfacecolor="blue", markersize=5 )
                # xticks = [0] + x
                # plt.xticks( x,xticks )
                plt.xlabel( "{} Count".format( str( str( key ).split( "_" )[0] ).capitalize( ) ) )
                plt.ylabel( "{} Time in Seconds".format( str( str( key ).split( "_" )[0] ).capitalize( ) ) )
                plt.title( "{} Time Elapsed Satistics".format( data_type.capitalize( ) ) )
                filename = os.path.join( new_folder_dir, "{}_{}_epoch_{}.jpg".format( str( data_type ).lower( ), key, epoch ) )
                plt.savefig( filename )    
                plt.show( )                
                fig.clf( )
                plt.clf( )
                plt.close( fig )
                
            if "loss" in key:                
                fig = plt.figure( )
                x = [ i for i in range( 1, len( data[key][data_type] )+1 ) ]
                y = data[key][data_type]
                plt.plot( x, y, color="green", linestyle="dashed", linewidth = 1, marker="o", markerfacecolor="blue", markersize=5 )
                # xticks = [0] + x
                # plt.xticks( x,xticks )
                plt.xlabel( "{} Iterations".format( str( str( key ).split( "_" )[0] ).capitalize( ) ) )
                plt.ylabel( "{} Loss".format( str( str( key ).split( "_" )[0] ).capitalize( ) ) )
                plt.title( "{} Data Loss Satistics".format( data_type.capitalize( ) ) )
                filename = os.path.join( new_folder_dir, "{}_{}_epoch_loss_epoch_{}.jpg".format( str( data_type ).lower( ), str( key ).split( "_" )[0].lower( ), epoch ) )
                plt.savefig( filename )    
                plt.show( )                
                fig.clf( )
                plt.clf( )
                plt.close( fig )
                
            if "map" in key:                
                fig = plt.figure( )
                x = [ i for i in range( 1, len( data[key][data_type] )+1 ) ]
                y = data[key][data_type]
                plt.plot( x, y, color="green", linestyle="dashed", linewidth = 1, marker="o", markerfacecolor="blue", markersize=5 )
                # xticks = [0] + x
                # plt.xticks( x,xticks )
                plt.xlabel( "{} Iterations".format( str( str( key ).split( "_" )[0] ).capitalize( ) ) )
                plt.ylabel( "{} Global MAP Score".format( str( str( key ).split( "_" )[0] ).capitalize( ) ) )
                plt.title( "{} Global MAP Satistics".format( data_type.capitalize( ) ) )
                filename = os.path.join( new_folder_dir, "{}_{}_global_map_score_epoch_{}.jpg".format( str( data_type ).lower( ), str( key ).split( "_" )[0].lower( ), epoch ) )
                plt.savefig( filename )    
                plt.show( )                
                fig.clf( )
                plt.clf( )
                plt.close( fig )
                       
    
    filename = "settings" + ".yaml"
    filename_path = os.path.join( new_folder_dir, filename )
    with open( filename_path, "w+" ) as yaml_file:
        yaml.dump( settings, yaml_file, allow_unicode=True, default_flow_style=False )

#%% calculate_iou

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
    
    for i in range( len( roc_groups ) ):
        for key in list(roc_groups[i].keys()):
            if len(roc_groups[i][key]) < 2:
                roc_groups[i].pop(key)
                
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
            roc_preds[image_number][key] = roc_preds[image_number][key].to( settings["DEVICE_GBL"] )
            
    return roc_preds