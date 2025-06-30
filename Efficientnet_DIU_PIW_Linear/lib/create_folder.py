# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 10:50:45 2023

@author: Shaik
"""
import os
from datetime import datetime


def create_folder(filename):
    
    
    dirr = os.path.join(os.getcwd(), filename)
    
    # if os.path.isdir(filename)== True:
    #     mydir = os.path.join(dirr,datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    #     os.makedirs(mydir)
    # else:
    #     os.makedirs(dirr)
    #     mydir = os.path.join(dirr,datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    #     os.makedirs(mydir)
        
    mydir = dirr
        
       
    path_list = []
    
    my_list = ['training_plots', 'validation_plots', 'model_info', 'loss_param', 'K_comp_plot', 'test_results']
    
    for i in my_list:
        path = os.path.join(mydir, i)
        os.makedirs(path)
        path_list.append(path)
        
        
    return path_list
            

        
        
        

    
    
    