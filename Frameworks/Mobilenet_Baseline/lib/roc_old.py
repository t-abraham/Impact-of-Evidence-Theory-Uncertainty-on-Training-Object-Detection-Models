# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 12:43:29 2023

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

import copy
import numpy as np
import pandas as pd
from itertools import chain, combinations


class __ds__ ():
    def __init__(self):
        super().__init__()
        
    def __ds_roc__ (self, m12: pd.DataFrame):
        roc_mat = { col:float() for col in m12.columns }
        K = float()
        
        for col in m12.columns:
            df = m12[col]
            
            if isinstance( col, str ):
                matcher_length = len( set( (col,) ) )
                for idx in df.index:
                    if isinstance( idx, str ):
                        if len( set( (col,) ) & set( (idx,) ) ) == matcher_length:
                            roc_mat[col] += df[idx]
                            K += df[idx]
                            # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
                    else:
                        if len( set( (col,) ) & set( idx ) ) == matcher_length:
                            roc_mat[col] += df[idx]
                            K += df[idx]
                            # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
            else:
                matcher_length = len( set( col ) )
                for idx in df.index:
                    if isinstance( idx, str ):
                        if len( set( col ) & set( (idx,) ) ) == matcher_length:
                            roc_mat[col] += df[idx]
                            K += df[idx]
                            # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
                    else:
                        if len( set( col ) & set( idx ) ) == matcher_length:
                            roc_mat[col] += df[idx]
                            K += df[idx]
                            # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
        
        for idx in m12.index:
            df = m12.loc[[idx]]
            
            if isinstance( idx, str ):
                matcher_length = len( set( (idx,) ) )
                
                for col in df.columns:
                    if isinstance( col, str ):
                        if len( set( (idx,) ) & set( (col,) ) ) == matcher_length:
                            roc_mat[idx] += df[col][idx]
                            K += df[col][idx]
                            # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
                    else:
                        if len( set( (idx,) ) & set( col ) ) == matcher_length:
                            roc_mat[idx] += df[col][idx]
                            K += df[col][idx]
                            # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
            else:
                matcher_length = len( set( idx ) )
                
                for col in df.columns:
                    if isinstance( col, str ):
                        pass
                    else:
                        if len( set( idx ) & set( col ) ) == matcher_length:
                            roc_mat[idx] += df[col][idx]
                            K += df[col][idx]
                            # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
        
        for col in m12.columns:
            
            K -= m12[col][col]

        K = 1 - K            
        
        for col in m12.columns:
            
            roc_mat[col] -= m12[col][col]
            roc_mat[col] = round ( roc_mat[col] / (1 - K), self.deci )      
            # print ( "REMOVED: m1 -> {}, m2 -> {}".format( col, col ) )
        
        return K, roc_mat        
    
class __ygr__ ():
    def __init__(self):        
        super().__init__()
        
    def __ygr_roc__ (self, m12: pd.DataFrame):
        roc_mat = { col:float() for col in m12.columns }
        K = float()
        
        for col in m12.columns:
            df = m12[col]
            
            if isinstance( col, str ):
                matcher_length = len( set( (col,) ) )
                for idx in df.index:
                    if isinstance( idx, str ):
                        if len( set( (col,) ) & set( (idx,) ) ) == matcher_length:
                            roc_mat[col] += df[idx]
                            K += df[idx]
                            # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
                    else:
                        if len( set( (col,) ) & set( idx ) ) == matcher_length:
                            roc_mat[col] += df[idx]
                            K += df[idx]
                            # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
            else:
                matcher_length = len( set( col ) )
                for idx in df.index:
                    if isinstance( idx, str ):
                        if len( set( col ) & set( (idx,) ) ) == matcher_length:
                            roc_mat[col] += df[idx]
                            K += df[idx]
                            # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
                    else:
                        if len( set( col ) & set( idx ) ) == matcher_length:
                            roc_mat[col] += df[idx]
                            K += df[idx]
                            # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
        
        for idx in m12.index:
            df = m12.loc[[idx]]
            
            if isinstance( idx, str ):
                matcher_length = len( set( (idx,) ) )
                
                for col in df.columns:
                    if isinstance( col, str ):
                        if len( set( (idx,) ) & set( (col,) ) ) == matcher_length:
                            roc_mat[idx] += df[col][idx]
                            K += df[col][idx]
                            # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
                    else:
                        if len( set( (idx,) ) & set( col ) ) == matcher_length:
                            roc_mat[idx] += df[col][idx]
                            K += df[col][idx]
                            # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
            else:
                matcher_length = len( set( idx ) )
                
                for col in df.columns:
                    if isinstance( col, str ):
                        pass
                    else:
                        if len( set( idx ) & set( col ) ) == matcher_length:
                            roc_mat[idx] += df[col][idx]
                            K += df[col][idx]
                            # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
        
        for col in m12.columns:
            
            K -= m12[col][col]
        
        K = 1 - K
            
        for col in m12.columns:
            
            roc_mat[col] -= m12[col][col]
            roc_mat[col] = round ( roc_mat[col], self.deci )      
            # print ( "REMOVED: m1 -> {}, m2 -> {}".format( col, col ) )
            
        total = sum( x for x in roc_mat.values() )
        roc_mat["Unknown"] = round ( 1 - total, self.deci )
        
        return K, roc_mat
    
class __smets__ ():
    def __init__(self):
        super().__init__()
        
    def __smets_roc__ (self, m12: pd.DataFrame):
        roc_mat = { col:float() for col in m12.columns }
        K = float()
        
        for col in m12.columns:
            df = m12[col]
            
            if isinstance( col, str ):
                matcher_length = len( set( (col,) ) )
                for idx in df.index:
                    if isinstance( idx, str ):
                        if len( set( (col,) ) & set( (idx,) ) ) == matcher_length:
                            roc_mat[col] += df[idx]
                            K += df[idx]
                            # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
                    else:
                        if len( set( (col,) ) & set( idx ) ) == matcher_length:
                            roc_mat[col] += df[idx]
                            K += df[idx]
                            # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
            else:
                matcher_length = len( set( col ) )
                for idx in df.index:
                    if isinstance( idx, str ):
                        if len( set( col ) & set( (idx,) ) ) == matcher_length:
                            roc_mat[col] += df[idx]
                            K += df[idx]
                            # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
                    else:
                        if len( set( col ) & set( idx ) ) == matcher_length:
                            roc_mat[col] += df[idx]
                            K += df[idx]
                            # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
        
        for idx in m12.index:
            df = m12.loc[[idx]]
            
            if isinstance( idx, str ):
                matcher_length = len( set( (idx,) ) )
                
                for col in df.columns:
                    if isinstance( col, str ):
                        if len( set( (idx,) ) & set( (col,) ) ) == matcher_length:
                            roc_mat[idx] += df[col][idx]
                            K += df[col][idx]
                            # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
                    else:
                        if len( set( (idx,) ) & set( col ) ) == matcher_length:
                            roc_mat[idx] += df[col][idx]
                            K += df[col][idx]
                            # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
            else:
                matcher_length = len( set( idx ) )
                
                for col in df.columns:
                    if isinstance( col, str ):
                        pass
                    else:
                        if len( set( idx ) & set( col ) ) == matcher_length:
                            roc_mat[idx] += df[col][idx]
                            K += df[col][idx]
                            # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
        
        for col in m12.columns:
            
            K -= m12[col][col]

        K = 1 - K        
    
        for col in m12.columns:
            
            roc_mat[col] -= m12[col][col]
            roc_mat[col] = round ( roc_mat[col], self.deci )      
            # print ( "REMOVED: m1 -> {}, m2 -> {}".format( col, col ) )
            
        total = sum( x for x in roc_mat.values() )
        roc_mat["Conflicts"] = K
        roc_mat["Unknown"] = 1 - total - K
        
        return K, roc_mat


class roc (
            __ds__,
            __ygr__,
            __smets__,
            ):
    
    def __init__(self, deci=5, mat_round=False):
        self.deci = deci
        self.mat_round = mat_round
        
        super().__init__()
        
    def alter_deci(self, deci):
        self.deci = deci
        
    def alter_mat_round(self, mat_round):
        self.mat_round = mat_round
        
    def perform_ds(self, **kargs):
        
        evidence = None
        K = None
        
        for key in kargs.keys():
            if isinstance (kargs[key], dict):
                if evidence is None:
                    evidence = kargs[key]
                else:
                    K, evidence = self.__ds_roc__ ( 
                                                         self.__key_balancer__( 
                                                                                     copy.deepcopy( evidence ), 
                                                                                     copy.deepcopy( kargs[key] )
                                                                                 ) 
                                                   )
                        
        return K, evidence
    
    def perform_ygr(self, **kargs):
        
        evidence = None
        K = None
        
        for key in kargs.keys():
            if isinstance (kargs[key], dict):
                if evidence is None:
                    evidence = kargs[key]
                else:
                    K, evidence = self.__ygr_roc__ ( 
                                                        self.__key_balancer__( 
                                                                                    copy.deepcopy( evidence ), 
                                                                                    copy.deepcopy( kargs[key] )
                                                                                ) 
                                                    )
                        
        return K, evidence
        
    def perform_smets(self, **kargs):
        
        evidence = None
        K = None
        
        for key in kargs.keys():
            if isinstance (kargs[key], dict):
                if evidence is None:
                    evidence = kargs[key]
                else:
                    K, evidence = self.__smets_roc__ ( 
                                                          self.__key_balancer__( 
                                                                                      copy.deepcopy( evidence ), 
                                                                                      copy.deepcopy( kargs[key] )
                                                                                  ) 
                                                      )
        return K, evidence
        
    def scorecard_merger(self, score_card: dict, score: float, weight: float, m1: dict):
        
        if score >= 100:
            score = 100
        elif score <= 0:
            score = 0
            
        grade_point = 1
        
        for current_bands in score_card.keys():
            both_bands = str(current_bands).split("-")
            lower_band = both_bands[0]
            upper_band = both_bands[1]
            if float( upper_band ) == 100:
                if self.deci == 1:
                    upper_band = 100 + float(f'{0.0:.{self.deci - 1}f}' + '.1')
                else:
                    upper_band = 100 + float(f'{0.0:.{self.deci - 1}f}' + '1')
                    
            if ( float( lower_band ) <= float( score ) < float( upper_band ) ) is True:
                grade_point = score_card[current_bands]
        
        return { key : value*grade_point*weight for key, value in m1.items() }
     
    def factor_generator(self, score_card: dict, score: float):
        
        if score >= 100:
            score = 100
        elif score <= 0:
            score = 0
            
        grade_point = 1
        
        for current_bands in score_card.keys():
            both_bands = str(current_bands).split("-")
            lower_band = both_bands[0]
            upper_band = both_bands[1]
                                
            if ( float( lower_band ) <= float( score ) < float( upper_band ) ) is True:
                grade_point = score_card[current_bands]
        
        return grade_point
        
        
    def __key_balancer__(self, m1: dict, m2: dict):
        
        # m1_keys = list( m1.keys() )
        # m2_keys = list( m2.keys() )
                
        for m1_key in list( m1.keys() ):
            if not m1_key in m2:
                m2[m1_key] = 0
                
        for m2_key in list( m2.keys() ):
            if not m2_key in m1:
                m1[m2_key] = 0
        
        all_keys = []
        all_keys += list( m1.keys() )
        all_keys += list( m2.keys() )
        
        all_keys = list( dict.fromkeys( all_keys ) )
        
        all_keys = [key for key in all_keys if isinstance( key, str )] 
        
        all_keys = self.__all_subsets__( all_keys )
        
        for key in all_keys:
            if len(key) == 1:
                key = key[0]
                
            if not key in m1:
                if isinstance(key, str):
                    m1[key] = 0
                else:
                    for i in key:
                        if not key in m1:
                            m1[key] = m1[i]
                        else:
                            m1[key] = m1[key] * m1[i]
                            
            if not key in m2:
                if isinstance(key, str):
                    m2[key] = 0
                else:
                    for i in key:
                        if not key in m2:
                            m2[key] = m2[i]
                        else:
                            m2[key] = m2[key] * m2[i]
        
        m1 = dict( zip( list( m1.keys() ), self.__softmax_stable__( list( m1.values() ) ) ) )
        m2 = dict( zip( list( m2.keys() ), self.__softmax_stable__( list( m2.values() ) ) ) )
        
        if self.mat_round is True:
            
            for m1_key in list( m1.keys() ):
                m1[m1_key] = self.__zeroing__ ( round ( m1[m1_key], self.deci ) )
                
            for m2_key in list( m2.keys() ):
                m2[m2_key] = self.__zeroing__ ( round ( m2[m2_key], self.deci ) )
        
        m12 = pd.DataFrame(
                                m2.values(), 
                                columns=["mass_values"], 
                                index=m2.keys()
                           ).dot(
                                   pd.DataFrame(    
                                                   m1.values(),
                                                   columns=["mass_values"],
                                                   index=m1.keys()
                                                ).transpose()
                                 )
        
        return m12   
    
    
    def __zeroing__ (self, K):
        
        if K == 1:
            if self.deci == 1:
                K = 1 - float(f'{0.0:.{self.deci - 1}f}' + '.1')
            else:
                K = 1 - float(f'{0.0:.{self.deci - 1}f}' + '1')
        elif K == 0:
            K = float(f'{0.0:.{self.deci - 1}f}' + '1')
        
        return K
    
    def __softmax_stable__(self, x):
        
        return ( np.exp( x - np.max( x ) ) / np.exp( x - np.max( x ) ).sum( ) )

    def __all_subsets__(self, ss):
        
        all_subsets = list ( chain( *map( lambda x: combinations( ss, x ), range( 0, len( ss ) + 1  ) ) ) )
        
        return all_subsets[1:]
    
#%% Standalone Run

if __name__ == "__main__":
    
    score_card = {
                    "80-100"    :       1,
                    "60-80"     :       0.90,
                    "40-60"     :       0.80,
                    "20-40"     :       0.70,
                    "0-20"      :       0.60,
                    
                    }
    
    m1_weight = 1
    m2_weight = 1
    m3_weight = 1
    
    
    # pred1 = {
    #             "label" : 2,
    #             "score" : 0.935,
    #             "bbox" : [212,548,6363,5223]
    #         }
    
    # pred2 = {
    #             "label" : 3,
    #             "score" : 0.435,
    #             "bbox" : [442,228,773,5003]
    #         }
    # #classification
    
    # f1scorem1 = 0.80
    # m1 = {
    #         "Walle" : 0.935
    #     }
    # f1scorem2 = 0.60
    # m2 = {
    #         "Donald" : 0.435
    #     }
    
    # #regtression
    
    # iouscorem1 = 0.72
    # m1 = {
    #         "[212,548,6363,5223]" : 0.935
    #     }
    
    # iouscorem2 = 0.98
    # m2 = {
    #         "[442,228,773,5003]" : 0.435
    #     }
    
    
    m1 =   {'angel_lucy': 0.69704544, 'grootz': 0.23224842}
          
    
    m2 =      {'angel_luczzzy': 0.9862074851989746}
          
    
    m3 = {
            "angel_lucy"       :       1,
           
          }
    
    a = roc(5)
    # m1 = {
    #         "Brain Tumor"       :       0.9929248
            
    #      }
    # m2 = {
    #         "Brain Tumor"       :       0.99986374
           
    #      }
    
   
    K_ds, roc_ds = a.perform_ds(
                                    m1 = a.scorecard_merger( score_card, 100, m1_weight, m1 ), 
                                    m2 = a.scorecard_merger( score_card, 100, m2_weight, m2 ), 
                                    # m3 = a.scorecard_merger( score_card, 100, m3_weight, m3 ),
                                )
    mul_factor = a.factor_generator(score_card, (max(roc_ds.values())) *100)
    
    K_ygr, roc_ygr = a.perform_ygr(
                                        m1 = a.scorecard_merger( score_card, 100, m1_weight, m1 ),
                                        m2 = a.scorecard_merger( score_card, 100, m2_weight, m2 ),
                                    )
    
    K_smets, roc_smets = a.perform_smets(
                                            m1 = a.scorecard_merger( score_card, 2, m1_weight, m1 ), 
                                            m2 = a.scorecard_merger( score_card, 55, m2_weight, m2 ),
                                          )