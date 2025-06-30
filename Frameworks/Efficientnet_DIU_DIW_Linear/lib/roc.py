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
        k = float()
        
        for col in m12.columns:
            selected_column = m12[col]
            for row in selected_column.index:
                if col == row:
                    roc_mat[col] += selected_column[row]
                else:
                    k += selected_column[row]
        
        for key in roc_mat.keys():
            roc_mat[key] = roc_mat[key] / (1-k)
        
        # for col in m12.columns:
        #     df = m12[col]
            
        #     if isinstance( col, str ):
        #         matcher_length = len( set( (col,) ) )
        #         for idx in df.index:
        #             if isinstance( idx, str ):
        #                 if len( set( (col,) ) & set( (idx,) ) ) == matcher_length:
        #                     roc_mat[col] += df[idx]
        #                     k += df[idx]
        #                     # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
        #             else:
        #                 if len( set( (col,) ) & set( idx ) ) == matcher_length:
        #                     roc_mat[col] += df[idx]
        #                     k += df[idx]
        #                     # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
        #     else:
        #         matcher_length = len( set( col ) )
        #         for idx in df.index:
        #             if isinstance( idx, str ):
        #                 if len( set( col ) & set( (idx,) ) ) == matcher_length:
        #                     roc_mat[col] += df[idx]
        #                     k += df[idx]
        #                     # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
        #             else:
        #                 if len( set( col ) & set( idx ) ) == matcher_length:
        #                     roc_mat[col] += df[idx]
        #                     k += df[idx]
        #                     # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
        
        # for idx in m12.index:
        #     df = m12.loc[[idx]]
            
        #     if isinstance( idx, str ):
        #         matcher_length = len( set( (idx,) ) )
                
        #         for col in df.columns:
        #             if isinstance( col, str ):
        #                 if len( set( (idx,) ) & set( (col,) ) ) == matcher_length:
        #                     roc_mat[idx] += df[col][idx]
        #                     k += df[col][idx]
        #                     # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
        #             else:
        #                 if len( set( (idx,) ) & set( col ) ) == matcher_length:
        #                     roc_mat[idx] += df[col][idx]
        #                     k += df[col][idx]
        #                     # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
        #     else:
        #         matcher_length = len( set( idx ) )
                
        #         for col in df.columns:
        #             if isinstance( col, str ):
        #                 pass
        #             else:
        #                 if len( set( idx ) & set( col ) ) == matcher_length:
        #                     roc_mat[idx] += df[col][idx]
        #                     k += df[col][idx]
        #                     # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
        
        # for col in m12.columns:
            
        #     k -= m12[col][col]

        # K = 1 - k
        # K = self.__zeroing__( K )            
        
        # for col in m12.columns:
        #     # print (roc_mat[col], m12[col][col])
        #     roc_mat[col] -= m12[col][col]
        #     # print (roc_mat[col])
        #     roc_mat[col] = self.__zeroing__( roc_mat[col] ) 
        #     # print (roc_mat[col])
        #     roc_mat[col] = roc_mat[col] / k
        #     # print (roc_mat[col])
        #     # print ( "REMOVED: m1 -> {}, m2 -> {}".format( col, col ) )
        
        return k, roc_mat        
    
class __ygr__ ():
    def __init__(self):        
        super().__init__()
        
    def __ygr_roc__ (self, m12: pd.DataFrame):
        roc_mat = { col:float() for col in m12.columns }
        k = float()
        
        for col in m12.columns:
            selected_column = m12[col]
            for row in selected_column.index:
                if col == row:
                    roc_mat[col] += selected_column[row]
                else:
                    k += selected_column[row]
        
        #roc_mat["Conflicts"] += k
        
        # for col in m12.columns:
        #     df = m12[col]
            
        #     if isinstance( col, str ):
        #         matcher_length = len( set( (col,) ) )
        #         for idx in df.index:
        #             if isinstance( idx, str ):
        #                 if len( set( (col,) ) & set( (idx,) ) ) == matcher_length:
        #                     roc_mat[col] += df[idx]
        #                     k += df[idx]
        #                     # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
        #             else:
        #                 if len( set( (col,) ) & set( idx ) ) == matcher_length:
        #                     roc_mat[col] += df[idx]
        #                     k += df[idx]
        #                     # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
        #     else:
        #         matcher_length = len( set( col ) )
        #         for idx in df.index:
        #             if isinstance( idx, str ):
        #                 if len( set( col ) & set( (idx,) ) ) == matcher_length:
        #                     roc_mat[col] += df[idx]
        #                     k += df[idx]
        #                     # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
        #             else:
        #                 if len( set( col ) & set( idx ) ) == matcher_length:
        #                     roc_mat[col] += df[idx]
        #                     k += df[idx]
        #                     # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
        
        # for idx in m12.index:
        #     df = m12.loc[[idx]]
            
        #     if isinstance( idx, str ):
        #         matcher_length = len( set( (idx,) ) )
                
        #         for col in df.columns:
        #             if isinstance( col, str ):
        #                 if len( set( (idx,) ) & set( (col,) ) ) == matcher_length:
        #                     roc_mat[idx] += df[col][idx]
        #                     k += df[col][idx]
        #                     # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
        #             else:
        #                 if len( set( (idx,) ) & set( col ) ) == matcher_length:
        #                     roc_mat[idx] += df[col][idx]
        #                     k += df[col][idx]
        #                     # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
        #     else:
        #         matcher_length = len( set( idx ) )
                
        #         for col in df.columns:
        #             if isinstance( col, str ):
        #                 pass
        #             else:
        #                 if len( set( idx ) & set( col ) ) == matcher_length:
        #                     roc_mat[idx] += df[col][idx]
        #                     k += df[col][idx]
        #                     # print ( "m1 -> {}, m2 -> {}".format( col, idx ) )
        
        # for col in m12.columns:
            
        #     k -= m12[col][col]
        
        # K = 1 - k
        # K = self.__zeroing__( K )
            
        # for col in m12.columns:
            
        #     roc_mat[col] -= m12[col][col]
        #     roc_mat[col] = self.__zeroing__( roc_mat[col] ) 
        #     roc_mat[col] = round ( roc_mat[col], self.deci )      
        #     # print ( "REMOVED: m1 -> {}, m2 -> {}".format( col, col ) )
            
        # total = sum( x for x in roc_mat.values() )
        # roc_mat["Unknown"] = round ( 1 - total, self.deci )
        
        return k, roc_mat


class roc (
            __ds__,
            __ygr__,
            ):
    
    def __init__(self, deci=5):
        self.deci = deci        
        super().__init__()
        
    def alter_deci(self, deci):
        if isinstance(deci, int):
            self.deci = deci
        
    def perform_ds(self, **kargs):
        
        evidence = None
        K = None
        
        for key in kargs.keys():
            mass = copy.deepcopy( kargs[key] )
            if isinstance (mass, dict):
                if evidence is None:
                    evidence = mass
                else:
                    K, evidence = self.__ds_roc__ ( 
                                                         self.__key_balancer__( 
                                                                                     copy.deepcopy( evidence ), 
                                                                                     copy.deepcopy( mass )
                                                                                 ) 
                                                   )
                        
        #print (f"K -> {K}")
        return K, evidence
    
    def perform_ygr(self, **kargs):
        
        evidence = None
        K = None
        
        for key in kargs.keys():
            mass = copy.deepcopy( kargs[key] )
            if isinstance (mass, dict):
                if not "Conflicts" in mass:
                    mass["Conflicts"] = 0.0                
                if evidence is None:
                    evidence = mass
                else:
                    K, evidence = self.__ygr_roc__ ( 
                                                        self.__key_balancer__( 
                                                                                    copy.deepcopy( evidence ), 
                                                                                    copy.deepcopy( mass )
                                                                                ) 
                                                    )
                        
        return K, evidence
        
    def scorecard_merger(self, score_card: dict, score: float, weight: float, m1: dict):
        
        if score > 100:
            score = 100
        elif score < 0:
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
        #print (f"Grade -> {grade_point}")
        return grade_point
    
    def factor_generator_v2(self, score_card: dict, score: float):
            
        w_max = score_card["factor_max"]
        w_min = score_card["factor_min"]
        k_max = 1
        k_min = 0
        k = score
        grade_point = ( ( ( w_max - w_min ) / ( k_max - k_min ) ) * ( k - k_min ) ) + w_min
        #print (f"Grade -> {grade_point}")
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
            
        for m1_key in list( m1.keys() ):
            m1[m1_key] = round( self.__zeroing__ ( m1[m1_key] ), self.deci )
            
        for m2_key in list( m2.keys() ):
            m2[m2_key] = round( self.__zeroing__ ( m2[m2_key] ), self.deci )

            
        if not sum(m1.values()) == 1:
            m1 = {k: v / total for total in (sum(m1.values()),) for k, v in m1.items()}
            
        if not sum(m2.values()) == 1:
            m2 = {k: v / total for total in (sum(m2.values()),) for k, v in m2.items()}
        
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
    
    def __zeroing__ (self, x):
        
        if x == 1:
            if self.deci == 1:
                x = 1 - float(f'{0.0:.{self.deci - 1}f}' + '.1')
            else:
                x = 1 - float(f'{0.0:.{self.deci - 1}f}' + '1')
        elif x == 0:
            x = float(f'{0.0:.{self.deci - 1}f}' + '1')
        
        return x
    
    def __normalizer_stable__(self, x):
        
        return ( np.array(x) / np.array(x).sum() )
    
    def __softmax_stable__(self, x):
        
        return ( np.exp( x - np.max( x ) ) / np.exp( x - np.max( x ) ).sum( ) )

    def __all_subsets__(self, ss):
        
        return list ( chain( *map( lambda x: combinations( ss, x ), range( 0, len( ss ) + 1  ) ) ) )[1:]
    
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
    
    
    m1 = {
            "A"         :       0,
            "B"         :       0.99,
            "C"         :       0.01,
            
         }    
    m2 = {
            "A"         :       0.99,
            "B"         :       0.0,
            "C"         :       0.01,
         }
    # m3 = {
    #         "A"         :       0.1,
    #         "B"         :       0.8,
    #         "C"         :       0.1,
    #       }    
    # m4 = {
    #         "A"         :       0.3,
    #         "B"         :       0.5,
    #         "C"         :       0.2,
    #       }
    # m5 = {
    #         "A"         :       0.07,
    #         "B"         :       0.9,
    #         "C"         :       0.03,
    #       }
    m3 = {}
    m4 = {}
    m5 = {}
    a = roc(5)
    
    K_ds, roc_ds = a.perform_ds(
                                    m1 =  m1, 
                                    m2 =  m2, 
                                    m3 =  m3, 
                                    m4 =  m4, 
                                    m5 =  m5, 
                                )
    K_ygr, roc_ygr = a.perform_ygr(
                                    m1 =  m1, 
                                    m2 =  m2, 
                                    m3 =  m3, 
                                    m4 =  m4, 
                                    m5 =  m5, 
                                )
    
    # K_ds, roc_ds = a.perform_ds(
    #                                 m1 = a.scorecard_merger( score_card, 100, m1_weight, m1 ), 
    #                                 m2 = a.scorecard_merger( score_card, 100, m2_weight, m2 ), 
    #                                 # m3 = a.scorecard_merger( score_card, 80, m3_weight, m3 ),
    #                             )
    
    # K_ygr, roc_ygr = a.perform_ygr(
    #                                     m1 = a.scorecard_merger( score_card, 100, m1_weight, m1 ),
    #                                     m2 = a.scorecard_merger( score_card, 100, m2_weight, m2 ),
    #                                 )
    
    # K_smets, roc_smets = a.perform_smets(
    #                                         m1 = a.scorecard_merger( score_card, 2, m1_weight, m1 ), 
    #                                         m2 = a.scorecard_merger( score_card, 55, m2_weight, m2 ),
    #                                       )
