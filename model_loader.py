from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import numpy as np
import time
import pickle
from statistics import mean

import model_builder

class model_loader:
    '''
    Mainly Go and look at the folder and file list

    '''
    def __init__(self,path,Desired_param=False):
        self.path=path
        self.file_directory=[]
        self.Param=[]
        self.Selected_Param=[]
        self.Param_pd=[]
        self.Par=[]
        self.Desired_param=Desired_param # wanted param such as pressue 1bar and offset 10mm
        self.Desired_file_directory=[] # Directory list based on wanted param
       
        self.total_df=[]
        
     ## 1. Searching the folder with different parameter    
    def file_searching(self,path):
        '''
        Input : path
        Output : split the directory name with '_' and make a directionary for this
        '''
        ## 1. Make file list
        file_lists=os.listdir(path)
        self.file_directory= file_lists

        ## 2. Parameter list
        self.Param=[]
        for file in self.file_directory:
            self.Param.append(file.split('_'))
        self.Param_pd=pd.DataFrame(self.Param)

        k=0
        print("model name format ex : ", self.file_directory[0])
        for i in range(len(self.Param[0])):
            '''if len(self.Param_pd[i].unique())==1:
                print("**Variable**",self.Param_pd[i].unique(),"\n")
                V=self.Param_pd[i].unique()
            else:
                k=k+1
                print(k,"Parameters",self.Param_pd[i].unique(),"\n")
                self.Par.append(self.Param_pd[i].unique())'''
            k=k+1
            print(k,"Parameters",self.Param_pd[i].unique(),"\n")
            self.Par.append(self.Param_pd[i].unique())
    ## 2. Make list of folder with desired parameter and list of parameter        
    def desired_file_path(self,Desired_param):
        if len(self.Par) != len(Desired_param):
            print(f'Your input length ({len(self.Desired_param)}) does not match to Param length ({len(self.Par)})')
        else:
            #print('Go')
            for i in range(len(self.Par)):
                #print(i)
                #print((A.Par[i][input_param[i][:]]))
                if i ==0:
                    filt=self.Param_pd[i].isin(self.Par[i][self.Desired_param[i][:]])
                    #print(filt)
                else:
                    filt_temp=self.Param_pd[i].isin(self.Par[i][self.Desired_param[i][:]])
                    #print(filt_temp)
                    filt=filt_temp&filt
                    #print('print filtered')
                    
                    self.Selected_Param=[self.Param[i] for i in list(self.Param_pd.index[filt])]
                    self.Desired_file_directory=[self.path+'/'+self.file_directory[i] for i in list(self.Param_pd.index[filt])]
    
        return self.Desired_file_directory
    
    ## 3. Make list of desired file's properties
    


