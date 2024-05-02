from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import numpy as np
import time
import pickle
from statistics import mean
from ros_data_reader import Vector_set
class data_loader:
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
        self.contents=[] # Force or TCP , ANalogue => file set
        self.total_df=[]
        
     ## 1. Searching the folder with different parameter    
    def file_searching(self,path):
        '''
        Input : path
        Output : split the directory name with '_' and make a directionary for this
        '''
        ## 1. Make file list
        file_lists=os.listdir(path)
        for file_list in file_lists:
            #print(file_list)
            # Only extract folder
            if os.path.isdir(os.path.join(path,file_list)): 
                self.file_directory.append(file_list)

        ## 2. Parameter list
        self.Param=[]
        for file in self.file_directory:
            self.Param.append(file.split('_'))
        self.Param_pd=pd.DataFrame(self.Param)

        k=0
        print("1st Folder name : ", self.file_directory[0])
        self.contents=os.listdir(os.path.join(self.path,file_list))
        print("Contents in the folder :", self.contents,"\n\n")
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
    ## 3. Read the file and add the parameters (Kevin code)
        # read the file and add the parameter values
        # append into one value
    def read_Data_pandas(self,Col,New_Columns):
        for j in range(len(self.Desired_file_directory)):
            
            ################## Should change this part ######################
            
            df = pd.DataFrame(Vector_set(self.Desired_file_directory[j]), columns=Col)
            
                
            ###################3## Until this length####################3
            for i in range(len(New_Columns)):
                if len(New_Columns)==len(self.Selected_Param[0]):
                    df.insert(len(Col)+i,New_Columns[i],self.Selected_Param[j][i])
                else :
                    print(f'Manual Input colum not matching Defined Additional col : {len(New_Columns)}   ||   Input Additional col : {len(self.Selected_Param[0])}  ')
            if j ==0:
                self.total_df= df
            else:
                self.total_df=pd.concat([self.total_df,df],ignore_index=True)
        
        return self.total_df

class Data:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return (self.X[index], self.y[index])

    def __len__(self):
        return len(self.X)
    
class data_loader:
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
        self.contents=[] # Force or TCP , ANalogue => file set
        self.total_df=[]
        
     ## 1. Searching the folder with different parameter    
    def file_searching(self,path):
        '''
        Input : path
        Output : split the directory name with '_' and make a directionary for this
        '''
        ## 1. Make file list
        file_lists=os.listdir(path)
        for file_list in file_lists:
            #print(file_list)
            # Only extract folder
            if os.path.isdir(os.path.join(path,file_list)): 
                self.file_directory.append(file_list)

        ## 2. Parameter list
        self.Param=[]
        for file in self.file_directory:
            self.Param.append(file.split('_'))
        self.Param_pd=pd.DataFrame(self.Param)

        k=0
        print("1st Folder name : ", self.file_directory[0])
        self.contents=os.listdir(os.path.join(self.path,file_list))
        print("Contents in the folder :", self.contents,"\n\n")
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
    ## 3. Read the file and add the parameters (Kevin code)
        # read the file and add the parameter values
        # append into one value
    def read_Data_pandas(self,Col,New_Columns):
        for j in range(len(self.Desired_file_directory)):
            
            ################## Should change this part ######################
            
            df = pd.DataFrame(Vector_set(self.Desired_file_directory[j]), columns=Col)
            
                
            ###################3## Until this length####################3
            for i in range(len(New_Columns)):
                if len(New_Columns)==len(self.Selected_Param[0]):
                    df.insert(len(Col)+i,New_Columns[i],self.Selected_Param[j][i])
                else :
                    print(f'Manual Input colum not matching Defined Additional col : {len(New_Columns)}   ||   Input Additional col : {len(self.Selected_Param[0])}  ')
            if j ==0:
                self.total_df= df
            else:
                self.total_df=pd.concat([self.total_df,df],ignore_index=True)
        
        return self.total_df