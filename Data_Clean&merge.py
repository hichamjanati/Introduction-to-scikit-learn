# -*- coding: utf-8 -*-

########################################################################
###  Nettoyage de la base initiale et création de la base de travail ###
########################################################################


import numpy as np
import pandas as pd
import os
from math import floor

################ The general home directory has to be ajusted
Home         = "HOME DIRECTORY WITH DATA FILE"
# The following directories don't have to be changed
NewHome      = Home+'/Restructured Data'
NewData      = NewHome + '/New Data'
NewLandmarks = NewHome + '/New Landmarks'


################ If the new directories that will contain our restructured data don't exist, let's create them: 
if not os.path.exists(NewHome):
    os.makedirs(NewHome)
    
if not os.path.exists(NewLandmarks):
    os.makedirs(NewLandmarks)

if not os.path.exists(NewData):
    os.makedirs(NewData)


################ Creation of a function, which imports and prepares the initial data for further use --> Data-Management

def DATA(newvar=0,labelfolder=Home+'/AU Labels',landmarksfolder=Home + '/Landmark Points',targetfolder=NewData):
# Create our final huge DATA dataframe:
    DATA = pd.DataFrame()
# For each file label:
    for h,dirs,files  in os.walk(landmarksfolder):
        for file in files:
            file='/'+ file
            if 'txt' in file.split('.'):
                 
                #############################################
                ############ LANDMARKS TREATMENT ############

                landmarks = pd.read_csv(landmarksfolder + file,sep=',')

                # Droping the last "unnamed"-column
                landmarks = landmarks[[c for c in landmarks.columns if "Unnamed" not in c]]
                # Naming the 44 columns (corresponding to coordinates of the 22 landmarks) and add a variable, which indicates the frame-number
                landmarks.columns=["x1","y1","x2","y2","x3","y3","x4","y4","x5","y5"
                                   ,"x6","y6","x7","y7","x8","y8","x9","y9","x10",
                                   "y10","x11","y11","x12","y12","x13","y13","x14",
                                   "y14","x15","y15","x16","y16","x17","y17","x18",
                                   "y18","x19","y19","x20","y20","x21","y21","x22","y22"]
                landmarks["frame"]=[i for i in range(0,len(landmarks))]
    

                ############################################
                ############# LABEL TREATMENT ##############

                # Get the number of file (cut the last part : '-landmarks.txt') so as to also read the related label file
                nameOffile = file.split('landmark')[0]
                label      = pd.read_csv(labelfolder+nameOffile+'label.csv',sep=',')
                
                # Keep the variables we're interested in:
                C          = label.columns
                if ("Smile" not in C) | ("Time" not in C) | ("Trackerfail" not in C): continue
                
                label      = label[["Time","Smile","Trackerfail"]]

                # Since the videos' frame rate is 25 frame per second; using a new index ("frame number")is necessary 
                # to group labels and landmarks :
                label.Time = label.Time*25
                label.Time = label.Time.apply(floor)
                label.rename(columns={'Time':'frame'},inplace=True)

                ###############################################################
                ############ GROUPING LABELS + LANDMARKS  ##############

                # Concat the two dataframes                     
                newdata = landmarks.merge(label, how='inner',on="frame")

                # Trackerfail > 50 means landmarks may not be correct
                newdata = newdata[newdata.Trackerfail<=50]
            
                # No need to keep Trackerfail anymore                
                newdata = newdata.drop('Trackerfail',axis=1)
                
                # If newdata doesn't contain enough observations, move forward in the loop
                if newdata.shape[0]<=10: continue 
                
               
                ##############################################################
                ####### NEW VARIABLES if newvar=1 (distance variables) #######
                
                if newvar:
                    # Grouping x,y variables in separate lists
                    eyes    = ['x9','y9','x10','y10','x11','y11','x12','y12']
                    eyebrows= ['x17','y17','x18','y18','x21','y21','x22','y22']
                    nose    = ['x4','y4','x19','y19','x20','y20']
                    upperlip= ['x5','y5','x7','y7','x8','y8','x13','y13','x14','y14']
                    lowerlip= ['x6','y6','x15','y15','x16','y16']
                    lips    = upperlip+lowerlip
                    
                    lips    = np.ravel(lips).reshape(8,2)
                
                    newvars = []
                    for i,a in enumerate(lips):
                        for b in lips[i+1:]:
                            newdata[a[0]+'-'+b[0]] = newdata[a[0]]-newdata[b[0]]                # Introduce the variable x_i-x_j
                            newdata[a[1]+'-'+b[1]] = newdata[a[1]]-newdata[b[1]]                # Introduce the variable y_i-y_j
                            newvars                = newvars+[a[0]+'-'+b[0],a[1]+'-'+b[1]]      # Add both variables to new variables' list   
      
                 
                l1      = [index for index,rows in newdata.iterrows() if 0 not in list(newdata.ix[index,'x1':'y22'])]
                newdata = newdata.ix[l1,:]
                
                # Unless less than 10 observations are left, we continue
                if newdata.shape[0]<=10: continue
                
                # If new variables are defined let's normalize them:
                if newvar:
                    newdata["nose"] = ((newdata.y4-newdata.y3)**2+(newdata.x3-newdata.x4)**2).apply(np.sqrt)  # Create a new variable equal to size of the nose
                    mean            =vnewdata["nose"].mean()           
                    for newvariable in newvars:
                        newdata[newvariable] *= mean/newdata["nose"]  #Get all distances to the same scale 
                    
                    newdata.drop(["nose"],axis=1,inplace=True)
                # Transform the smile variable into binary (if 20% of evaluators agree)
                newdata.loc[newdata.Smile<20,"Smile"] = 0
                newdata.loc[newdata.Smile>=20,"Smile"]= 1
    
                # Saving newdata for each file:
                newdata.to_csv(targetfolder+nameOffile+'-newdata.txt',sep=',')
    

                ##############################################################
                ############### GROUPING ALL NEW DATA FILES ##################
                
                # Keep the video's reference of each file in case we need to trace back a specific frame
                newdata['File'] = file.split('-landmarks')[0]              
                
                # If it's empty (first time in the loop), DATA == the first dataframe newdata
                if DATA.size==0:
                    DATA = newdata
                # Then, we keep adding dataframes ...
                else:
                    DATA = pd.concat((DATA,newdata))
    
    
    
    # Reindexing of our final data 
    DATA.index=[i for i in range(len(DATA))]
    
   
    # Frames indexation starts with 1 instead of 0
    DATA.frame.apply(lambda e: e+1)
    
    # Save in the same folder:
    DATA.to_csv(targetfolder+"/Data"+str(newvar)+".txt",sep=",")
    return DATA



# Execution of the function:        
d0=DATA(newvar=0)
#d1=DATA(newvar=1)




