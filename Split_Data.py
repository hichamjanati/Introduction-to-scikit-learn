# -*- coding: utf-8 -*-

########################################################################
##########   Split of training data and test data             ##########
########################################################################

import pandas as pd

################ The general home directory has to be ajusted 
Home    = "HOME DIRECTORY WITH DATA FILE (OUTPUT FROM Final_Data.py)"
# The following directories don't have to be changed
NewHome = Home+'/Restructured Data'
NewData = NewHome + '/New Data'
File    = NewData+'/DATA.txt'


################ Import the updated database
df  = pd.read_csv(File,sep=',')
df  = df.drop("Unnamed: 0",axis=1)


################ Building up a function that splits a certain well-defined percentage gamma of our dataframe which we you as a test dataset lateron
def split(df, gamma):                                                            
    if type(df)!=pd.core.frame.DataFrame or 1<=gamma<=0: 
        return("First argument has to be a pandas-Dataframe and second a real percentage.")
    else:
        B           = df.copy(deep=True)
        B1, B0      = B[B.Smile==1], B[B.Smile==0]
        total_size  = B1.shape[0], B0.shape[0]
        Bsample     = B1.sample(int(total_size[0]*gamma)),B0.sample(int(total_size[1]*gamma))
        randchoice  = pd.concat(Bsample)
        B           = B.drop([index for index,rows in randchoice.iterrows()],axis=0)    
        randchoice.index=[i for i in range(len(randchoice))]
        
        # Exporting the datasets: Training and Test
        B.to_csv(NewData+"/TRAINING_DATASET.txt",sep=",")
        randchoice.to_csv(NewData+"/TEST_DATASET.txt",sep=",") 
        
        print("First argument of outcome is the training dataset and second is the test dataset.")
        return(B, randchoice)
        
  
      
# Execution of the function        
B, randchoice = split(df, 0.15)
  
  

    
    
    
    
    
    
    
    
    
    
    
    
   