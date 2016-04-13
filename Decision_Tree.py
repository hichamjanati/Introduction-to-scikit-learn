# -*- coding: utf-8 -*-


########################################################################
##########   Classification à l'aide des arbres de décision   ##########
########################################################################


import numpy as np
import pandas as pd
import os
from sklearn import tree, grid_search
from sklearn.metrics import mean_squared_error


################ The general home directory has to be ajusted
Home            = "GENERAL HOME DIRECTORY which has been chosen before as well"
# The following directories don't have to be changed
NewHome         = Home+'/Restructured Data'
NewData         = NewHome+'/New Data'
Training_File   = NewData+'/TRAINING_DATASET.txt'
Test_File       = NewData+'/TEST_DATASET.txt'


################ Import the updated and splitted databases
df_training         = pd.read_csv(Training_File,sep=',')
df_test             = pd.read_csv(Test_File,sep=',')
X_training, X_test  = df_training.ix[:,"x1":"y22"], df_test.ix[:,"x1":"y22"]
Y_training, Y_test  = df_training.ix[:,"Smile"], df_test.ix[:,"Smile"]

################ Fitting the model on the training dataset (first part is for optimisation only, to comment if wished)
param_grid  = {'max_depth': np.arange(3,20)}
grid        = grid_search.GridSearchCV(tree.DecisionTreeClassifier(),param_grid)
result_grid = grid.fit(X_training, Y_training)
opt_grid    = result_grid.best_params_["max_depth"]
print("The optimal depth of the decision tree is: "+ str(opt_grid))

rndf        = tree.DecisionTreeClassifier(max_depth=opt_grid)
rndf        = rndf.fit(X_training, Y_training)

### First small prediction on one randomly chosen value, which seems to work pretty well
#ind     = df_test[df_test.Smile==1].index
#a       = np.random.randint(0,len(ind))
#b       = ind[a]
#topred  = X_test.ix[b,:]
#print(rndf.predict(topred))

################# MSE control on test data
Y_test_pred = rndf.predict(X_test)
score       = mean_squared_error(Y_test,Y_test_pred)

print("The mean squared error is: "+str(score))

################# Graphical representation of the trees
dotfile = open(Home+"/Code/tree.dot", 'w')
tree.export_graphviz(rndf, out_file=dotfile, feature_names = X_test.columns, class_names=['No smile','Smile'],filled=True, rounded=True,  
                         special_characters=True, max_depth=3)
dotfile.close()

cwd = os.getcwd()
from subprocess import check_call
check_call(['dot','-Tpng','tree.dot','-o','tree.png'])



# Export of the predicted values 
np.savetxt(NewData+"/Prediction_Tree.txt", Y_test_pred, delimiter=";",header="Smile")

