# -*- coding: utf-8 -*-


########################################################################
##########   Comparaison of SVM and Decision Tree model       ##########
########################################################################

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, roc_curve, auc
import matplotlib.pyplot as plt


################ The general home directory has to be ajusted
Home            = "GENERAL HOME DIRECTORY which has been chosen before as well"
# The following directories don't have to be changed
NewHome         = Home+'/Restructured Data'
NewData         = NewHome+'/New Data'
Training_File   = NewData+'/TRAINING_DATASET.txt'
Test_File       = NewData+'/TEST_DATASET.txt'

################  Import the prediction outputs of each method as well as the true values from the Test dataset
df_test         = pd.read_csv(Test_File,sep=',')
true_values     = df_test["Smile"]
pred_svm        = pd.read_csv(NewData+"/Prediction_SVM.txt",sep=',')
pred_tree       = pd.read_csv(NewData+"/Prediction_Tree.txt",sep=',')

################ Confusion matrices
# SVM
print('The classification report and the confusion matrix for SVM:')
print(classification_report(true_values, pred_svm))
print(confusion_matrix(true_values, pred_svm))
# Decision Tree
print('The classification report and the confusion matrix for Decision Tree:')
print(classification_report(true_values, pred_tree))
print(confusion_matrix(true_values, pred_tree))

################  ROC curves
false_positive_rate_svm, true_positive_rate_svm, thresholds1   = roc_curve(true_values, pred_svm['# Smile'])
roc_auc_svm                                                    = auc(false_positive_rate_svm, true_positive_rate_svm)
false_positive_rate_tree, true_positive_rate_tree, thresholds2 = roc_curve(true_values, pred_tree['# Smile'])
roc_auc_tree                                                   = auc(false_positive_rate_tree, true_positive_rate_tree)

fig = plt.figure()
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate_svm, true_positive_rate_svm, 'b', label='SVM'% roc_auc_svm)
plt.plot(false_positive_rate_tree, true_positive_rate_tree, 'y', label='Decision Tree'% roc_auc_tree)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
fig.savefig('ROC curve.png')

################  Metrics of interest
# Create a vector of differences indication whether the method predicted in the right way
diff_svm  = (np.array(pred_svm['# Smile'])-np.array(true_values))
diff_tree = (np.array(pred_tree['# Smile'])-np.array(true_values))

mse_svm             = mean_squared_error(true_values, pred_svm)
first_err_svm       = ((diff_svm==-1).sum())/len(true_values)
second_err_svm      = ((diff_svm==1).sum())/len(true_values)
print('The error rate for SVM is:  '+ str(mse_svm))
print('The first type error rate for SVM is:  '+ str(first_err_svm))
print('The second type error rate for SVM is:  '+ str(second_err_svm))

mse_tree            = mean_squared_error(true_values, pred_tree)
first_err_tree      = ((diff_tree==-1).sum())/len(true_values)
second_err_tree     = ((diff_tree==1).sum())/len(true_values)
print('The error rate for Decision Tree is:  '+ str(mse_tree))
print('The first type error rate for Decision Tree is:  '+ str(first_err_tree))
print('The second type error rate for Decision Tree is:  '+ str(second_err_tree))


########### Retrouver les frames correspondant à une prédiction précise
#def right_frame(df_test,diff_svm,diff_tree,rightpre_svm=True,rightpre_tree=True):
#    diff_svm    = abs(diff_svm)
#    diff_tree   = abs(diff_tree)*2
#    diff        = pd.DataFrame(diff_svm-diff_tree)
#    pred        = []
#    diction     = {}
#    if rightpre_svm and rightpre_tree:
#        pred = [i for i in diff.index if int(diff.ix[i])==0]
#    if rightpre_svm and rightpre_tree==False:
#        pred = [i for i in diff.index if int(diff.ix[i])==-2]
#    if rightpre_svm==False and rightpre_tree:
#        pred = [i for i in diff.index if int(diff.ix[i])==1]
#    if rightpre_svm==False and rightpre_tree==False:
#        pred = [i for i in diff.index if int(diff.ix[i])==-1]
#    new_test=df_test.ix[pred]
#    k=new_test.loc[:,['File','frame']]
#    for i in k.index: 
#        diction.setdefault(str(k.loc[i,'File']), []).append(int(k.loc[i,'frame']))
#    return diction
#
#k = right_frame(df_test,diff_svm,diff_tree,True,True)
#
#outfile = open( 'dict.txt', 'w' )
#for key, value in sorted( k.items() ):
#    outfile.write( str(key) + '\t' + str(value) + '\n' )

    




