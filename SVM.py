# -*- coding: utf-8 -*-


########################################################################
##########   Classification à l'aide de SVM       ######################
########################################################################


import pandas as pd
from sklearn import svm,metrics

################ The general home directory has to be ajusted
Home            = "GENERAL HOME DIRECTORY which has been chosen before as well"
# The following directories don't have to be changed
NewHome         = Home+'/Restructured Data'
NewData         = NewHome+'/New Data'
Training_File   = NewData+'/TRAINING_DATASET.txt'
Test_File       = NewData+'/TEST_DATASET.txt'

################ Import the updated and splitted databases
Train = pd.read_csv(Training_File,sep=',')
test  = pd.read_csv(Test_File,sep=',')
X_Train, X_test = Train.ix[:,"x1":"y22"], test.ix[:,"x1":"y22"]
y_Train, y_test = Train.ix[:,"Smile"], test.ix[:,"Smile"]

################ Fitting the model on the training dataset
classifier = svm.SVC()
classifier.fit(X_Train,y_Train)

################# MSE and other measures on performing (control on test data)
predicted = classifier.predict(X_test)
expected = y_test

score = metrics.mean_squared_error(expected,predicted)

print(metrics.classification_report(expected, predicted))
print("\nConfusion matrix:\n{}".format(metrics.confusion_matrix(expected,predicted)))
print("\n\nmean squarred error: {}".format(score))

f = open(NewData+'/Prediction_SVM.txt','w')

f.write(metrics.classification_report(expected, predicted))                                 # Python will convert \n to os.linesep
f.write("\nConfusion matrix:\n{}".format(metrics.confusion_matrix(expected,predicted)))
f.write("\n\nmean squarred error: {}".format(score))

f.close()

