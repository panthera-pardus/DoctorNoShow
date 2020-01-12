# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:22:40 2017

@author: Alassane-anand
"""
#%% Import and read data
from sklearn.model_selection import train_test_split as split
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import statsmodels.api as sm
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import itertools

cd = os.chdir('/Users/panthera_pardus/Documents/ds_projects/doctor_appointment')
doctor = pickle.load(open( "data/transformed_data_pickle", "rb" ))


#%% Analyse the data with a logistic regression
####################### Analysis ############

#For now a simple split between training, validation and test set shall be done
training, test = split(doctor, test_size = 0.2)
training, validation = split(training, test_size = 0.25 )

# Clean the datasets
training.drop(['AppointmentID', 'ScheduledDay', 'AppointmentDay', 'Neighbourhood','time_diff' ], axis = 1)
validation.drop(['AppointmentID', 'ScheduledDay', 'AppointmentDay', 'Neighbourhood','time_diff' ], axis = 1)
test.drop(['AppointmentID', 'ScheduledDay', 'AppointmentDay', 'Neighbourhood','time_diff' ], axis = 1)

# Rearrange col order for simplicity
col = ['Noshow',
 'PatientId',
 'Gender',
 'Age',
 'Scholarship',
 'Hipertension',
 'Diabetes',
 'Alcoholism',
 'Handcap',
 'SMS_received',
 'time_diff_mean',
 'neighb_cat',
 'dayofweek']

training = training[col]
validation = validation[col]
test = test [col]

training['intercept'] = 1
validation['intercept'] = 1
test['intercept'] = 1

#Fit the logistic regression on the training set
#from sklearn.linear_model import LogisticRegression as logit

#X = np.array([training['PatientId'],training['Gender'], training['Age'], training['Scholarship'], training['Hipertension'], training['Diabetes'], training['Alcoholism'], training['Handcap'], training['SMS_received'], training['time_diff_mean'],training['neighb_cat']])
#X = np.transpose(X)

#y = np.array(training['Noshow'])
#y = np.transpose(y)

#model = logit()
#model.fit(X, y)
#print(model.score(X,y))

#Through statsmod
#Note that adding an intercept to the model removed much of the significance

endo = 'Noshow'
exog1 = training.columns.tolist()
exog1.remove('Noshow')
exog1.remove('dayofweek')


logit1 = sm.Logit(training[endo],training[exog1])
result1 = logit1.fit()
print(result1.summary())

#In order to answer the causal questions we must evaluate the logit model (coefficients of the logistic regression) and their significance
#We will use p-values to estimate the causal effect of coeff of interest
#The variables that are linked to gender and neighbourhood are insignificant.
#Although it could be interesting to dig deeper for this study we shall focus on the significant variables and drop the others

exog2 = training.columns.tolist()
exog2 = [element for element in exog2 if element not in ('Noshow', 'Hipertension','PatientId','Gender','neighb_cat','dayofweek')]

logit2 = sm.Logit(training[endo],training[exog2])
result2 = logit2.fit()
print(result2.summary())

#Days of the week model
exog3 = training.columns.tolist()
exog3 = [element for element in exog3 if element not in ('time_diff_mean', 'Noshow')]

logit3 = sm.Logit(training[endo],training[exog3])
result3 = logit3.fit()
print(result3.summary())


#Now we can compare this to the validation set and see how well we perform
#Goodness of fit of the model ono the validation set
predicted1 = result1.predict(validation[exog1])
predicted2 = result2.predict(validation[exog2])
predicted3 = result3.predict(validation[exog3])

#Now we have the probability of no_show. We want a binary outcome to compare to validation set series
# Symmetric threshold implemented
predicted_choice1 = []

for i in predicted1:
    if i > 0.5:
        i = 1
    else:
        i = 0
    predicted_choice1.append(i)

predicted_choice2 = []

for i in predicted2:
    if i > 0.5:
        i = 1
    else:
        i = 0
    predicted_choice2.append(i)

predicted_choice3 = []

for i in predicted3:
    if i > 0.5:
        i = 1
    else:
        i = 0
    predicted_choice3.append(i)

MSE1 = metrics.mean_squared_error(validation['Noshow'],predicted_choice1)
MSE2 = metrics.mean_squared_error(validation['Noshow'],predicted_choice2)
MSE3 = metrics.mean_squared_error(validation['Noshow'],predicted_choice3) # the models are the same

#Obtain a confusion matrix


#Functions written by sklearn dev
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#Confusion Matrix - deos not match report [set seed importance]
cm_3 = confusion_matrix(validation["Noshow"], predicted_choice3)
plot_confusion_matrix(cm = cm_3, classes = ["Show", "No show"])
