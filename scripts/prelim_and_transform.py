# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:22:40 2017

@author: Alassane-anand
"""

#%% Import cells
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle

cd = os.chdir('/Users/panthera_pardus/Documents/ds_projects/doctor_appointment')
cwd = os.getcwd()

#%% Read and preliminary look at data
#Open the data as a dataframe
doctor = pd.read_csv('doctor_no_show.csv', sep = ',')

####################### Preliminary Analysis ############

# Read the first few lines and describe the data
doctor.head()
description = doctor.describe() #certain min and max stand out
print(description)
doctor.count() # ensure data is complete

#Preliminary graphs
#Difference between neighborhoods:

a = doctor['Neighbourhood'].value_counts().plot(kind = 'bar',
                                                rot = 80,
                                                title = 'Count of patients by Neighbourhood',
                                                colormap = 'summer',
                                                width = 0.2,
                                                figsize = (17,10)
                                                )
a = a.get_figure()
a.savefig(cwd + "/neighbourhood.png")

#Gender differences
b = doctor['Gender'].value_counts().plot(
                                         kind='bar',
                                         rot = 2,
                                         title = 'Count of gender',
                                         colormap = 'summer',
                                         width = 0.9)
b = b.get_figure()
b.savefig(cwd + "/gender.png")

#Distribution of Age
sns.set_style("white")
age = sns.distplot(doctor['Age'], color = 'mediumseagreen')
age = age.get_figure()
age.savefig(cwd + "/age.png")

#%% Data transformation
####################### Data transfomation ############
#First we will get rid of the problematic observation (as there is only one : where Age == -1)

doctor = doctor.drop(doctor.index[doctor['Age'] == -1].tolist()[0], axis = 0) # the method drop takes an integer or list as the first argument and an axis = 0 for index and axis = 1 for columns
# We now transform the categorical varibles into a numerical form

doctor['Noshow'] = pd.get_dummies(doctor.Noshow).Yes #no_show = 1 if no show and 0 otherwise
doctor['Gender'] = pd.get_dummies(doctor.Gender).M # gender = 1 if male, 0 otherwise

# Now we deal with the timestamps
doctor['ScheduledDay'] = pd.to_datetime(doctor['ScheduledDay'])
doctor['AppointmentDay'] = pd.to_datetime(doctor['AppointmentDay'])

#Also note the first year of smaple is 1970 and last is in 2016
#we note that the longest time between a scheduled apoint and the actual is less than 1 year
max(doctor['AppointmentDay'] - doctor['ScheduledDay'])
doctor['time_diff'] = doctor['AppointmentDay'] - doctor['ScheduledDay'] # we have negative days difference
doctor['time_diff_days'] = doctor['time_diff'].dt.days # This is a column only with the difference in days as floats

# we show the proportion of negative days (measurement errors probbly)
sum = 0
for i in doctor.time_diff_days.value_counts().iteritems(): #the list is short so we loop
    if i[0] < 0:
        sum += i[1]

percentage_errors = (sum/len(doctor)) * 100

# There are too many errors to drop ~ 35%
# Instead we will replace by mean values (note mode is -1 so cannot be used)

doctor.rename(columns = {'time_diff_days' : 'time_diff_mean'}, inplace = True)
doctor.loc[doctor['time_diff_mean']<0,'time_diff_mean'] = doctor['time_diff_mean'].mean() #it might be important to find another imputation method
doctor['dayofweek'] = doctor['ScheduledDay'].dt.dayofweek


#Now we deal with the regions : For now we will create a catagorical varible with 3 bins
#We shall divide the regions in 3 catagories : high_people = 2 {first 26 regions}, averge_people = 1{second 26}, low_people = 0 {last 30}
#doctor = doctor.drop(doctor['Neighbourhood'] == 1, 0) #drop neighbourhood errors (erreur de manip)

doctor.Neighbourhood.value_counts() #shows that there are 82 region
group_1 = doctor['Neighbourhood'].value_counts()[0:25].index.tolist() #Highest group
group_2 = doctor['Neighbourhood'].value_counts()[25:50].index.tolist() #Middle group
group_3 = doctor['Neighbourhood'].value_counts()[50:82].index.tolist() #Lowest group

doctor['neighb_cat'] = np.nan
doctor.loc[doctor['Neighbourhood'].isin(group_1),'neighb_cat'] = 2
doctor.loc[doctor['Neighbourhood'].isin(group_2),'neighb_cat'] = 1
doctor.loc[doctor['Neighbourhood'].isin(group_3), 'neighb_cat'] = 0


# Now we deal with handicap to make if binary
doctor.loc[doctor['Handcap'] > 1, :] = 1

#Now save the whole as pickle file
doctor.to_pickle('transformed_data_pickle')
