import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from tableone import TableOne, load_dataset
import urllib.request
import os
import seaborn
import researchpy as rp
import patsy
from scipy import stats
from pandas.plotting import scatter_matrix

#load dataset
sparcs = pd.read_csv('https://health.data.ny.gov/resource/gnzp-ekau.csv')

#Examine the dataset contents
sparcs.columns
sparcs.shape
sparcs.dtypes

#Cleaning
sparcs.columns = sparcs.columns.str.replace('[^A-Za-z0-9]+', '_')
sparcs.columns = sparcs.columns.str.lower()
sparcs.columns #check if cleaned properly

############ descriptive analysis #############

#overall dataset statistics
sparcs.mean()
sparcs.var()
sparcs.describe()

#Length of stay analysis
los_mean = sparcs['length_of_stay'].mean()
los_median = sparcs['length_of_stay'].median()
los_describe = sparcs['length_of_stay'].describe()
boxplot = seaborn.boxplot(data=sparcs, x="length_of_stay", y="age_group") #Length of stay boxplot by age group
plt.show()

#Total costs analysis
costs_mean = sparcs['total_costs'].mean()
costs_median = sparcs['total_costs'].median()
costs_describe = sparcs['total_costs'].describe()
boxplot = seaborn.boxplot(data=sparcs, x="total_costs", y="age_group") #Length of stay boxplot by age group
plt.show()

#total charges analysis
charges_mean = sparcs['total_charges'].mean()
charges_median = sparcs['total_charges'].mean()
charges_describe = sparcs['total_charges'].mean()
boxplot = seaborn.boxplot(data=sparcs, x="total_charges", y="age_group") #Length of stay boxplot by age group
plt.show() 

#average costs based on risk of mortality
groupby_mortality = sparcs.groupby('apr_risk_of_mortality')
for mortality, value in groupby_mortality['total_charges']:
    print(mortality, value.mean())

#Gender charges analysis
groupby_gender = sparcs.groupby('gender')
for gender, value in groupby_gender['length_of_stay']:
    print((gender, value.mean()))
groupby_gender.mean()
boxplot = seaborn.boxplot(data=sparcs, x="total_charges", y="gender") #male charges vs female charges
plt.show()

#risk of mortality bar chart
sparcs['apr_risk_of_mortality'].value_counts()
risk = ['Minor', 'Moderate', 'Major', 'Exreme']
count = [623, 190, 154, 33]
plt.bar(risk, count)
plt.title('Risk of Mortality Among Hospital Discharges')
plt.xlabel('Risk of Mortality')
plt.ylabel('Number of Patients')
plt.show()

#ethnicity and race pie charts
#ethnicity
sparcs['ethnicity'].value_counts()
ethnicities = np.array([947, 35, 18])
labels = ['Not Span/Hispanic', 'Spanish/Hispanic', 'Unknown']
plt.pie(ethnicities, labels = labels)
plt.legend(title = "Ethnicities")
plt.show()

#race
sparcs['race'].value_counts()
races = np.array([992, 7, 1])
labels = ['White', 'Black/African American', 'Other Race']
plt.pie(races, labels = labels)
plt.legend(title = "races")
plt.show()

########## correlation analysis ###########

#2-sample t-test
stats.ttest_1samp(sparcs.dropna()['Weight'], 0)
female_costs = sparcs.dropna()[sparcs['gender'] == 'F']['total_costs']
male_costs = sparcs.dropna()[sparcs['gender'] == 'M']['total_costs']
stats.ttest_ind(female_costs, male_costs) #p = 0.11

######### tableone #############
sparcs_columns = ['age_group', 'gender', 'race', 'type_of_admission', 'length_of_stay']
categorical = ['gender', 'race', 'type_of_admission', 'age_group']
groupby = ['age_group']

table1 = TableOne(sparcs, columns=sparcs_columns, categorical=categorical, groupby=groupby, pval=False)
print(table1.tabulate(tablefmt = "fancy_grid"))
table1.to_excel('data/tableone_sparcs.xlsx')

############# researchpy #############
rp.summary_cont(sparcs[['length_of_stay', 'total_costs']])