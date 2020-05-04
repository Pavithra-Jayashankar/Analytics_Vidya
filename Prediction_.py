import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import os
os.chdir("D:\Analytics Vidhya\Loan Prediction III")
os.getcwd()

import pandas as pd
import matplotlib.pyplot as plt

Data = pd.read_csv("train.csv")
Test_Data = pd.read_csv("test.csv")

Data.shape
Test_Data.shape

#checking for missing value 

Data.isnull().sum()
Data.info()

Data['Loan_Status'].value_counts()

#Deleting columns with greater than 50% missing values

half_count=len(Data)/2
Data1 = Data.dropna(thresh = half_count,axis=1)
Data1.isnull().sum()

#missing value imputation

for col in Data1.columns:
    if(Data1[col].dtype == 'object'):
        temp_mode = Data1.loc[:,col].mode()[0]
        Data1[col] = Data1[col].fillna(temp_mode)
    else:
        temp_median = Data1.loc[:,col].median()
        Data1[col].fillna(temp_median, inplace = True)

Data1.isnull().sum()

color = ['blue','green']
plt.pie(Data1['Loan_Status'].value_counts(), labels = ['Yes','No'], colors = color, autopct = "%.2f%%")

pd.crosstab(Data1.Gender,Data1.Loan_Status).plot(kind='bar')
plt.xlabel('Gender')
plt.ylabel('Loan_Status')

pd.crosstab(Data1.Married,Data1.Loan_Status).plot(kind='bar')
plt.xlabel('Matrial Status')
plt.ylabel('Loan_Status')

pd.crosstab(Data1.Education,Data1.Loan_Status).plot(kind='bar')
plt.xlabel('Education')
plt.ylabel('Loan_Status')

pd.crosstab(Data1.Property_Area,Data1.Loan_Status).plot(kind='bar')
plt.xlabel('Area')
plt.ylabel('Loan_Status')

pd.crosstab(Data1.Self_Employed,Data1.Loan_Status).plot(kind='bar')
plt.xlabel('Employment')
plt.ylabel('Loan_Status')

Data1['Dependents'].value_counts()

def Dependents(x):
    if type(x)==float: 
        return x
    if x=='3+':
        return 3
    else:
        return int(x.split(' ')[0])

Data1['Dependents']=Data1['Dependents'].apply(Dependents)
Data1['Dependents'].unique()

#Label Encoding

Data1['Gender'].value_counts()
Gender_final={'Male':1,'Female':2}
Data1.Gender=[Gender_final[item]for item in Data1.Gender]
Data1['Gender'].unique()

Data1['Married'].value_counts()
married_final={'Yes':1,'No':2}
Data1.Married=[married_final[item]for item in Data1.Married]
Data1['Married'].unique()

Data1['Education'].value_counts()
Education_final={'Graduate':1,'Not Graduate':2}
Data1.Education=[Education_final[item]for item in Data1.Education]
Data1['Education'].unique()

Data1['Self_Employed'].value_counts()
Self_Employed_final={'Yes':1,'No':2}
Data1.Self_Employed=[Self_Employed_final[item]for item in Data1.Self_Employed]
Data1['Self_Employed'].unique()

Data1['Property_Area'].value_counts()
Property_Area_final={'Rural':1,'Urban':2,'Semiurban':3}
Data1.Property_Area=[Property_Area_final[item]for item in Data1.Property_Area]
Data1['Property_Area'].unique()

Data1.drop(['Loan_ID'], axis = 1, inplace= True)

Data1.columns
Data1['Gender'].unique()
Data1['Married'].unique()
Data1['Dependents'].unique()
Data1['Education'].unique()
Data1['Self_Employed'].unique()
(Data1['ApplicantIncome'].unique()).sum()
(Data1['CoapplicantIncome'].unique()).sum()
(Data1['LoanAmount'].unique()).sum()
(Data1['Loan_Amount_Term'].unique()).sum()
Data1['Credit_History'].unique()
Data1['Property_Area'].unique()
Data1['Loan_Status'].unique()

Data1['Loan_Status'].value_counts()
Loan_Status_final={'Y':1,'N':0}
Data1.Loan_Status=[Loan_Status_final[item]for item in Data1.Loan_Status]
Data1['Loan_Status'].unique()

import seaborn as sns
corrDf1 = Data1.corr()
sns.heatmap(corrDf1, xticklabels = corrDf1.columns, yticklabels = corrDf1.columns, cmap = 'gist_earth_r')


from sklearn.model_selection import train_test_split
Trainset, Testset = train_test_split(Data1, test_size = 0.3)
Trainset.shape
Testset.shape

Train_x = Trainset.drop(['Loan_Status'], axis = 1).copy()
Train_x.shape
Test_x = Testset.drop(['Loan_Status'], axis = 1).copy()
Test_x.shape
Train_y = Trainset['Loan_Status'].copy()
Train_y.shape
Test_y = Testset['Loan_Status'].copy()
Test_y.shape

#VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor

Max_VIF = 10
Train_X_Copy = Train_x.copy()
counter = 1
High_VIF_Column_Names = []

while (Max_VIF >= 10):
    
    print(counter)
    
    VIF_Df = pd.DataFrame()   
    VIF_Df['VIF'] = [variance_inflation_factor(Train_X_Copy.values, i) for i in range(Train_X_Copy.shape[1])]  
    VIF_Df['Column_Name'] = Train_X_Copy.columns
    
    Max_VIF = max(VIF_Df['VIF'])
    Temp_Column_Name = VIF_Df.loc[VIF_Df['VIF'] == Max_VIF, 'Column_Name']
    print(Temp_Column_Name, ": ", Max_VIF)
    
    if (Max_VIF >= 10): # This condition will ensure that ONLY columns having VIF lower than 10 are NOT dropped
        print(Temp_Column_Name, Max_VIF)
        Train_X_Copy = Train_X_Copy.drop(Temp_Column_Name, axis = 1)    
        High_VIF_Column_Names.extend(Temp_Column_Name)

Train_x.drop(['Loan_Amount_Term','Self_Employed','Gender'], axis = 1, inplace=True)
Test_x.drop(['Loan_Amount_Term','Self_Employed','Gender'], axis = 1, inplace=True)

Train_x.shape
Test_x.shape

from statsmodels.api import Logit
Model1 = Logit(Train_y, Train_x).fit()
Model1.summary()

col_names = ['ApplicantIncome','Dependents']
Model2 = Logit(Train_y,Train_x.drop( col_names, axis= 1)).fit()
Model2.summary()

Test_x.drop(['ApplicantIncome','Dependents'], axis = 1, inplace=True)

Test_x['Predit'] = Model2.predict(Test_x)
Test_x.columns
Test_x['Predit'][0:6]

import numpy as np
Test_x['Test_class']=np.where(Test_x['Predit']>=0.5, 1, 0)

import pandas as pd
confusion_matrix = pd.crosstab(Test_x['Test_class'], Test_y)
confusion_matrix

accuracy = sum(np.diagonal(confusion_matrix))/Test_x.shape[0]*100
accuracy #82.70

from sklearn.metrics import f1_score, precision_score, recall_score
f1_score(Test_y,Test_x['Test_class'])  
precision_score(Test_y,Test_x['Test_class'])  
recall_score(Test_y,Test_x['Test_class']) 

from sklearn.metrics import roc_curve, auc

Train_prob = Model1.predict(Train_x)
fpr,tpr,cutoff = roc_curve(Train_y,Train_prob)

cutoff_table = pd.DataFrame()
cutoff_table['FPR'] = fpr
cutoff_table['TPR'] = tpr
cutoff_table['Cutoff'] = cutoff

import seaborn as sns
sns.lineplot(cutoff_table['FPR'], cutoff_table['TPR'])

auc(fpr,tpr)

cutoff_table['Difference'] = cutoff_table['TPR'] - cutoff_table['FPR']
max(cutoff_table['Difference'])

cutoff_table['Distance'] = np.sqrt((1-cutoff_table['TPR'])**2 + (0-cutoff_table['FPR']**2))
min(cutoff_table['Distance'])

#### Based of Max difference 

Test_x['Test_class1']=np.where(Test_x['Predit']>=0.4, 1, 0)

confusion_matrix1 = pd.crosstab(Test_x['Test_class1'], Test_y)

sum(np.diagonal(confusion_matrix))/Test_x.shape[0]*100

from sklearn.metrics import f1_score, precision_score, recall_score
f1_score(Test_y,Test_x['Test_class1']) 
precision_score(Test_y,Test_x['Test_class1'])  
recall_score(Test_y,Test_x['Test_class1'])

auc(fpr,tpr)

# applying on test dataset

Test_Data.isnull().sum()
Test_Data.info()

#Deleting columns with greater than 50% missing values

half_count=len(Test_Data)/2
Test_Data1 = Test_Data.dropna(thresh = half_count,axis=1)
Test_Data1.isnull().sum()

#missing value imputation

for col in Test_Data1.columns:
    if(Test_Data1[col].dtype == 'object'):
        temp_mode = Test_Data1.loc[:,col].mode()[0]
        Test_Data1[col] = Test_Data1[col].fillna(temp_mode)
    else:
        temp_median = Test_Data1.loc[:,col].median()
        Test_Data1[col].fillna(temp_median, inplace = True)

Test_Data1.isnull().sum()
Test_Data1['Dependents'].value_counts()

def Dependents(x):
    if type(x)==float: 
        return x
    if x=='3+':
        return 3
    else:
        return int(x.split(' ')[0])

Test_Data1['Dependents']=Test_Data1['Dependents'].apply(Dependents)
Test_Data1['Dependents'].unique()

#Label Encoding

Test_Data1['Gender'].value_counts()
Gender_final={'Male':1,'Female':2}
Test_Data1.Gender=[Gender_final[item]for item in Test_Data1.Gender]
Test_Data1['Gender'].unique()

Test_Data1['Married'].value_counts()
married_final={'Yes':1,'No':2}
Test_Data1.Married=[married_final[item]for item in Test_Data1.Married]
Test_Data1['Married'].unique()

Test_Data1['Education'].value_counts()
Education_final={'Graduate':1,'Not Graduate':2}
Test_Data1.Education=[Education_final[item]for item in Test_Data1.Education]
Test_Data1['Education'].unique()

Test_Data1['Self_Employed'].value_counts()
Self_Employed_final={'Yes':1,'No':2}
Test_Data1.Self_Employed=[Self_Employed_final[item]for item in Test_Data1.Self_Employed]
Test_Data1['Self_Employed'].unique()

Test_Data1['Property_Area'].value_counts()
Property_Area_final={'Rural':1,'Urban':2,'Semiurban':3}
Test_Data1.Property_Area=[Property_Area_final[item]for item in Test_Data1.Property_Area]
Test_Data1['Property_Area'].unique()

Test_Data1.shape

Test_Data1.drop(['Loan_ID'], axis = 1, inplace= True)

Test_Data1.columns
Test_x.columns

Test_Data1.drop(['Loan_Amount_Term','Self_Employed','Gender','ApplicantIncome','Dependents'], axis = 1, inplace=True)

Test_Data1['Predit'] = Model2.predict(Test_Data1)
Test_Data1.columns
Test_Data1['Predit'][0:6]

Test_Data1['Test_class1']=np.where(Test_Data1['Predit']>=0.4, 1, 0)

Submission = pd.DataFrame()
Submission['Loan_ID'] = Test_Data['Loan_ID']
Submission['Loan_Status'] = Test_Data1['Test_class1']
Submission['Loan_Status'].replace(0, 'N',inplace=True)
Submission['Loan_Status'].replace(1, 'Y',inplace=True)
Submission['Loan_Status'].value_counts()
Submission.to_csv("Submission.csv", index=False)

