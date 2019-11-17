import pandas as pd
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt
import csv
dataset = pd.read_csv("C:\project\data2.csv")
inputs=dataset.drop('Suggested Job Role',axis='columns')
target=dataset['Suggested Job Role']
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_percentage_in_Operating_Systems=LabelEncoder()
le_percentage_in_Algorithms=LabelEncoder()
le_Percentage_in_Programming_Concepts=LabelEncoder()
le_Percentage_in_Software_Engineering=LabelEncoder()
le_Percentage_in_Computer_Networks=LabelEncoder()
le_Percentage_in_Electronics =LabelEncoder()
le_Percentage_in_Mathematics=LabelEncoder()
le_Percentage_in_Communication_skills=LabelEncoder()
le_coding_skills_rating=LabelEncoder()
le_public_speaking_points=LabelEncoder()
le_work_long_time_before_system=LabelEncoder()
le_self_learning_capability=LabelEncoder()
le_workshops=LabelEncoder()
le_reading_writing_skills=LabelEncoder()
le_memory_capability_score=LabelEncoder()
le_Interested_subjects=LabelEncoder()
le_Type_company_settle_in=LabelEncoder()
le_Interested_Type_Books=LabelEncoder()
le_Gentle_Tuff=LabelEncoder()
le_Management_Technical=LabelEncoder()
le_Salary_work=LabelEncoder()
le_hard_smart=LabelEncoder()
le_worked_in_teams=LabelEncoder()
le_Introvert=LabelEncoder()
inputs['can work long time before system?_n']=le_work_long_time_before_system.fit_transform(inputs['can work long time before system?'])
inputs['self-learning capability?_n']=le_self_learning_capability.fit_transform(inputs['self-learning capability?'])
inputs['workshops_n']=le_workshops.fit_transform(inputs['workshops'])
inputs['reading and writing skills_n']=le_reading_writing_skills.fit_transform(inputs['reading and writing skills'])
inputs['memory capability score_n']=le_memory_capability_score.fit_transform(inputs['memory capability score'])
inputs['Interested subjects_n']=le_Interested_subjects.fit_transform(inputs['Interested subjects'])
inputs['Type of company want to settle in?_n']=le_Type_company_settle_in.fit_transform(inputs['Type of company want to settle in?'])
inputs['Interested Type of Books_n']=le_Interested_Type_Books.fit_transform(inputs['Interested Type of Books'])
inputs['Gentle or Tuff behaviour?_n']=le_Gentle_Tuff.fit_transform(inputs['Gentle or Tuff behaviour?'])
inputs['Management or Technical_n']=le_Management_Technical.fit_transform(inputs['Management or Technical'])
inputs['Salary/work_n']=le_Salary_work.fit_transform(inputs['Salary/work'])
inputs['hard/smart worker_n']=le_hard_smart.fit_transform(inputs['hard/smart worker'])
inputs['worked in teams ever?_n']=le_worked_in_teams.fit_transform(inputs['worked in teams ever?'])
inputs['Introvert_n']=le_Introvert.fit_transform(inputs['Introvert'])
inputs_n=inputs.drop(['can work long time before system?','self-learning capability?','workshops',
       'reading and writing skills', 'memory capability score',
       'Interested subjects',
       'Type of company want to settle in?',
        'Interested Type of Books',
        'Gentle or Tuff behaviour?',
       'Management or Technical', 'Salary/work', 'hard/smart worker',
       'worked in teams ever?', 'Introvert'],axis='columns')
pd.set_option('display.max_columns', None)
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.metrics import accuracy_score
modelv=tree.DecisionTreeClassifier()
modelv.fit(inputs_n, target)
import pickle
with open('modelv.pkl', 'wb') as file:
    pickle.dump(modelv, file)
model=pickle.load(open('modelv.pkl','rb'))
