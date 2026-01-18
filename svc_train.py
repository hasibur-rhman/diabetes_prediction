import pandas as pd
import numpy as np
 
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

df=pd.read_csv('diabetes.csv')
#Imputation
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
#Feature engineering

df_imputed['Body_Fat_Index'] = df_imputed['BMI'] * df_imputed['SkinThickness']


df_imputed['Metabolic_Index'] = df_imputed['Glucose'] * df_imputed['Insulin']
#remove outlier
Q1 = df_imputed.quantile(0.25)
Q3 = df_imputed.quantile(0.75)
IQR = Q3 - Q1
df_cleaned = df_imputed[~((df_imputed < (Q1 - 1.5 * IQR)) |(df_imputed > (Q3 + 1.5 * IQR))).any(axis=1)]

X=df_cleaned.drop('Outcome', axis=1)
y=df_cleaned['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
    )

#pipeline
def get_pipeline(model):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
svc_pipeline=get_pipeline(SVC(kernel='linear',C=0.1,probability=True))
svc_pipeline.fit(X_train, y_train)

import pickle
with open('diabetes_model.pkl', 'wb') as file:
    pickle.dump(svc_pipeline, file)