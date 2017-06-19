import re
import xgboost as xgb
import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import sklearn.preprocessing as pp
from sklearn import metrics
import matplotlib.pylab as plt

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
total=pd.concat([train,test])
PassengerId=total['PassengerId']

def fill_features(df):
    df['Embarked'].fillna('S',inplace=True)
    df['Fare'].fillna(df['Fare'].median(),inplace=True)
    mu=df['Age'].mean()
    delta=df['Age'].std()
    rand_list=np.random.randint(mu-delta,mu+delta,size=df['Age'].isnull().sum())
    df.loc[df['Age'].isnull(), 'Age']=rand_list
    df['Age']=df['Age'].astype(int)
    return df

def add_features(df):
    df['FamilySize']=df['SibSp']+df['Parch']+1
    df['IsAlone']=1
    df.loc[df['FamilySize']>1, 'IsAlone']=0
    df['HasCabin']=df['Cabin'].apply(lambda x: 0 if type(x)==float else 1)
    return df

def simplify_features(df):
    df=simplify_fare(df)
    df=simplify_ages(df)
    df=simplify_title(df)
    return df

def simplify_fare(df):
    bins=(-1,0,7.896,14.454,31.275,512.4)
    tags=['unknown', 'first','second','third','forth']
    df['Fare']=pd.cut(df['Fare'], bins, tags)
    return df	

def simplify_ages(df):
    bins=(-1,0,5,12,18,25,35,60,120)
    tags = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    df['Age'] = pd.cut(df['Age'],bins , labels=tags) 
    return df


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

def simplify_title(df):
    df['Title'] = df['Name'].apply(get_title)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    return df

def drop_features(df):
    df.drop('PassengerId',axis=1,inplace=True)
    df.drop('Name',axis=1,inplace=True)
    df.drop('Cabin',axis=1,inplace=True)
    df.drop('SibSp',axis=1,inplace=True)
    #total.drop('Parch',axis=1,inplace=True)
    df.drop('Ticket',axis=1,inplace=True)
    return df

def encode_features(df):
    features = ['Fare', 'Age', 'Embarked', 'Sex', 'Title']
    for feature in features:
        le=pp.LabelEncoder()
        le=le.fit(df[feature])
        df[feature]=le.transform(df[feature])
    return df

total=fill_features(total)
total=add_features(total)
total=simplify_features(total)
total=drop_features(total)
total=encode_features(total)

train_y = train['Survived']
total.drop('Survived', axis=1, inplace=True)
train_x=total.iloc[range(len(train_y))]

param_test1 = {
	'gamma':[0.175,0.2,0.225]
}
gsearch1 = ms.GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=46, max_depth=3,
 min_child_weight=0, gamma=0.2, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', n_jobs=4, scale_pos_weight=1,random_state=29), 
 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(train_x, train_y)
print gsearch1.cv_results_
print gsearch1.best_params_
print gsearch1.best_score_

"""
#param_grid = {'max_depth':range(3,10,1), 'min_child_weight':range(1,6,1)} # 3,4
#param_grid = {'gamma':[i/10.0 for i in range(0,7)]} # 0.6
#param_grid = {'subsample':[i/100.0 for i in range(60,100,5)], 'colsample_bytree':[i/100.0 for i in range(60,100,5)]} # 0.85,0.75
#param_grid = {'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]}

model = xgb.XGBClassifier(learning_rate=0.1, 
                      n_estimators=5000, 
                      max_depth=3,
                      min_child_weight=4, 
                      gamma=0.6, 
                      subsample=0.85, 
                      colsample_bytree=0.75,
                      reg_alpha=0.1,
                      objective= 'binary:logistic', 
                      n_jobs=4, 
                      scale_pos_weight=1)

grid = ms.GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', n_jobs=-1, verbose=1)
grid.fit(train_x,train_y)
model.fit(train_x,train_y)
predictions = model.predict(test_x)
submission = pd.DataFrame({ 'PassengerId': test['PassengerId'],
                            'Survived': predictions })
submission.to_csv("submission.csv", index=False)
"""
