import xgboost as xgb
import pandas as pd
import numpy as np
import sklearn.model_selection as ms



train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
total=pd.concat([train,test])


total['FamilySize']=total['SibSp']+total['Parch']+1
total['IsAlone']=1
total.loc[total['FamilySize']>1, 'IsAlone']=0
total['Embarked'].fillna('S',inplace=True)
total['Fare'].fillna(total['Fare'].median(),inplace=True)
mu=total['Age'].mean()
delta=total['Age'].std()
rand_list=np.random.randint(mu-delta,mu+delta,size=total['Age'].isnull().sum())
total.loc[total['Age'].isnull(), 'Age']=rand_list
total['Age']=total['Age'].astype(int)
total.drop('PassengerId',axis=1,inplace=True)
total.drop('Name',axis=1,inplace=True)
total.drop('Cabin',axis=1,inplace=True)
total.drop('SibSp',axis=1,inplace=True)
total.drop('Parch',axis=1,inplace=True)
total.drop('Ticket',axis=1,inplace=True)


age_bins=[-1,16,32,48,64,80]
total['Age'] = pd.cut(total['Age'],age_bins , labels=range(len(age_bins)-1)) 

fare_bins=[-1,7.896,14.454,31.275,512.3292]
total['Fare']=pd.cut(total['Fare'], fare_bins, labels=range(len(fare_bins)-1))

total['Embarked']=total['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2 }).astype(int)
total['Sex']=total['Sex'].map( { 'female' : 0 , 'male' : 1 } ).astype(int)


train_y = train['Survived']

total.drop('Survived', axis=1, inplace=True)
total_array = total.as_matrix()

train_x = total_array[:len(train_y),:]
test_x = total_array[len(train_y):,:]

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

#grid = ms.GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', n_jobs=-1, verbose=1)
#grid.fit(train_x,train_y)
model.fit(train_x,train_y)
predictions = model.predict(test_x)
submission = pd.DataFrame({ 'PassengerId': test['PassengerId'],
                            'Survived': predictions })
submission.to_csv("submission.csv", index=False)

