import re
import xgboost as xgb
import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import sklearn.preprocessing as pp
import sklearn.feature_selection as fs
from sklearn import metrics
import matplotlib.pylab as plt
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

#from features import fill_features, add_features, simplify_features, drop_features, encode_features
from features1 import add_title,fill_age,fill_fare,fill_embarked,fill_cabin,fill_ticket,add_family


# import data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
total=pd.concat([train,test])
PassengerId=total['PassengerId']

# for feature1.py
total=add_title(total)
total.drop('Name',axis=1,inplace=True)# Fill
total=add_family(total)
total=fill_age(total)
total=fill_fare(total)
total=fill_embarked(total)
total=fill_cabin(total)
total=fill_ticket(total)
total = pd.concat([total,pd.get_dummies(total['Title'],prefix='Title')],axis=1)
total = pd.concat([total,pd.get_dummies(total['Embarked'],prefix='Embarked')],axis=1)
total = pd.concat([total,pd.get_dummies(total['Cabin'], prefix='Cabin')], axis=1)
total = pd.concat([total,pd.get_dummies(total['Pclass'], prefix="Pclass")],axis=1) 
total['Sex'] = total['Sex'].map({'male':1,'female':0})
total = pd.concat([total, pd.get_dummies(total['Ticket'], prefix='Ticket')], axis=1)
total.drop('Title',axis=1,inplace=True)
total.drop('Embarked',axis=1,inplace=True)
total.drop('Cabin', axis=1, inplace=True)
total.drop('Pclass',axis=1,inplace=True)
total.drop('Ticket', inplace=True, axis=1)
total.drop('PassengerId', inplace=True, axis=1)
total.drop('Survived', axis=1, inplace=True)
data_y=train['Survived']
data_x=total.iloc[range(len(data_y))]
test_x=total.iloc[len(data_y):]

cv_params = {
	'max_depth':[3,5,7],
	'min_child_weight':[1,3,5],
}
fixed_params = {
	'learning_rate':0.1,
	'n_estimators':1000,
	'seed':0,
	'subsample':0.8,
	'colsample_bytree':0.8,
	'objective':'binaty:logistic',
}

optimized_gbm = GridSearhCV(xgb.XGBClassifier(**fixed_params),
							cv_params,
							scoring='accuracy',
							cv=5,
							n_jobs=-1)













"""
model = xgb.XGBClassifier(learning_rate =0.1, 
                      n_estimators=1000,
                      reg_alpha=0.01,
                      colsample_bytree=0.9,
                      min_child_weight=8,
                      subsample=0.7,
                      max_depth=9,
                      gamma=0.0,
                      objective= 'binary:logistic', 
                      n_jobs=1, 
                      scale_pos_weight=1,
                      random_state = 5)
xgbTrain = xgb.DMatrix(data_x, label=data_y)
xgbParams = model.get_xgb_params()
cv_result=xgb.cv(xgbParams,
                 xgbTrain,
                 stratified=True,
                 num_boost_round=model.get_params()['n_estimators'],
                 nfold= 5,
                 metrics='auc',
                 early_stopping_rounds=50,
                 callbacks=[xgb.callback.print_evaluation(show_stdv=False), xgb.callback.early_stop(50)],
                 seed=11
                 )

model.set_params(n_estimators=cv_result.shape[0])
model.fit(data_x,data_y, eval_metric='auc')

predictions = model.predict(data_x)
pred_prob = model.predict_proba(data_x)[:,1]


print "\nModel Report"
print "Accuracy : %.4g" % metrics.accuracy_score(data_y, predictions)
print "AUC Score (Train): %f" % metrics.roc_auc_score(data_y, pred_prob)


model_s = fs.SelectFromModel(model, prefit=True)
data_x_reduced = model_s.transform(data_x)
test_x_reduced = model_s.transform(test_x)
param_test1 = {
# 'max_depth':[9],
# 'min_child_weight':[8],
# 'gamma':[i/10.0 for i in range(0,5)],
# 'subsample':[i/10.0 for i in range(6,10)],
# 'colsample_bytree':[0.8,0.9,1.0],
#  'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
}

cv_buckets = StratifiedKFold(data_y, n_folds=5,random_state=4)
gsearch1 = ms.GridSearchCV(estimator = model, 
 param_grid = param_test1, scoring='roc_auc',n_jobs=-1,iid=False, cv=cv_buckets)
gsearch1.fit(data_x_reduced, data_y)
print gsearch1.cv_results_
print gsearch1.best_params_
print gsearch1.best_score_
clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
cl=clf.fit(data_x,data_y)

model = SelectFromModel(clf, prefit=True)
data_x_reduced = model.transform(data_x)
test_x_reduced = model.transform(test_x)
if False:
	parameter_grid = {
		         'max_depth' : [8,9,10],#9
		         'n_estimators': [25,30,35,40],#30
		         'max_features': ['log2'],
		         'min_samples_split': [9,10,11,12],#11
		         'min_samples_leaf': [1,2,3,4],#2
		         'bootstrap': [True, False],#false
		         }
	forest = RandomForestClassifier()

	cross_validation = StratifiedKFold(data_y, n_folds=5)

	grid_search = GridSearchCV(forest,
		                       scoring='accuracy',
		                       param_grid=parameter_grid,
		                       cv=cross_validation,
		                       n_jobs=-1)

	grid_search.fit(data_x_reduced, data_y)
	model = grid_search
	parameters = grid_search.best_params_

	print('Best score: {}'.format(grid_search.best_score_))
	print('Best parameters: {}'.format(grid_search.best_params_))

else :
	#parameters = {'bootstrap': True, 'min_samples_leaf': 2, 'n_estimators': 35, 
	#	          'min_samples_split': 11, 'max_features': 'log2', 'max_depth': 8}
	parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, 
		          'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}	
	model = RandomForestClassifier(**parameters)
	model.fit(data_x_reduced, data_y)

def compute_score(clf, X, y, scoring='accuracy'):
	xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
	return np.mean(xval)

print compute_score(model, data_x_reduced, data_y, scoring='accuracy')






model = XGBClassifier()
n_estimators = range(50, 400, 50)
param_grid = dict(n_estimators=n_estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X, label_encoded_y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))
"""


"""
# for feature.py


total=fill_features(total)
total=add_features(total)
total=simplify_features(total)
total=drop_features(total)
total=encode_features(total)

train_y = train['Survived']
total.drop('Survived', axis=1, inplace=True)
train_x=total.iloc[range(len(train_y))]



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
predictions = model.predict(test_x_reduced)
submission = pd.DataFrame({ 'PassengerId': test['PassengerId'],
                            'Survived': predictions })
submission.to_csv("submission.csv", index=False)
print "saved submission"
 	"""

