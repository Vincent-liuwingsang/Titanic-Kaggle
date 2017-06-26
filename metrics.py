import re
import seaborn as sns
import xgboost as xgb
import pandas as pd
import numpy as np
import sklearn.model_selection as ms
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn import metrics
from scipy import interp

import matplotlib.pylab as plt

def plot_calibration_curve(est, name, fig_index,X,y,train_i,test_i):
    X_train=X.iloc[train_i]
    X_test=X.iloc[test_i]
    y_train=y.iloc[train_i]
    y_test=y.iloc[test_i]
    isotonic = CalibratedClassifierCV(est, cv=5, method='isotonic')
    sigmoid = CalibratedClassifierCV(est, cv=5, method='sigmoid')

    fig = plt.figure(fig_index, figsize=(20, 20))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    for clf, name in [(est, name),
                      (isotonic, name + ' + Isotonic'),
                      (sigmoid, name + ' + Sigmoid')]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y.max())
        print("%s:" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()


	
def cross_validate(alg, X,y,seed=0,rep=10,cv=5):   
	#base_fpr = np.linspace(0, 1, 101)
	#plt.figure(figsize=(10, 10))

	tprs = []
	accuracies = []
	roc_auc_scores = []
	log_losses = []
	confuses=[]
	f1s=[]
	for i in np.random.randint(0,high=10000, size=rep):
		kf = ms.KFold(n_splits=cv, shuffle=True, random_state=seed+i)
		for train_i, test_i in kf.split(X):
			alg.fit(X.iloc[train_i], y.iloc[train_i])
#			alg.fit(X.iloc[train_i], y.iloc[train_i],eval_metric='auc') for xgb_classifer
			predictions = alg.predict(X.iloc[test_i])
			predict_prob = alg.predict_proba(X.iloc[test_i])[:,1]   
			roc_auc_score = metrics.roc_auc_score(y.iloc[test_i],predict_prob)
			accuracy = metrics.accuracy_score(y.iloc[test_i],predictions)
			log_loss = metrics.log_loss(y.iloc[test_i], predict_prob, eps=1e-15, normalize=True)
			f1_score = metrics.f1_score(y.iloc[test_i],predictions)
#			fpr, tpr, _ = metrics.roc_curve(y.iloc[test_i], predict_prob)
#			plt.plot(fpr, tpr, 'b', alpha=0.15)
#			tpr = interp(base_fpr, fpr, tpr)
#			tpr[0] = 0.0
#			tprs.append(tpr)	
			accuracies.append(accuracy)
			roc_auc_scores.append(roc_auc_score)
			log_losses.append(log_loss)
			confuses.append(metrics.confusion_matrix(y.iloc[test_i], predictions))
			f1s.append(f1_score)
	x=np.array(confuses)[:,:,:].sum(axis=0)/len(confuses)

	return (np.mean(accuracies),np.mean(roc_auc_scores),np.mean(log_losses),1.0*x.item(3)/(x.item(3)+x.item(2)),1.0*x.item(3)/(x.item(3)+x.item(1)),np.mean(f1s))
"""
	# graph plotting for ROC
	tprs = np.array(tprs)
	mean_tprs = tprs.mean(axis=0)
	std = tprs.std(axis=0)
	tprs_upper = np.minimum(mean_tprs + std, 1)
	tprs_lower = mean_tprs - std

	plt.plot(base_fpr, mean_tprs, 'b')
	plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.1)
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([-0.01, 1.01])
	plt.ylim([-0.01, 1.01])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.axes().set_aspect('equal', 'datalim')
	plt.show()

	#Print model report:
	print "Model Report"
	print "Mean Accuray: %f" % (np.mean(accuracies))
	print "Mean AUC: %f" % (np.mean(roc_auc_scores))
 	print "Mean Logloss %f" % (np.mean(log_losses))
	
	x=np.array(confuses)[:,:,:].sum(axis=0)/len(confuses)   
	for r in x:
		print r
	print "Mean Precision: %f" % (1.0*x.item(3)/(x.item(3)+x.item(2)))
	print "Mean Recall: %f"% (1.0*x.item(3)/(x.item(3)+x.item(1)))       
"""
 
