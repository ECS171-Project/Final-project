import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import scikitplot

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn import model_selection

np.random.seed(1)


## function to turn parameters for Random Forest
def RF_tune_parameter(X, y, metric, n_est, cv):
    print('Tuning...\nn varies from %d to %d'%(n_est[0], n_est[-1]))
    score = []
    for n in n_est:
        print('n =', n)
        model = RandomForestClassifier(n_estimators = n, criterion='entropy')
        s = model_selection.cross_validate(model, X, y, scoring=metric, cv=cv)
        score.append(np.mean(s['test_score']))
    return np.array(score)

# Find out the n_estimators with the highest F1 score
def find_best_parameter(n_est, score, figure):
    print('n_est: ', n_est,'\nscore:', score)
    index = np.argmax(score)
    print('The best n_estimators :', n_est[index])
    return n_est[index]


# ========================== Training and testing set ============================================================
## vectorization CountVectorizer()
with open('train_data.pkl', 'rb') as f:
    X_tr, y_tr = pickle.load(f)

with open('test_data.pkl', 'rb') as f:
    X_te, y_te = pickle.load(f)

## vectorization TfidfVectorizer()
with open('train_data2.pkl', 'rb') as f:
    X_tr2, y_tr2 = pickle.load(f)

with open('test_data2.pkl', 'rb') as f:
    X_te2, y_te2 = pickle.load(f)

# ========================= Data set 1 ==============================================================================
print('CountVectorizer: Data set 1')
# ========================= Random Forest classier starts ===========================================================
n_est = np.arange(60, 100, 5)
S1 = RF_tune_parameter(X = X_tr, y = y_tr, metric = 'f1',n_est = n_est, cv = 5)
best_n_est1 = find_best_parameter(n_est, S1, 5)

classifier = RandomForestClassifier(n_estimators = best_n_est1, criterion='entropy')
classifier.fit(X_tr,y_tr)
y_pr_rt = classifier.fit(X_tr, y_tr).predict_proba(X_te)
y_pr = classifier.predict(X_te)


# ========================= precision_recall_curve plotting =========================================================
pr, rc, thresholds = precision_recall_curve(y_te, y_pr_rt[:, 1])
print("roc_auc for class 1:", roc_auc_score(y_te,y_pr_rt[:, 1]))
# Figure 1: precision, recall vs thresholds

scikitplot.metrics.plot_precision_recall(y_te, y_pr_rt, plot_micro = False)


# Figure 2: ROC
scikitplot.metrics.plot_roc(y_te, y_pr_rt, plot_micro = False, plot_macro = False)


# ========================== classification report and Confusion matrix ============================================
print(classification_report(y_te, y_pr))

m_confusion_test = confusion_matrix(y_te, y_pr)
print(pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0', 'Predicted 1'],
            index = ['Actual 0', 'Actual 1']))

print('\n')
# ==================================================================================================================


# ========================= Data set 2 ==============================================================================
print('TfidVectorizer: Data set 2')
# ========================= Random Forest classier starts ===========================================================
n_est = np.arange(40, 80, 3)
S2 = RF_tune_parameter(X = X_tr2, y = y_tr2, metric = 'f1',n_est = n_est, cv = 5)
best_n_est2 = find_best_parameter(n_est, S2, 6)
classifier2 = RandomForestClassifier(n_estimators = best_n_est2, criterion = 'entropy')
classifier2.fit(X_tr2,y_tr2)
y_pr_rt2 = classifier.fit(X_tr2, y_tr2).predict_proba(X_te2)
y_pr2 = classifier.predict(X_te2)

# ========================= precision_recall_curve plotting =========================================================
pr, rc, thresholds = precision_recall_curve(y_te2, y_pr_rt2[:, 1])
print("roc_auc for class 1:", roc_auc_score(y_te2,y_pr_rt2[:, 1]))
# Figure 1: precision, recall vs thresholds

scikitplot.metrics.plot_precision_recall(y_te2, y_pr_rt2, plot_micro = False)

# Figure 2: ROC
scikitplot.metrics.plot_roc(y_te2, y_pr_rt2, plot_micro = False, plot_macro = False)


# ========================== classification report and Confusion matrix ============================================
print(classification_report(y_te2, y_pr2))

m_confusion_test = confusion_matrix(y_te2, y_pr2)
print(pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0', 'Predicted 1'],
            index = ['Actual 0', 'Actual 1']))


# ==================================================================================================================
## store four models
with open('RF_model/RF_count.pkl', 'wb') as f:
    pickle.dump((classifier), f)

with open('RF_model/RF_tfid.pkl', 'wb') as f:
    pickle.dump((classifier2), f)

plt.show()
