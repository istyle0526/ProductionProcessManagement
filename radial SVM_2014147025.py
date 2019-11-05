import numpy as np
from sklearn.svm import SVC
import pandas as pd

traindata = pd.read_csv('./semicon_balanced_train.csv')
X_train = traindata.iloc[:, 1:]
Y_train = traindata.iloc[:, 0]

testdata = pd.read_csv('./semicon_balanced_test.csv')
X_test = testdata.iloc[:,1:]
Y_test = testdata.iloc[:,0]

svc = SVC(kernel='rbf')
svc = svc.fit(X_train,Y_train)
estimates = svc.predict(X_test)
estimates_2 = svc.predict(X_train)

from sklearn.metrics import confusion_matrix

C=confusion_matrix(Y_test,estimates)
TN, FP, FN, TP = C.ravel()

Accuracy= svc.score(X_test,Y_test)
Precision=float(TP/(TP+FP))
Recall=float(TP/(TP+FN))
Specificity=float(TN/(TN+FP))
F1measure=float(2*Precision*Recall/(Precision+Recall))
Gmean=float(np.sqrt(Precision*Recall))

print("\n"
      "\n"
      "This solution is computed using test data")
print(C)
print("Accuracy using train data is: %.3f"%(Accuracy))
print("Precision : %.3f, Recall : %.3f, Specificity : %.3f, F1measure : %.3f, G-mean : %.3f" %(Precision, Recall, Specificity, F1measure, Gmean))
print("Type 1 error : %.3f, Type 2 error : %.3f"%(1-Specificity, 1-Recall))

C_2 =confusion_matrix(Y_train,estimates_2)
TN_2, FP_2, FN_2, TP_2 = C_2.ravel()

accuracy_2=float(svc.score(X_train,Y_train))
Precision_2=float(TP_2/(TP_2+FP_2))
Recall_2=float(TP_2/(TP_2+FN_2))
Specificity_2=float(TN_2/(TN_2+FP_2))
F1measure_2=float(2*Precision_2*Recall_2/(Precision_2+Recall_2))
Gmean_2=float(np.sqrt(Precision_2*Recall_2))

print("\n"
      "\n"
      "This solution is computed using train data")
print(C_2)
print("Accuracy using train data is: %.3f"%(accuracy_2))
print("Precision : %.3f, Recall : %.3f, Specificity: %.3f, F1measure : %.3f, G-mean : %.3f" %(Precision_2, Recall_2, Specificity_2, F1measure_2, Gmean_2))
print("Type 1 error : %.3f, Type 2 error : %.3f"%(1-Specificity_2, 1-Recall_2))
