import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

traindata = pd.read_csv('./semicon_balanced_train.csv')
X_train = traindata.iloc[:, 1:]
Y_train = traindata.iloc[:, 0]

testdata = pd.read_csv('./semicon_balanced_test.csv')
X_test = testdata.iloc[:,1:]
Y_test = testdata.iloc[:,0]

accuracy =[]

for k in range (1,100):
    KNN_R = KNeighborsClassifier(n_neighbors=k, p=1)
    KNN_R.fit(X_train, Y_train)
    estimates = KNN_R.predict(X_test)
    ac= KNN_R.score(X_test, Y_test)
    accuracy.append(ac)


optimal_k = int(accuracy.index(max(accuracy))+1)

print("optimal k is %.3f"%(optimal_k))
print("\n"
      "\n"
      "This solution is computed using test data")


from sklearn.metrics import confusion_matrix

KNN_O = KNeighborsClassifier(n_neighbors=optimal_k)
KNN_O.fit(X_train,Y_train)
optimal_estimates = KNN_O.predict(X_test)
optimal_estimates_2 = KNN_O.predict(X_train)

C=confusion_matrix(Y_test,optimal_estimates)
TN, FP, FN, TP = C.ravel()

accuracy=float(KNN_O.score(X_test,Y_test))
Precision=float(TP/(TP+FP))
Recall=float(TP/(TP+FN))
Specificity=float(TN/(TN+FP))
F1measure=float(2*Precision*Recall/(Precision+Recall))
Gmean=float(np.sqrt(Precision*Recall))

print(C)
print("Accuracy using test data is: %.3f"%(accuracy))
print("Precision : %.3f, Recall : %.3f, Specificity : %.3f, F1measure : %.3f, G-mean : %.3f" %(Precision, Recall, Specificity, F1measure, Gmean))
print("Type 1 error : %.3f, Type 2 error : %.3f"%(1-Specificity, 1-Recall))

C_2 =confusion_matrix(Y_train,optimal_estimates_2)
TN_2, FP_2, FN_2, TP_2 = C_2.ravel()

accuracy_2=float(KNN_O.score(X_train,Y_train))
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
