```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
```


```python
wine = load_wine()
wine.keys()
```




    dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names'])




```python
#Feature Data 지정하기
wine_data = wine.data
print(wine_data.shape)
```

    (178, 13)



```python
wine_data[0]
```




    array([1.423e+01, 1.710e+00, 2.430e+00, 1.560e+01, 1.270e+02, 2.800e+00,
           3.060e+00, 2.800e-01, 2.290e+00, 5.640e+00, 1.040e+00, 3.920e+00,
           1.065e+03])




```python
#라벨 저장
wine_label = wine.target
print(wine_label.shape)
```

    (178,)



```python
wine_label[0:20]
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])




```python
#Target Names 출력
wine.target_names
```




    array(['class_0', 'class_1', 'class_2'], dtype='<U7')




```python
#데이터 Describe 

print(wine.DESCR)
```

    .. _wine_dataset:
    
    Wine recognition dataset
    ------------------------
    
    **Data Set Characteristics:**
    
        :Number of Instances: 178 (50 in each of three classes)
        :Number of Attributes: 13 numeric, predictive attributes and the class
        :Attribute Information:
     		- Alcohol
     		- Malic acid
     		- Ash
    		- Alcalinity of ash  
     		- Magnesium
    		- Total phenols
     		- Flavanoids
     		- Nonflavanoid phenols
     		- Proanthocyanins
    		- Color intensity
     		- Hue
     		- OD280/OD315 of diluted wines
     		- Proline
    
        - class:
                - class_0
                - class_1
                - class_2
    		
        :Summary Statistics:
        
        ============================= ==== ===== ======= =====
                                       Min   Max   Mean     SD
        ============================= ==== ===== ======= =====
        Alcohol:                      11.0  14.8    13.0   0.8
        Malic Acid:                   0.74  5.80    2.34  1.12
        Ash:                          1.36  3.23    2.36  0.27
        Alcalinity of Ash:            10.6  30.0    19.5   3.3
        Magnesium:                    70.0 162.0    99.7  14.3
        Total Phenols:                0.98  3.88    2.29  0.63
        Flavanoids:                   0.34  5.08    2.03  1.00
        Nonflavanoid Phenols:         0.13  0.66    0.36  0.12
        Proanthocyanins:              0.41  3.58    1.59  0.57
        Colour Intensity:              1.3  13.0     5.1   2.3
        Hue:                          0.48  1.71    0.96  0.23
        OD280/OD315 of diluted wines: 1.27  4.00    2.61  0.71
        Proline:                       278  1680     746   315
        ============================= ==== ===== ======= =====
    
        :Missing Attribute Values: None
        :Class Distribution: class_0 (59), class_1 (71), class_2 (48)
        :Creator: R.A. Fisher
        :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
        :Date: July, 1988
    
    This is a copy of UCI ML Wine recognition datasets.
    https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data
    
    The data is the results of a chemical analysis of wines grown in the same
    region in Italy by three different cultivators. There are thirteen different
    measurements taken for different constituents found in the three types of
    wine.
    
    Original Owners: 
    
    Forina, M. et al, PARVUS - 
    An Extendible Package for Data Exploration, Classification and Correlation. 
    Institute of Pharmaceutical and Food Analysis and Technologies,
    Via Brigata Salerno, 16147 Genoa, Italy.
    
    Citation:
    
    Lichman, M. (2013). UCI Machine Learning Repository
    [https://archive.ics.uci.edu/ml]. Irvine, CA: University of California,
    School of Information and Computer Science. 
    
    .. topic:: References
    
      (1) S. Aeberhard, D. Coomans and O. de Vel, 
      Comparison of Classifiers in High Dimensional Settings, 
      Tech. Rep. no. 92-02, (1992), Dept. of Computer Science and Dept. of  
      Mathematics and Statistics, James Cook University of North Queensland. 
      (Also submitted to Technometrics). 
    
      The data was used with many others for comparing various 
      classifiers. The classes are separable, though only RDA 
      has achieved 100% correct classification. 
      (RDA : 100%, QDA 99.4%, LDA 98.9%, 1NN 96.1% (z-transformed data)) 
      (All results using the leave-one-out technique) 
    
      (2) S. Aeberhard, D. Coomans and O. de Vel, 
      "THE CLASSIFICATION PERFORMANCE OF RDA" 
      Tech. Rep. no. 92-01, (1992), Dept. of Computer Science and Dept. of 
      Mathematics and Statistics, James Cook University of North Queensland. 
      (Also submitted to Journal of Chemometrics).
    



```python
#X_train, X_test, y_train, y_test를 생성

X_train, X_test, y_train, y_test = train_test_split(wine_data, 
                                                    wine_label, 
                                                    test_size=0.2, 
                                                    random_state=32)
print('X_train 개수: ', len(X_train),', X_test 개수: ', len(X_test))
```

    X_train 개수:  142 , X_test 개수:  36



```python
X_train.shape, y_train.shape
```




    ((142, 13), (142,))




```python
y_train, y_test
```




    (array([2, 0, 2, 2, 0, 1, 2, 2, 0, 0, 0, 2, 0, 1, 1, 1, 1, 1, 0, 2, 0, 1,
            0, 0, 1, 1, 1, 0, 2, 0, 1, 1, 0, 1, 2, 0, 0, 1, 1, 1, 2, 0, 0, 2,
            1, 1, 1, 0, 0, 1, 1, 2, 2, 2, 2, 1, 2, 0, 1, 2, 2, 2, 2, 1, 2, 0,
            0, 2, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 2, 0, 1, 1, 0, 1, 0,
            0, 0, 2, 1, 1, 0, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 0, 2, 1, 1,
            2, 2, 1, 1, 0, 1, 2, 1, 0, 1, 2, 1, 2, 1, 0, 0, 0, 2, 1, 2, 2, 0,
            0, 0, 1, 1, 2, 1, 1, 0, 2, 0]),
     array([1, 1, 0, 2, 2, 0, 0, 2, 0, 0, 1, 2, 0, 0, 0, 2, 1, 1, 0, 0, 1, 2,
            2, 2, 1, 2, 0, 0, 0, 0, 1, 0, 2, 1, 0, 1]))




```python
#Decision Tree 사용
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(random_state=32)
print(decision_tree._estimator_type)

```

    classifier



```python
decision_tree.fit(X_train, y_train)
```




    DecisionTreeClassifier(random_state=32)




```python
#예측하기

y_pred = decision_tree.predict(X_test)
y_pred
```




    array([1, 1, 0, 1, 2, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 1, 1, 0, 0, 1, 2,
           2, 2, 1, 2, 0, 0, 0, 0, 2, 0, 2, 1, 0, 1])




```python
# 정확도 비교

decision_accuracy = accuracy_score(y_test, y_pred)
print('의사결정나무의 정확도 : ',decision_accuracy)

```

    의사결정나무의 정확도 :  0.9166666666666666



```python
## 분석결과 확인

decision_report = classification_report(y_test, y_pred)
print(decision_report)
```

                  precision    recall  f1-score   support
    
               0       0.94      1.00      0.97        16
               1       0.89      0.80      0.84        10
               2       0.90      0.90      0.90        10
    
        accuracy                           0.92        36
       macro avg       0.91      0.90      0.90        36
    weighted avg       0.92      0.92      0.91        36
    



```python
#Random Forest 사용

from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(wine_data, 
                                                    wine_label, 
                                                    test_size=0.2, 
                                                    random_state=32)

random_forest = RandomForestClassifier(random_state=32)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)

print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        16
               1       1.00      0.90      0.95        10
               2       0.91      1.00      0.95        10
    
        accuracy                           0.97        36
       macro avg       0.97      0.97      0.97        36
    weighted avg       0.97      0.97      0.97        36
    



```python
#예측하기

y_pred = random_forest.predict(X_test)
y_pred
```




    array([1, 1, 0, 2, 2, 0, 0, 2, 0, 0, 1, 2, 0, 0, 0, 2, 1, 1, 0, 0, 1, 2,
           2, 2, 1, 2, 0, 0, 0, 0, 2, 0, 2, 1, 0, 1])




```python
#정확도 비교
random_accuracy = accuracy_score(y_test, y_pred)
print('랜덤포레스트의 정확도 : ',random_accuracy)
```

    랜덤포레스트의 정확도 :  0.9722222222222222



```python
# 분석결과 확인

random_report = classification_report(y_test, y_pred)
print(random_report)

```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        16
               1       1.00      0.90      0.95        10
               2       0.91      1.00      0.95        10
    
        accuracy                           0.97        36
       macro avg       0.97      0.97      0.97        36
    weighted avg       0.97      0.97      0.97        36
    



```python
#SVM 사용

from sklearn import svm
svm_model = svm.SVC()

print(svm_model._estimator_type)

svm_model.fit(X_train, y_train)
```

    classifier





    SVC()




```python
#예측하기

y_pred = svm_model.predict(X_test)
y_pred
```




    array([1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1,
           1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1])




```python
#정확도 비교

svm_accuracy = accuracy_score(y_test, y_pred)
print('SVM의 정확도 : ',svm_accuracy)
```

    SVM의 정확도 :  0.6111111111111112



```python
# 분석결과 확인

svm_report = classification_report(y_test, y_pred)
print(svm_report)
```

                  precision    recall  f1-score   support
    
               0       0.93      0.81      0.87        16
               1       0.41      0.90      0.56        10
               2       0.00      0.00      0.00        10
    
        accuracy                           0.61        36
       macro avg       0.45      0.57      0.48        36
    weighted avg       0.53      0.61      0.54        36
    


    /opt/conda/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1308: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /opt/conda/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1308: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /opt/conda/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1308: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))



```python
#SGD Classifier 사용

from sklearn.linear_model import SGDClassifier
sgd_model = SGDClassifier()

print(sgd_model._estimator_type)

sgd_model.fit(X_train, y_train)
```

    classifier





    SGDClassifier()




```python
#예측하기

y_pred = sgd_model.predict(X_test)
y_pred 
```




    array([1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0,
           1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1])




```python
#정확도 비교

sgd_accuracy = accuracy_score(y_test, y_pred)
print('sgd의 정확도 : ',sgd_accuracy)
```

    sgd의 정확도 :  0.6388888888888888



```python
# 분석결과 확인

sgd_report = classification_report(y_test, y_pred)
print(sgd_report)
```

                  precision    recall  f1-score   support
    
               0       0.87      0.81      0.84        16
               1       0.48      1.00      0.65        10
               2       0.00      0.00      0.00        10
    
        accuracy                           0.64        36
       macro avg       0.45      0.60      0.49        36
    weighted avg       0.52      0.64      0.55        36
    


    /opt/conda/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1308: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /opt/conda/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1308: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /opt/conda/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1308: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))



```python
# Logistic Regression 사용

from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()

print(logistic_model._estimator_type)

```

    classifier



```python
logistic_model.fit(X_train, y_train)
```

    /opt/conda/lib/python3.9/site-packages/sklearn/linear_model/_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(





    LogisticRegression()




```python

```


```python
#예측하기

y_pred = logistic_model.predict(X_test)
y_pred
```




    array([1, 1, 0, 2, 2, 0, 0, 2, 0, 0, 1, 2, 0, 0, 0, 1, 0, 1, 0, 0, 1, 2,
           2, 2, 1, 2, 0, 0, 0, 0, 2, 0, 2, 1, 0, 1])




```python
#정확도 비교

logistic_accuracy = accuracy_score(y_test, y_pred)
print('LogisticRegression의 정확도 : ',logistic_accuracy)
```

    LogisticRegression의 정확도 :  0.9166666666666666



```python
# 분석결과 확인

logistic_report = classification_report(y_test, y_pred)
print(logistic_report)
```

                  precision    recall  f1-score   support
    
               0       0.94      1.00      0.97        16
               1       0.89      0.80      0.84        10
               2       0.90      0.90      0.90        10
    
        accuracy                           0.92        36
       macro avg       0.91      0.90      0.90        36
    weighted avg       0.92      0.92      0.91        36
    



```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
