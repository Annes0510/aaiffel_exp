```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
```


```python
digits = load_digits()
digits.keys()
```




    dict_keys(['data', 'target', 'frame', 'feature_names', 'target_names', 'images', 'DESCR'])




```python
#Feature Data 지정하기
digits_data = digits.data
digits_data.shape
```




    (1797, 64)




```python
digits_data[0]
```




    array([ 0.,  0.,  5., 13.,  9.,  1.,  0.,  0.,  0.,  0., 13., 15., 10.,
           15.,  5.,  0.,  0.,  3., 15.,  2.,  0., 11.,  8.,  0.,  0.,  4.,
           12.,  0.,  0.,  8.,  8.,  0.,  0.,  5.,  8.,  0.,  0.,  9.,  8.,
            0.,  0.,  4., 11.,  0.,  1., 12.,  7.,  0.,  0.,  2., 14.,  5.,
           10., 12.,  0.,  0.,  0.,  0.,  6., 13., 10.,  0.,  0.,  0.])




```python
#라벨 저장
digits_label = digits.target
print(digits_label.shape)
digits_label[:20]
```

    (1797,)





    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
new_label = [3 if i == 3 else 0 for i in digits_label]
new_label[:20]
```




    [0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0]




```python
#Target Names 출력

digits.target_names
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
#데이터 Describe 

print(digits.DESCR)
```

    .. _digits_dataset:
    
    Optical recognition of handwritten digits dataset
    --------------------------------------------------
    
    **Data Set Characteristics:**
    
        :Number of Instances: 1797
        :Number of Attributes: 64
        :Attribute Information: 8x8 image of integer pixels in the range 0..16.
        :Missing Attribute Values: None
        :Creator: E. Alpaydin (alpaydin '@' boun.edu.tr)
        :Date: July; 1998
    
    This is a copy of the test set of the UCI ML hand-written digits datasets
    https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits
    
    The data set contains images of hand-written digits: 10 classes where
    each class refers to a digit.
    
    Preprocessing programs made available by NIST were used to extract
    normalized bitmaps of handwritten digits from a preprinted form. From a
    total of 43 people, 30 contributed to the training set and different 13
    to the test set. 32x32 bitmaps are divided into nonoverlapping blocks of
    4x4 and the number of on pixels are counted in each block. This generates
    an input matrix of 8x8 where each element is an integer in the range
    0..16. This reduces dimensionality and gives invariance to small
    distortions.
    
    For info on NIST preprocessing routines, see M. D. Garris, J. L. Blue, G.
    T. Candela, D. L. Dimmick, J. Geist, P. J. Grother, S. A. Janet, and C.
    L. Wilson, NIST Form-Based Handprint Recognition System, NISTIR 5469,
    1994.
    
    .. topic:: References
    
      - C. Kaynak (1995) Methods of Combining Multiple Classifiers and Their
        Applications to Handwritten Digit Recognition, MSc Thesis, Institute of
        Graduate Studies in Science and Engineering, Bogazici University.
      - E. Alpaydin, C. Kaynak (1998) Cascading Classifiers, Kybernetika.
      - Ken Tang and Ponnuthurai N. Suganthan and Xi Yao and A. Kai Qin.
        Linear dimensionalityreduction using relevance weighted LDA. School of
        Electrical and Electronic Engineering Nanyang Technological University.
        2005.
      - Claudio Gentile. A New Approximate Maximal Margin Classification
        Algorithm. NIPS. 2000.
    



```python
#X_train, X_test, y_train, y_test를 생성

X_train, X_test, y_train, y_test = train_test_split(digits_data, 
                                                    digits_label, 
                                                    test_size=0.2, 
                                                    random_state=32)
print('X_train 개수: ', len(X_train),', X_test 개수: ', len(X_test))
```

    X_train 개수:  1437 , X_test 개수:  360



```python
X_train.shape, y_train.shape
```




    ((1437, 64), (1437,))




```python
y_train, y_test
```




    (array([7, 4, 0, ..., 1, 0, 7]),
     array([3, 9, 4, 4, 7, 8, 8, 6, 1, 1, 8, 0, 5, 8, 3, 0, 6, 3, 9, 7, 3, 2,
            2, 7, 4, 4, 9, 9, 5, 1, 0, 3, 0, 3, 1, 4, 5, 9, 6, 5, 6, 2, 6, 9,
            1, 6, 8, 8, 5, 5, 5, 3, 7, 7, 5, 1, 6, 3, 2, 9, 6, 3, 3, 1, 8, 4,
            3, 4, 3, 6, 4, 4, 0, 7, 7, 8, 0, 6, 3, 6, 3, 0, 0, 7, 3, 0, 7, 3,
            5, 6, 3, 6, 1, 6, 0, 8, 9, 3, 0, 5, 9, 0, 4, 2, 9, 3, 8, 3, 9, 3,
            3, 6, 3, 1, 0, 0, 5, 5, 2, 7, 0, 0, 7, 7, 1, 5, 7, 8, 0, 3, 0, 7,
            3, 3, 9, 9, 0, 8, 3, 2, 1, 8, 2, 0, 6, 5, 3, 1, 9, 0, 6, 6, 3, 7,
            4, 8, 8, 0, 5, 1, 3, 0, 7, 0, 3, 9, 5, 8, 9, 9, 3, 3, 6, 2, 1, 2,
            2, 7, 0, 6, 9, 5, 6, 9, 4, 1, 9, 3, 4, 4, 3, 1, 2, 9, 0, 9, 5, 9,
            1, 5, 4, 4, 2, 0, 5, 7, 0, 1, 4, 6, 4, 1, 3, 9, 3, 2, 3, 8, 3, 1,
            5, 7, 2, 9, 2, 6, 4, 2, 5, 6, 5, 7, 6, 8, 9, 4, 0, 3, 2, 7, 7, 0,
            2, 4, 6, 3, 7, 4, 1, 5, 1, 8, 0, 7, 9, 5, 9, 6, 1, 3, 5, 1, 0, 2,
            5, 4, 8, 7, 3, 9, 4, 4, 5, 0, 3, 9, 7, 8, 2, 8, 2, 9, 1, 7, 7, 8,
            9, 1, 5, 1, 0, 3, 6, 4, 2, 0, 1, 5, 2, 1, 6, 1, 1, 5, 6, 1, 9, 3,
            5, 4, 7, 3, 3, 2, 8, 4, 5, 8, 3, 2, 5, 8, 2, 1, 0, 3, 8, 1, 9, 5,
            5, 7, 2, 2, 0, 6, 4, 3, 9, 7, 7, 7, 3, 2, 0, 7, 2, 3, 1, 1, 3, 6,
            6, 3, 2, 4, 6, 9, 3, 4]))




```python
#Decision Tree 사용
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(random_state=32)
print(decision_tree._estimator_type)

decision_tree.fit(X_train, y_train)
```

    classifier





    DecisionTreeClassifier(random_state=32)




```python
#예측하기

y_pred = decision_tree.predict(X_test)
y_pred
```




    array([3, 9, 4, 4, 7, 3, 8, 6, 1, 1, 1, 0, 5, 1, 3, 0, 6, 3, 9, 7, 9, 2,
           2, 7, 4, 4, 3, 9, 5, 1, 0, 3, 0, 9, 2, 4, 5, 9, 6, 5, 6, 5, 6, 9,
           1, 6, 8, 8, 5, 5, 5, 3, 7, 7, 5, 1, 6, 3, 2, 9, 6, 3, 3, 1, 8, 4,
           3, 7, 3, 6, 4, 4, 0, 7, 7, 8, 0, 6, 3, 6, 3, 8, 0, 7, 3, 0, 7, 3,
           5, 4, 3, 6, 1, 6, 0, 8, 9, 3, 5, 5, 9, 0, 4, 2, 9, 3, 8, 2, 9, 9,
           3, 6, 3, 1, 0, 0, 5, 5, 2, 7, 2, 0, 7, 7, 1, 5, 7, 8, 0, 3, 0, 7,
           3, 3, 9, 9, 0, 7, 1, 2, 1, 8, 2, 0, 6, 5, 3, 1, 9, 0, 6, 6, 3, 2,
           4, 8, 8, 0, 5, 1, 3, 0, 7, 0, 3, 5, 5, 8, 1, 9, 3, 3, 6, 2, 1, 2,
           2, 7, 0, 4, 1, 5, 6, 9, 4, 1, 9, 3, 4, 6, 2, 1, 2, 9, 0, 9, 5, 9,
           1, 9, 4, 4, 2, 0, 5, 7, 0, 1, 4, 6, 4, 1, 3, 9, 3, 2, 9, 1, 3, 1,
           5, 9, 1, 9, 2, 6, 4, 2, 5, 6, 5, 7, 6, 8, 3, 4, 0, 3, 2, 7, 7, 0,
           2, 4, 6, 3, 7, 4, 4, 5, 2, 1, 0, 7, 9, 5, 9, 6, 1, 3, 5, 2, 0, 2,
           5, 4, 8, 7, 3, 9, 4, 4, 5, 0, 9, 9, 1, 8, 2, 8, 2, 9, 1, 7, 7, 8,
           9, 1, 5, 1, 2, 3, 6, 9, 1, 0, 1, 5, 2, 1, 6, 4, 1, 5, 6, 1, 9, 3,
           5, 4, 7, 3, 3, 2, 8, 4, 5, 8, 3, 8, 5, 8, 2, 1, 0, 3, 8, 1, 4, 5,
           5, 7, 2, 2, 0, 6, 4, 3, 9, 3, 7, 7, 3, 8, 0, 7, 2, 3, 2, 1, 3, 6,
           6, 3, 2, 4, 6, 9, 3, 4])




```python
# 정확도 비교

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
accuracy
```




    0.875




```python
## 분석결과 확인

decision_report = classification_report(y_test, y_pred)
print(decision_report)
```

                  precision    recall  f1-score   support
    
               0       1.00      0.89      0.94        38
               1       0.75      0.83      0.79        36
               2       0.75      0.84      0.79        32
               3       0.92      0.86      0.89        56
               4       0.85      0.90      0.88        31
               5       0.92      0.97      0.95        36
               6       0.97      0.94      0.96        34
               7       0.94      0.88      0.91        34
               8       0.88      0.78      0.82        27
               9       0.79      0.83      0.81        36
    
        accuracy                           0.88       360
       macro avg       0.88      0.87      0.87       360
    weighted avg       0.88      0.88      0.88       360
    



```python
#Random Forest 사용

from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(digits_data, 
                                                    digits_label, 
                                                    test_size=0.2, 
                                                    random_state=32)

random_forest = RandomForestClassifier(random_state=32)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)

print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        38
               1       0.97      1.00      0.99        36
               2       0.97      1.00      0.98        32
               3       1.00      1.00      1.00        56
               4       1.00      0.97      0.98        31
               5       1.00      0.97      0.99        36
               6       1.00      1.00      1.00        34
               7       0.97      1.00      0.99        34
               8       0.96      0.89      0.92        27
               9       0.97      1.00      0.99        36
    
        accuracy                           0.99       360
       macro avg       0.98      0.98      0.98       360
    weighted avg       0.99      0.99      0.99       360
    



```python
#예측하기

y_pred = random_forest.predict(X_test)
y_pred
```




    array([3, 9, 4, 4, 7, 2, 8, 6, 1, 1, 8, 0, 5, 8, 3, 0, 6, 3, 9, 7, 3, 2,
           2, 7, 4, 4, 9, 9, 5, 1, 0, 3, 0, 3, 1, 4, 5, 9, 6, 5, 6, 2, 6, 9,
           1, 6, 8, 8, 5, 5, 5, 3, 7, 7, 5, 1, 6, 3, 2, 9, 6, 3, 3, 1, 8, 4,
           3, 4, 3, 6, 4, 4, 0, 7, 7, 8, 0, 6, 3, 6, 3, 0, 0, 7, 3, 0, 7, 3,
           5, 6, 3, 6, 1, 6, 0, 8, 9, 3, 0, 5, 9, 0, 4, 2, 9, 3, 8, 3, 9, 3,
           3, 6, 3, 1, 0, 0, 5, 5, 2, 7, 0, 0, 7, 7, 1, 5, 7, 8, 0, 3, 0, 7,
           3, 3, 9, 9, 0, 7, 3, 2, 1, 8, 2, 0, 6, 5, 3, 1, 9, 0, 6, 6, 3, 7,
           4, 8, 8, 0, 5, 1, 3, 0, 7, 0, 3, 9, 5, 8, 9, 9, 3, 3, 6, 2, 1, 2,
           2, 7, 0, 6, 9, 5, 6, 9, 4, 1, 9, 3, 8, 4, 3, 1, 2, 9, 0, 9, 5, 9,
           1, 5, 4, 4, 2, 0, 5, 7, 0, 1, 4, 6, 4, 1, 3, 9, 3, 2, 3, 1, 3, 1,
           9, 7, 2, 9, 2, 6, 4, 2, 5, 6, 5, 7, 6, 8, 9, 4, 0, 3, 2, 7, 7, 0,
           2, 4, 6, 3, 7, 4, 1, 5, 1, 8, 0, 7, 9, 5, 9, 6, 1, 3, 5, 1, 0, 2,
           5, 4, 8, 7, 3, 9, 4, 4, 5, 0, 3, 9, 7, 8, 2, 8, 2, 9, 1, 7, 7, 8,
           9, 1, 5, 1, 0, 3, 6, 4, 2, 0, 1, 5, 2, 1, 6, 1, 1, 5, 6, 1, 9, 3,
           5, 4, 7, 3, 3, 2, 8, 4, 5, 8, 3, 2, 5, 8, 2, 1, 0, 3, 8, 1, 9, 5,
           5, 7, 2, 2, 0, 6, 4, 3, 9, 7, 7, 7, 3, 2, 0, 7, 2, 3, 1, 1, 3, 6,
           6, 3, 2, 4, 6, 9, 3, 4])




```python
#정확도 비교
random_accuracy = accuracy_score(y_test, y_pred)
print('랜덤포레스트의 정확도 : ',random_accuracy)
```

    랜덤포레스트의 정확도 :  0.9861111111111112



```python
# 분석결과 확인

random_report = classification_report(y_test, y_pred)
print(random_report)

```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        38
               1       0.97      1.00      0.99        36
               2       0.97      1.00      0.98        32
               3       1.00      1.00      1.00        56
               4       1.00      0.97      0.98        31
               5       1.00      0.97      0.99        36
               6       1.00      1.00      1.00        34
               7       0.97      1.00      0.99        34
               8       0.96      0.89      0.92        27
               9       0.97      1.00      0.99        36
    
        accuracy                           0.99       360
       macro avg       0.98      0.98      0.98       360
    weighted avg       0.99      0.99      0.99       360
    



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




    array([3, 9, 4, 4, 7, 8, 8, 6, 1, 1, 8, 0, 5, 8, 3, 0, 6, 3, 9, 7, 3, 2,
           2, 7, 4, 4, 9, 9, 5, 1, 0, 3, 0, 3, 1, 4, 5, 9, 6, 5, 6, 2, 6, 9,
           1, 6, 8, 8, 5, 5, 5, 3, 7, 7, 5, 1, 6, 3, 2, 9, 6, 3, 3, 1, 8, 4,
           3, 4, 3, 6, 4, 4, 0, 7, 7, 8, 0, 6, 3, 6, 3, 0, 0, 7, 3, 0, 7, 3,
           5, 6, 3, 6, 1, 6, 0, 8, 9, 3, 0, 5, 9, 0, 4, 2, 9, 3, 8, 3, 9, 3,
           3, 6, 3, 1, 0, 0, 5, 5, 2, 7, 0, 0, 7, 7, 1, 5, 7, 8, 0, 3, 0, 7,
           3, 3, 9, 9, 0, 8, 3, 2, 1, 8, 2, 0, 6, 5, 3, 1, 9, 0, 6, 6, 3, 7,
           4, 8, 8, 0, 5, 1, 3, 0, 7, 0, 3, 9, 5, 8, 9, 9, 3, 3, 6, 2, 1, 2,
           2, 7, 0, 6, 9, 5, 6, 9, 4, 1, 9, 3, 9, 4, 3, 1, 2, 9, 0, 9, 5, 9,
           1, 5, 4, 4, 2, 0, 5, 7, 0, 1, 4, 6, 4, 1, 3, 9, 3, 2, 3, 8, 3, 1,
           9, 7, 2, 9, 2, 6, 4, 2, 5, 6, 5, 7, 6, 8, 9, 4, 0, 3, 2, 7, 7, 0,
           2, 4, 6, 3, 7, 4, 1, 5, 1, 8, 0, 7, 9, 5, 9, 6, 1, 3, 5, 1, 0, 2,
           5, 4, 8, 7, 3, 9, 4, 4, 5, 0, 3, 9, 7, 8, 2, 8, 2, 9, 1, 7, 7, 8,
           9, 1, 5, 1, 0, 3, 6, 4, 2, 0, 1, 5, 2, 1, 6, 1, 1, 5, 6, 1, 9, 3,
           5, 4, 7, 3, 3, 2, 8, 4, 5, 8, 3, 2, 5, 8, 2, 1, 0, 3, 8, 1, 9, 5,
           5, 7, 2, 2, 0, 6, 4, 3, 9, 7, 7, 7, 3, 2, 0, 7, 2, 3, 1, 1, 3, 6,
           6, 3, 2, 4, 6, 9, 3, 4])




```python
#정확도 비교

svm_accuracy = accuracy_score(y_test, y_pred)
print('SVM의 정확도 : ',svm_accuracy)
```

    SVM의 정확도 :  0.9944444444444445



```python
# 분석결과 확인

svm_report = classification_report(y_test, y_pred)
print(svm_report)
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        38
               1       1.00      1.00      1.00        36
               2       1.00      1.00      1.00        32
               3       1.00      1.00      1.00        56
               4       1.00      0.97      0.98        31
               5       1.00      0.97      0.99        36
               6       1.00      1.00      1.00        34
               7       1.00      1.00      1.00        34
               8       1.00      1.00      1.00        27
               9       0.95      1.00      0.97        36
    
        accuracy                           0.99       360
       macro avg       0.99      0.99      0.99       360
    weighted avg       0.99      0.99      0.99       360
    



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




    array([3, 9, 4, 4, 7, 8, 8, 6, 1, 1, 8, 0, 5, 8, 3, 0, 6, 3, 9, 7, 3, 2,
           2, 7, 4, 4, 9, 9, 5, 1, 0, 3, 0, 3, 1, 4, 5, 9, 6, 5, 6, 2, 6, 9,
           1, 6, 8, 8, 5, 5, 5, 3, 7, 7, 5, 1, 6, 3, 2, 9, 6, 3, 3, 1, 8, 4,
           3, 4, 3, 6, 4, 4, 5, 7, 7, 8, 0, 6, 3, 6, 3, 0, 0, 7, 3, 0, 7, 3,
           5, 6, 3, 6, 1, 6, 0, 3, 9, 3, 0, 5, 9, 0, 4, 2, 9, 3, 8, 3, 9, 3,
           3, 6, 3, 9, 0, 0, 5, 5, 2, 7, 0, 0, 7, 7, 1, 1, 7, 8, 0, 3, 0, 7,
           3, 3, 9, 9, 0, 8, 3, 2, 1, 8, 2, 0, 6, 5, 3, 1, 9, 0, 6, 6, 3, 7,
           4, 8, 8, 0, 5, 1, 3, 0, 7, 0, 3, 9, 5, 8, 9, 9, 3, 3, 6, 2, 1, 2,
           2, 7, 0, 6, 9, 5, 6, 9, 4, 3, 9, 3, 9, 4, 3, 1, 2, 9, 0, 9, 5, 9,
           1, 5, 4, 4, 2, 0, 5, 7, 0, 1, 4, 6, 4, 1, 3, 9, 3, 2, 3, 8, 3, 1,
           9, 7, 2, 9, 2, 6, 4, 2, 5, 6, 5, 7, 6, 8, 9, 4, 0, 3, 2, 7, 7, 0,
           2, 4, 6, 3, 7, 4, 1, 5, 1, 1, 0, 7, 9, 5, 9, 6, 1, 3, 5, 1, 0, 2,
           5, 4, 8, 7, 3, 9, 4, 4, 5, 0, 3, 9, 7, 8, 2, 8, 2, 9, 1, 7, 7, 8,
           9, 1, 5, 1, 0, 3, 6, 4, 2, 0, 1, 5, 2, 1, 6, 1, 1, 5, 6, 1, 9, 3,
           5, 4, 7, 3, 3, 2, 8, 4, 5, 8, 3, 2, 5, 8, 2, 1, 0, 3, 3, 1, 9, 5,
           5, 7, 2, 2, 0, 6, 4, 3, 9, 7, 7, 7, 3, 2, 0, 7, 2, 3, 1, 1, 3, 6,
           6, 3, 2, 4, 6, 9, 3, 4])




```python
#정확도 비교

sgd_accuracy = accuracy_score(y_test, y_pred)
print('sgd의 정확도 : ',sgd_accuracy)
```

    sgd의 정확도 :  0.975



```python
# 분석결과 확인

sgd_report = classification_report(y_test, y_pred)
print(sgd_report)
```

                  precision    recall  f1-score   support
    
               0       1.00      0.97      0.99        38
               1       0.94      0.94      0.94        36
               2       1.00      1.00      1.00        32
               3       0.95      1.00      0.97        56
               4       1.00      0.97      0.98        31
               5       0.97      0.94      0.96        36
               6       1.00      1.00      1.00        34
               7       1.00      1.00      1.00        34
               8       1.00      0.89      0.94        27
               9       0.92      1.00      0.96        36
    
        accuracy                           0.97       360
       macro avg       0.98      0.97      0.97       360
    weighted avg       0.98      0.97      0.97       360
    



```python
# Logistic Regression 사용

from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()

print(logistic_model._estimator_type)

logistic_model.fit(X_train, y_train)
```

    classifier


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




    array([3, 9, 4, 4, 7, 8, 8, 6, 1, 1, 8, 0, 5, 8, 3, 0, 6, 3, 9, 7, 3, 2,
           2, 7, 4, 4, 9, 9, 5, 1, 0, 3, 0, 3, 1, 4, 5, 9, 6, 5, 6, 2, 6, 9,
           1, 6, 8, 8, 5, 5, 5, 3, 7, 7, 5, 1, 6, 3, 2, 9, 6, 3, 3, 1, 8, 4,
           3, 4, 3, 6, 4, 4, 5, 7, 7, 8, 0, 6, 3, 6, 3, 0, 0, 7, 3, 0, 7, 3,
           5, 6, 3, 6, 1, 6, 0, 8, 9, 3, 0, 5, 9, 0, 4, 2, 9, 3, 8, 3, 9, 3,
           3, 6, 3, 9, 0, 0, 5, 5, 2, 7, 0, 5, 7, 7, 1, 1, 7, 8, 0, 3, 0, 7,
           3, 3, 9, 9, 0, 8, 3, 2, 1, 8, 2, 0, 6, 9, 3, 1, 9, 0, 6, 6, 9, 7,
           4, 8, 8, 0, 5, 1, 3, 0, 7, 0, 3, 9, 5, 8, 9, 9, 3, 3, 6, 2, 1, 2,
           2, 9, 0, 6, 9, 5, 6, 9, 4, 1, 9, 3, 8, 4, 3, 1, 2, 9, 0, 9, 5, 9,
           1, 5, 4, 4, 2, 0, 5, 7, 0, 1, 4, 6, 4, 1, 3, 9, 3, 2, 3, 8, 3, 1,
           9, 7, 2, 9, 2, 6, 4, 2, 5, 6, 5, 7, 6, 8, 9, 4, 0, 3, 2, 7, 7, 0,
           2, 4, 6, 3, 7, 4, 1, 5, 1, 8, 0, 7, 9, 5, 9, 6, 1, 3, 5, 1, 0, 2,
           5, 4, 8, 7, 3, 9, 4, 4, 5, 0, 3, 9, 7, 8, 2, 8, 2, 9, 1, 7, 7, 8,
           9, 1, 5, 1, 0, 3, 6, 4, 2, 0, 1, 5, 2, 1, 6, 1, 1, 5, 6, 1, 9, 3,
           5, 4, 7, 3, 3, 1, 8, 4, 5, 8, 3, 2, 5, 8, 2, 1, 0, 3, 8, 1, 9, 5,
           5, 7, 2, 2, 0, 6, 4, 3, 9, 7, 7, 7, 3, 2, 0, 7, 2, 3, 1, 1, 3, 6,
           6, 3, 2, 4, 6, 9, 3, 4])




```python
#정확도 비교

logistic_accuracy = accuracy_score(y_test, y_pred)
print('LogisticRegression의 정확도 : ',logistic_accuracy)
```

    LogisticRegression의 정확도 :  0.9722222222222222



```python
# 분석결과 확인

logistic_report = classification_report(y_test, y_pred)
print(logistic_report)
```

                  precision    recall  f1-score   support
    
               0       1.00      0.95      0.97        38
               1       0.95      0.97      0.96        36
               2       1.00      0.97      0.98        32
               3       1.00      0.98      0.99        56
               4       1.00      0.97      0.98        31
               5       0.94      0.92      0.93        36
               6       1.00      1.00      1.00        34
               7       1.00      0.97      0.99        34
               8       0.96      1.00      0.98        27
               9       0.88      1.00      0.94        36
    
        accuracy                           0.97       360
       macro avg       0.97      0.97      0.97       360
    weighted avg       0.97      0.97      0.97       360
    



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
