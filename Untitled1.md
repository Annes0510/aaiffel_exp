```python
def numbers():
    X=[]
    while True:
        number = input("Enter a number (<Enter key> to quit)")
        while number !="":
            try: 
                x = float(number)
                X.append(x) 
            except ValueError:
                print('>>> NOT a number! lgnored..')
            number = input("Enter a number (<Enter Key> to quit)")
        if len(X) > 1: 
            reterun X
X=numbers()

print('X :', X)
```


      File "/tmp/ipykernel_13/271302374.py", line 13
        reterun X
                ^
    SyntaxError: invalid syntax




```python
# 2개 이상의 숫자를 입력받아 리스트에 저장하는 함수
def numbers():
    X=[]    # X에 빈 리스트를 할당합니다.
    while True:
        number = input("Enter a number (<Enter key> to quit)") 
        while number !="":
            try:
                x = float(number)
                X.append(x)    # float형으로 변환한 숫자 입력을 리스트에 추가합니다.
            except ValueError:
                print('>>> NOT a number! Ignored..')
            number = input("Enter a number (<Enter key> to quit)")
        if len(X) > 1:  # 저장된 숫자가 2개 이상일 때만 리턴합니다.
            return X

X=numbers()

print('X :', X)
```


```python
total = 0.0
for i in range(len(X)):
    total = total + X[i]
mean = total / len(X)

print('sum of X: ', total)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    /tmp/ipykernel_14/3570914483.py in <module>
          1 total = 0.0
    ----> 2 for i in range(len(X)):
          3     total = total + X[i]
          4 mean = total / len(X)
          5 


    TypeError: object of type 'float' has no len()



```python
med = median(X)
avg = means(X)
std = std_dev(X, avg)
print("당신이 입력한 숫자{}의 ".format(X))
print("중앙값은{}, 평균은{}, 표준편차는{}입니다.".format(med, avg, std))
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /tmp/ipykernel_14/2527917235.py in <module>
    ----> 1 med = median(X)
          2 avg = means(X)
          3 std = std_dev(X, avg)
          4 print("당신이 입력한 숫자{}의 ".format(X))
          5 print("중앙값은{}, 평균은{}, 표준편차는{}입니다.".format(med, avg, std))


    NameError: name 'median' is not defined



```python

```


```python

```


```python

```
