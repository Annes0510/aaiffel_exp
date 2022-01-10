```python
import os
import glob 
from PIL import Image
import matplotlib.pyplot as plt
print("PIL 라이브러리 import 완료!")
```

    PIL 라이브러리 import 완료!



```python

# 가위 이미지가 저장된 디렉토리 아래의 모든 jpg 파일을 읽어들여서
image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/scissor"
print("이미지 디렉토리 경로: ", image_dir_path)

images=glob.glob(image_dir_path + "/*.png")  

# 파일마다 모두 28x28 사이즈로 바꾸어 저장합니다.
target_size=(28,28)
for img in images:
    old_img=Image.open(img)
    new_img=old_img.resize(target_size,Image.ANTIALIAS)
    new_img.save(img,"PNG")

print("가위 이미지 resize 완료!")

# 바위 이미지가 저장된 디렉토리 아래의 모든 jpg 파일을 읽어들여서
image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/rock"
print("이미지 디렉토리 경로: ", image_dir_path)

images=glob.glob(image_dir_path + "/*.png")  

# 파일마다 모두 28x28 사이즈로 바꾸어 저장합니다.
target_size=(28,28)
for img in images:
    old_img=Image.open(img)
    new_img=old_img.resize(target_size,Image.ANTIALIAS)
    new_img.save(img,"PNG")

print("바위 이미지 resize 완료!")

# 보 이미지가 저장된 디렉토리 아래의 모든 jpg 파일을 읽어들여서
image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/paper"
print("이미지 디렉토리 경로: ", image_dir_path)

images=glob.glob(image_dir_path + "/*.png")  

# 파일마다 모두 28x28 사이즈로 바꾸어 저장합니다.
target_size=(28,28)
for img in images:
    old_img=Image.open(img)
    new_img=old_img.resize(target_size,Image.ANTIALIAS)
    new_img.save(img,"PNG")

print("보 이미지 resize 완료!")
```

    이미지 디렉토리 경로:  /aiffel/aiffel/rock_scissor_paper/scissor
    가위 이미지 resize 완료!
    이미지 디렉토리 경로:  /aiffel/aiffel/rock_scissor_paper/rock
    바위 이미지 resize 완료!
    이미지 디렉토리 경로:  /aiffel/aiffel/rock_scissor_paper/paper
    보 이미지 resize 완료!



```python
import os
import glob 
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np

def load_data(img_path):
    # 가위 : 0, 바위 : 1, 보 : 2
    number_of_data=2188   # 가위바위보 이미지 개수 총합에 주의하세요.
    img_size=28
    color=3
    #이미지 데이터와 라벨(가위 : 0, 바위 : 1, 보 : 2) 데이터를 담을 행렬(matrix) 영역을 생성합니다.
    imgs=np.zeros(number_of_data*img_size*img_size*color,dtype=np.int32).reshape(number_of_data,img_size,img_size,color)
    labels=np.zeros(number_of_data,dtype=np.int32)

    idx=2187
    for file in glob.iglob(img_path+'/scissor/*.png'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=0   # 가위 : 0
        idx=idx+1

    for file in glob.iglob(img_path+'/rock/*.png'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=1   # 바위 : 1
        idx=idx+1       
    
    for file in glob.iglob(img_path+'/paper/*.png'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=2   # 보 : 2
        idx=idx+1
        
    print("학습데이터(x_train)의 이미지 개수는",  idx,"입니다.")
    return imgs, labels

image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper"
(x_train, y_train)=load_data(image_dir_path)
x_train_norm = x_train/255.0   # 입력은 0~1 사이의 값으로 정규화

print("x_train shape: {}".format(x_train.shape))
print("y_train shape: {}".format(y_train.shape))
```

    학습데이터(x_train)의 이미지 개수는 2187 입니다.
    x_train shape: (2188, 28, 28, 3)
    y_train shape: (2188,)



```python
kernel_regularizer = keras.regularizers.L1(0.001)

model=keras.models.Sequential()
model.add(keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28,28,3)))

model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(32, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))


print('Model에 추가된 Layer 개수: ', len(model.layers))
```

    Model에 추가된 Layer 개수:  7



```python
model.summary()
```

    Model: "sequential_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_6 (Conv2D)            (None, 26, 26, 16)        448       
    _________________________________________________________________
    max_pooling2d_6 (MaxPooling2 (None, 13, 13, 16)        0         
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 11, 11, 32)        4640      
    _________________________________________________________________
    max_pooling2d_7 (MaxPooling2 (None, 5, 5, 32)          0         
    _________________________________________________________________
    flatten_3 (Flatten)          (None, 800)               0         
    _________________________________________________________________
    dense_6 (Dense)              (None, 32)                25632     
    _________________________________________________________________
    dense_7 (Dense)              (None, 10)                330       
    =================================================================
    Total params: 31,050
    Trainable params: 31,050
    Non-trainable params: 0
    _________________________________________________________________



```python

```


```python
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1000)
```

    Epoch 1/1000
    69/69 [==============================] - 3s 4ms/step - loss: 2.2423 - accuracy: 1.0000
    Epoch 2/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.1206 - accuracy: 1.0000
    Epoch 3/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.0018 - accuracy: 1.0000
    Epoch 4/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.8862 - accuracy: 1.0000
    Epoch 5/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.7742 - accuracy: 1.0000
    Epoch 6/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.6659 - accuracy: 1.0000
    Epoch 7/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.5617 - accuracy: 1.0000
    Epoch 8/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.4617 - accuracy: 1.0000
    Epoch 9/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.3661 - accuracy: 1.0000
    Epoch 10/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.2752 - accuracy: 1.0000
    Epoch 11/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.1889 - accuracy: 1.0000
    Epoch 12/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.1074 - accuracy: 1.0000
    Epoch 13/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.0307 - accuracy: 1.0000
    Epoch 14/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.9587 - accuracy: 1.0000
    Epoch 15/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.8914 - accuracy: 1.0000
    Epoch 16/1000
    69/69 [==============================] - 0s 5ms/step - loss: 0.8286 - accuracy: 1.0000
    Epoch 17/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.7703 - accuracy: 1.0000
    Epoch 18/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.7161 - accuracy: 1.0000
    Epoch 19/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.6659 - accuracy: 1.0000
    Epoch 20/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.6195 - accuracy: 1.0000
    Epoch 21/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.5767 - accuracy: 1.0000
    Epoch 22/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.5371 - accuracy: 1.0000
    Epoch 23/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.5007 - accuracy: 1.0000
    Epoch 24/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.4671 - accuracy: 1.0000
    Epoch 25/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.4362 - accuracy: 1.0000
    Epoch 26/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.4076 - accuracy: 1.0000
    Epoch 27/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.3814 - accuracy: 1.0000
    Epoch 28/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.3571 - accuracy: 1.0000
    Epoch 29/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.3347 - accuracy: 1.0000
    Epoch 30/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.3141 - accuracy: 1.0000
    Epoch 31/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.2950 - accuracy: 1.0000
    Epoch 32/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.2774 - accuracy: 1.0000
    Epoch 33/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.2610 - accuracy: 1.0000
    Epoch 34/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.2459 - accuracy: 1.0000
    Epoch 35/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.2318 - accuracy: 1.0000
    Epoch 36/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.2188 - accuracy: 1.0000
    Epoch 37/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.2067 - accuracy: 1.0000
    Epoch 38/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.1954 - accuracy: 1.0000
    Epoch 39/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.1849 - accuracy: 1.0000
    Epoch 40/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.1751 - accuracy: 1.0000
    Epoch 41/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.1660 - accuracy: 1.0000
    Epoch 42/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.1574 - accuracy: 1.0000
    Epoch 43/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.1494 - accuracy: 1.0000
    Epoch 44/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.1419 - accuracy: 1.0000
    Epoch 45/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.1349 - accuracy: 1.0000
    Epoch 46/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.1283 - accuracy: 1.0000
    Epoch 47/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.1221 - accuracy: 1.0000
    Epoch 48/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.1163 - accuracy: 1.0000
    Epoch 49/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.1108 - accuracy: 1.0000
    Epoch 50/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.1056 - accuracy: 1.0000
    Epoch 51/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.1008 - accuracy: 1.0000
    Epoch 52/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0962 - accuracy: 1.0000
    Epoch 53/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0918 - accuracy: 1.0000
    Epoch 54/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0878 - accuracy: 1.0000
    Epoch 55/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0839 - accuracy: 1.0000
    Epoch 56/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0802 - accuracy: 1.0000
    Epoch 57/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0767 - accuracy: 1.0000
    Epoch 58/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0734 - accuracy: 1.0000
    Epoch 59/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0703 - accuracy: 1.0000
    Epoch 60/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0673 - accuracy: 1.0000
    Epoch 61/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0645 - accuracy: 1.0000
    Epoch 62/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0619 - accuracy: 1.0000
    Epoch 63/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0593 - accuracy: 1.0000
    Epoch 64/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0569 - accuracy: 1.0000
    Epoch 65/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0546 - accuracy: 1.0000
    Epoch 66/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0524 - accuracy: 1.0000
    Epoch 67/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0503 - accuracy: 1.0000
    Epoch 68/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0483 - accuracy: 1.0000
    Epoch 69/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0464 - accuracy: 1.0000
    Epoch 70/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0446 - accuracy: 1.0000
    Epoch 71/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0428 - accuracy: 1.0000
    Epoch 72/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0412 - accuracy: 1.0000
    Epoch 73/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0396 - accuracy: 1.0000
    Epoch 74/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0381 - accuracy: 1.0000
    Epoch 75/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0366 - accuracy: 1.0000
    Epoch 76/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0352 - accuracy: 1.0000
    Epoch 77/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0339 - accuracy: 1.0000
    Epoch 78/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0326 - accuracy: 1.0000
    Epoch 79/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0314 - accuracy: 1.0000
    Epoch 80/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0303 - accuracy: 1.0000
    Epoch 81/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0291 - accuracy: 1.0000
    Epoch 82/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0281 - accuracy: 1.0000
    Epoch 83/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0270 - accuracy: 1.0000
    Epoch 84/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0260 - accuracy: 1.0000
    Epoch 85/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0251 - accuracy: 1.0000
    Epoch 86/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0242 - accuracy: 1.0000
    Epoch 87/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0233 - accuracy: 1.0000
    Epoch 88/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0225 - accuracy: 1.0000
    Epoch 89/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0217 - accuracy: 1.0000
    Epoch 90/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0209 - accuracy: 1.0000
    Epoch 91/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0202 - accuracy: 1.0000
    Epoch 92/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0194 - accuracy: 1.0000
    Epoch 93/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0188 - accuracy: 1.0000
    Epoch 94/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0181 - accuracy: 1.0000
    Epoch 95/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0175 - accuracy: 1.0000
    Epoch 96/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0168 - accuracy: 1.0000
    Epoch 97/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0163 - accuracy: 1.0000
    Epoch 98/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0157 - accuracy: 1.0000
    Epoch 99/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0151 - accuracy: 1.0000
    Epoch 100/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0146 - accuracy: 1.0000
    Epoch 101/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0141 - accuracy: 1.0000
    Epoch 102/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0136 - accuracy: 1.0000
    Epoch 103/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0132 - accuracy: 1.0000
    Epoch 104/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0127 - accuracy: 1.0000
    Epoch 105/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0123 - accuracy: 1.0000
    Epoch 106/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0119 - accuracy: 1.0000
    Epoch 107/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0114 - accuracy: 1.0000
    Epoch 108/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0111 - accuracy: 1.0000
    Epoch 109/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0107 - accuracy: 1.0000
    Epoch 110/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0103 - accuracy: 1.0000
    Epoch 111/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0100 - accuracy: 1.0000
    Epoch 112/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0096 - accuracy: 1.0000
    Epoch 113/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0093 - accuracy: 1.0000
    Epoch 114/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0090 - accuracy: 1.0000
    Epoch 115/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0087 - accuracy: 1.0000
    Epoch 116/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0084 - accuracy: 1.0000
    Epoch 117/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0081 - accuracy: 1.0000
    Epoch 118/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0078 - accuracy: 1.0000
    Epoch 119/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0076 - accuracy: 1.0000
    Epoch 120/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0073 - accuracy: 1.0000
    Epoch 121/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0071 - accuracy: 1.0000
    Epoch 122/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0068 - accuracy: 1.0000
    Epoch 123/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0066 - accuracy: 1.0000
    Epoch 124/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0064 - accuracy: 1.0000
    Epoch 125/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0062 - accuracy: 1.0000
    Epoch 126/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0060 - accuracy: 1.0000
    Epoch 127/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0058 - accuracy: 1.0000
    Epoch 128/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0056 - accuracy: 1.0000
    Epoch 129/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0054 - accuracy: 1.0000
    Epoch 130/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0052 - accuracy: 1.0000
    Epoch 131/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0051 - accuracy: 1.0000
    Epoch 132/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0049 - accuracy: 1.0000
    Epoch 133/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0047 - accuracy: 1.0000
    Epoch 134/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0046 - accuracy: 1.0000
    Epoch 135/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0044 - accuracy: 1.0000
    Epoch 136/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0043 - accuracy: 1.0000
    Epoch 137/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0041 - accuracy: 1.0000
    Epoch 138/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0040 - accuracy: 1.0000
    Epoch 139/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0039 - accuracy: 1.0000
    Epoch 140/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0037 - accuracy: 1.0000
    Epoch 141/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0036 - accuracy: 1.0000
    Epoch 142/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0035 - accuracy: 1.0000
    Epoch 143/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0034 - accuracy: 1.0000
    Epoch 144/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0033 - accuracy: 1.0000
    Epoch 145/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0032 - accuracy: 1.0000
    Epoch 146/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0031 - accuracy: 1.0000
    Epoch 147/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0030 - accuracy: 1.0000
    Epoch 148/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0029 - accuracy: 1.0000
    Epoch 149/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0028 - accuracy: 1.0000
    Epoch 150/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0027 - accuracy: 1.0000
    Epoch 151/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0026 - accuracy: 1.0000
    Epoch 152/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0025 - accuracy: 1.0000
    Epoch 153/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0024 - accuracy: 1.0000
    Epoch 154/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0024 - accuracy: 1.0000
    Epoch 155/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0023 - accuracy: 1.0000
    Epoch 156/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0022 - accuracy: 1.0000
    Epoch 157/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0021 - accuracy: 1.0000
    Epoch 158/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0021 - accuracy: 1.0000
    Epoch 159/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0020 - accuracy: 1.0000
    Epoch 160/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0019 - accuracy: 1.0000
    Epoch 161/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0019 - accuracy: 1.0000
    Epoch 162/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0018 - accuracy: 1.0000
    Epoch 163/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0017 - accuracy: 1.0000
    Epoch 164/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0017 - accuracy: 1.0000
    Epoch 165/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0016 - accuracy: 1.0000
    Epoch 166/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0016 - accuracy: 1.0000
    Epoch 167/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0015 - accuracy: 1.0000
    Epoch 168/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0015 - accuracy: 1.0000
    Epoch 169/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0014 - accuracy: 1.0000
    Epoch 170/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0014 - accuracy: 1.0000
    Epoch 171/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0013 - accuracy: 1.0000
    Epoch 172/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0013 - accuracy: 1.0000
    Epoch 173/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0013 - accuracy: 1.0000
    Epoch 174/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0012 - accuracy: 1.0000
    Epoch 175/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0012 - accuracy: 1.0000
    Epoch 176/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0011 - accuracy: 1.0000
    Epoch 177/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0011 - accuracy: 1.0000
    Epoch 178/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0011 - accuracy: 1.0000
    Epoch 179/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0010 - accuracy: 1.0000
    Epoch 180/1000
    69/69 [==============================] - 0s 3ms/step - loss: 9.9847e-04 - accuracy: 1.0000
    Epoch 181/1000
    69/69 [==============================] - 0s 3ms/step - loss: 9.6622e-04 - accuracy: 1.0000
    Epoch 182/1000
    69/69 [==============================] - 0s 3ms/step - loss: 9.3503e-04 - accuracy: 1.0000
    Epoch 183/1000
    69/69 [==============================] - 0s 3ms/step - loss: 9.0485e-04 - accuracy: 1.0000
    Epoch 184/1000
    69/69 [==============================] - 0s 3ms/step - loss: 8.7568e-04 - accuracy: 1.0000
    Epoch 185/1000
    69/69 [==============================] - 0s 3ms/step - loss: 8.4744e-04 - accuracy: 1.0000
    Epoch 186/1000
    69/69 [==============================] - 0s 3ms/step - loss: 8.2014e-04 - accuracy: 1.0000
    Epoch 187/1000
    69/69 [==============================] - 0s 3ms/step - loss: 7.9371e-04 - accuracy: 1.0000
    Epoch 188/1000
    69/69 [==============================] - 0s 3ms/step - loss: 7.6814e-04 - accuracy: 1.0000
    Epoch 189/1000
    69/69 [==============================] - 0s 3ms/step - loss: 7.4339e-04 - accuracy: 1.0000
    Epoch 190/1000
    69/69 [==============================] - 0s 3ms/step - loss: 7.1942e-04 - accuracy: 1.0000
    Epoch 191/1000
    69/69 [==============================] - 0s 3ms/step - loss: 6.9624e-04 - accuracy: 1.0000
    Epoch 192/1000
    69/69 [==============================] - 0s 3ms/step - loss: 6.7379e-04 - accuracy: 1.0000
    Epoch 193/1000
    69/69 [==============================] - 0s 3ms/step - loss: 6.5210e-04 - accuracy: 1.0000
    Epoch 194/1000
    69/69 [==============================] - 0s 3ms/step - loss: 6.3107e-04 - accuracy: 1.0000
    Epoch 195/1000
    69/69 [==============================] - 0s 3ms/step - loss: 6.1074e-04 - accuracy: 1.0000
    Epoch 196/1000
    69/69 [==============================] - 0s 3ms/step - loss: 5.9107e-04 - accuracy: 1.0000
    Epoch 197/1000
    69/69 [==============================] - 0s 3ms/step - loss: 5.7203e-04 - accuracy: 1.0000
    Epoch 198/1000
    69/69 [==============================] - 0s 3ms/step - loss: 5.5366e-04 - accuracy: 1.0000
    Epoch 199/1000
    69/69 [==============================] - 0s 3ms/step - loss: 5.3583e-04 - accuracy: 1.0000
    Epoch 200/1000
    69/69 [==============================] - 0s 3ms/step - loss: 5.1861e-04 - accuracy: 1.0000
    Epoch 201/1000
    69/69 [==============================] - 0s 3ms/step - loss: 5.0193e-04 - accuracy: 1.0000
    Epoch 202/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.8578e-04 - accuracy: 1.0000
    Epoch 203/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.7015e-04 - accuracy: 1.0000
    Epoch 204/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.5502e-04 - accuracy: 1.0000
    Epoch 205/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.4039e-04 - accuracy: 1.0000
    Epoch 206/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.2622e-04 - accuracy: 1.0000
    Epoch 207/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.1251e-04 - accuracy: 1.0000
    Epoch 208/1000
    69/69 [==============================] - 0s 3ms/step - loss: 3.9923e-04 - accuracy: 1.0000
    Epoch 209/1000
    69/69 [==============================] - 0s 3ms/step - loss: 3.8640e-04 - accuracy: 1.0000
    Epoch 210/1000
    69/69 [==============================] - 0s 3ms/step - loss: 3.7395e-04 - accuracy: 1.0000
    Epoch 211/1000
    69/69 [==============================] - 0s 3ms/step - loss: 3.6194e-04 - accuracy: 1.0000
    Epoch 212/1000
    69/69 [==============================] - 0s 3ms/step - loss: 3.5030e-04 - accuracy: 1.0000
    Epoch 213/1000
    69/69 [==============================] - 0s 3ms/step - loss: 3.3905e-04 - accuracy: 1.0000
    Epoch 214/1000
    69/69 [==============================] - 0s 3ms/step - loss: 3.2814e-04 - accuracy: 1.0000
    Epoch 215/1000
    69/69 [==============================] - 0s 3ms/step - loss: 3.1761e-04 - accuracy: 1.0000
    Epoch 216/1000
    69/69 [==============================] - 0s 3ms/step - loss: 3.0738e-04 - accuracy: 1.0000
    Epoch 217/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.9752e-04 - accuracy: 1.0000
    Epoch 218/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.8793e-04 - accuracy: 1.0000
    Epoch 219/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.7869e-04 - accuracy: 1.0000
    Epoch 220/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.6974e-04 - accuracy: 1.0000
    Epoch 221/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.6107e-04 - accuracy: 1.0000
    Epoch 222/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.5269e-04 - accuracy: 1.0000
    Epoch 223/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.4458e-04 - accuracy: 1.0000
    Epoch 224/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.3673e-04 - accuracy: 1.0000
    Epoch 225/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.2912e-04 - accuracy: 1.0000
    Epoch 226/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.2177e-04 - accuracy: 1.0000
    Epoch 227/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.1465e-04 - accuracy: 1.0000
    Epoch 228/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.0776e-04 - accuracy: 1.0000
    Epoch 229/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.0109e-04 - accuracy: 1.0000
    Epoch 230/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.9462e-04 - accuracy: 1.0000
    Epoch 231/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.8836e-04 - accuracy: 1.0000
    Epoch 232/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.8232e-04 - accuracy: 1.0000
    Epoch 233/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.7647e-04 - accuracy: 1.0000
    Epoch 234/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.7080e-04 - accuracy: 1.0000
    Epoch 235/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.6533e-04 - accuracy: 1.0000
    Epoch 236/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.6002e-04 - accuracy: 1.0000
    Epoch 237/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.5488e-04 - accuracy: 1.0000
    Epoch 238/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.4991e-04 - accuracy: 1.0000
    Epoch 239/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.4510e-04 - accuracy: 1.0000
    Epoch 240/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.4044e-04 - accuracy: 1.0000
    Epoch 241/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.3593e-04 - accuracy: 1.0000
    Epoch 242/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.3158e-04 - accuracy: 1.0000
    Epoch 243/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.2735e-04 - accuracy: 1.0000
    Epoch 244/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.2325e-04 - accuracy: 1.0000
    Epoch 245/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.1931e-04 - accuracy: 1.0000
    Epoch 246/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.1548e-04 - accuracy: 1.0000
    Epoch 247/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.1179e-04 - accuracy: 1.0000
    Epoch 248/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.0820e-04 - accuracy: 1.0000
    Epoch 249/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.0472e-04 - accuracy: 1.0000
    Epoch 250/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.0136e-04 - accuracy: 1.0000
    Epoch 251/1000
    69/69 [==============================] - 0s 3ms/step - loss: 9.8108e-05 - accuracy: 1.0000
    Epoch 252/1000
    69/69 [==============================] - 0s 3ms/step - loss: 9.4984e-05 - accuracy: 1.0000
    Epoch 253/1000
    69/69 [==============================] - 0s 3ms/step - loss: 9.1927e-05 - accuracy: 1.0000
    Epoch 254/1000
    69/69 [==============================] - 0s 3ms/step - loss: 8.8969e-05 - accuracy: 1.0000
    Epoch 255/1000
    69/69 [==============================] - 0s 3ms/step - loss: 8.6107e-05 - accuracy: 1.0000
    Epoch 256/1000
    69/69 [==============================] - 0s 3ms/step - loss: 8.3362e-05 - accuracy: 1.0000
    Epoch 257/1000
    69/69 [==============================] - 0s 3ms/step - loss: 8.0680e-05 - accuracy: 1.0000
    Epoch 258/1000
    69/69 [==============================] - 0s 3ms/step - loss: 7.8102e-05 - accuracy: 1.0000
    Epoch 259/1000
    69/69 [==============================] - 0s 3ms/step - loss: 7.5582e-05 - accuracy: 1.0000
    Epoch 260/1000
    69/69 [==============================] - 0s 3ms/step - loss: 7.3158e-05 - accuracy: 1.0000
    Epoch 261/1000
    69/69 [==============================] - 0s 3ms/step - loss: 7.0802e-05 - accuracy: 1.0000
    Epoch 262/1000
    69/69 [==============================] - 0s 3ms/step - loss: 6.8546e-05 - accuracy: 1.0000
    Epoch 263/1000
    69/69 [==============================] - 0s 3ms/step - loss: 6.6340e-05 - accuracy: 1.0000
    Epoch 264/1000
    69/69 [==============================] - 0s 3ms/step - loss: 6.4215e-05 - accuracy: 1.0000
    Epoch 265/1000
    69/69 [==============================] - 0s 3ms/step - loss: 6.2157e-05 - accuracy: 1.0000
    Epoch 266/1000
    69/69 [==============================] - 0s 3ms/step - loss: 6.0182e-05 - accuracy: 1.0000
    Epoch 267/1000
    69/69 [==============================] - 0s 3ms/step - loss: 5.8236e-05 - accuracy: 1.0000
    Epoch 268/1000
    69/69 [==============================] - 0s 3ms/step - loss: 5.6378e-05 - accuracy: 1.0000
    Epoch 269/1000
    69/69 [==============================] - 0s 3ms/step - loss: 5.4549e-05 - accuracy: 1.0000
    Epoch 270/1000
    69/69 [==============================] - 0s 3ms/step - loss: 5.2825e-05 - accuracy: 1.0000
    Epoch 271/1000
    69/69 [==============================] - 0s 3ms/step - loss: 5.1123e-05 - accuracy: 1.0000
    Epoch 272/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.9481e-05 - accuracy: 1.0000
    Epoch 273/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.7884e-05 - accuracy: 1.0000
    Epoch 274/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.6367e-05 - accuracy: 1.0000
    Epoch 275/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.4878e-05 - accuracy: 1.0000
    Epoch 276/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.3439e-05 - accuracy: 1.0000
    Epoch 277/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.2043e-05 - accuracy: 1.0000
    Epoch 278/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.0698e-05 - accuracy: 1.0000
    Epoch 279/1000
    69/69 [==============================] - 0s 3ms/step - loss: 3.9381e-05 - accuracy: 1.0000
    Epoch 280/1000
    69/69 [==============================] - 0s 3ms/step - loss: 3.8124e-05 - accuracy: 1.0000
    Epoch 281/1000
    69/69 [==============================] - 0s 3ms/step - loss: 3.6917e-05 - accuracy: 1.0000
    Epoch 282/1000
    69/69 [==============================] - 0s 3ms/step - loss: 3.5741e-05 - accuracy: 1.0000
    Epoch 283/1000
    69/69 [==============================] - 0s 3ms/step - loss: 3.4575e-05 - accuracy: 1.0000
    Epoch 284/1000
    69/69 [==============================] - 0s 3ms/step - loss: 3.3476e-05 - accuracy: 1.0000
    Epoch 285/1000
    69/69 [==============================] - 0s 3ms/step - loss: 3.2408e-05 - accuracy: 1.0000
    Epoch 286/1000
    69/69 [==============================] - 0s 3ms/step - loss: 3.1369e-05 - accuracy: 1.0000
    Epoch 287/1000
    69/69 [==============================] - 0s 3ms/step - loss: 3.0345e-05 - accuracy: 1.0000
    Epoch 288/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.9375e-05 - accuracy: 1.0000
    Epoch 289/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.8444e-05 - accuracy: 1.0000
    Epoch 290/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.7571e-05 - accuracy: 1.0000
    Epoch 291/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.6630e-05 - accuracy: 1.0000
    Epoch 292/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.5809e-05 - accuracy: 1.0000
    Epoch 293/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.4958e-05 - accuracy: 1.0000
    Epoch 294/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.4191e-05 - accuracy: 1.0000
    Epoch 295/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.3399e-05 - accuracy: 1.0000
    Epoch 296/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.2670e-05 - accuracy: 1.0000
    Epoch 297/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.1897e-05 - accuracy: 1.0000
    Epoch 298/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.1240e-05 - accuracy: 1.0000
    Epoch 299/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.0545e-05 - accuracy: 1.0000
    Epoch 300/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.9917e-05 - accuracy: 1.0000
    Epoch 301/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.9232e-05 - accuracy: 1.0000
    Epoch 302/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.8612e-05 - accuracy: 1.0000
    Epoch 303/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.8055e-05 - accuracy: 1.0000
    Epoch 304/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.7440e-05 - accuracy: 1.0000
    Epoch 305/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.6933e-05 - accuracy: 1.0000
    Epoch 306/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.6337e-05 - accuracy: 1.0000
    Epoch 307/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.5874e-05 - accuracy: 1.0000
    Epoch 308/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.5313e-05 - accuracy: 1.0000
    Epoch 309/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.4878e-05 - accuracy: 1.0000
    Epoch 310/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.4376e-05 - accuracy: 1.0000
    Epoch 311/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.3947e-05 - accuracy: 1.0000
    Epoch 312/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.3433e-05 - accuracy: 1.0000
    Epoch 313/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.2993e-05 - accuracy: 1.0000
    Epoch 314/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.2656e-05 - accuracy: 1.0000
    Epoch 315/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.2244e-05 - accuracy: 1.0000
    Epoch 316/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.1802e-05 - accuracy: 1.0000
    Epoch 317/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.1457e-05 - accuracy: 1.0000
    Epoch 318/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.1084e-05 - accuracy: 1.0000
    Epoch 319/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.0729e-05 - accuracy: 1.0000
    Epoch 320/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.0391e-05 - accuracy: 1.0000
    Epoch 321/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.0106e-05 - accuracy: 1.0000
    Epoch 322/1000
    69/69 [==============================] - 0s 3ms/step - loss: 9.6559e-06 - accuracy: 1.0000
    Epoch 323/1000
    69/69 [==============================] - 0s 3ms/step - loss: 9.4789e-06 - accuracy: 1.0000
    Epoch 324/1000
    69/69 [==============================] - 0s 3ms/step - loss: 9.1226e-06 - accuracy: 1.0000
    Epoch 325/1000
    69/69 [==============================] - 0s 3ms/step - loss: 8.7922e-06 - accuracy: 1.0000
    Epoch 326/1000
    69/69 [==============================] - 0s 3ms/step - loss: 8.5830e-06 - accuracy: 1.0000
    Epoch 327/1000
    69/69 [==============================] - 0s 3ms/step - loss: 8.2805e-06 - accuracy: 1.0000
    Epoch 328/1000
    69/69 [==============================] - 0s 3ms/step - loss: 8.0341e-06 - accuracy: 1.0000
    Epoch 329/1000
    69/69 [==============================] - 0s 3ms/step - loss: 7.7194e-06 - accuracy: 1.0000
    Epoch 330/1000
    69/69 [==============================] - 0s 3ms/step - loss: 7.5102e-06 - accuracy: 1.0000
    Epoch 331/1000
    69/69 [==============================] - 0s 3ms/step - loss: 7.3402e-06 - accuracy: 1.0000
    Epoch 332/1000
    69/69 [==============================] - 0s 3ms/step - loss: 7.0117e-06 - accuracy: 1.0000
    Epoch 333/1000
    69/69 [==============================] - 0s 3ms/step - loss: 6.9141e-06 - accuracy: 1.0000
    Epoch 334/1000
    69/69 [==============================] - 0s 3ms/step - loss: 6.4443e-06 - accuracy: 1.0000
    Epoch 335/1000
    69/69 [==============================] - 0s 3ms/step - loss: 6.4373e-06 - accuracy: 1.0000
    Epoch 336/1000
    69/69 [==============================] - 0s 3ms/step - loss: 6.2464e-06 - accuracy: 1.0000
    Epoch 337/1000
    69/69 [==============================] - 0s 3ms/step - loss: 5.9598e-06 - accuracy: 1.0000
    Epoch 338/1000
    69/69 [==============================] - 0s 3ms/step - loss: 5.8412e-06 - accuracy: 1.0000
    Epoch 339/1000
    69/69 [==============================] - 0s 3ms/step - loss: 5.5666e-06 - accuracy: 1.0000
    Epoch 340/1000
    69/69 [==============================] - 0s 3ms/step - loss: 5.3644e-06 - accuracy: 1.0000
    Epoch 341/1000
    69/69 [==============================] - 0s 3ms/step - loss: 5.3644e-06 - accuracy: 1.0000
    Epoch 342/1000
    69/69 [==============================] - 0s 3ms/step - loss: 5.1386e-06 - accuracy: 1.0000
    Epoch 343/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.8876e-06 - accuracy: 1.0000
    Epoch 344/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.7928e-06 - accuracy: 1.0000
    Epoch 345/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.7588e-06 - accuracy: 1.0000
    Epoch 346/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.2915e-06 - accuracy: 1.0000
    Epoch 347/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.2915e-06 - accuracy: 1.0000
    Epoch 348/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.2915e-06 - accuracy: 1.0000
    Epoch 349/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.2261e-06 - accuracy: 1.0000
    Epoch 350/1000
    69/69 [==============================] - 0s 3ms/step - loss: 3.8147e-06 - accuracy: 1.0000
    Epoch 351/1000
    69/69 [==============================] - 0s 3ms/step - loss: 3.8147e-06 - accuracy: 1.0000
    Epoch 352/1000
    69/69 [==============================] - 0s 3ms/step - loss: 3.6990e-06 - accuracy: 1.0000
    Epoch 353/1000
    69/69 [==============================] - 0s 3ms/step - loss: 3.6955e-06 - accuracy: 1.0000
    Epoch 354/1000
    69/69 [==============================] - 0s 3ms/step - loss: 3.3721e-06 - accuracy: 1.0000
    Epoch 355/1000
    69/69 [==============================] - 0s 3ms/step - loss: 3.2186e-06 - accuracy: 1.0000
    Epoch 356/1000
    69/69 [==============================] - 0s 3ms/step - loss: 3.2186e-06 - accuracy: 1.0000
    Epoch 357/1000
    69/69 [==============================] - 0s 3ms/step - loss: 3.2186e-06 - accuracy: 1.0000
    Epoch 358/1000
    69/69 [==============================] - 0s 3ms/step - loss: 3.2186e-06 - accuracy: 1.0000
    Epoch 359/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.9929e-06 - accuracy: 1.0000
    Epoch 360/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.7418e-06 - accuracy: 1.0000
    Epoch 361/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.7418e-06 - accuracy: 1.0000
    Epoch 362/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.6889e-06 - accuracy: 1.0000
    Epoch 363/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.6226e-06 - accuracy: 1.0000
    Epoch 364/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.6226e-06 - accuracy: 1.0000
    Epoch 365/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.5363e-06 - accuracy: 1.0000
    Epoch 366/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.1458e-06 - accuracy: 1.0000
    Epoch 367/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.1458e-06 - accuracy: 1.0000
    Epoch 368/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.1458e-06 - accuracy: 1.0000
    Epoch 369/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.1458e-06 - accuracy: 1.0000
    Epoch 370/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.1458e-06 - accuracy: 1.0000
    Epoch 371/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.1458e-06 - accuracy: 1.0000
    Epoch 372/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.1458e-06 - accuracy: 1.0000
    Epoch 373/1000
    69/69 [==============================] - 0s 3ms/step - loss: 2.0037e-06 - accuracy: 1.0000
    Epoch 374/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.6689e-06 - accuracy: 1.0000
    Epoch 375/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.6689e-06 - accuracy: 1.0000
    Epoch 376/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.6689e-06 - accuracy: 1.0000
    Epoch 377/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.6689e-06 - accuracy: 1.0000
    Epoch 378/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.6317e-06 - accuracy: 1.0000
    Epoch 379/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.5497e-06 - accuracy: 1.0000
    Epoch 380/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.5497e-06 - accuracy: 1.0000
    Epoch 381/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.5497e-06 - accuracy: 1.0000
    Epoch 382/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.5497e-06 - accuracy: 1.0000
    Epoch 383/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.5497e-06 - accuracy: 1.0000
    Epoch 384/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.2403e-06 - accuracy: 1.0000
    Epoch 385/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.0729e-06 - accuracy: 1.0000
    Epoch 386/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.0729e-06 - accuracy: 1.0000
    Epoch 387/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.0729e-06 - accuracy: 1.0000
    Epoch 388/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.0729e-06 - accuracy: 1.0000
    Epoch 389/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.0729e-06 - accuracy: 1.0000
    Epoch 390/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.0729e-06 - accuracy: 1.0000
    Epoch 391/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.0729e-06 - accuracy: 1.0000
    Epoch 392/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.0729e-06 - accuracy: 1.0000
    Epoch 393/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.0729e-06 - accuracy: 1.0000
    Epoch 394/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.0729e-06 - accuracy: 1.0000
    Epoch 395/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.0729e-06 - accuracy: 1.0000
    Epoch 396/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.0729e-06 - accuracy: 1.0000
    Epoch 397/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.0729e-06 - accuracy: 1.0000
    Epoch 398/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.0729e-06 - accuracy: 1.0000
    Epoch 399/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.0729e-06 - accuracy: 1.0000
    Epoch 400/1000
    69/69 [==============================] - 0s 3ms/step - loss: 8.6803e-07 - accuracy: 1.0000
    Epoch 401/1000
    69/69 [==============================] - 0s 3ms/step - loss: 5.9605e-07 - accuracy: 1.0000
    Epoch 402/1000
    69/69 [==============================] - 0s 3ms/step - loss: 5.9605e-07 - accuracy: 1.0000
    Epoch 403/1000
    69/69 [==============================] - 0s 3ms/step - loss: 5.9605e-07 - accuracy: 1.0000
    Epoch 404/1000
    69/69 [==============================] - 0s 3ms/step - loss: 5.9605e-07 - accuracy: 1.0000
    Epoch 405/1000
    69/69 [==============================] - 0s 3ms/step - loss: 5.9605e-07 - accuracy: 1.0000
    Epoch 406/1000
    69/69 [==============================] - 0s 3ms/step - loss: 5.9605e-07 - accuracy: 1.0000
    Epoch 407/1000
    69/69 [==============================] - 0s 3ms/step - loss: 5.9605e-07 - accuracy: 1.0000
    Epoch 408/1000
    69/69 [==============================] - 0s 3ms/step - loss: 5.9605e-07 - accuracy: 1.0000
    Epoch 409/1000
    69/69 [==============================] - 0s 3ms/step - loss: 5.9605e-07 - accuracy: 1.0000
    Epoch 410/1000
    69/69 [==============================] - 0s 3ms/step - loss: 5.9605e-07 - accuracy: 1.0000
    Epoch 411/1000
    69/69 [==============================] - 0s 3ms/step - loss: 5.9605e-07 - accuracy: 1.0000
    Epoch 412/1000
    69/69 [==============================] - 0s 3ms/step - loss: 5.9605e-07 - accuracy: 1.0000
    Epoch 413/1000
    69/69 [==============================] - 0s 3ms/step - loss: 5.9605e-07 - accuracy: 1.0000
    Epoch 414/1000
    69/69 [==============================] - 0s 3ms/step - loss: 5.6575e-07 - accuracy: 1.0000
    Epoch 415/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.7684e-07 - accuracy: 1.0000
    Epoch 416/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.7684e-07 - accuracy: 1.0000
    Epoch 417/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.7684e-07 - accuracy: 1.0000
    Epoch 418/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.7684e-07 - accuracy: 1.0000
    Epoch 419/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.7684e-07 - accuracy: 1.0000
    Epoch 420/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.7684e-07 - accuracy: 1.0000
    Epoch 421/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.7684e-07 - accuracy: 1.0000
    Epoch 422/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.7684e-07 - accuracy: 1.0000
    Epoch 423/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.7684e-07 - accuracy: 1.0000
    Epoch 424/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.7684e-07 - accuracy: 1.0000
    Epoch 425/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.7684e-07 - accuracy: 1.0000
    Epoch 426/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.7684e-07 - accuracy: 1.0000
    Epoch 427/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.7684e-07 - accuracy: 1.0000
    Epoch 428/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.7684e-07 - accuracy: 1.0000
    Epoch 429/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.7684e-07 - accuracy: 1.0000
    Epoch 430/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.7684e-07 - accuracy: 1.0000
    Epoch 431/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.7684e-07 - accuracy: 1.0000
    Epoch 432/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.7684e-07 - accuracy: 1.0000
    Epoch 433/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.7684e-07 - accuracy: 1.0000
    Epoch 434/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.7684e-07 - accuracy: 1.0000
    Epoch 435/1000
    69/69 [==============================] - 0s 3ms/step - loss: 4.7684e-07 - accuracy: 1.0000
    Epoch 436/1000
    69/69 [==============================] - 0s 3ms/step - loss: 1.3948e-08 - accuracy: 1.0000
    Epoch 437/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 438/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 439/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 440/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 441/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 442/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 443/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 444/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 445/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 446/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 447/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 448/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 449/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 450/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 451/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 452/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 453/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 454/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 455/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 456/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 457/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 458/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 459/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 460/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 461/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 462/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 463/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 464/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 465/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 466/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 467/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 468/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 469/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 470/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 471/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 472/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 473/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 474/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 475/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 476/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 477/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 478/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 479/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 480/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 481/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 482/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 483/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 484/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 485/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 486/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 487/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 488/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 489/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 490/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 491/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 492/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 493/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 494/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 495/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 496/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 497/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 498/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 499/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 500/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 501/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 502/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 503/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 504/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 505/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 506/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 507/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 508/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 509/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 510/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 511/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 512/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 513/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 514/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 515/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 516/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 517/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 518/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 519/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 520/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 521/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 522/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 523/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 524/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 525/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 526/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 527/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 528/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 529/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 530/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 531/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 532/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 533/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 534/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 535/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 536/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 537/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 538/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 539/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 540/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 541/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 542/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 543/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 544/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 545/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 546/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 547/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 548/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 549/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 550/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 551/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 552/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 553/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 554/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 555/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 556/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 557/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 558/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 559/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 560/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 561/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 562/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 563/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 564/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 565/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 566/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 567/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 568/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 569/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 570/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 571/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 572/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 573/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 574/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 575/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 576/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 577/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 578/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 579/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 580/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 581/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 582/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 583/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 584/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 585/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 586/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 587/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 588/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 589/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 590/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 591/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 592/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 593/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 594/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 595/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 596/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 597/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 598/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 599/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 600/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 601/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 602/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 603/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 604/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 605/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 606/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 607/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 608/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 609/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 610/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 611/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 612/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 613/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 614/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 615/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 616/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 617/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 618/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 619/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 620/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 621/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 622/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 623/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 624/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 625/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 626/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 627/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 628/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 629/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 630/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 631/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 632/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 633/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 634/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 635/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 636/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 637/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 638/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 639/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 640/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 641/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 642/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 643/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 644/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 645/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 646/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 647/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 648/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 649/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 650/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 651/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 652/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 653/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 654/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 655/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 656/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 657/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 658/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 659/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 660/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 661/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 662/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 663/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 664/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 665/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 666/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 667/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 668/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 669/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 670/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 671/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 672/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 673/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 674/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 675/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 676/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 677/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 678/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 679/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 680/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 681/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 682/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 683/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 684/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 685/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 686/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 687/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 688/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 689/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 690/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 691/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 692/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 693/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 694/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 695/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 696/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 697/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 698/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 699/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 700/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 701/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 702/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 703/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 704/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 705/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 706/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 707/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 708/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 709/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 710/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 711/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 712/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 713/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 714/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 715/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 716/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 717/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 718/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 719/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 720/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 721/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 722/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 723/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 724/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 725/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 726/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 727/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 728/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 729/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 730/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 731/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 732/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 733/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 734/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 735/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 736/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 737/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 738/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 739/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 740/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 741/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 742/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 743/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 744/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 745/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 746/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 747/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 748/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 749/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 750/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 751/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 752/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 753/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 754/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 755/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 756/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 757/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 758/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 759/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 760/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 761/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 762/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 763/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 764/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 765/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 766/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 767/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 768/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 769/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 770/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 771/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 772/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 773/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 774/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 775/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 776/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 777/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 778/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 779/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 780/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 781/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 782/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 783/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 784/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 785/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 786/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 787/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 788/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 789/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 790/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 791/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 792/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 793/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 794/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 795/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 796/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 797/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 798/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 799/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 800/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 801/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 802/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 803/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 804/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 805/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 806/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 807/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 808/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 809/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 810/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 811/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 812/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 813/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 814/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 815/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 816/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 817/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 818/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 819/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 820/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 821/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 822/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 823/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 824/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 825/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 826/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 827/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 828/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 829/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 830/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 831/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 832/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 833/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 834/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 835/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 836/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 837/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 838/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 839/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 840/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 841/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 842/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 843/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 844/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 845/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 846/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 847/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 848/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 849/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 850/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 851/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 852/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 853/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 854/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 855/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 856/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 857/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 858/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 859/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 860/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 861/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 862/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 863/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 864/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 865/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 866/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 867/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 868/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 869/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 870/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 871/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 872/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 873/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 874/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 875/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 876/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 877/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 878/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 879/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 880/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 881/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 882/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 883/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 884/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 885/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 886/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 887/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 888/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 889/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 890/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 891/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 892/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 893/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 894/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 895/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 896/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 897/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 898/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 899/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 900/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 901/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 902/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 903/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 904/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 905/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 906/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 907/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 908/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 909/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 910/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 911/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 912/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 913/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 914/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 915/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 916/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 917/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 918/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 919/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 920/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 921/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 922/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 923/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 924/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 925/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 926/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 927/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 928/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 929/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 930/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 931/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 932/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 933/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 934/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 935/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 936/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 937/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 938/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 939/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 940/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 941/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 942/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 943/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 944/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 945/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 946/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 947/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 948/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 949/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 950/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 951/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 952/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 953/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 954/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 955/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 956/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 957/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 958/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 959/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 960/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 961/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 962/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 963/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 964/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 965/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 966/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 967/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 968/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 969/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 970/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 971/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 972/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 973/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 974/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 975/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 976/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 977/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 978/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 979/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 980/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 981/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 982/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 983/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 984/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 985/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 986/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 987/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 988/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 989/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 990/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 991/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 992/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 993/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 994/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 995/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 996/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 997/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 998/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 999/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1000/1000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000





    <keras.callbacks.History at 0x7f45242262e0>




```python
test_loss, test_accuracy = model.evaluate(x_train, y_train, verbose=2)
print("test_loss: {} ".format(test_loss))
print("test_accuracy: {}".format(test_accuracy))
```

    69/69 - 0s - loss: 0.0000e+00 - accuracy: 1.0000
    test_loss: 0.0 
    test_accuracy: 1.0



```python

```


```python
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import glob 
from PIL import Image
import matplotlib.pyplot as plt

def load_data2(img_path):
    # 가위 : 0, 바위 : 1, 보 : 2
    number_of_data=2188   # 가위바위보 이미지 개수 총합에 주의하세요.
    img_size=28
    color=3
    #이미지 데이터와 라벨(가위 : 0, 바위 : 1, 보 : 2) 데이터를 담을 행렬(matrix) 영역을 생성합니다.
    imgs=np.zeros(number_of_data*img_size*img_size*color,dtype=np.int32).reshape(number_of_data,img_size,img_size,color)
    labels=np.zeros(number_of_data,dtype=np.int32)

    idx=2187
    for file in glob.iglob(img_path+'/scissor/*.png'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=0   # 가위 : 0
        idx=idx+1

    for file in glob.iglob(img_path+'/rock/*.png'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=1   # 바위 : 1
        idx=idx+1       
    
    for file in glob.iglob(img_path+'/paper/*.png'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=2   # 보 : 2
        idx=idx+1
        
    print("학습데이터(x_test)의 이미지 개수는",  idx,"입니다.")
    return imgs, labels

image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper"
(x_test, y_test)=load_data2(image_dir_path)
x_test_norm = x_test/255.0   # 입력은 0~1 사이의 값으로 정규화

print("x_test shape: {}".format(x_test.shape))
print("y_test shape: {}".format(y_test.shape))
```

    학습데이터(x_test)의 이미지 개수는 2187 입니다.
    x_test shape: (2188, 28, 28, 3)
    y_test shape: (2188,)



```python
import os
import tensorflow as tf
from tensorflow import keras

kernel_regularizer = keras.regularizers.L1(0.001)


n_channel_1=16
n_channel_2=32
n_dense=32
n_test_epoch=2000

model=keras.models.Sequential()
model.add(keras.layers.Conv2D(n_channel_1, (3,3), activation='relu', input_shape=(28,28,3)))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(n_channel_2, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(n_dense, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()


```

    Model: "sequential_4"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_8 (Conv2D)            (None, 26, 26, 16)        448       
    _________________________________________________________________
    max_pooling2d_8 (MaxPooling2 (None, 13, 13, 16)        0         
    _________________________________________________________________
    conv2d_9 (Conv2D)            (None, 11, 11, 32)        4640      
    _________________________________________________________________
    max_pooling2d_9 (MaxPooling2 (None, 5, 5, 32)          0         
    _________________________________________________________________
    flatten_4 (Flatten)          (None, 800)               0         
    _________________________________________________________________
    dense_8 (Dense)              (None, 32)                25632     
    _________________________________________________________________
    dense_9 (Dense)              (None, 10)                330       
    =================================================================
    Total params: 31,050
    Trainable params: 31,050
    Non-trainable params: 0
    _________________________________________________________________



```python
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(x_test, y_test, epochs=2000)
```

    Epoch 1/2000
    69/69 [==============================] - 1s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 2/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 3/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 4/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 5/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 6/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 7/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 8/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 9/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 10/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 11/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 12/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 13/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 14/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 15/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 16/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 17/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 18/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 19/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 20/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 21/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 22/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 23/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 24/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 25/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 26/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 27/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 28/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 29/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 30/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 31/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 32/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 33/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 34/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 35/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 36/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 37/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 38/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 39/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 40/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 41/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 42/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 43/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 44/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 45/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 46/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 47/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 48/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 49/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 50/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 51/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 52/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 53/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 54/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 55/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 56/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 57/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 58/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 59/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 60/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 61/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 62/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 63/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 64/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 65/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 66/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 67/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 68/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 69/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 70/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 71/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 72/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 73/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 74/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 75/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 76/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 77/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 78/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 79/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 80/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 81/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 82/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 83/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 84/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 85/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 86/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 87/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 88/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 89/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 90/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 91/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 92/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 93/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 94/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 95/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 96/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 97/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 98/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 99/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 100/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 101/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 102/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 103/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 104/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 105/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 106/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 107/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 108/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 109/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 110/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 111/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 112/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 113/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 114/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 115/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 116/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 117/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 118/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 119/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 120/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 121/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 122/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 123/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 124/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 125/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 126/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 127/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 128/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 129/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 130/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 131/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 132/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 133/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 134/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 135/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 136/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 137/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 138/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 139/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 140/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 141/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 142/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 143/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 144/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 145/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 146/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 147/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 148/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 149/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 150/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 151/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 152/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 153/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 154/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 155/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 156/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 157/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 158/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 159/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 160/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 161/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 162/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 163/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 164/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 165/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 166/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 167/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 168/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 169/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 170/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 171/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 172/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 173/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 174/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 175/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 176/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 177/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 178/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 179/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 180/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 181/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 182/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 183/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 184/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 185/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 186/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 187/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 188/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 189/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 190/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 191/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 192/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 193/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 194/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 195/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 196/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 197/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 198/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 199/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 200/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 201/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 202/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 203/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 204/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 205/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 206/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 207/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 208/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 209/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 210/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 211/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 212/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 213/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 214/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 215/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 216/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 217/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 218/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 219/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 220/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 221/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 222/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 223/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 224/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 225/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 226/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 227/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 228/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 229/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 230/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 231/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 232/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 233/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 234/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 235/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 236/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 237/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 238/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 239/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 240/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 241/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 242/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 243/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 244/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 245/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 246/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 247/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 248/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 249/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 250/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 251/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 252/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 253/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 254/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 255/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 256/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 257/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 258/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 259/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 260/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 261/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 262/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 263/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 264/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 265/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 266/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 267/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 268/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 269/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 270/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 271/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 272/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 273/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 274/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 275/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 276/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 277/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 278/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 279/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 280/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 281/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 282/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 283/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 284/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 285/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 286/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 287/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 288/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 289/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 290/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 291/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 292/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 293/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 294/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 295/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 296/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 297/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 298/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 299/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 300/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 301/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 302/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 303/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 304/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 305/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 306/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 307/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 308/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 309/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 310/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 311/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 312/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 313/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 314/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 315/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 316/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 317/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 318/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 319/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 320/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 321/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 322/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 323/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 324/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 325/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 326/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 327/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 328/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 329/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 330/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 331/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 332/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 333/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 334/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 335/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 336/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 337/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 338/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 339/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 340/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 341/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 342/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 343/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 344/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 345/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 346/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 347/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 348/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 349/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 350/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 351/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 352/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 353/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 354/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 355/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 356/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 357/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 358/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 359/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 360/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 361/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 362/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 363/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 364/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 365/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 366/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 367/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 368/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 369/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 370/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 371/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 372/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 373/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 374/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 375/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 376/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 377/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 378/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 379/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 380/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 381/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 382/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 383/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 384/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 385/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 386/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 387/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 388/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 389/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 390/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 391/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 392/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 393/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 394/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 395/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 396/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 397/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 398/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 399/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 400/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 401/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 402/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 403/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 404/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 405/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 406/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 407/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 408/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 409/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 410/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 411/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 412/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 413/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 414/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 415/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 416/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 417/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 418/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 419/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 420/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 421/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 422/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 423/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 424/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 425/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 426/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 427/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 428/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 429/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 430/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 431/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 432/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 433/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 434/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 435/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 436/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 437/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 438/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 439/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 440/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 441/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 442/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 443/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 444/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 445/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 446/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 447/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 448/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 449/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 450/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 451/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 452/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 453/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 454/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 455/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 456/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 457/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 458/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 459/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 460/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 461/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 462/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 463/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 464/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 465/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 466/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 467/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 468/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 469/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 470/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 471/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 472/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 473/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 474/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 475/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 476/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 477/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 478/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 479/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 480/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 481/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 482/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 483/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 484/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 485/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 486/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 487/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 488/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 489/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 490/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 491/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 492/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 493/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 494/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 495/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 496/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 497/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 498/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 499/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 500/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 501/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 502/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 503/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 504/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 505/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 506/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 507/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 508/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 509/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 510/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 511/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 512/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 513/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 514/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 515/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 516/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 517/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 518/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 519/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 520/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 521/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 522/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 523/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 524/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 525/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 526/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 527/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 528/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 529/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 530/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 531/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 532/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 533/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 534/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 535/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 536/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 537/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 538/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 539/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 540/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 541/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 542/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 543/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 544/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 545/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 546/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 547/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 548/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 549/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 550/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 551/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 552/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 553/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 554/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 555/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 556/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 557/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 558/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 559/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 560/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 561/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 562/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 563/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 564/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 565/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 566/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 567/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 568/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 569/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 570/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 571/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 572/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 573/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 574/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 575/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 576/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 577/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 578/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 579/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 580/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 581/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 582/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 583/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 584/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 585/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 586/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 587/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 588/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 589/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 590/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 591/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 592/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 593/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 594/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 595/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 596/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 597/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 598/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 599/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 600/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 601/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 602/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 603/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 604/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 605/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 606/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 607/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 608/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 609/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 610/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 611/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 612/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 613/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 614/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 615/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 616/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 617/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 618/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 619/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 620/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 621/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 622/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 623/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 624/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 625/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 626/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 627/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 628/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 629/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 630/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 631/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 632/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 633/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 634/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 635/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 636/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 637/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 638/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 639/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 640/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 641/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 642/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 643/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 644/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 645/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 646/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 647/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 648/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 649/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 650/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 651/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 652/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 653/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 654/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 655/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 656/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 657/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 658/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 659/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 660/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 661/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 662/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 663/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 664/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 665/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 666/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 667/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 668/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 669/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 670/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 671/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 672/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 673/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 674/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 675/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 676/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 677/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 678/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 679/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 680/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 681/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 682/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 683/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 684/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 685/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 686/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 687/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 688/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 689/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 690/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 691/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 692/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 693/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 694/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 695/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 696/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 697/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 698/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 699/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 700/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 701/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 702/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 703/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 704/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 705/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 706/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 707/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 708/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 709/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 710/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 711/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 712/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 713/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 714/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 715/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 716/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 717/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 718/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 719/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 720/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 721/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 722/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 723/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 724/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 725/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 726/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 727/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 728/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 729/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 730/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 731/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 732/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 733/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 734/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 735/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 736/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 737/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 738/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 739/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 740/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 741/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 742/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 743/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 744/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 745/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 746/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 747/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 748/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 749/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 750/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 751/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 752/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 753/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 754/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 755/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 756/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 757/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 758/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 759/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 760/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 761/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 762/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 763/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 764/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 765/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 766/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 767/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 768/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 769/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 770/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 771/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 772/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 773/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 774/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 775/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 776/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 777/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 778/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 779/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 780/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 781/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 782/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 783/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 784/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 785/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 786/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 787/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 788/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 789/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 790/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 791/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 792/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 793/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 794/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 795/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 796/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 797/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 798/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 799/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 800/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 801/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 802/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 803/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 804/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 805/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 806/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 807/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 808/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 809/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 810/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 811/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 812/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 813/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 814/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 815/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 816/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 817/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 818/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 819/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 820/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 821/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 822/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 823/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 824/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 825/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 826/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 827/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 828/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 829/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 830/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 831/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 832/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 833/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 834/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 835/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 836/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 837/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 838/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 839/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 840/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 841/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 842/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 843/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 844/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 845/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 846/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 847/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 848/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 849/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 850/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 851/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 852/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 853/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 854/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 855/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 856/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 857/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 858/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 859/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 860/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 861/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 862/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 863/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 864/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 865/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 866/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 867/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 868/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 869/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 870/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 871/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 872/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 873/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 874/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 875/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 876/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 877/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 878/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 879/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 880/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 881/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 882/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 883/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 884/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 885/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 886/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 887/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 888/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 889/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 890/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 891/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 892/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 893/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 894/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 895/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 896/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 897/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 898/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 899/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 900/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 901/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 902/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 903/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 904/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 905/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 906/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 907/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 908/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 909/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 910/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 911/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 912/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 913/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 914/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 915/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 916/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 917/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 918/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 919/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 920/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 921/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 922/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 923/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 924/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 925/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 926/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 927/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 928/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 929/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 930/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 931/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 932/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 933/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 934/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 935/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 936/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 937/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 938/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 939/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 940/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 941/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 942/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 943/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 944/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 945/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 946/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 947/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 948/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 949/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 950/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 951/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 952/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 953/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 954/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 955/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 956/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 957/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 958/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 959/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 960/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 961/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 962/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 963/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 964/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 965/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 966/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 967/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 968/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 969/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 970/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 971/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 972/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 973/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 974/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 975/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 976/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 977/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 978/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 979/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 980/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 981/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 982/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 983/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 984/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 985/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 986/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 987/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 988/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 989/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 990/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 991/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 992/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 993/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 994/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 995/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 996/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 997/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 998/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 999/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1000/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1001/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1002/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1003/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1004/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1005/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1006/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1007/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1008/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1009/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1010/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1011/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1012/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1013/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1014/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1015/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1016/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1017/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1018/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1019/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1020/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1021/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1022/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1023/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1024/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1025/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1026/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1027/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1028/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1029/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1030/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1031/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1032/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1033/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1034/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1035/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1036/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1037/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1038/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1039/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1040/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1041/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1042/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1043/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1044/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1045/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1046/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1047/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1048/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1049/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1050/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1051/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1052/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1053/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1054/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1055/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1056/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1057/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1058/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1059/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1060/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1061/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1062/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1063/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1064/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1065/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1066/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1067/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1068/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1069/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1070/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1071/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1072/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1073/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1074/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1075/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1076/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1077/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1078/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1079/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1080/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1081/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1082/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1083/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1084/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1085/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1086/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1087/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1088/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1089/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1090/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1091/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1092/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1093/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1094/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1095/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1096/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1097/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1098/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1099/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1100/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1101/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1102/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1103/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1104/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1105/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1106/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1107/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1108/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1109/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1110/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1111/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1112/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1113/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1114/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1115/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1116/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1117/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1118/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1119/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1120/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1121/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1122/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1123/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1124/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1125/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1126/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1127/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1128/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1129/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1130/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1131/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1132/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1133/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1134/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1135/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1136/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1137/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1138/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1139/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1140/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1141/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1142/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1143/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1144/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1145/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1146/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1147/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1148/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1149/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1150/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1151/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1152/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1153/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1154/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1155/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1156/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1157/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1158/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1159/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1160/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1161/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1162/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1163/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1164/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1165/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1166/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1167/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1168/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1169/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1170/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1171/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1172/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1173/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1174/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1175/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1176/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1177/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1178/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1179/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1180/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1181/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1182/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1183/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1184/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1185/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1186/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1187/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1188/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1189/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1190/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1191/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1192/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1193/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1194/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1195/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1196/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1197/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1198/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1199/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1200/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1201/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1202/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1203/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1204/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1205/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1206/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1207/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1208/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1209/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1210/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1211/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1212/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1213/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1214/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1215/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1216/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1217/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1218/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1219/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1220/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1221/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1222/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1223/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1224/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1225/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1226/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1227/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1228/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1229/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1230/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1231/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1232/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1233/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1234/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1235/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1236/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1237/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1238/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1239/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1240/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1241/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1242/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1243/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1244/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1245/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1246/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1247/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1248/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1249/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1250/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1251/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1252/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1253/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1254/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1255/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1256/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1257/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1258/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1259/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1260/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1261/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1262/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1263/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1264/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1265/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1266/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1267/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1268/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1269/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1270/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1271/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1272/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1273/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1274/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1275/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1276/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1277/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1278/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1279/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1280/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1281/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1282/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1283/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1284/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1285/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1286/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1287/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1288/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1289/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1290/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1291/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1292/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1293/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1294/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1295/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1296/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1297/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1298/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1299/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1300/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1301/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1302/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1303/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1304/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1305/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1306/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1307/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1308/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1309/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1310/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1311/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1312/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1313/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1314/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1315/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1316/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1317/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1318/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1319/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1320/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1321/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1322/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1323/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1324/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1325/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1326/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1327/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1328/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1329/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1330/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1331/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1332/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1333/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1334/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1335/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1336/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1337/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1338/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1339/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1340/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1341/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1342/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1343/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1344/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1345/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1346/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1347/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1348/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1349/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1350/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1351/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1352/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1353/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1354/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1355/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1356/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1357/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1358/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1359/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1360/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1361/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1362/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1363/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1364/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1365/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1366/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1367/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1368/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1369/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1370/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1371/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1372/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1373/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1374/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1375/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1376/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1377/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1378/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1379/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1380/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1381/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1382/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1383/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1384/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1385/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1386/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1387/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1388/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1389/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1390/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1391/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1392/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1393/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1394/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1395/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1396/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1397/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1398/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1399/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1400/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1401/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1402/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1403/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1404/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1405/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1406/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1407/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1408/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1409/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1410/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1411/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1412/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1413/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1414/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1415/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1416/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1417/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1418/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1419/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1420/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1421/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1422/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1423/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1424/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1425/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1426/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1427/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1428/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1429/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1430/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1431/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1432/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1433/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1434/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1435/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1436/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1437/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1438/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1439/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1440/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1441/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1442/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1443/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1444/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1445/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1446/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1447/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1448/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1449/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1450/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1451/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1452/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1453/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1454/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1455/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1456/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1457/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1458/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1459/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1460/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1461/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1462/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1463/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1464/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1465/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1466/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1467/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1468/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1469/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1470/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1471/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1472/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1473/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1474/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1475/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1476/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1477/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1478/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1479/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1480/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1481/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1482/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1483/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1484/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1485/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1486/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1487/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1488/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1489/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1490/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1491/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1492/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1493/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1494/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1495/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1496/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1497/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1498/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1499/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1500/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1501/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1502/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1503/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1504/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1505/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1506/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1507/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1508/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1509/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1510/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1511/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1512/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1513/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1514/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1515/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1516/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1517/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1518/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1519/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1520/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1521/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1522/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1523/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1524/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1525/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1526/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1527/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1528/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1529/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1530/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1531/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1532/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1533/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1534/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1535/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1536/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1537/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1538/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1539/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1540/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1541/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1542/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1543/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1544/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1545/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1546/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1547/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1548/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1549/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1550/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1551/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1552/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1553/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1554/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1555/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1556/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1557/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1558/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1559/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1560/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1561/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1562/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1563/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1564/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1565/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1566/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1567/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1568/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1569/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1570/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1571/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1572/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1573/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1574/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1575/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1576/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1577/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1578/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1579/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1580/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1581/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1582/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1583/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1584/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1585/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1586/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1587/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1588/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1589/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1590/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1591/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1592/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1593/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1594/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1595/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1596/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1597/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1598/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1599/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1600/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1601/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1602/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1603/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1604/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1605/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1606/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1607/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1608/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1609/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1610/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1611/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1612/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1613/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1614/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1615/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1616/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1617/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1618/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1619/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1620/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1621/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1622/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1623/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1624/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1625/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1626/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1627/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1628/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1629/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1630/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1631/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1632/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1633/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1634/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1635/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1636/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1637/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1638/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1639/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1640/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1641/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1642/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1643/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1644/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1645/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1646/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1647/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1648/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1649/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1650/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1651/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1652/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1653/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1654/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1655/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1656/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1657/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1658/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1659/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1660/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1661/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1662/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1663/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1664/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1665/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1666/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1667/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1668/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1669/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1670/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1671/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1672/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1673/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1674/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1675/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1676/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1677/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1678/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1679/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1680/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1681/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1682/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1683/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1684/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1685/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1686/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1687/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1688/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1689/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1690/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1691/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1692/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1693/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1694/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1695/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1696/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1697/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1698/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1699/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1700/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1701/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1702/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1703/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1704/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1705/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1706/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1707/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1708/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1709/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1710/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1711/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1712/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1713/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1714/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1715/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1716/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1717/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1718/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1719/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1720/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1721/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1722/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1723/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1724/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1725/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1726/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1727/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1728/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1729/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1730/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1731/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1732/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1733/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1734/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1735/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1736/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1737/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1738/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1739/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1740/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1741/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1742/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1743/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1744/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1745/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1746/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1747/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1748/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1749/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1750/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1751/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1752/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1753/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1754/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1755/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1756/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1757/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1758/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1759/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1760/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1761/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1762/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1763/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1764/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1765/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1766/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1767/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1768/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1769/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1770/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1771/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1772/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1773/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1774/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1775/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1776/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1777/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1778/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1779/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1780/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1781/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1782/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1783/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1784/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1785/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1786/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1787/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1788/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1789/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1790/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1791/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1792/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1793/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1794/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1795/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1796/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1797/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1798/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1799/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1800/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1801/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1802/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1803/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1804/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1805/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1806/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1807/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1808/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1809/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1810/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1811/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1812/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1813/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1814/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1815/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1816/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1817/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1818/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1819/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1820/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1821/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1822/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1823/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1824/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1825/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1826/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1827/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1828/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1829/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1830/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1831/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1832/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1833/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1834/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1835/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1836/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1837/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1838/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1839/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1840/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1841/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1842/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1843/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1844/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1845/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1846/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1847/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1848/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1849/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1850/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1851/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1852/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1853/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1854/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1855/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1856/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1857/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1858/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1859/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1860/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1861/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1862/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1863/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1864/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1865/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1866/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1867/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1868/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1869/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1870/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1871/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1872/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1873/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1874/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1875/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1876/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1877/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1878/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1879/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1880/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1881/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1882/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1883/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1884/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1885/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1886/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1887/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1888/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1889/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1890/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1891/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1892/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1893/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1894/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1895/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1896/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1897/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1898/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1899/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1900/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1901/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1902/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1903/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1904/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1905/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1906/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1907/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1908/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1909/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1910/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1911/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1912/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1913/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1914/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1915/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1916/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1917/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1918/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1919/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1920/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1921/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1922/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1923/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1924/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1925/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1926/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1927/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1928/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1929/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1930/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1931/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1932/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1933/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1934/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1935/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1936/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1937/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1938/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1939/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1940/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1941/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1942/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1943/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1944/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1945/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1946/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1947/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1948/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1949/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1950/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1951/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1952/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1953/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1954/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1955/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1956/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1957/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1958/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1959/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1960/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1961/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1962/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1963/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1964/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1965/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1966/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1967/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1968/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1969/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1970/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1971/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1972/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1973/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1974/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1975/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1976/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1977/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1978/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1979/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1980/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1981/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1982/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1983/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1984/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1985/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1986/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1987/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1988/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1989/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1990/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1991/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1992/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1993/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1994/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1995/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1996/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1997/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1998/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 1999/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000
    Epoch 2000/2000
    69/69 [==============================] - 0s 3ms/step - loss: 0.0000e+00 - accuracy: 1.0000





    <keras.callbacks.History at 0x7f4598d2a430>




```python

```


```python
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print("test_loss: {} ".format(test_loss))
print("test_accuracy: {}".format(test_accuracy))
```

    69/69 - 0s - loss: 0.0000e+00 - accuracy: 1.0000
    test_loss: 0.0 
    test_accuracy: 1.0



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
