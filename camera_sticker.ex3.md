```python
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import dlib
print("๐ซ๐ธ")
```

    ๐ซ๐ธ



```python
my_image_path = os.getenv('HOME')+'/aiffel/camera_sticker/images/anes.png'
img_bgr = cv2.imread(my_image_path)    # OpenCV๋ก ์ด๋ฏธ์ง๋ฅผ ๋ถ๋ฌ์ต๋๋ค
img_show = img_bgr.copy()      # ์ถ๋ ฅ์ฉ ์ด๋ฏธ์ง๋ฅผ ๋ฐ๋ก ๋ณด๊ดํฉ๋๋ค
plt.imshow(img_bgr)
plt.show()
```


    
![png](output_1_0.png)
    



```python
# plt.imshow ์ด์ ์ RGB ์ด๋ฏธ์ง๋ก ๋ฐ๊พธ๋ ๊ฒ์ ์์ง๋ง์ธ์. 
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()
```


    
![png](output_2_0.png)
    



```python
# detector๋ฅผ ์ ์ธํฉ๋๋ค
detector_hog = dlib.get_frontal_face_detector()
print("๐ซ๐ธ")
```

    ๐ซ๐ธ



```python
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
dlib_rects = detector_hog(img_rgb, 1)   # (image, num of image pyramid)
print("๐ซ๐ธ")
```

    ๐ซ๐ธ



```python
# ์ฐพ์ ์ผ๊ตด ์์ญ ๋ฐ์ค ๋ฆฌ์คํธ
# ์ฌ๋ฌ ์ผ๊ตด์ด ์์ ์ ์์ต๋๋ค
print(dlib_rects)   

for dlib_rect in dlib_rects:
    l = dlib_rect.left()
    t = dlib_rect.top()
    r = dlib_rect.right()
    b = dlib_rect.bottom()

    cv2.rectangle(img_show, (l,t), (r,b), (0,255,0), 2, lineType=cv2.LINE_AA)

img_show_rgb =  cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
plt.imshow(img_show_rgb)
plt.show()
```

    rectangles[[(132, 59) (798, 724)]]



    
![png](output_5_1.png)
    



```python
model_path = os.getenv('HOME')+'/aiffel/camera_sticker/models/shape_predictor_68_face_landmarks.dat'
landmark_predictor = dlib.shape_predictor(model_path)
print("๐ซ๐ธ")
```

    ๐ซ๐ธ



```python
list_landmarks = []

# ์ผ๊ตด ์์ญ ๋ฐ์ค ๋ง๋ค face landmark๋ฅผ ์ฐพ์๋๋๋ค
for dlib_rect in dlib_rects:
    points = landmark_predictor(img_rgb, dlib_rect)
    # face landmark ์ขํ๋ฅผ ์ ์ฅํด๋ก๋๋ค
    list_points = list(map(lambda p: (p.x, p.y), points.parts()))
    list_landmarks.append(list_points)

print(len(list_landmarks[0]))
```

    68



```python
for landmark in list_landmarks:
    for point in landmark:
        cv2.circle(img_show, point, 2, (0, 255, 255), -1)

img_show_rgb = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
plt.imshow(img_show_rgb)
plt.show()
```


    
![png](output_8_0.png)
    



```python
for dlib_rect, landmark in zip(dlib_rects, list_landmarks):
    print (landmark[30]) # ์ฝ์ index๋ 30 ์๋๋ค
    x = landmark[30][0]
    y = landmark[30][1] - dlib_rect.height()//2
    w = h = dlib_rect.width()
    print ('(x,y) : (%d,%d)'%(x,y))
    print ('(w,h) : (%d,%d)'%(w,h))
```

    (454, 419)
    (x,y) : (454,86)
    (w,h) : (667,667)



```python

```


```python
refined_x = x - w // 2
refined_y = y
print ('(x,y) : (%d,%d)'%(refined_x, refined_y))
```

    (x,y) : (121,86)



```python
if refined_x < 0: 
    img_sticker = img_sticker[:, -refined_x:]
    refined_x = 0
if refined_y < 0:
    img_sticker = img_sticker[-refined_y:, :]
    refined_y = 0

print ('(x,y) : (%d,%d)'%(refined_x, refined_y))
```

    (x,y) : (121,86)



```python
sticker_area = img_rgb[refined_y:img_sticker.shape[0] + refined_y, refined_x:refined_x + img_sticker.shape[1]]
img_rgb[refined_y:img_sticker.shape[0] + refined_y, refined_x:refined_x + img_sticker.shape[1]] = np.where(img_sticker == 255, sticker_area, img_sticker).astype(np.uint8)
```


```python
plt.imshow(img_rgb)
plt.show()
```


    
![png](output_14_0.png)
    



```python
sticker_area = img_bgr[refined_y:refined_y +img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]
img_bgr[refined_y:refined_y +img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = \
    np.where(img_sticker==0,sticker_area,img_sticker).astype(np.uint8)
plt.imshow(img_rgb)
plt.show()
```


    
![png](output_15_0.png)
    



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
