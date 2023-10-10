import tensorflow as tf
from tensorflow.keras import layers
import cv2
import numpy 

img1 = cv2.imread("1(1).jpg")
img2 = cv2.imread("2(2).jpg")
img3 = cv2.imread("3(1).jpg")

orb = cv2.ORB_create()

kp1,des1 = orb.detectAndCompute(img1,None)
kp2,des2 = orb.detectAndCompute(img2,None)
kp3,des3 = orb.detectAndCompute(img3,None)

imgkp1 = cv2.drawKeypoints(img1,kp1,None)
imgkp2 = cv2.drawKeypoints(img2,kp2,None)
imgkp3 = cv2.drawKeypoints(img3,kp3,None)


# cv2.imshow("Imgkp1",imgkp1)
# cv2.imshow("Imgkp2",imgkp2)
# cv2.imshow("Img1",img1)
# cv2.imshow("Img2",img2)
cv2.imshow("Imgkp3",imgkp3)
cv2.imshow("Img3",img3)

cv2.waitKey(0)

model = tf.keras.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1))  # Output layer for soil moisture prediction

model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

