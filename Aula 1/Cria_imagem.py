import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from os import path

"""
#Exemplo 1
array=np.zeros((200,400),np.uint8)
array[50:150,20:380]=255
cv2.imshow("Bin",array)
cv2.waitKey(0)
"""

"""
#Exemplo 2
array=np.zeros((900,1000),np.uint8)
r=100
x_init=500
y_init=450
points=[]
for i in range(0,360,1):
    y_var=int(r*math.sin(math.radians(i)))
    x_var=int(r*math.cos(math.radians(i)))
    array[y_init-y_var,x_init+x_var]=255
    
cv2.imshow("Bin",array)
cv2.waitKey(0)
"""

"""
#Exercicio 3
n=400
m=200
array=np.zeros((m,n),np.uint8)
for i in range(0,n):
    array[:,i]=np.uint8((i/(n-1))*255)

cv2.imshow("Bin",array)
cv2.waitKey(0)
"""

"""
#Exercicio 4
img=cv2.imread("Banco de imagens/quadrados_cinzas.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Teste",img)
cv2.waitKey(0)
"""

"""
#Exercicio 5
array=np.zeros((200,400,3),np.uint8)
array[50:150,20:380]=[0,255,255]
cv2.imshow("Colorido",array)
cv2.waitKey(0)
"""

"""
#Exercicio 6
img=cv2.imread("Banco de imagens/quadrados.png")

cv2.imshow("Teste",img)
cv2.waitKey(0)
"""


#Exercicio 7
img=cv2.imread("Banco de imagens/quadrados.png")
img2=np.float32(img/255)
img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
cv2.imshow("Teste",img)
cv2.imshow("Algo",img2)

cv2.waitKey(0)