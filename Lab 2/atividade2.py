import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

def canny(I1,sigma,ti,ts):
    linhas, colunas = I1.shape
    w=int(6*sigma+1)
    kernel = np.zeros((w,w), np.float32)

    v1 = 1/(2 * np.pi * (sigma**2))
    v2 = -1/(2 * (sigma**2))

    for y in np.arange(0,w):
        for x in np.arange(0,w):

            # corrige coordenadas para a fun¸c~ao gaussiana
            j = y - (w-1)/2
            i = x - (w-1)/2

            kernel[y,x] = v1 * np.exp(v2*(j**2 + i**2))

            kernel = kernel/np.sum(kernel)
    I1=cv2.filter2D(I1,-1,kernel, borderType=cv2.BORDER_REFLECT101)

    grad_X=cv2.Sobel(I1,cv2.CV_64F,1,0,ksize=w)
    grad_y=cv2.Sobel(I1,cv2.CV_64F,0,1,ksize=w)

    grad_mag=np.sqrt(grad_X**2+grad_y**2)
    grad_dir=np.arctan2(grad_y,grad_X)* 180 / np.pi

    I2=np.zeros([linhas,colunas],np.uint8)
    Is=np.zeros([linhas,colunas],np.uint8)
    Ii=np.zeros([linhas,colunas],np.uint8)
    Ib=np.zeros([linhas,colunas],np.uint8)

    grad_mag=((grad_mag/(np.max(grad_mag)+1e-10))*255).astype(np.uint8)

    for i in range(1,linhas-1):
        for j in range(1,colunas-1):
            theta=grad_dir[i,j]
            mag=grad_mag[i,j]
            #print(theta)
            #print(teste[i,j])
            if ((theta < -22.5) and (theta>-67.5) or (theta < 157.5) and (theta>112.5)):
                u=grad_mag[i+1,j+1]
                v=grad_mag[i-1,j-1]
            elif ((theta < -112.5) and (theta>-157.5) or (theta < 67.5) and (theta>22.5)):
                u=grad_mag[i+1,j-1]
                v=grad_mag[i-1,j+1]
            elif ((theta < -67.5) and (theta>-112.5) or (theta < 112.5) and (theta>67.5)):
                u=grad_mag[i+1,j]
                v=grad_mag[i-1,j]
            else:
                u=grad_mag[i,j-1]
                v=grad_mag[i,j+1]
            if (u<=mag)and(v<=mag):
                I2[i,j]=mag

    Is[I2>ts]=I2[I2>ts]
    Ii[I2>ti]=I2[I2>ti]
    Ii=Ii-Is
    for i in range(linhas):
        for j in range(colunas):
            if(Is[i,j]>0):
                Ib[i-1:i+2,j-1:j+2]=Ii[i-1:i+2,j-1:j+2]
    Ib=Ib+Is

    teste=Ii
    
    return Ib,teste

I1=cv2.imread("Lab 2\Moodle\castle.jpg",cv2.IMREAD_GRAYSCALE)
w=3
sigma=(w-1)/6
ti=30
ts=80

Icanny=cv2.Canny(I1,50,150)

Ib, teste=canny(I1,sigma,ti,ts)

"""fig = plt.figure()
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)

ax1.clear()
ax1.imshow(Icanny, cmap='gray')

ax2.clear()
ax2.imshow(Ib, cmap='gray')

ax3.clear()
ax3.imshow(teste, cmap='gray')"""

plt.figure()
plt.imshow(Ib, cmap='gray')
plt.figure()
plt.imshow(teste, cmap='gray')

plt.show()