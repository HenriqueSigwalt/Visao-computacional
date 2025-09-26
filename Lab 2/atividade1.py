import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

I1=cv2.imread("Lab 2\Moodle\Tatoo1.jpg")

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

linhas, colunas, camadas = I1.shape

R=140
G=110
B=100

Ri=np.float32(I1[:,:,2])
Gi=np.float32(I1[:,:,1])
Bi=np.float32(I1[:,:,0])

dist=((Bi-B)**2 + (Gi-G)**2 + (Ri-R)**2)**(1/2)
index=(dist<30)

h=1
w=2*h+1

#y = 82 - 560    x = 255 - 425

kernel=np.zeros([w,w],np.float32)
target=np.zeros([w,w,3],np.float32)
target[:,:]=[160,160,180]

ax1.clear()
ax1.imshow(cv2.cvtColor(I1, cv2.COLOR_BGR2RGB))
Inova=cv2.imread("Lab 2\Moodle\Tatoo1.jpg")

for i in range(linhas-h):
    for j in range(colunas-h):
        if(i>82 and i<560) and (j>255 and j<425):
            if(True in index[i-h:i+h+1,j-h:j+h+1]):
                O=I1[i-h:i+h+1,j-h:j+h+1]
                O[O==0]=1
                dist=target[:,:,0]-O[:,:,0]
                sup=np.unravel_index(np.argmin(dist, axis=None), dist.shape)
                target[:,:,0]=O[sup[0],sup[1],0]
                kernel=(target[:,:,0])/(O[:,:,0])
                Inova[i,j,0]=np.average(O[:,:,0]*kernel)
                dist=target[:,:,1]-O[:,:,1]
                sup=np.unravel_index(np.argmin(dist, axis=None), dist.shape)
                target[:,:,1]=O[sup[0],sup[1],1]
                kernel=(target[:,:,1])/(O[:,:,1])
                Inova[i,j,1]=np.average(O[:,:,1]*kernel)
                dist=target[:,:,2]-O[:,:,2]
                sup=np.unravel_index(np.argmin(dist, axis=None), dist.shape)
                target[:,:,2]=O[sup[0],sup[1],2]
                kernel=(target[:,:,2])/(O[:,:,2])
                Inova[i,j,2]=np.average(O[:,:,2]*kernel)
                #print("resultado")
                #print(dist)
                #print(O[:,:,0])

ax2.clear()
ax2.imshow(cv2.cvtColor(Inova, cv2.COLOR_BGR2RGB))

plt.show()