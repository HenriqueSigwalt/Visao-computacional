import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

#Carrega a Imagem
I1=cv2.imread("Lab 2\Moodle\Tatoo1.jpg")

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

linhas, colunas, camadas = I1.shape

h=1
w=2*h+1

#y = 82 - 560    x = 255 - 425

#inicializa o kernel
kernel=np.zeros([w,w],np.float32)
#cria uma matriz "Alvo" com a cor da pele
tar=np.zeros([w,w,3],np.float32)
tar[:,:]=[160,160,180]

#Plota a imagem original
ax1.clear()
ax1.imshow(cv2.cvtColor(I1, cv2.COLOR_BGR2RGB))
Inova=cv2.imread("Lab 2\Moodle\Tatoo1.jpg")

#Percorre todos os pixels, exceto os da borda da imagem, aplicando a filtragem
for i in range(linhas-h):
    for j in range(colunas-h):
        #Se estiver na região em torno da imagem
        if(i>82 and i<560) and (j>255 and j<425):
            #Pega a região em torno do pixel a ser analisado
            O=I1[i-h:i+h+1,j-h:j+h+1]
            O[O==0]=1
            #Calcula a distancia da cor de cada pixel da área para a cor do "alvo"
            dist=tar[:,:,0]-O[:,:,0]
            #Pega o index da menor distancia e define ela como a cor desejada
            sup=np.unravel_index(np.argmin(dist, axis=None), dist.shape)
            target=O[sup[0],sup[1],0]
            #Cria um Kernel que leva a cor dos pixels para a cor desejada
            kernel=(target)/(O[:,:,0])
            #Aplica o Kernel na imagem
            Inova[i,j,0]=np.average(O[:,:,0]*kernel)
            #Repete o processo para cada uma das camadas de cor da imagem
            dist=tar[:,:,1]-O[:,:,1]
            sup=np.unravel_index(np.argmin(dist, axis=None), dist.shape)
            target=O[sup[0],sup[1],1]
            kernel=(target)/(O[:,:,1])
            Inova[i,j,1]=np.average(O[:,:,1]*kernel)
            dist=tar[:,:,2]-O[:,:,2]
            sup=np.unravel_index(np.argmin(dist, axis=None), dist.shape)
            target=O[sup[0],sup[1],2]
            kernel=(target)/(O[:,:,2])
            Inova[i,j,2]=np.average(O[:,:,2]*kernel)

#Mostra a imagem suavizada
ax2.clear()
ax2.imshow(cv2.cvtColor(Inova, cv2.COLOR_BGR2RGB))

plt.show()