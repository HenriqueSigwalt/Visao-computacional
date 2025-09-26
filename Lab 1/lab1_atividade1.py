import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# ----------------------------------------------------------
cap = cv2.VideoCapture("Lab 1//Moodle//video2.mp4")

n_linhas = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_colunas  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# ----------------------------------------------------------
# Gera figura
fig = plt.figure()
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3, adjustable='box', aspect=0.9)

fig_width = 20
fig_height = 6
fig.set_size_inches((fig_width/2.54, fig_height/2.54))
fig.tight_layout()

# ----------------------------------------------------------
ret = True

R=23 

G=45

B=63

hasBall=False

ballCount=0

# R = 150 - 210; G = 160 - 220; B = 50 - 90;

# Seção de variáveis
# IMPLEMENTE O SEU CÓDIGO AQUI
e1=cv2.getTickCount()
time=[]

while ret:

    # lê frame do vídeo
    ret, I1 = cap.read()
    
    if I1 is None:
        break            
    
    # Algoritmo de detecção de objetos
    # IMPLEMENTE O SEU CÓDIGO AQUI]
    
    linhas, colunas, camadas = I1.shape
    
    Igray=np.zeros([linhas,colunas],np.uint8)
    
    Ri=np.float32(I1[:,:,2])
    Gi=np.float32(I1[:,:,1])
    Bi=np.float32(I1[:,:,0])
    
    dist=((Bi-B)**2 + (Gi-G)**2 + (Ri-R)**2)**(1/2)
    index=(dist<30)
    Igray[index]=255
    
    """
    for i in range(linhas):
        for a in range(colunas):
            Ri=float(I1[i][a][2])
            Gi=float(I1[i][a][1])
            Bi=float(I1[i][a][0])
            d=math.pow((math.pow((Bi-B),2)+math.pow((Gi-G),2)+math.pow((Ri-R),2)),(1/2))
            if(d<40):
                Igray[i][a]=255"""
    
    # atualiza plot
    ax1.clear()
    ax1.imshow(cv2.cvtColor(I1, cv2.COLOR_BGR2RGB))

    """
    Igray=cv2.cvtColor(I1,cv2.COLOR_BGR2GRAY)
    ret2,Igray=cv2.threshold(Igray,160,255,cv2.THRESH_BINARY)
    """

    ax2.clear()
    ax2.imshow(Igray,cmap="gray")
    ax2.plot([n_colunas/2, n_colunas/2], [0, n_linhas-1], ':')
    
    found=False
    ref_linha=np.zeros(linhas)
    
    for i in range(linhas):
        ref_linha[i]=Igray[i][round(colunas/2)]
        if(ref_linha[i]==255):
            found=True
    if(found):
        if(hasBall==False):
            ballCount+=1
            hasBall=True
    else:
        hasBall=False

    ax3.clear()
    ax3.plot(np.arange(0, ref_linha.size), ref_linha)
    ax3.set_ylim([0, 260])
    ax3.set_xlim([0, ref_linha.size])
    ax3.set_title('Coluna central da\n imagem binária')

    plt.pause(0.05)
    e2=cv2.getTickCount()
    time.append((e2-e1)/cv2.getTickFrequency())
    e1=e2

print('Processo finalizado!')
print(ballCount)
print(np.average(time))
cv2.waitKey(0)