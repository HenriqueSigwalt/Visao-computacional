import cv2
import numpy as np

# abri arquivos de videos
cap1 = cv2.VideoCapture('Lab 1\Moodle\Chromakey.mp4')
cap2 = cv2.VideoCapture('Lab 1\Moodle\Clouds.mp4')

n_linhas = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_colunas  = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('resultado_atividade2.avi', fourcc, 24, (n_colunas, n_linhas))
e1=cv2.getTickCount()
time=[]
# processa cada quadro dos vídeos
ret = True
while ret:

    # lê frame dos vídeos
    ret, I1 = cap1.read()
    _, I2 = cap2.read()

    if I1 is None:
        break
    
    R=80
    G=230
    B=190
    
    Ri=np.float32(I1[:,:,2])
    Gi=np.float32(I1[:,:,1])
    Bi=np.float32(I1[:,:,0])
    
    dist=((Bi-B)**2 + (Gi-G)**2 + (Ri-R)**2)**(1/2)
    index=(dist<50)
    mask1=np.zeros([n_linhas,n_colunas],np.uint8)
    mask1[index]=255
    index=(dist>50)
    mask2=np.zeros([n_linhas,n_colunas],np.uint8)
    mask2[index]=255
    
    I_step=cv2.bitwise_and(I1, I1, mask=mask2)
    #cv2.imshow('Imagem', mask2)
    I_step2=cv2.bitwise_and(I2, I2, mask=mask1)
    #cv2.imshow('Imagem', I_step2)
    I_final=cv2.bitwise_or(I_step,I_step2)
    #cv2.imshow('Imagem', I_final)
    
    video.write(I_final)
    e2=cv2.getTickCount()
    time.append((e2-e1)/cv2.getTickFrequency())
    e1=e2
    if cv2.waitKey(5) == ord('q'):
        break

print(np.average(time))
cv2.destroyAllWindows()
video.release()