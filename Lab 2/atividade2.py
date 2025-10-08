import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

#Função de encontrar bordas
def canny(I1,sigma,ti,ts):
    linhas, colunas = I1.shape
    #Define tamanho do kernel
    w=int(6*sigma+1)
    kernel = np.zeros((w,w), np.float32)
    #Aplicação do kernel gaussiano
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

    # Derivada em relação a x (Sobel)
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    grad_X = cv2.filter2D(I1, cv2.CV_32F, Kx, borderType=cv2.BORDER_REFLECT101)

    # Derivada em relação a y (Sobel)
    Ky = Kx.T
    grad_y = cv2.filter2D(I1, cv2.CV_32F, Ky, borderType=cv2.BORDER_REFLECT101)

    #Magnitude do gradiente
    grad_mag=np.sqrt(grad_X**2+grad_y**2)
    #fase do gradiente
    grad_dir=np.arctan2(grad_y,grad_X)* 180 / np.pi

    #Imagem
    I2=np.zeros([linhas,colunas],np.uint8)
    Is=np.zeros([linhas,colunas],np.uint8)
    Ii=np.zeros([linhas,colunas],np.uint8)
    Ib=np.zeros([linhas,colunas],np.uint8)

    #Normaliza a magnitude (Não lembro o pq)
    grad_mag=((grad_mag/(np.max(grad_mag)+1e-10))*255).astype(np.uint8)

    for i in range(1,linhas-1):
        for j in range(1,colunas-1):
            theta=grad_dir[i,j]
            mag=grad_mag[i,j]
            #Determina quais são os visinhos com base no intervalo onde estão os 
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
            #Se o pixel atual tiver gradiente maior que os vizinhos coloca ele na imagem I2
            if (u<=mag)and(v<=mag):
                I2[i,j]=mag

    #Faz a limiarização superior da imagem
    Is[I2>ts]=I2[I2>ts]
    #Faz a limiarização inferior da imagem
    Ii[I2>ti]=I2[I2>ti]
    #Calcula a diferença das duas limiarizações
    Ii=Ii-Is
    
    #Percorre toda a imagem superior
    for i in range(linhas):
        for j in range(colunas):
            #Se o pixel foi identificado como borda
            if(Is[i,j]>0):
                #Defini todos os pixel em volta dele como borda
                Ib[i-1:i+2,j-1:j+2]=Ii[i-1:i+2,j-1:j+2]
    #Defini todos os pixels da imagem superior como borda
    Ib=Ib+Is

    #Retorna as imagens geradas
    return Ib

I1=cv2.imread("Lab 2\Moodle\castle.jpg",cv2.IMREAD_GRAYSCALE)
#tamanho do kernel
w=3
sigma=(w-1)/6

#Limiares inferiores e superiores
ti=0.01
ts=0.1

#Realiza a detecção de bordas
Ib=canny(I1,sigma,ti,ts)

plt.figure()
plt.imshow(Ib, cmap='gray')
plt.show()