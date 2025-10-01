import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


def Canny(Ie, sig, th, tl):
    #KERNEL GAUSSIANO
    
    w = 6*sig +1
    h = w//2

    j, i = np.mgrid[-h:h+1, -h:h+1]  
    kernelg = (1/(2*np.pi*sig**2)) * np.exp(-(i**2 + j**2)/(2*sig**2))
    kernelg = kernelg/kernelg.sum()

    #FILTRO 2D
    Is = cv2.filter2D(Ie, -1, kernelg, borderType=cv2.BORDER_REFLECT101)

    #cv2.imshow("Imagem suavizada", Is)

    #DETECÇÃO DE BORDA

    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.transpose(Kx)

    Ix = cv2.filter2D(Is, -1, Kx, borderType=cv2.BORDER_REFLECT101)
    Iy = cv2.filter2D(Is, -1, Ky, borderType=cv2.BORDER_REFLECT101)

    #cv2.imshow("Ix", Ix)
    #cv2.imshow("Iy", Iy)

    #MAGNITUDE

    Im = ((Ix**2)+(Iy**2))**(1/2)
    #cv2.imshow("Im", Im)

    #GRADIENTE

    Ig = np.arctan2(Iy,Ix)

    #plt.imshow(Ig, cmap='jet')
    #plt.colorbar()
    #plt.show()

    #SUPRESSÃO DE NÃO MÁXIMOS 

    #Como só importa a direção e não a orientação (sinal), é possivel fazer apenas uma normalização e comparar o "excedente"
    theta = (np.rad2deg(Ig) + 180.0) % 180.0

    dir = np.zeros(theta.shape, np.uint8) # Já que dir inicia zerado, os intervalos para 0°/180° (theta < 22.5 $ theta > 157.5), que dariam 0, já estão certos.
    dir[(theta >= 22.5) & (theta < 67.5)]   = 45
    dir[(theta >= 67.5) & (theta < 112.5)]  = 90    #substitui direto no dir os valores certos
    dir[(theta >= 112.5) & (theta < 157.5)] = 135

    M = Im.astype(np.float32)
    Gn = np.zeros(M.shape, np.float32)

    pc = M[1:-1, 1:-1]      #ignora as bordas da imagem para fazer a supressão
    dc = dir[1:-1, 1:-1]

    leste      =   M[1:-1, 2:  ]     
    oeste      =   M[1:-1, 0:-2]     
    norte      =   M[0:-2,  1:-1]    
    sul        =   M[2:  ,  1:-1]    
    nordeste   =   M[0:-2,  2:  ]    
    sudoeste   =   M[2:  ,  0:-2]    
    noroeste   =   M[0:-2,  0:-2]    
    sudeste    =   M[2:  ,  2:  ]    

    Gc = np.zeros(pc.shape, np.float32)

    hor = (dc == 0)
    Gc[hor] = pc[hor]*(pc[hor] >= leste[hor])*(pc[hor] >= oeste[hor])

    diag1 = (dc == 45)
    Gc[diag1] = pc[diag1]*(pc[diag1] >= nordeste[diag1])*(pc[diag1] >= sudeste[diag1])

    ver = (dc == 90)
    Gc[ver] = pc[ver]*(pc[ver] >= norte[ver])*(pc[ver] >= sul[ver])

    diag2 = (dc == 135)
    Gc[diag2] = pc[diag2]*(pc[diag2] >= noroeste[diag2])*(pc[diag2] >= sudoeste[diag2])

    Gn[1:-1, 1:-1] = Gc

    Ign = (Gn / (Gn.max() + 1e-8) * 255).astype(np.uint8)
    #cv2.imshow("Ign", Ign)


    #LIMIALIZAÇÃO DUPLO

    Gnh = (Gn >= th).astype(np.uint8) * 255          
    Gnl = (Gn >= tl).astype(np.uint8) * 255
    #cv2.imshow("Gnl ", Gnl)

    Gnl = Gnl - Gnh

    #cv2.imshow("Gnh ", Gnh)
    #cv2.imshow("Gnl ", Gnl)

    #CONECTIVIDADE

    H, W = Gnh.shape
    val = np.zeros((H, W), dtype=bool)   # mapa de válidos pro Gnl

    for u, v in np.argwhere(Gnh > 0):

        r0, r1 = max(0, u-1), min(H, u+2)
        c0, c1 = max(0, v-1), min(W, v+2)

        val[r0:r1, c0:c1] |= (Gnl[r0:r1, c0:c1] > 0)

    Gnl = (val.astype(np.uint8) * 255)

    If = cv2.add(Gnh, Gnl)

    cv2.waitKey(0)
    return If   


Ie = cv2.imread(r"C:\Users\avgui\OneDrive\Documentos\CAC3040\Banco de imagens-20250821\castle.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
#cv2.imshow("Imagem de entrada", Ie)
If = Canny(Ie, 1.4, 0.3, 0.14 )

#cv2.imshow("Imagem de saída", If)

plt.imshow(If, cmap='gray', vmin=0, vmax=255)
plt.show()

cv2.waitKey(0)
