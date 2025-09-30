import cv2
import numpy as np
from matplotlib import pyplot as plt

# Carregar imagem 
I1 = cv2.imread("Lab 2\Moodle\castle.jpg",cv2.IMREAD_GRAYSCALE)
plt.figure()
plt.title("Imagem Original")
plt.imshow(I1, cmap='gray')

# Kernel gaussiano
w = 5
sigma = (w-1)/6
kernel = np.zeros((w,w), np.float32)

v1 = 1/(2 * np.pi * (sigma**2))
v2 = -1/(2 * (sigma**2))

for y in range(w):
    for x in range(w):
        j = y - (w-1)/2
        i = x - (w-1)/2
        kernel[y,x] = v1 * np.exp(v2*(j*2 + i*2))

kernel /= np.sum(kernel)

# Imagem filtrada (suavizada)
Is = cv2.filter2D(I1, -1, kernel, borderType=cv2.BORDER_REFLECT101)
plt.figure()
plt.title("Imagem Suavizada")
plt.imshow(Is, cmap='gray')

# Derivada em relação a x (Sobel)
Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
Ix = cv2.filter2D(Is, cv2.CV_32F, Kx, borderType=cv2.BORDER_REFLECT101)
plt.figure()
plt.title("Derivada em X")
plt.imshow(Ix, cmap='gray')

# Derivada em relação a y (Sobel)
Ky = Kx.T
Iy = cv2.filter2D(Is, cv2.CV_32F, Ky, borderType=cv2.BORDER_REFLECT101)
plt.figure()
plt.title("Derivada em Y")
plt.imshow(Iy, cmap='gray')

# Magnitude do gradiente
M = np.sqrt(Ix**2 + Iy**2)

# Normaliza magnitude para [0,255]
M_norm = (M / (np.max(M) + 1e-10)) * 255
M_norm = M_norm.astype(np.uint8)
plt.figure()
plt.title("Magnitude do Gradiente Normalizada")
plt.imshow(M_norm, cmap='gray')

# Ângulo do gradiente em graus [0,180)
alfa = np.arctan2(Iy, Ix)
angle = np.rad2deg(alfa)
angle[angle < 0] += 180

# Supressão de não máximos
Gn = np.zeros(Is.shape, np.uint8)
lin, col = Is.shape
for y in range(1, lin-1):
    for x in range(1, col-1):
        ang = angle[y, x]
        if (0 <= ang < 22.5) or (157.5 <= ang < 180):
            u, v = M_norm[y, x+1], M_norm[y, x-1]
        elif (22.5 <= ang < 67.5):
            u, v = M_norm[y-1, x+1], M_norm[y+1, x-1]
        elif (67.5 <= ang < 112.5):
            u, v = M_norm[y-1, x], M_norm[y+1, x]
        else:
            u, v = M_norm[y-1, x-1], M_norm[y+1, x+1]
        if (M_norm[y, x] >= u) and (M_norm[y, x] >= v):
            Gn[y, x] = M_norm[y, x]

plt.figure()
plt.title("Supressão de Não Máximos")
plt.imshow(Gn, cmap='gray')

# Limiarização dupla correta
Th = 70  # limiar alto
Tl = 30  # limiar baixo

Gnh = (Gn >= Th).astype(np.uint8)*255
Gnl = ((Gn >= Tl) & (Gn < Th)).astype(np.uint8)*255

plt.figure()
plt.title("Bordas Fortes (limiar alto)")
plt.imshow(Gnh, cmap='gray')

plt.figure()
plt.title("Bordas Fracas (entre limiares)")
plt.imshow(Gnl, cmap='gray')

# Análise de conectividade: promove bordas fracas conectadas a fortes
Ib = np.copy(Gnh)
for y in range(1, lin-1):
    for x in range(1, col-1):
        if Gnh[y, x] == 255:
            for j in [-1, 0, 1]:
                for i in [-1, 0, 1]:
                    if Gnl[y+j, x+i] == 255:
                        Ib[y+j, x+i] = 255

# Garantir que Ib seja binária (0 ou 255)
Ib = (Ib > 0).astype(np.uint8)*255

plt.figure()
plt.title("Imagem de Bordas Final (após conectividade)")
plt.imshow(Ib, cmap='gray')

# Comparação com Canny do OpenCV
I_teste = cv2.Canny(I1, 50, 100, apertureSize=3, L2gradient=False)
plt.figure()
plt.title("Canny do OpenCV")
plt.imshow(I_teste, cmap='gray')

plt.show()