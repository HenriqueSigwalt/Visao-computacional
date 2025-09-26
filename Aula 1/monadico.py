import cv2
import numpy as np
from matplotlib import pyplot as plt

# le arquivo de imagem
I1 = cv2.imread('Banco de imagens/castle.jpg', cv2.IMREAD_GRAYSCALE)
# obt´em o histograma da imagem I1
hist1 = cv2.calcHist([I1],[0],None,[256],[0,256])
# converte para array de 1 dimens~ao
hist1 = hist1[:,0]
# fun¸c~ao densidade de probabilidade (pdf)
pdf = hist1/np.sum(hist1)
# fun¸c~ao de distribui¸c~ao acumulada (cdf)
cdf = np.cumsum(pdf)
#fun¸c~ao de processamento (array de mapeamento)
f = np.uint8(256 * cdf)
# equaliza¸c~ao de histograma
n_linhas, n_colunas = I1.shape
I2 = np.zeros((n_linhas, n_colunas), np.uint8)
for y in np.arange(0, n_linhas):
    for x in np.arange(0, n_colunas):
        I2[y,x] = f[I1[y,x]]
# apresenta imagens na tela
cv2.imshow('Imagem original', I1)
cv2.imshow('Imagem processada', I2)
cv2.waitKey(0)
# calcula o histograma da imagem em escala de cinza
hist1 = cv2.calcHist([I1],[0],None,[256],[0,256])
hist2 = cv2.calcHist([I2],[0],None,[256],[0,256])
#apresenta histograma
plt.bar(np.arange(0,256), hist1[:,0])
plt.bar(np.arange(0,256), hist2[:,0])
plt.xlim([0,255])
plt.title('Histograma')
plt.ylabel('Contagem dos pixels')
plt.xlabel('Valores do pixels')
plt.show()
cv2.waitKey(0)