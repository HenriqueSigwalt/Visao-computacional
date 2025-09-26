import cv2
import numpy as np

# l^e arquivo de imagem
I1 = cv2.imread("Banco de imagens/castle.jpg")
cv2.imshow('Imagem de entrada', I1)

kernel = np.ones((3,3),np.float32)
# os elementos do kernel devem ser do tipo np.float32
# filtragem linear 2D
I2 = cv2.filter2D(I1, -1, kernel, borderType=cv2.BORDER_REFLECT101)
cv2.imshow('Imagem filtrada', I2)
cv2.waitKey(0)