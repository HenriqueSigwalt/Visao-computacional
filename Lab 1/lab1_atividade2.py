import visaoComputacional as visco
import cv2
import numpy as np

# abri arquivos de videos
cap1 = cv2.VideoCapture('Chromakey.mp4')
cap2 = cv2.VideoCapture('Clouds.mp4')

n_linhas = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_colunas  = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('resultado_atividade2.avi', fourcc, 24, (n_colunas, n_linhas))

# processa cada quadro dos vídeos
ret = True
while ret:

    # lê frame dos vídeos
    ret, I1 = cap1.read()
    _, I2 = cap2.read()

    if I1 is None:
        break

    # Efeito de chromakey
    
    #video.write(I_final)

    if cv2.waitKey(5) == ord('q'):
        break

cv2.destroyAllWindows()
video.release()