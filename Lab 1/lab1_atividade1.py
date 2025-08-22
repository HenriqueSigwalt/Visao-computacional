import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# ----------------------------------------------------------
cap = cv2.VideoCapture('sequencia1.mp4')

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

# Seção de variáveis
# IMPLEMENTE O SEU CÓDIGO AQUI

while ret:

    # lê frame do vídeo
    ret, I1 = cap.read()

    if I1 is None:
        break

    # Algoritmo de detecção de objetos
    # IMPLEMENTE O SEU CÓDIGO AQUI

    # atualiza plot
    ax1.clear()
    ax1.imshow(cv2.cvtColor(I1, cv2.COLOR_BGR2RGB))

    ax2.clear()
    ax2.imshow(I2, cmap='gray')
    ax2.plot([n_colunas/2, n_colunas/2], [0, n_linhas-1], ':')

    '''ax3.clear()
    ax3.plot(np.arange(0, ref_linha.size), ref_linha)
    ax3.set_ylim([0, 260])
    ax3.set_xlim([0, ref_linha.size])
    ax3.set_title('Coluna central da\n imagem binária')'''

    plt.pause(0.05)

print('Processo finalizado!')
cv2.waitKey(0)