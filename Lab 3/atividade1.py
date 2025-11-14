import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from analyse import analyse_img

template=cv2.imread("Lab 3/Moodle/banco_de_imagens/template_letras.png", cv2.IMREAD_GRAYSCALE)
template=cv2.bitwise_not(template)

template_lin, template_col = template.shape
template_cor=np.zeros((template_lin,template_col,3),np.uint8)
#Binariza e inverte a imagem para permitir encontrar contornos
for i in range(template_lin):
    for j in range(template_col):
        if(template[i,j]>0):
            template_cor[i,j]=(255,255,255)

#Encontra contornos
contours, hierarchy = cv2.findContours(template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#Dicionario de letras
letters=[{"nome":"A"},{"nome":"B"},{"nome":"C"},{"nome":"D"},{"nome":"E"},{"nome":"F"},{"nome":"G"},{"nome":"H"},{"nome":"I"},{"nome":"J"},{"nome":"K"},{"nome":"L"},{"nome":"M"},
         {"nome":"N"},{"nome":"O"},{"nome":"P"},{"nome":"Q"},{"nome":"R"},{"nome":"S"},{"nome":"T"},{"nome":"U"},{"nome":"V"},{"nome":"X"},{"nome":"Z"},{"nome":"W"}]
letter_count=0
#ENcontra e ordena as regiões que contem as letras com base na coordenada x
bounds=[cv2.boundingRect(i) for i in contours]
bounds=sorted(bounds,key=lambda bound:bound[0],reverse=False)

#Adiciona as letras no dicionario e marca a posição na tela
for i in bounds:
    cv2.rectangle(template_cor,(i[0],i[1]),(i[0]+i[2],i[1]+i[3]),(0,255,0),2)
    letters[letter_count]["region"]=template[i[1]:i[1]+i[3],i[0]:i[0]+i[2]]
    letter_count+=1

#plt.imshow(template_cor)
#plt.show()

#Processa as etiquetas
for i in range(8):
    Itarget=cv2.imread(f"Lab 3/Moodle/banco_de_imagens/im{i+1}.png")
    #plt.imshow(Itarget)
    #plt.show()

    analyse_img(Itarget,letters,i+1)