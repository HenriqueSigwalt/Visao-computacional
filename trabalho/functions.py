import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def identifyDict():

    #Carrega a imagem do template
    template=cv2.imread("trabalho/banco_de_imagens/fonte_mercosul.png")
    #plt.imshow(template)
    #plt.show()
    
    #Prepara o vetor de letras identificadas
    letters=[{"Nome":"A"},{"Nome":"B"},{"Nome":"C"},{"Nome":"D"},{"Nome":"E"},{"Nome":"F"},
             {"Nome":"G"},{"Nome":"H"},{"Nome":"I"},{"Nome":"J"},{"Nome":"K"},{"Nome":"L"},
             {"Nome":"M"},{"Nome":"N"},{"Nome":"O"},{"Nome":"P"},{"Nome":"Q"},{"Nome":"R"},
             {"Nome":"S"},{"Nome":"T"},{"Nome":"U"},{"Nome":"V"},{"Nome":"W"},{"Nome":"X"},
             {"Nome":"Y"},{"Nome":"Z"},{"Nome":"0"},{"Nome":"1"},{"Nome":"2"},{"Nome":"3"},
             {"Nome":"4"},{"Nome":"5"},{"Nome":"6"},{"Nome":"7"},{"Nome":"8"},{"Nome":"9"}]
    
    #Binariza e inverte  imagem
    lin, col, cam = template.shape
    template_inv=np.zeros((lin,col),np.uint8)
    for i in range(lin):
        for j in range(col):
            if np.all(template[i,j]==0):
                template_inv[i,j]=255
    #plt.imshow(template_inv,cmap="gray")
    #plt.show()

    #Encontra os contornos e divide as linhas
    contours, hierarchy=cv2.findContours(template_inv,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    bounds=[cv2.boundingRect(i) for i in contours]
    bounds=sorted(bounds,key=lambda bound:bound[0],reverse=False)

    linhas=[]

    for i in bounds:
        found=False
        cv2.rectangle(template,(i[0],i[1]),(i[0]+i[2],i[1]+i[3]),(0,255,0),2)
        if not(linhas):
            new_linha={"Start":i[1],"End":i[1]+i[3],"Letters":[i]}
            linhas.append(new_linha)
        else:
            for j in linhas:
                if i[1]<j["End"] and (i[1]+i[3])>j["Start"]:
                    j["Letters"].append(i)
                    found=True
            if not found:
                new_linha={"Start":i[1],"End":i[1]+i[3],"Letters":[i]}
                linhas.append(new_linha)
    linhas=sorted(linhas,key=lambda linha:linha["Start"],reverse=False)
    
    #Associa cada letra a uma região
    letter_count=0
    for i in linhas:
        for j in i["Letters"]:
            letters[letter_count]["Region"]=template_inv[j[1]:j[1]+j[3],j[0]:j[0]+j[2]]
            letter_count+=1

    #plt.imshow(template)
    #plt.show()

    return letters