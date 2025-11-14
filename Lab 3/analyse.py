import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

R=120 
G=180
B=106

#Compara letras com o template
def matchLetters(region,letters):
    letter={"nome":"","fit":1}
    l, w = region.shape
    if l>40: #Ignora :
        for i in letters: #Testa todas as letras do template
            linha, coluna = i["region"].shape
            #Ajusta o tamanho da letra para permitir comparação correta e binariz novamente
            new_region=cv2.resize(region,(coluna,linha))
            for a in range(linha):
                for j in range(coluna):
                    if new_region[a,j]>0:
                        new_region[a,j]=255
            #Compara as duas letras
            compare=i["region"]-new_region
            #if i["nome"]=="R":
                #plt.imshow(region,cmap="gray")
                #plt.figure()
                #plt.imshow(new_region,cmap="gray")
                #plt.figure()
                #plt.imshow(compare,cmap="gray")
                #plt.show()
            #Verifica qual a porcentagem de match
            remain=np.count_nonzero(compare)
            percent=remain/(linha*coluna)
            if(percent<letter["fit"]):#Se for o mais parecido, salva como resultado
                letter={"nome":i["nome"],"fit":percent}
    return letter["nome"]

def analyse_img(Itarget, letters, count):
    flag=False
    while not flag:
        if not flag:
            linha, coluna, layers = Itarget.shape
            Ibin = np.zeros((linha,coluna),np.uint8)
            #Binariza e inverte a etiqueta
            Ri=np.float32(Itarget[:,:,2])
            Gi=np.float32(Itarget[:,:,1])
            Bi=np.float32(Itarget[:,:,0])
            dist=((Bi-B)**2 + (Gi-G)**2 + (Ri-R)**2)**(1/2)
            index=(dist<30)
            Ibin[index]=255
            #plt.imshow(Ibin,cmap="gray")
            #plt.show()
            #Encontra e ordena as regiões de cada canto da etiqueta com base na coordenada x
            contours, hierarchy = cv2.findContours(Ibin,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            corners=[cv2.boundingRect(i) for i in contours]
            corners=sorted(corners,key=lambda corner:corner[0],reverse=False)

            #Define qual contorno corresponde a cada canto com base nas coordenadas x e y
            if(corners[0][1]>corners[1][1]):
                up_left=corners[1]
                down_left=corners[0]
            else:
                up_left=corners[0]
                down_left=corners[1]
            if(corners[2][1]>corners[3][1]):
                up_right=corners[3]
                down_right=corners[2]
            else:
                up_right=corners[2]
                down_right=corners[3]
            Itarget_draw=Itarget.copy()
            cv2.rectangle(Itarget_draw,(up_left[0],up_left[1]),(up_left[0]+up_left[2],up_left[1]+up_left[3]),(255,0,0),2)
            cv2.rectangle(Itarget_draw,(down_left[0],down_left[1]),(down_left[0]+down_left[2],down_left[1]+down_left[3]),(255,0,0),2)
            cv2.rectangle(Itarget_draw,(up_right[0],up_right[1]),(up_right[0]+up_right[2],up_right[1]+up_right[3]),(255,0,0),2)
            cv2.rectangle(Itarget_draw,(down_right[0],down_right[1]),(down_right[0]+down_right[2],down_right[1]+down_right[3]),(255,0,0),2)
            if(up_left[0]==down_left[0] and up_left[1]==up_right[1]): #Para a repetição se as pontas estiverem alinhadas
                flag=True
                break
            pts_src=np.array([[up_left[0],up_left[1]],[down_left[0],down_left[1]+down_left[3]],
                            [up_right[0]+up_right[2],up_right[1]],[down_right[0]+down_right[2],down_right[1]+down_right[3]]])
            pts_dst=np.array([[25,25],[25,linha-25],[coluna-25,25],[coluna-25,linha-25]])
            #Ajusta a imagem para não ficar distorcida
            h, _ = cv2.findHomography(pts_src,pts_dst)
            Itarget=cv2.warpPerspective(Itarget,h,(coluna,linha))
        plt.imshow(Itarget_draw)
        plt.show()

    #Remove a região externa da imagem, binariza e inverte
    Inova=Itarget[up_left[3]+25:linha-down_right[3]-25,up_left[2]+25:coluna-down_right[2]-25]
    #plt.imshow(Inova)
    #plt.show()
    
    linha, coluna, layers = Inova.shape
    Igray=np.ones((linha,coluna),np.uint8)*255
    Ri=np.float32(Inova[:,:,2])
    Gi=np.float32(Inova[:,:,1])
    Bi=np.float32(Inova[:,:,0])
    dist=((Bi-255)**2 + (Gi-255)**2 + (Ri-255)**2)**(1/2)
    index=(dist<30)
    Igray[index]=0

    #plt.imshow(Igray,cmap="gray")
    #plt.show()

    #Encontra os ocontornos das letras e ordena com base na coordenada x
    contours, hierarchy = cv2.findContours(Igray,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounds=[cv2.boundingRect(i) for i in contours]
    bounds=sorted(bounds, key=lambda bound:bound[0],reverse=False)
    linhas=[]
    for i in bounds: #Para cada letra
        found=False
        cv2.rectangle(Inova,(i[0],i[1]),(i[0]+i[2],i[1]+i[3]),(0,255,0),2)
        if (not linhas): #Cria uma linha se nenhuma existe
           linhas.append({"Linha":len(linhas),"Start":i[1],"Finish":i[1]+i[3],"End":i[0]+i[2],"Letters":"","Count":0})
        for linha in linhas: 
            #Se uma ou mais linhas existem, procura a qual delas a letra pertence baseado nas coordenadas y de seu começo e fim
            if(linha["Start"]<i[1]+i[3] and linha["Finish"]>i[1]):
                if(linha["End"]<i[0]-15):
                    #Adiciona um espaço caso a distancia entre as linhas seja maior que o padrão
                    linha["Letters"]+=" "
                #Adiciona a letra na frase da linha
                linha["Letters"]+=matchLetters(Igray[i[1]-1:i[1]+i[3]+1,i[0]-1:i[0]+i[2]+1],letters)
                linha["Count"]+=1
                linha["End"]=i[0]+i[2]
                found=True
        if(not found):
            #Caso a letra não pertença a nenhuma linha existente, cria uma nova para adicioná-la
            linhas.append({"Linha":len(linhas),"Start":i[1],"Finish":i[1]+i[3],"End":i[0]+i[2],"Letters":"","Count":0})
            linhas[-1]["Letters"]+=matchLetters(Igray[i[1]-1:i[1]+i[3]+1,i[0]-1:i[0]+i[2]+1],letters)
            linhas[-1]["Count"]+=1
            linhas[-1]["End"]=i[0]+i[2]
            linhas=sorted(linhas,key=lambda line:line["Start"])
    
    print(f"Resultado imagem {count}")
    for i in linhas:
        #Mostra as linhas identificadas, excluindo linhas invalidas
        if i["Letters"] != "" and i["Letters"] != " ":
            print(i["Letters"])
    #plt.imshow(Inova)
    #plt.show()
    
    return