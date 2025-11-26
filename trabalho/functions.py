import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

#Função que identifica as letras do template
def identifyDict(img):

    #Carrega a imagem do template
    template=cv2.imread(img)
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

#Compara letras com o template
def matchLetters(region,letters):
    letter={"nome":"","fit":1}
    l, w = region.shape
    for i in letters: #Testa todas as letras do template
        linha, coluna = i["Region"].shape
        #Ajusta o tamanho da letra para permitir comparação correta e binariz novamente
        new_region=cv2.resize(region,(coluna,linha))
        for a in range(linha):
            for j in range(coluna):
                if new_region[a,j]>0:
                    new_region[a,j]=255
        #Compara as duas letras
        compare=i["Region"]-new_region
        """if i["Nome"]=="W":
            plt.imshow(region,cmap="gray")
            plt.figure()
            plt.imshow(new_region,cmap="gray")
            plt.figure()
            plt.imshow(compare,cmap="gray")
            plt.show()"""
        #Verifica qual a porcentagem de match
        remain=np.count_nonzero(compare)
        percent=remain/(linha*coluna)
        if(percent<letter["fit"]):#Se for o mais parecido, salva como resultado
            letter={"nome":i["Nome"],"fit":percent}
    return letter["nome"]

def platePrepare(img_url):

    #Pega uma imagem como base da aparencia da placa
    img_base=cv2.imread("trabalho/banco_de_imagens/placa_base.jpg")
    base_bin=cv2.cvtColor(img_base,cv2.COLOR_BGR2GRAY)
    base_lin, base_col, _= img_base.shape

    #Prepara a placa a ser analizada
    img=cv2.imread(img_url)
    img_bin=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #plt.imshow(img_bin, cmap="gray")

    #Procura a imagem base na imagem sendo analisada
    sift=cv2.SIFT_create()
    k1, des1 = sift.detectAndCompute(base_bin, None)
    k2, des2 = sift.detectAndCompute(img_bin, None)
    bf=cv2.BFMatcher(cv2.NORM_L2,crossCheck=True)
    matches=bf.match(des1,des2)
    matches=sorted(matches, key= lambda match:match.distance, reverse=False)
    #show_match=cv2.drawMatches(img_base,k1,img,k2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #plt.imshow(cv2.cvtColor(show_match, cv2.COLOR_BGR2RGB))
    #plt.figure()

    #Realiza a detecção dos pontos de interesse
    src_pts = np.zeros((100,1,2), np.float32)
    dst_pts = np.zeros((100,1,2), np.float32)
    new_size=np.float32([[0,0],[0,base_lin-1],[base_col-1,base_lin-1],[base_col-1,0]]).reshape(-1,1,2)

    for i in range(100):
        src_pts[i] = np.float32(k1[matches[i].queryIdx].pt).reshape(-1, 1, 2)
        dst_pts[i] = np.float32(k2[matches[i].trainIdx].pt).reshape(-1, 1, 2)

    m, _ = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5)
    lines=cv2.perspectiveTransform(new_size,m)
    #img_rect=cv2.polylines(img.copy(),[np.int32(lines)],True,(0,255,0),2,cv2.LINE_AA)
    #plt.imshow(img_rect)

    #Define região de interesse a ser usada com base nos pontos encontrados
    xbound=sorted(lines,key=lambda line:line[0,0],reverse=False)
    ybound=sorted(lines,key=lambda line:line[0,1],reverse=False)
    top_left=sorted(xbound[0:2], key=lambda line:line[0,1],reverse=False)[0][0]
    top_right=sorted(xbound[2:4],key=lambda line:line[0,1],reverse=False)[0][0]
    bot_left=sorted(xbound[0:2], key=lambda line:line[0,1],reverse=True)[0][0]
    bot_right=sorted(xbound[2:4],key=lambda line:line[0,1],reverse=True)[0][0]
    length=round(xbound[3][0,0]-xbound[0][0,0])
    height=round(ybound[3][0,1]-ybound[0][0,1])

    #length_img, height_img,_=img.shape

    #Ajusta a perspectiva da imagem
    pts_src=np.array([top_left,bot_left,top_right,bot_right])
    pts_dst=np.array([[0,0],[0,height],[length,0],[length,height]])
    h,_ = cv2.findHomography(pts_src,pts_dst)
    img_new=cv2.warpPerspective(img,h,(length,height))
    #plt.imshow(img_new)
    
    kernel=np.ones([2,3],np.uint8)
    img_new=cv2.erode(img_new,kernel)

    plt.show()
    plate=img_new
    return plate

def readPlate(plate, letters):
    text=""

    #Binariza região de interesse
    lin, col, _ = plate.shape
    new_bin=np.zeros((lin,col),np.uint8)
    for i in range(lin):
        for j in range(col):
            if(np.all(plate[i,j]>=150)):
                new_bin[i,j]=255
    new_bin=cv2.bitwise_not(new_bin)
    plt.imshow(new_bin,cmap="gray")
    plt.figure()
    
    #Encontra contorno da placa e ajusta perspectiva até estar reta
    contours, _ = cv2.findContours(new_bin, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    bounds=[cv2.boundingRect(i) for i in contours]
    bounds=sorted(bounds, key=lambda x:x[0],reverse=False)
    for i in bounds:
        x=i[0]
        if x<=0:
            x=1
        y=i[1]
        if y<=0:
            y=1
        alt=i[3]
        if y+alt>lin:
            alt=lin-y
        compr=i[2]
        if x+compr>col:
            compr=col-x
        if (alt>round(lin*0.5)) and (alt<round(lin*0.8)) and (compr>round(col*0.01))and (compr<round(col*0.5)): #Remove contornos indesejados
            cv2.rectangle(plate,(x,y),(x+i[2],y+alt),(0,255,0),2)
            text+=matchLetters(new_bin[y-1:y+alt+1,x-1:x+i[2]+1],letters)
        else:
            cv2.rectangle(plate,(x,y),(x+i[2],y+alt),(255,0,0),2)
    plt.imshow(plate)

    plt.show()

    return text