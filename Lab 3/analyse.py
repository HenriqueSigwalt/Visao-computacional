import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

R=120 
G=180
B=106

def matchLetters(region,letters):
    letter={"nome":"","fit":1}
    l, w = region.shape
    if l>40:
        #plt.imshow(region,cmap="gray")
        #plt.show()
        for i in letters:
            linha, coluna = i["region"].shape
            new_region=cv2.resize(region,(coluna,linha))
            for a in range(linha):
                for j in range(coluna):
                    if new_region[a,j]>0:
                        new_region[a,j]=255
            compare=i["region"]-new_region
            #if i["nome"]=="R":
                #plt.imshow(region,cmap="gray")
                #plt.figure()
                #plt.imshow(new_region,cmap="gray")
                #plt.figure()
                #plt.imshow(compare,cmap="gray")
                #plt.show()
            #plt.imshow(compare,cmap="gray")
            #plt.show()
            remain=np.count_nonzero(compare)
            percent=remain/(linha*coluna)
            if(percent<letter["fit"]):    
                letter={"nome":i["nome"],"fit":percent}
    return letter["nome"]

def analyse_img(Itarget, letters):
    linha, coluna, layers = Itarget.shape
    Ibin = np.zeros((linha,coluna),np.uint8)

    Ri=np.float32(Itarget[:,:,2])
    Gi=np.float32(Itarget[:,:,1])
    Bi=np.float32(Itarget[:,:,0])
    dist=((Bi-B)**2 + (Gi-G)**2 + (Ri-R)**2)**(1/2)
    index=(dist<30)
    Ibin[index]=255
    #plt.imshow(Itarget)
    #plt.show()

    contours, hierarchy = cv2.findContours(Ibin,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    corners=[cv2.boundingRect(i) for i in contours]
    corners=sorted(corners,key=lambda corner:corner[0],reverse=False)

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

    cv2.rectangle(Itarget,(up_left[0],up_left[1]),(up_left[0]+up_left[2],up_left[1]+up_left[3]),(255,0,0),2)
    cv2.rectangle(Itarget,(down_left[0],down_left[1]),(down_left[0]+down_left[2],down_left[1]+down_left[3]),(255,0,0),2)
    cv2.rectangle(Itarget,(up_right[0],up_right[1]),(up_right[0]+up_right[2],up_right[1]+up_right[3]),(255,0,0),2)
    cv2.rectangle(Itarget,(down_right[0],down_right[1]),(down_right[0]+down_right[2],down_right[1]+down_right[3]),(255,0,0),2)
    #plt.imshow(Itarget)
    #plt.show()
    
    pts_src=np.array([[up_left[0],up_left[1]],[down_left[0],down_left[1]+down_left[3]],
                     [up_right[0]+up_right[2],up_right[1]],[down_right[0]+down_right[2],down_right[1]+down_right[3]]])
    pts_dst=np.array([[0,0],[0,linha],[coluna,0],[coluna,linha]])

    h, _ = cv2.findHomography(pts_src,pts_dst)
    Itarget=cv2.warpPerspective(Itarget,h,(coluna,linha))
    #plt.imshow(Itarget)
    #plt.show()

    Inova=Itarget[up_left[3]:linha-down_right[3],up_left[2]:coluna-down_right[2]]
    plt.imshow(Inova)
    plt.show()
    linha, coluna, layers = Inova.shape
    Igray=np.ones((linha,coluna),np.uint8)*255
    for i in range(linha):
        for j in range(coluna):
            if(Inova[i,j,0]==255 and Inova[i,j,1]==255 and Inova[i,j,2]==255):
                Igray[i,j]=0
    #plt.imshow(Igray,cmap="gray")
    #plt.show()

    contours, hierarchy = cv2.findContours(Igray,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounds=[cv2.boundingRect(i) for i in contours]
    bounds=sorted(bounds, key=lambda bound:bound[0],reverse=False)
    linhas=[]
    for i in bounds:
        found=False
        cv2.rectangle(Inova,(i[0],i[1]),(i[0]+i[2],i[1]+i[3]),(0,255,0),2)
        if (not linhas):
           linhas.append({"Linha":len(linhas),"Start":i[1],"Finish":i[1]+i[3],"End":i[0]+i[2],"Letters":"","Count":0})
        for linha in linhas:
            if(linha["Start"]<i[1]+i[3] and linha["Finish"]>i[1]):
                if(linha["End"]<i[0]-15):
                    linha["Letters"]+=" "
                linha["Letters"]+=matchLetters(Igray[i[1]-1:i[1]+i[3]+1,i[0]-1:i[0]+i[2]+1],letters)
                linha["Count"]+=1
                linha["End"]=i[0]+i[2]
                found=True
        if(not found):
            linhas.append({"Linha":len(linhas),"Start":i[1],"Finish":i[1]+i[3],"End":i[0]+i[2],"Letters":"","Count":0})
            linhas[-1]["Letters"]+=matchLetters(Igray[i[1]-1:i[1]+i[3]+1,i[0]-1:i[0]+i[2]+1],letters)
            linhas[-1]["Count"]+=1
            linhas[-1]["End"]=i[0]+i[2]
            linhas=sorted(linhas,key=lambda line:line["Start"])
    
    for i in linhas:
        print(i["Letters"])
    plt.imshow(Inova)
    plt.show()
    
    return