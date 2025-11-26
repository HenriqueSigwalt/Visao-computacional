import cv2
import numpy as np
from matplotlib import pyplot as plt


def centroide(I1):
    # Cálculo do Momento
    m00 = mpq(I1, 0, 0)
    m01 = mpq(I1, 0, 1)
    m10 = mpq(I1, 1, 0)

    # Centroide
    xc = m10/m00
    yc = m01/m00

    centro = np.array([xc, yc])
    return centro

def mpq(I1, p, q):
    y, x = np.where(I1)
    mpq = np.float32(np.sum((x**p) * (y**q) * I1[y,x]))
    return mpq

def bounding_box(I1):
    y, x = np.where(I1)
    ymin = np.min(y)
    ymax = np.max(y)
    xmin = np.min(x)
    xmax = np.max(x)

    p0 = np.array([xmin, ymin])
    p1 = np.array([xmax, ymax])

    return p0, p1

def analisaRegioes(I1):
    InfoRegioes = []

    # Rotulamento
    num_elem, I_labels = cv2.connectedComponents(I1)


    for i in np.arange(1, num_elem):
        # Dados do componente
        dados_do_componente = dict()

        # imagem do componente conectado
        I_componente = np.uint8(I_labels == i) * 255
        dados_do_componente['imagem'] = I_componente

        # Boudingbox
        p0, p1 = bounding_box(I_componente)
        dados_do_componente['bb_p0'] = p0
        dados_do_componente['bb_p1'] = p1

        # Area
        m00 = mpq(I_componente, 0, 0)
        dados_do_componente['area'] = m00

        # Centroide
        c0 = centroide(I_componente)
        dados_do_componente['centroide'] = c0

        # Momentos Centrais
        u20 = upq(I_componente, 2, 0)
        u02 = upq(I_componente, 0, 2)
        u11 = upq(I_componente, 1, 1)

        # Matriz de inércia da região
        J = np.array([[u20, u11], [u11, u02]])

        # Matriz de inercia da elipse equivalente
        Je = 4/m00 * J

        a_valores, a_vetores = np.linalg.eig(Je)

        # Raios da elipse
        raios = np.sort(np.sqrt(a_valores))
        menor_raio = raios[0]
        maior_raio = raios[1]

        pos = np.argmax(a_valores)
        vx = a_vetores[0, pos]
        vy = a_vetores[1, pos]
        orientacao = np.rad2deg(np.atan2(vy, vx))

        dados_do_componente['razao_raios'] = menor_raio/maior_raio
        dados_do_componente['orientacao'] = orientacao

        # Contornos do componente
        contorno, hierarquia = cv2.findContours(I_componente, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        x = contorno[0][:,0,0]
        y = contorno[0][:,0,1]
        # cv2.drawContours(I_componente, contorno, 0, color, thick)

        # Perímetro
        perimetro = np.sqrt((y[-1] - y[0])**2 + (x[-1] - x[0])**2)
        N = len(y)

        for n in np.arange(0, N-1):
            perimetro = perimetro + np.sqrt((y[n] - y[n+1])**2 + (x[n] - x[n+1])**2)

        dados_do_componente['perimetro'] = perimetro

        # Cálculo da circularidade
        rho = 4*np.pi*mpq(I_componente, 0, 0)/(perimetro**2)

        dados_do_componente['circularidade'] = rho

        # Curva de distancia
        xc = c0[0]
        yc = c0[1]

        curva_distancia = np.zeros(N)
        for n in np.arange(0, N):
            curva_distancia[n] = np.sqrt((y[n] - yc)**2 + (x[n] - xc)**2)
        
        # Interpolação
        N = 200
        N1= len(curva_distancia)
        curva_distancia_i = np.interp(np.linspace(0, N1 - 1, N), np.arange(0, N1), curva_distancia)

        # Normalização
        cd_zoff = curva_distancia_i - np.mean(curva_distancia_i)

        # Energia igual a 1
        curva_distancia = cd_zoff/np.sqrt(np.sum(cd_zoff*cd_zoff))

        dados_do_componente['curva_distancia'] = curva_distancia

        InfoRegioes.append(dados_do_componente.copy())
    
    return InfoRegioes

def desenhaa_bb(I1, InfoRegioes, color, thick):
    num_regioes = len(InfoRegioes)
    for i in np.arange(0, num_regioes):
        p0 = InfoRegioes[i]['bb_p0']
        p1 = InfoRegioes[i]['bb_p1']
        cv2.rectangle(I1, p0, p1, color, thick)

def desenha_centroide(I1, InfoRegioes, color, radius):
    num_regioes = len(InfoRegioes)

    for i in np.arange(0, num_regioes):
        cv2.circle(I1, np.uint32(InfoRegioes[i]['centroide']), radius, color, -1)

def upq(I1, p, q):
    c0 = centroide(I1)
    xc = c0[0]
    yc = c0[1]

    y, x = np.where(I1)

    momento_central = np.sum(((x - xc)**p) * ((y - yc)**q))
    
    return momento_central

def compute_match(y1, y2):
    N = len(y1)
    correlacao = np.zeros(N)
    
    for k in np.arange(0, N):
        correlacao[k] = np.sum(np.roll(y1, k) * y2)

    return np.max(correlacao)

curva_dist_quadrado = np.array([0.15175637,  0.13629238,  0.12082839,  0.10570478,  0.09122263,  0.07674049,
  0.06305336,  0.04971786,  0.03643567,  0.02443268,  0.01242968,  0.00102042,
 -0.00944706, -0.01991454, -0.02910064, -0.03782077, -0.04638456, -0.05315038,
 -0.0599162 , -0.06576991, -0.07039798, -0.07502605, -0.07789362, -0.08024519,
 -0.08231457, -0.08231457, -0.08231457, -0.08121717, -0.0788656 , -0.07651404,
 -0.07231092, -0.06768285, -0.06271274, -0.05594692, -0.0491811 , -0.04142509,
 -0.03270496, -0.02398482, -0.01377362, -0.00330614,  0.00746844,  0.01947144,
  0.03147443,  0.04420586,  0.05754135,  0.07087685,  0.08523668,  0.09971882,
  0.11443661,  0.1299006 ,  0.14536458,  0.14268416,  0.12722017,  0.11175619,
  0.09720859,  0.08272644,  0.06856536,  0.05522987,  0.04189437,  0.02939391,
  0.01739092,  0.00538793, -0.0051205 , -0.01558798, -0.02549631, -0.03421645,
 -0.04293658, -0.05035384, -0.05711966, -0.06385698, -0.06848505, -0.07311311,
 -0.07692164, -0.07927321, -0.08162478, -0.08231457, -0.08231457, -0.08218915,
 -0.07983759, -0.07748602, -0.07422385, -0.06959578, -0.06496771, -0.05874346,
 -0.05197764, -0.04502941, -0.03630928, -0.02758915, -0.01810018, -0.0076327 ,
  0.00283478,  0.0145102 ,  0.0265132 ,  0.03869386,  0.05202935,  0.06536484,
  0.07925073,  0.09373287,  0.10821502,  0.12350882,  0.1389728 ,  0.14907594,
  0.13361195,  0.11814797,  0.10319454,  0.0887124 ,  0.07423025,  0.06074187,
  0.04740638,  0.03435515,  0.02235216,  0.01034916, -0.00079394, -0.01126142,
 -0.02172891, -0.03061213, -0.03933226, -0.0475573 , -0.05432312, -0.06108894,
 -0.06657211, -0.07120018, -0.07582825, -0.07830123, -0.0806528 , -0.08231457,
 -0.08231457, -0.08231457, -0.08080957, -0.078458  , -0.07610643, -0.07150872,
 -0.06688065, -0.06154   , -0.05477418, -0.04800836, -0.0399136 , -0.03119347,
 -0.02242674, -0.01195926, -0.00149178,  0.00954896,  0.02155196,  0.03355495,
  0.04651735,  0.05985284,  0.07326478,  0.08774692,  0.10222906,  0.11711703,
  0.13258102,  0.14804501,  0.14000374,  0.12453975,  0.10918049,  0.09469835,
  0.08021621,  0.06625388,  0.05291838,  0.03958289,  0.02731339,  0.0153104 ,
  0.00353262, -0.00693487, -0.01740235, -0.0270078 , -0.03572794, -0.04444807,
 -0.05152658, -0.0582924 , -0.06465918, -0.06928724, -0.07391531, -0.07732925,
 -0.07968081, -0.08203238, -0.08231457, -0.08231457, -0.08178155, -0.07942998,
 -0.07707841, -0.07342165, -0.06879358, -0.06416552, -0.05757072, -0.0508049 ,
 -0.04351792, -0.03479779, -0.02607766, -0.01628582, -0.00581833,  0.00464915,
  0.01659072,  0.02859371,  0.04100534,  0.05434084,  0.06767633,  0.08176097,
  0.09624311,  0.11072525
])

def perspectiva(I1):
    delta = 50
    # Cor de referência dos marcadores
    color = [106, 180, 120]  

    # Posição de referência dos marcadores, base imagem1
    pos_ref = np.array([[28, 26],[790, 24],[790, 344],[26, 344]])

    # Selecionando apenas os pixeis com a cor de referência para cada camada
    B = np.uint8(I1[:,:,0] == color[0])*255
    G = np.uint8(I1[:,:,1] == color[1])*255
    R = np.uint8(I1[:,:,2] == color[2])*255
    Mask_base = cv2.bitwise_and(cv2.bitwise_and(B, G), R)

    # Avalia os componentes da imagem binária resultante
    num_elem, I_labels  = cv2.connectedComponents(Mask_base)

    # Extrai informacoes
    infoRegioes_base = analisaRegioes(Mask_base)
    correlacao = np.zeros(len(infoRegioes_base))
    dist = np.zeros(len(infoRegioes_base))

    for n in np.arange(0, len(infoRegioes_base)):
        correlacao[n] = compute_match(curva_dist_quadrado, infoRegioes_base[n]['curva_distancia'])

    # O canto sup_esq é o que possui o quadrado, o centroide é do indice com maior correlação do a curva base do quadrado
    sup_esq = infoRegioes_base[np.argmax(correlacao)]['centroide']

    # Calcula a distância de todos os pontos até o canto sup_esq
    for n in np.arange(0, len(infoRegioes_base)):
        dist[n] = np.sqrt((sup_esq[0] - infoRegioes_base[n]['centroide'][0])**2 + (sup_esq[1] - infoRegioes_base[n]['centroide'][1])**2)

    # Analisando a figura o ponto mais distante é o inf_dir, após isso o sup_dir e por fim o inf_esq
    inf_dir = infoRegioes_base[np.argmax(dist)]['centroide']
    dist[np.argmax(dist)] = 0

    sup_dir = infoRegioes_base[np.argmax(dist)]['centroide']
    dist[np.argmax(dist)] = 0

    inf_esq = infoRegioes_base[np.argmax(dist)]['centroide']
    dist[np.argmax(dist)] = 0

    # Array com os cantos da figura
    pos_base = np.array([sup_esq, sup_dir, inf_dir, inf_esq])

    # Realização da homografia entre os pontos da imagem alvo e da imagem base(im1.png)
    h, status = cv2.findHomography(pos_base, pos_ref)
    I_final = cv2.warpPerspective(I1,  h, (834, 382))


    return I_final[delta:-delta,delta:-delta,:]

def ordena_lista_coluna_centroide(I_dici):
    delta = 5
    info_alfabeto = analisaRegioes(I_dici)

    info_alfabeto_ordenado = sorted(info_alfabeto, key=lambda x: x['centroide'][0], reverse=False)

    for k in np.arange(0, len(info_alfabeto_ordenado)):
        p0 = info_alfabeto_ordenado[k]['bb_p0']
        p1 = info_alfabeto_ordenado[k]['bb_p1']
        info_alfabeto_ordenado[k]['imagem'] = cv2.resize(info_alfabeto_ordenado[k]['imagem'][p0[1]-delta:p1[1]+delta,p0[0]-delta:p1[0]+delta], (50,50))

    return info_alfabeto_ordenado

def monta_palavra(info_palavra, info_alfabeto, I_base, dots):
    string_palavra = np.zeros(len(info_palavra) - 2*dots)
    base_lin, _ = I_base.shape

    for i in np.arange(0, len(info_palavra) - 2*dots):
        p0 = info_palavra[i]['bb_p0']
        p1 = info_palavra[i]['bb_p1']
        I_template = info_palavra[i]['imagem']
        # for l in np.arange(0, len(info_palavra)):
        #     plt.figure(f"Imagem {l}")
        #     plt.imshow(info_palavra[l]['imagem'])

        _, tem_col = I_template.shape
        I_similaridade = cv2.matchTemplate(I_base, I_template, cv2.TM_SQDIFF_NORMED)
        _, col_temp = I_template.shape
        D = I_similaridade <= np.min(I_similaridade)
        _, x0 = np.where(D)
        dist = np.zeros(len(info_alfabeto))

        for k in np.arange(0, len(info_alfabeto)):
            dist[k] = info_alfabeto[k]['centroide'][0] - (x0[0] + col_temp/2)

        dist = abs(dist)
        string_palavra[i] = np.argmin(dist)

        # plt.figure(f"Template {i}")
        # plt.imshow(I_template, cmap='gray')

        # plt.figure(f"Match {i}")
        # plt.imshow(info_alfabeto[np.argmin(dist)]['imagem'], cmap='gray')

    # plt.figure("Base")
    # plt.imshow(I_base, cmap='gray')
    if dots == 1:
        string_palavra = np.append(string_palavra, len(alfabeto)-1)
    palavra = "".join(alfabeto[i] for i in np.uint32(string_palavra))
    #print(string_palavra)
    return palavra

### Tratamento do Template

S = np.uint8(np.array([[0,0,0],[1,1,1],[0,0,0]]))
alfabeto = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","X","W","Y","Z","0"
            ,"1", "2", "3","4","5","6","7","8","9"]

ImgFM = cv2.imread('trabalho/banco_de_imagens/fonte_mercosul.png', cv2.IMREAD_GRAYSCALE)

ret , FMbinary = cv2.threshold(ImgFM, 127, 255, cv2.THRESH_BINARY)
Binaria = cv2.bitwise_not(FMbinary)
kernelhorizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (10,1))
FMdilate = cv2.dilate(Binaria, kernelhorizontal, None, iterations= 5)
      
num_elem, I_labels = cv2.connectedComponents(FMdilate)

rotulo = 3
I_componente = np.uint8(I_labels == rotulo)**255
p0, p1 = bounding_box(I_componente)
I2 = ImgFM.copy
I2 = cv2.cvtColor(ImgFM, cv2.COLOR_GRAY2BGR)
color = (255, 0, 0)
thickness = 1

roi = ImgFM[0:60, 0:500]
roi2 = ImgFM[60:120, 0:500]
roi3 = ImgFM[120:180, 0:355]

I_base = cv2.bitwise_not(roi)
I_base2 = cv2.bitwise_not(roi2)
I_base3 = cv2.bitwise_not(roi3)
BASE = np.concatenate((I_base, I_base2, I_base3), axis=1)

info_alfabeto = ordena_lista_coluna_centroide(BASE)

### Tratamento Placa

I_placaTM = cv2.imread('trabalho/banco_de_imagens/nivel_1/placa3.jpg') 
I_placaTM = cv2.resize(I_placaTM, (0,0), fx = 1, fy = 1)
I_placaTM  = cv2.cvtColor(I_placaTM, cv2.COLOR_BGR2GRAY)
I_comparar = cv2.imread('trabalho/banco_de_imagens/nivel_2/placa11.jpg')
I_comparar = cv2.resize(I_comparar, (0,0), fx = 1, fy = 1)
I_comparar  = cv2.cvtColor(I_comparar, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp1, descritores1 = sift.detectAndCompute(I_placaTM, None)
kp2, descritores2 = sift.detectAndCompute(I_comparar, None)


bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

matches = bf.match(descritores1, descritores2)
matches = sorted(matches, key = lambda x:x.distance)


I_RTM = cv2.drawMatches(I_placaTM, kp1, I_comparar, kp2, matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#plt.figure()
#plt.imshow(cv2.cvtColor(I_RTM, cv2.COLOR_BGR2RGB))

# obtem matriz de homografia

N = 100

src_pts = np.zeros((100,1,2), np.float32)
dst_pts = np.zeros((100,1,2), np.float32)

for i in np.arange(0,N):
    src_pts[i] = np.float32(kp1[matches[i].queryIdx].pt).reshape(-1, 1, 2)
    dst_pts[i] = np.float32(kp2[matches[i].trainIdx].pt).reshape(-1, 1, 2)

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)

h, w = I_placaTM.shape
pts = np.float32([[0,0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts, M)
dstx = dst[:,0,0]
dsty = dst[:,0,1]

pxinf = round(np.min(dstx))
pxsup = round(np.max(dstx))
pyinf =  round(np.min(dsty)) 
pysup = round(np.max(dsty))

I4 = cv2.cvtColor(I_comparar, cv2.COLOR_GRAY2BGR)

I4  = cv2.polylines(I4, [np.int32(dst)], True, (0,0,255), 2, cv2.LINE_AA)

I_NOVA = I_comparar[pyinf:pysup , pxinf:pxsup]
plt.imshow(I_NOVA)
"""
pontosQ = [(round(dstx[0]), round(dsty[0])), (round(dstx[1]), round(dsty[1])), (round(dstx[2]), round(dsty[2])), (round(dstx[3]), round(dsty[3]))]

'''print(pxinf)
print(pxsup)
print(pyinf)
print(pysup)
print(dstx)
print(dsty)
print(pontosQ)'''

I_NOVAp = cv2.cvtColor(I_comparar, cv2.COLOR_GRAY2BGR)   

for pontos in pontosQ:
    I_NOVAp = cv2.circle(I_NOVAp, pontos, 2, (106, 180, 120), -1)


I_NOVAp = cv2.rectangle(I_NOVAp, (pxinf,pyinf), (pxinf+10,pyinf+10), (106, 180, 120), thickness=cv2.FILLED)

I_NOVAper = perspectiva(I_NOVAp)

plt.figure('I_NOVAp')
plt.imshow(cv2.cvtColor(I_NOVAp, cv2.COLOR_BGR2RGB))

plt.figure('I_NOVAper')
plt.imshow(cv2.cvtColor(I_NOVAper, cv2.COLOR_BGR2RGB))

ret1 , I_NOVABYNARY = cv2.threshold(I_NOVA, 127, 255, cv2.THRESH_BINARY)
BinariaInova = cv2.bitwise_not(I_NOVABYNARY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
I_novadilatada = cv2.dilate(BinariaInova, kernel, None, iterations= 2)

plt.figure('Inova')
plt.imshow(I_NOVA, cmap='gray')

plt.figure('Inovabinaria')
plt.imshow(I_NOVABYNARY, cmap='gray')

plt.figure('Inovadilatada')
plt.imshow(I_novadilatada, cmap='gray')

plt.figure()
plt.imshow(cv2.cvtColor(I4, cv2.COLOR_BGR2RGB))

plt.figure()
plt.imshow(I_placaTM, cmap= 'gray')

plt.figure()
plt.imshow(I_comparar, cmap= 'gray')"""
plt.show()