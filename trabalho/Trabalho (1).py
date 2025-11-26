import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# ---------------------------------------------------------------------------------
# ETAPA 1: FUNÇÕES AUXILIARES (Inalteradas, com a correção do KeyError)
# ---------------------------------------------------------------------------------

def bounding_box(I1):
    y, x = np.where(I1)
    if x.size == 0 or y.size == 0:
        return np.array([0, 0]), np.array([0, 0])
    ymin = np.min(y)
    ymax = np.max(y)
    xmin = np.min(x)
    xmax = np.max(x)

    p0 = np.array([xmin, ymin])
    p1 = np.array([xmax, ymax])

    return p0, p1

def mpq(I1, p, q):
    y, x = np.where(I1)
    mpq = np.float32(np.sum((x**p)*(y**q)))
    return mpq

def centroide(I1):
    m00 = mpq(I1, 0, 0)
    # Evita divisão por zero se a região for vazia
    if m00 == 0:
        return np.array([0, 0])

    m10 = mpq(I1, 1, 0)
    m01 = mpq(I1, 0, 1)

    xc = m10/m00
    yc = m01/m00

    c0 = np.array([xc, yc])
    return c0

def upq(I1, p, q):
    c0 = centroide(I1)
    xc = c0[0]
    yc = c0[1]

    y, x = np.where(I1)

    momento_centroide = np.sum(((x-xc)**p) * ((y-yc)**q))
    return momento_centroide

# --- FUNÇÃO ANALISAREGIOES (COM A CORREÇÃO DO KEYERROR) ---
def analisaRegioes(I1):
    infoRegioes = []

    # rotulamento
    num_elem, I_labels = cv2.connectedComponents(I1)

    for i in np.arange(1, num_elem):
        # dados dos componentes
        dados_do_compenente = dict( )
        # imagem do componente cenectado (região de interesse)
        I_componente = np.uint8(I_labels == i)*255
        dados_do_compenente['imagem'] = I_componente

        #bounding box
        p0, p1 = bounding_box(I_componente)

        dados_do_compenente['bb_p0'] = p0
        dados_do_compenente['bb_p1'] = p1

        # area
        m00 = mpq(I_componente, 0, 0)

        # *** CORREÇÃO DO KEYERROR ESTÁ AQUI ***
        # A área (m00) DEVE ser atribuída ANTES do 'continue'
        dados_do_compenente['area'] = m00

        # Pula regiões muito pequenas ou vazias
        if m00 < 10:
            continue

        # centroide
        c0 = centroide(I_componente)
        dados_do_compenente['centroide'] = c0

        # momentos centrais de segunda ordem
        u20 = upq(I_componente,2,0)
        u02 = upq(I_componente,0,2)
        u11 = upq(I_componente,1,1)

        # matriz de inércia da região
        J = np.array([[u20, u11], [u11, u02]])

        # matriz de inércia da elipse equivalente
        Je = 4/m00 * J
        avalores, avetores = np.linalg.eig(Je)

        # raios da elipse
        raios = np.sort(np.sqrt(np.abs(avalores)))
        menor_raio = raios[0]
        maior_raio = raios[1]

        dados_do_compenente['maior_raio'] = maior_raio
        dados_do_compenente['menor_raio'] = menor_raio

        # orientação da elipse
        pos = np.argmax(avalores)

        vx = avetores[0, pos]
        vy = avetores[1, pos]
        orientacao = np.rad2deg(np.atan2(vy, vx))

        # Evita divisão por zero
        if maior_raio > 0:
            dados_do_compenente['razao_raios'] = menor_raio/maior_raio
        else:
            dados_do_compenente['razao_raios'] = 0

        dados_do_compenente['orientacao'] = orientacao

        # extrai coordenas dos pixels de borda
        contours, hierarchy = cv2.findContours(I_componente, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        if not contours: # Se não houver contornos
            continue

        x = contours[0][:,0,0]
        y = contours[0][:,0,1]

        # Armazena as coordenadas do contorno para uso na curva de distância
        dados_do_compenente['contour_x'] = x
        dados_do_compenente['contour_y'] = y

        # cálculo do perímetro
        perimetro = np.sqrt((y[-1] - y[0])**2 + (x[-1]- x[0])**2)
        N = len(y)

        for n in np.arange(0,N-1):
            perimetro = perimetro + np.sqrt((y[n] - y[n+1])**2 + (x[n]- x[n+1])**2)

        dados_do_compenente['perimetro'] = perimetro

        # circularidade
        if perimetro > 0:
            circularidade = 4*np.pi*m00/(perimetro**2)
            dados_do_compenente['circularidade'] = circularidade
        else:
            dados_do_compenente['circularidade'] = 0


        # curva de distância
        xc = c0[0]
        yc = c0[1]

        curva_distancia = np.zeros(N)
        for n in np.arange(0, N):
            curva_distancia[n] = np.sqrt((y[n]-yc)**2 + (x[n]-xc)**2)

        dados_do_compenente['curva_distancia'] = curva_distancia

        # adiciona o dicionário na lista infoRegioes
        infoRegioes.append(dados_do_compenente.copy())

    return infoRegioes

def desenha_boundingBoxes(I1, infoRegioes, color, thickness):
    num_regioes = len(infoRegioes)

    for i in np.arange(0, num_regioes):
        p0 = infoRegioes[i]['bb_p0']
        p1 = infoRegioes[i]['bb_p1']

        cv2.rectangle(I1, tuple(p0.astype(int)), tuple(p1.astype(int)), color, thickness)

    return I1

def desenha_centroide(I1,infoRegioes, color, radius):
    num_regioes  = len(infoRegioes)
    for i in np.arange(0, num_regioes):
        c0 = infoRegioes[i]['centroide']
        cv2.circle(I1, np.int32(c0), radius, color, -1)

    return I1


def normalizar_letra(img_letra, tamanho_final=(50,50)):
    img_letra = cv2.copyMakeBorder(img_letra, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0)
    contours, _ = cv2.findContours(img_letra, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(tamanho_final, dtype=np.uint8)
    contorno_maior = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contorno_maior)
    letra_recortada = img_letra[y:y + h, x:x + w]
    if letra_recortada.size == 0:
        return np.zeros(tamanho_final, dtype=np.uint8)
    h_rec, w_rec = letra_recortada.shape
    canvas = np.zeros(tamanho_final, dtype=np.uint8)
    fator_ocupacao = 0.85
    escala = min(tamanho_final[0] * fator_ocupacao / h_rec, tamanho_final[1] * fator_ocupacao / w_rec)
    novo_w = max(1, int(w_rec * escala))
    novo_h = max(1, int(h_rec * escala))
    letra_redim = cv2.resize(letra_recortada, (novo_w, novo_h), interpolation=cv2.INTER_AREA)
    x_pos = (tamanho_final[1] - novo_w) // 2
    y_pos = (tamanho_final[0] - novo_h) // 2
    canvas[y_pos:y_pos + novo_h, x_pos:x_pos + novo_w] = letra_redim
    return canvas

def imreconstruction(Mask, Marker):

    # Operação morfológica de reconstrução
    num_pixels_brancos = 0
    kernel = np.ones((3,3), np.float32)

    # critério de parada: quando não houver mais alteração na imagem Marker
    while num_pixels_brancos != np.sum(Marker):
        num_pixels_brancos = np.sum(Marker)
        Marker = cv2.dilate(Marker, kernel, iterations=1)
        Marker = cv2.bitwise_and(Mask, Marker)

    return Marker

def imclearboard(I):
    # imagem Marker
    Marker = I.copy()
    Marker[1:-1, 1:-1] = 0

    I2 = imreconstruction(I, Marker)
    I3 = I - I2

    return I3

def detectar_retangulo_preto(I_bgr):
    I_gray = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2GRAY)

    # realça linhas escuras sobre fundo claro
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    blackhat = cv2.morphologyEx(I_gray, cv2.MORPH_BLACKHAT, kernel)

    blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
    _, bw = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel2, iterations=2)

    conts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    melhor = None
    melhor_area = 0
    bbox = None

    for c in conts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)

        # placa brasileira é mais larga que alta
        if w / h < 1.5:
            continue

        area = w * h
        if area > melhor_area:
            melhor_area = area
            melhor = approx
            bbox = (x, y, w, h)

    return bbox


# ---------------------------------------------------------------------------------
# ETAPA 2: Extração dos templates
# ---------------------------------------------------------------------------------
try:
    imagem_templates = cv2.imread("trabalho/banco_de_imagens/fonte_mercosul.png",cv2.IMREAD_GRAYSCALE)
    if imagem_templates is None: raise FileNotFoundError
except FileNotFoundError:
    print("Erro: Não foi possível encontrar 'fonte_mercosul.png'. Verifique o caminho.")
    exit()

tamanho_template = (50,50)
templates_order = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

_, th = cv2.threshold(imagem_templates, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Morfologia: fecha buracos e remove pequenos ruídos
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

# ---------- encontra contornos externos ----------
contours, _ = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("[INFO] Contornos externos encontrados:", len(contours))

# filtra por área mínima
min_area = 200
filtered = [c for c in contours if cv2.contourArea(c) >= min_area]
print("[INFO] Após filtro por área mínima:", len(filtered))

if len(filtered) != len(templates_order):
    print("[AVISO] Número de contornos filtrados != número de templates esperados (36).")
    print("         Você pode ajustar min_area ou visualizar os contornos para depuração.")

# ---------- calc centroid Y para ordenar por linhas ------------
boxes = []
for c in filtered:
    x, y, w, h = cv2.boundingRect(c)
    cx = x + w/2
    cy = y + h/2
    boxes.append({'contour': c, 'x': x, 'y': y, 'w': w, 'h': h, 'cx': cx, 'cy': cy})

# Ordena por y (linha), mas agrupando contornos próximos em mesma linha
boxes_sorted = sorted(boxes, key=lambda b: (b['cy'], b['x']))

# agrupar em linhas: quando diferença de cy for grande, começa nova linha
rows = []
row_thresh = np.mean([b['h'] for b in boxes_sorted]) * 0.8
current_row = [boxes_sorted[0]]
for b in boxes_sorted[1:]:
    if abs(b['cy'] - np.mean([r['cy'] for r in current_row])) > row_thresh:
        rows.append(sorted(current_row, key=lambda r: r['x']))
        current_row = [b]
    else:
        current_row.append(b)
rows.append(sorted(current_row, key=lambda r: r['x']))

# aplaina linhas em ordem (top->bottom, left->right)
final_boxes = []
for row in rows:
    final_boxes.extend(row)

print(f"[INFO] Linhas detectadas: {len(rows)}; total final_boxes: {len(final_boxes)}")

# salva/mostra templates (associa na ordem final localizada)
caracteres = {}
for i, b in enumerate(final_boxes):
    x, y, w, h = b['x'], b['y'], b['w'], b['h']
    recorte = th[y:y+h, x:x+w]
    normalized = normalizar_letra(recorte, tamanho_template)
    label = templates_order[i] if i < len(templates_order) else f"#{i}"
    caracteres[label] = normalized

    # exibe (opcional)
    # plt.figure()
    # plt.imshow(normalized, cmap='gray')
    # plt.title(f"{i} -> {label}")
    # plt.axis('off')
#plt.show()

# ---------------------------------------------------------------------------------
# ETAPA 3: Segmentação Binária Pura, Recorte Agressivo e Ramer-Douglas-Peucker
# ---------------------------------------------------------------------------------

path_img = "trabalho/banco_de_imagens/nivel_2/placa8.jpg"


if not os.path.exists(path_img):
    print(f"Erro: Imagem não encontrada em: {path_img}")
else:
    I_entrada = cv2.imread(path_img)
    I_gray = cv2.cvtColor(I_entrada, cv2.COLOR_BGR2GRAY)

    img_h, img_w = I_gray.shape
    plt.figure()
    plt.imshow(I_gray)
    plt.figure()
    plt.imshow(I_gray, cmap='gray')

    # --------------------------------------------------------------------
    # 1) TRATAMENTO DA IMAGEM BINÁRIA
    # --------------------------------------------------------------------
    I_inv = cv2.bitwise_not(I_gray)
    plt.figure()
    plt.imshow(I_inv, cmap = 'gray')

    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    I_clean = cv2.morphologyEx(I_inv, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # DILATAÇÃO: Reforça o que é branco (neste caso, as letras invertidas e bordas)
    # Isso garante que os caracteres fiquem grossos e legíveis na binarização
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    I_dilatada = cv2.dilate(I_clean, kernel_dil, iterations=2)
    plt.figure()
    plt.imshow(I_dilatada, cmap = 'gray')

    # Binarização (Otsu) na imagem invertida/dilatada
    # Resulta em: Fundo/Placa = Preto (0), Letras = Branco (255)
    _, binaria_inv = cv2.threshold(I_dilatada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.figure()
    plt.imshow(binaria_inv, cmap = 'gray')

    # Invertemos de volta para o padrão humano/Ramer:
    # Placa = Branco (255), Letras = Preto (0)
    I_binaria_final = cv2.bitwise_not(binaria_inv)



    # Visualiza a imagem binária base que usaremos para tudo
    plt.figure(figsize=(8, 4))
    plt.title("Imagem Binária Base (Placa Branca / Letras Grossas)")
    plt.imshow(I_binaria_final, cmap='gray')
    plt.show()

    # -------------------------------------------------------------
    # IDENTIFICAÇÃO E RECORTE
    # -------------------------------------------------------------
    # Caminho para a placa molde (imagem frontal, limpa, qualquer placa Mercosul)
    template_placa_path = "trabalho/banco_de_imagens/nivel_1/placa3.jpg"

    template = cv2.imread(template_placa_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print("[ERRO] Template da placa não encontrado.")
        exit()

    # Reduza ruído
    tpl = cv2.GaussianBlur(template, (3, 3), 0)
    scene = cv2.GaussianBlur(I_gray, (3, 3), 0)

    # ---- Detector SIFT ----
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(tpl, None)
    kp2, des2 = sift.detectAndCompute(scene, None)

    if des1 is None or des2 is None:
        print("[ERRO] Não foi possível extrair descritores SIFT.")
        exit()

    # ---- Match BFMatcher (L2, igual seu exemplo) ----
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)

    # Ordena por distância
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 10:
        print("[ERRO] Poucos matches SIFT. Não foi possível localizar a placa.")
        exit()

    # Usa somente os melhores matches
    N = min(80, len(matches))
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:N]]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:N]]).reshape(-1, 1, 2)

    # ---- Homografia ----
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        print("[ERRO] Homografia falhou.")
        exit()

    # ---- Warp dos cantos do template ----
    h_t, w_t = tpl.shape
    pts = np.float32([[0, 0], [0, h_t - 1], [w_t - 1, h_t - 1], [w_t - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H).reshape(4, 2)

    # ---- Bounding Box ----
    xs = dst[:, 0]
    ys = dst[:, 1]
    x_min = int(max(0, np.min(xs)))
    y_min = int(max(0, np.min(ys)))
    x_max = int(min(scene.shape[1] - 1, np.max(xs)))
    y_max = int(min(scene.shape[0] - 1, np.max(ys)))

    w_found = x_max - x_min
    h_found = y_max - y_min

    if w_found < 30 or h_found < 10:
        print("[ERRO] BBox muito pequena. Homografia ruim.")
        exit()

    # ---- Recorte final da placa ----
    I_recortada_bin = I_binaria_final[y_min:y_max, x_min:x_max].copy()

    plt.figure()
    plt.title("Recorte baseado no Template + SIFT + Homografia")
    plt.imshow(I_recortada_bin, cmap='gray')
    plt.axis('off')
    plt.show()

    # --------------------------------------------------------------------
    # 3) LIMPEZA DE BORDAS (imclearboard)
    # --------------------------------------------------------------------
    h_roi, w_roi = I_recortada_bin.shape

    # Desenhar um retângulo branco na borda da ROI
    # Isso garante que imclearboard NÃO consiga entrar
    cv2.rectangle(I_recortada_bin, (1, 1), (w_roi - 2, h_roi - 2), 255, 2)

    h_roi, w_roi = I_recortada_bin.shape

    # faixa preta só nas bordas
    I_recortada_bin[0:3, :] = 0
    I_recortada_bin[-3:, :] = 0
    I_recortada_bin[:, 0:3] = 0
    I_recortada_bin[:, -3:] = 0
    I_recortada_bin = imclearboard(I_recortada_bin)

    plt.figure()
    plt.title("Recorte após limpeza de bordas")
    plt.imshow(I_recortada_bin, cmap='gray')

    # --------------------------------------------------------------------
    # 4) RAMER-DOUGLAS-PEUCKER FINAL
    # --------------------------------------------------------------------
    contoursPlaca, _ = cv2.findContours(I_recortada_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contoursPlaca:
        cnt = max(contoursPlaca, key=cv2.contourArea)

        approx = None
        peri = cv2.arcLength(cnt, True)

        for k in np.linspace(0.01, 0.1, 50):
            approx_candidate = cv2.approxPolyDP(cnt, k * peri, True)
            if len(approx_candidate) == 4:
                approx = approx_candidate
                break

        if approx is None:
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # Visualização
        I_vis_ramer = cv2.cvtColor(I_recortada_bin, cv2.COLOR_GRAY2BGR)
        cv2.polylines(I_vis_ramer, [np.int32(approx)], True, (0, 255, 0), 3, cv2.LINE_AA)

        # Máscara final
        mask_ramer = np.zeros_like(I_recortada_bin)
        cv2.fillPoly(mask_ramer, [np.int32(approx)], 255)

        I_final_output = cv2.bitwise_and(I_recortada_bin, I_recortada_bin, mask=mask_ramer)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 3, 1)
        plt.title("Recorte Limpo")
        plt.imshow(I_recortada_bin, cmap='gray')

        plt.subplot(1, 3, 2)
        plt.title("Ramer Detectado")
        plt.imshow(cv2.cvtColor(I_vis_ramer, cv2.COLOR_BGR2RGB))

        plt.subplot(1, 3, 3)
        plt.title("Resultado Final")
        plt.imshow(I_final_output, cmap='gray')
        plt.show()

    else:
        print("Erro: Nenhum contorno encontrado após limpeza.")
