import cv2
import numpy as np


# --------------------------------------------------------------------------------#
# Função 1: DETECTAR BORDAS (CANNY)
# --------------------------------------------------------------------------------#
def detectar_bordas_canny(imagem):
    # ATENÇÃO: OpenCV lê vídeos em formato BGR, não RGB.
    # Corrigindo para o formato correto.
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)  # <-- ALTERAÇÃO SUTIL, MAS IMPORTANTE
    imagem_blur = cv2.GaussianBlur(imagem_cinza, (7, 7), 0)
    imagem_canny = cv2.Canny(imagem_blur, 50, 150)
    return imagem_canny


# --------------------------------------------------------------------------------#
# Função 2: DEFINIR REGIÃO DE INTERESSE (ROI)
# --------------------------------------------------------------------------------#
def definir_regiao_de_interesse(imagem):
    altura = imagem.shape[0]
    largura = imagem.shape[1]

    # É aqui que os vértices são criados!
    # A variável 'poligono' contém as coordenadas que queremos desenhar.
    poligono = np.array([
        [(0, altura), (largura, altura), (int(largura * 0.5), int(altura * 0.6))]
    ])

    mascara = np.zeros_like(imagem)
    cv2.fillPoly(mascara, poligono, 255)
    imagem_com_mascara = cv2.bitwise_and(imagem, mascara)

    # <-- ALTERAÇÃO PRINCIPAL: Retornamos não só a imagem mascarada,
    # mas também a variável 'poligono' que contém os vértices.
    return imagem_com_mascara, poligono


# --------------------------------------------------------------------------------#
# Função 3 e 4 (Cálculo das linhas) - Sem alterações
# --------------------------------------------------------------------------------#
def calcular_coordenadas(imagem, parametros_linha):
    try:
        inclinacao, intercepto = parametros_linha
    except (TypeError, ValueError):
        inclinacao, intercepto = 0.001, 0
    y1 = imagem.shape[0]
    y2 = int(y1 * 0.6)
    x1 = int((y1 - intercepto) / inclinacao)
    x2 = int((y2 - intercepto) / inclinacao)
    return np.array([x1, y1, x2, y2])


def calcular_media_linhas(imagem, linhas):
    faixa_esquerda = []
    faixa_direita = []
    if linhas is None:
        return None
    for linha in linhas:
        x1, y1, x2, y2 = linha.reshape(4)
        parametros = np.polyfit((x1, x2), (y1, y2), 1)
        inclinacao = parametros[0]
        intercepto = parametros[1]
        if inclinacao < 0:
            faixa_esquerda.append((inclinacao, intercepto))
        else:
            faixa_direita.append((inclinacao, intercepto))

    # Adicionando verificações para evitar crash se uma faixa não for detectada
    if not faixa_esquerda or not faixa_direita:
        return None  # Retorna None se não encontrar ambas as faixas

    media_faixa_esquerda = np.average(faixa_esquerda, axis=0)
    media_faixa_direita = np.average(faixa_direita, axis=0)
    linha_esquerda = calcular_coordenadas(imagem, media_faixa_esquerda)
    linha_direita = calcular_coordenadas(imagem, media_faixa_direita)
    return np.array([linha_esquerda, linha_direita])


# --------------------------------------------------------------------------------#
# Função 5: DESENHAR AS LINHAS NA IMAGEM
# --------------------------------------------------------------------------------#
def desenhar_linhas(imagem, linhas):
    imagem_com_linhas = np.zeros_like(imagem)
    if linhas is not None:
        for linha in linhas:
            if linha is not None:
                x1, y1, x2, y2 = linha
                cv2.line(imagem_com_linhas, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return imagem_com_linhas


# --------------------------------------------------------------------------------#
# Função 6: DESENHAR A REGIÃO DE INTERESSE
# --------------------------------------------------------------------------------#
def desenhar_roi(imagem, vertices):
    # A função em si está correta. Ela recebe a imagem e os vértices.
    cv2.polylines(imagem, [vertices], isClosed=True, color=(0, 255, 0), thickness=2)
    return imagem

# --------------------------------------------------------------------------------#
# Função 7: LINHAS BRUTAS
# --------------------------------------------------------------------------------#
def linhas_brutas(image, linhas):
    line_image = np.zeros_like(image)

    if linhas is not None:
        for linha in linhas:
            x1,y1,x2,y2 = linha.reshape(4)
            cv2.line(line_image, (x1,y1) , (x2,y2), (255,0,0), 10)
    return line_image


# ================================================================================#
# SCRIPT PRINCIPAL (LOOP DE VÍDEO EM TEMPO REAL)
# ================================================================================#
cap = cv2.VideoCapture("Video3.mp4")

if not cap.isOpened():
    print("Erro: Não foi possível abrir o vídeo.")
    exit()

while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        # Se o vídeo acabar, reinicia do primeiro frame para loop infinito
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # --- FLUXO DE PROCESSAMENTO ---
    try:
        # 1. Detectar bordas
        imagem_canny = detectar_bordas_canny(frame)

        # 2. Isolar a região de interesse E OBTER OS VÉRTICES
        # <-- ALTERAÇÃO: Agora capturamos os dois valores retornados.
        imagem_roi, vertices_poligono = definir_regiao_de_interesse(imagem_canny)

        # 3. Detectar linhas
        linhas_detectadas = cv2.HoughLinesP(imagem_roi, 2, np.pi / 180, 100,
                                            np.array([]), minLineLength=20, maxLineGap=5)

        # 4. Calcular média das linhas
        linhas_otimizadas = calcular_media_linhas(frame, linhas_detectadas)

        #Função extra:
        imagem_com_linha_bruta = linhas_brutas(frame, linhas_detectadas)

        # 5. Desenhar linhas
        imagem_com_linhas = desenhar_linhas(frame, linhas_otimizadas)

        # 6. Mesclar imagem original com as linhas
        imagem_final = cv2.addWeighted(frame, 0.8, imagem_com_linhas, 1, 1)

        # 7. DESENHAR A ROI NA IMAGEM FINAL
        # <-- ALTERAÇÃO: Agora chamamos a função 'desenhar_roi' corretamente,
        # passando a imagem final e os 'vertices_poligono' que obtivemos no passo 2.
        imagem_final_com_roi = desenhar_roi(imagem_final, vertices_poligono)

        # Exibe o resultado final, agora com a ROI desenhada
        cv2.imshow("Detector de Faixas", imagem_final_com_roi)
        cv2.imshow("Detector de Faixas2",imagem_com_linha_bruta )
        cv2.imshow("Detector de Faixa3", imagem_roi)

    except (ValueError, TypeError, np.linalg.LinAlgError) as e:
        # Se ocorrer um erro (ex: nenhuma linha detectada), apenas exibe o frame original
        cv2.imshow("Detector de Faixas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()