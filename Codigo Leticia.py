import cv2
import numpy as np
from collections import deque

# Variáveis globais para os pontos do polígono (simplificado para 4 parâmetros)
roi_params = {
    'largura_base': 80,  # Largura da base do trapézio (%)
    'largura_topo': 20,  # Largura do topo do trapézio (%)
    'altura_topo': 55,  # Altura onde começa o topo (%)
    'posicao_x': 50  # Posição horizontal central (%)
}


# Sistema de memória para suavização das linhas
class MemoriaLinhas:
    def __init__(self, tamanho_memoria=10):
        self.tamanho_memoria = tamanho_memoria
        self.historico_esquerda = deque(maxlen=tamanho_memoria)
        self.historico_direita = deque(maxlen=tamanho_memoria)
        self.linha_esquerda_media = None
        self.linha_direita_media = None

    def adicionar_linhas(self, linha_esquerda, linha_direita):
        """Adiciona novas linhas ao histórico e calcula a média"""
        if linha_esquerda is not None:
            self.historico_esquerda.append(linha_esquerda)

        if linha_direita is not None:
            self.historico_direita.append(linha_direita)

        # Calcular médias
        self._calcular_medias()

    def _calcular_medias(self):
        """Calcula a média das coordenadas armazenadas"""
        if len(self.historico_esquerda) > 0:
            self.linha_esquerda_media = np.mean(self.historico_esquerda, axis=0).astype(int)
        else:
            self.linha_esquerda_media = None

        if len(self.historico_direita) > 0:
            self.linha_direita_media = np.mean(self.historico_direita, axis=0).astype(int)
        else:
            self.linha_direita_media = None

    def obter_linhas_suavizadas(self):
        """Retorna as linhas suavizadas"""
        linhas_suavizadas = []

        if self.linha_esquerda_media is not None:
            linhas_suavizadas.append(self.linha_esquerda_media)

        if self.linha_direita_media is not None:
            linhas_suavizadas.append(self.linha_direita_media)

        return np.array(linhas_suavizadas) if linhas_suavizadas else None

    def obter_status(self):
        """Retorna informações sobre o estado da memória"""
        return {
            'frames_esquerda': len(self.historico_esquerda),
            'frames_direita': len(self.historico_direita),
            'memoria_cheia': len(self.historico_esquerda) == self.tamanho_memoria and
                             len(self.historico_direita) == self.tamanho_memoria
        }

    def limpar_memoria(self):
        """Limpa todo o histórico"""
        self.historico_esquerda.clear()
        self.historico_direita.clear()
        self.linha_esquerda_media = None
        self.linha_direita_media = None


# Instanciar o sistema de memória
memoria_linhas = MemoriaLinhas(tamanho_memoria=10)


# --------------------------------------------------------------------------------#
# Função para criar as trackbars na janela principal
# --------------------------------------------------------------------------------#
def criar_trackbars():
    cv2.createTrackbar('Largura Base (%)', 'Detector de Faixas', roi_params['largura_base'], 100, lambda x: None)
    cv2.createTrackbar('Largura Topo (%)', 'Detector de Faixas', roi_params['largura_topo'], 100, lambda x: None)
    cv2.createTrackbar('Altura Topo (%)', 'Detector de Faixas', roi_params['altura_topo'], 100, lambda x: None)
    cv2.createTrackbar('Posicao X (%)', 'Detector de Faixas', roi_params['posicao_x'], 100, lambda x: None)


# --------------------------------------------------------------------------------#
# Função para atualizar os parâmetros da ROI
# --------------------------------------------------------------------------------#
def atualizar_roi_params():
    roi_params['largura_base'] = cv2.getTrackbarPos('Largura Base (%)', 'Detector de Faixas')
    roi_params['largura_topo'] = cv2.getTrackbarPos('Largura Topo (%)', 'Detector de Faixas')
    roi_params['altura_topo'] = cv2.getTrackbarPos('Altura Topo (%)', 'Detector de Faixas')
    roi_params['posicao_x'] = cv2.getTrackbarPos('Posicao X (%)', 'Detector de Faixas')


# --------------------------------------------------------------------------------#
# Função 1: DETECTAR BORDAS (CANNY)
# --------------------------------------------------------------------------------#
def detectar_bordas_canny(imagem):
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    imagem_blur = cv2.GaussianBlur(imagem_cinza, (5, 5), 0)
    imagem_canny = cv2.Canny(imagem_blur, 100, 100)
    return imagem_canny


# --------------------------------------------------------------------------------#
# Função 2: DEFINIR REGIÃO DE INTERESSE (ROI) - SIMPLIFICADA
# --------------------------------------------------------------------------------#
def definir_regiao_de_interesse(imagem):
    altura = imagem.shape[0]
    largura = imagem.shape[1]

    centro_x = int(largura * roi_params['posicao_x'] / 100)

    largura_base_pixels = int(largura * roi_params['largura_base'] / 100)
    base_esquerda = max(0, centro_x - largura_base_pixels // 2)
    base_direita = min(largura, centro_x + largura_base_pixels // 2)

    altura_topo_pixels = int(altura * roi_params['altura_topo'] / 100)
    largura_topo_pixels = int(largura * roi_params['largura_topo'] / 100)
    topo_esquerda = max(0, centro_x - largura_topo_pixels // 2)
    topo_direita = min(largura, centro_x + largura_topo_pixels // 2)

    poligono = np.array([
        [
            (base_esquerda, altura - 1),
            (topo_esquerda, altura_topo_pixels),
            (topo_direita, altura_topo_pixels),
            (base_direita, altura - 1)
        ]
    ])

    mascara = np.zeros_like(imagem)
    cv2.fillPoly(mascara, poligono, 255)
    imagem_com_mascara = cv2.bitwise_and(imagem, mascara)
    return imagem_com_mascara, poligono


# --------------------------------------------------------------------------------#
# Função 3: CALCULAR COORDENADAS DAS LINHAS
# --------------------------------------------------------------------------------#
def calcular_coordenadas(imagem, parametros_linha):
    try:
        inclinacao, intercepto = parametros_linha
    except (TypeError, ValueError):
        inclinacao, intercepto = 0.001, 0

    if abs(inclinacao) < 0.001:
        inclinacao = 0.001 if inclinacao >= 0 else -0.001

    y1 = imagem.shape[0]
    y2 = int(y1 * 0.6)
    x1 = int((y1 - intercepto) / inclinacao)
    x2 = int((y2 - intercepto) / inclinacao)
    return np.array([x1, y1, x2, y2])


# --------------------------------------------------------------------------------#
# Função 4: CALCULAR MÉDIA DAS LINHAS - MODIFICADA PARA USAR MEMÓRIA
# --------------------------------------------------------------------------------#
def calcular_media_linhas(imagem, linhas):
    faixa_esquerda = []
    faixa_direita = []

    if linhas is None:
        return None, None

    for linha in linhas:
        x1, y1, x2, y2 = linha.reshape(4)
        if x2 == x1:
            continue

        parametros = np.polyfit((x1, x2), (y1, y2), 1)
        inclinacao = parametros[0]
        intercepto = parametros[1]

        if abs(inclinacao) < 0.1:
            continue

        if inclinacao < 0:
            faixa_esquerda.append((inclinacao, intercepto))
        else:
            faixa_direita.append((inclinacao, intercepto))

    # Calcular linhas individuais para este frame
    linha_esquerda_atual = None
    linha_direita_atual = None

    if faixa_esquerda:
        media_esquerda = np.average(faixa_esquerda, axis=0)
        linha_esquerda_atual = calcular_coordenadas(imagem, media_esquerda)

    if faixa_direita:
        media_direita = np.average(faixa_direita, axis=0)
        linha_direita_atual = calcular_coordenadas(imagem, media_direita)

    return linha_esquerda_atual, linha_direita_atual


# --------------------------------------------------------------------------------#
# Função 5: DESENHAR AS LINHAS NA IMAGEM - MODIFICADA
# --------------------------------------------------------------------------------#
def desenhar_linhas(imagem, linhas, cor=(0, 255, 0), espessura=10, alpha=1.0):
    imagem_com_linhas = np.zeros_like(imagem)
    if linhas is not None:
        for linha in linhas:
            x1, y1, x2, y2 = linha
            cv2.line(imagem_com_linhas, (x1, y1), (x2, y2), cor, espessura)

    # Aplicar transparência se necessário
    if alpha < 1.0:
        return cv2.addWeighted(imagem, 1 - alpha, imagem_com_linhas, alpha, 0)
    else:
        return imagem_com_linhas


# --------------------------------------------------------------------------------#
# Função 6: DESENHAR A REGIÃO DE INTERESSE
# --------------------------------------------------------------------------------#
def desenhar_roi(imagem, vertices):
    cv2.polylines(imagem, [vertices], isClosed=True, color=(255, 0, 0), thickness=3)

    cores_pontos = [(0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 0)]
    labels = ['Base Esq', 'Topo Esq', 'Topo Dir', 'Base Dir']

    for i, ponto in enumerate(vertices[0]):
        cv2.circle(imagem, tuple(ponto), 8, cores_pontos[i], -1)
        cv2.putText(imagem, labels[i], (ponto[0] + 10, ponto[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cores_pontos[i], 2)

    return imagem


# --------------------------------------------------------------------------------#
# Função 7: LINHAS BRUTAS (sem média)
# --------------------------------------------------------------------------------#
def linhas_brutas(imagem, linhas):
    imagem_linhas = np.zeros_like(imagem)
    if linhas is not None:
        for linha in linhas:
            x1, y1, x2, y2 = linha.reshape(4)
            cv2.line(imagem_linhas, (x1, y1), (x2, y2), (255, 0, 0), 3)
    return imagem_linhas


# --------------------------------------------------------------------------------#
# Função para exibir informações na tela - MODIFICADA
# --------------------------------------------------------------------------------#
def exibir_info(imagem):
    status_memoria = memoria_linhas.obter_status()

    # Fundo semi-transparente para o texto
    overlay = imagem.copy()
    cv2.rectangle(overlay, (10, 10), (450, 220), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, imagem, 0.3, 0, imagem)

    info_text = [
        "=== CONTROLES ROI ===",
        f"Largura Base: {roi_params['largura_base']}%",
        f"Largura Topo: {roi_params['largura_topo']}%",
        f"Altura Topo: {roi_params['altura_topo']}%",
        f"Posicao X: {roi_params['posicao_x']}%",
        "",
        "=== MEMORIA DE LINHAS ===",
        f"Frames Esquerda: {status_memoria['frames_esquerda']}/10",
        f"Frames Direita: {status_memoria['frames_direita']}/10",
        f"Memoria Completa: {'SIM' if status_memoria['memoria_cheia'] else 'NAO'}",
        "",
        "Teclas: 'q'=sair | 'r'=reset | 'c'=limpar"
    ]

    y_offset = 25
    for i, text in enumerate(info_text):
        if "===" in text:
            color = (0, 255, 255)
        elif "Memoria Completa: SIM" in text:
            color = (0, 255, 0)
        elif "Memoria Completa: NAO" in text:
            color = (0, 165, 255)
        else:
            color = (255, 255, 255)

        cv2.putText(imagem, text, (15, y_offset + i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return imagem


# --------------------------------------------------------------------------------#
# Função para resetar ROI para valores padrão
# --------------------------------------------------------------------------------#
def resetar_roi():
    cv2.setTrackbarPos('Largura Base (%)', 'Detector de Faixas', 80)
    cv2.setTrackbarPos('Largura Topo (%)', 'Detector de Faixas', 20)
    cv2.setTrackbarPos('Altura Topo (%)', 'Detector de Faixas', 55)
    cv2.setTrackbarPos('Posicao X (%)', 'Detector de Faixas', 50)


# ================================================================================#
# SCRIPT PRINCIPAL (LOOP DE VÍDEO EM TEMPO REAL)
# ================================================================================#

cap = cv2.VideoCapture("Video3.mp4")

if not cap.isOpened():
    print("Erro: Não foi possível abrir a câmera.")
    exit()

cv2.namedWindow('Detector de Faixas', cv2.WINDOW_NORMAL)
criar_trackbars()

print("=== DETECTOR DE FAIXAS COM MEMORIA DE COORDENADAS ===")
print("Funcionalidades:")
print("- Armazena as últimas 10 detecções de cada linha")
print("- Calcula média móvel para suavizar as linhas")
print("- Trackbars para ajuste da ROI em tempo real")
print("- Visualização do status da memória")
print("\nControles:")
print("- Trackbars: Ajustam a forma da ROI")
print("- 'q': Sair do programa")
print("- 'r': Reset da ROI")
print("- 'c': Limpar memória de linhas")

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame_count += 1

    try:
        # Atualizar parâmetros da ROI das trackbars
        atualizar_roi_params()

        # 1. Detectar bordas
        imagem_canny = detectar_bordas_canny(frame)

        # 2. Isolar a região de interesse e obter os vértices
        imagem_roi, vertices_poligono = definir_regiao_de_interesse(imagem_canny)

        # 3. Detectar linhas com Hough Transform
        linhas_detectadas = cv2.HoughLinesP(imagem_roi, 2, np.pi / 180, 100,
                                            np.array([]), minLineLength=20, maxLineGap=2)

        # 4. Calcular linhas do frame atual
        linha_esquerda_atual, linha_direita_atual = calcular_media_linhas(frame, linhas_detectadas)

        # 5. Adicionar à memória e obter linhas suavizadas
        memoria_linhas.adicionar_linhas(linha_esquerda_atual, linha_direita_atual)
        linhas_suavizadas = memoria_linhas.obter_linhas_suavizadas()

        # 6. Desenhar diferentes tipos de linhas
        imagem_linha_bruta = linhas_brutas(frame, linhas_detectadas)

        # Linhas do frame atual (semi-transparentes)
        linhas_atuais = []
        if linha_esquerda_atual is not None:
            linhas_atuais.append(linha_esquerda_atual)
        if linha_direita_atual is not None:
            linhas_atuais.append(linha_direita_atual)

        imagem_linhas_atuais = desenhar_linhas(frame, np.array(linhas_atuais) if linhas_atuais else None,
                                               cor=(255, 255, 0), espessura=6)

        # Linhas suavizadas (principais)
        imagem_linhas_suavizadas = desenhar_linhas(frame, linhas_suavizadas,
                                                   cor=(0, 255, 0), espessura=10)

        # 7. Combinar todas as imagens
        imagem_final = cv2.addWeighted(frame, 0.7, imagem_linhas_atuais, 0.3, 0)
        imagem_final = cv2.addWeighted(imagem_final, 0.8, imagem_linhas_suavizadas, 1, 1)

        # 8. Desenhar a região de interesse
        imagem_final_com_roi = desenhar_roi(imagem_final.copy(), vertices_poligono)

        # 9. Adicionar informações na tela
        imagem_final_com_info = exibir_info(imagem_final_com_roi)

        # 10. Exibir as imagens
        cv2.imshow("Detector de Faixas", imagem_final_com_info)

        # Janelas auxiliares menores
        cv2.imshow("ROI (Canny)", cv2.resize(imagem_roi, (320, 240)))
        cv2.imshow("Linhas Brutas", cv2.resize(imagem_linha_bruta, (320, 240)))

    except Exception as e:
        print(f"[Erro no frame {frame_count}]: {e}")
        cv2.imshow("Detector de Faixas", frame)

    # Controles de teclado
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        resetar_roi()
        print("ROI resetada para valores padrão")
    elif key == ord('c'):
        memoria_linhas.limpar_memoria()
        print("Memória de linhas limpa")

cap.release()
cv2.destroyAllWindows()