import cv2
import numpy as np
import collections


class AdvancedLaneDetector:
    """
    Detector de Faixas Avançado e Robusto.

    Este detector implementa um pipeline completo para identificar e desenhar faixas de
    rolamento em um vídeo de estrada, otimizado para execução em tempo real.
    """

    def __init__(self, smooth_factor=15):
        # --- Configurações e Variáveis de Estado ---
        self.image_shape = None
        self.roi_points = None
        self.transform_matrix = None
        self.inverse_transform_matrix = None

        # Fator de suavização: armazena os 'n' últimos ajustes para calcular a média.
        # Um valor maior resulta em mais suavidade, mas menor reatividade.
        self.smooth_factor = smooth_factor
        self.left_fit_history = collections.deque(maxlen=smooth_factor)
        self.right_fit_history = collections.deque(maxlen=smooth_factor)

        # Coeficientes médios do polinômio para as faixas
        self.left_fit_avg = None
        self.right_fit_avg = None

    def _initialize_geometry(self, image):
        """
        Define a geometria (formato da imagem, ROI, matrizes de transformação)
        com base no primeiro frame do vídeo. É executado apenas uma vez.
        """
        self.image_shape = image.shape
        h, w = self.image_shape[0], self.image_shape[1]

        # 1. Definir a Região de Interesse (ROI) como um trapézio.
        # Estes pontos podem precisar de ajuste para diferentes vídeos/câmeras.
        self.roi_points = np.float32([
            (int(w * 0.45), int(h * 0.62)),  # Topo esquerdo
            (int(w * 0.55), int(h * 0.62)),  # Topo direito
            (int(w * 0.95), h),  # Base direita
            (int(w * 0.05), h)  # Base esquerda
        ])

        # 2. Definir o destino da transformação para a visão de pássaro (um retângulo).
        dst_points = np.float32([
            (0, 0),
            (w, 0),
            (w, h),
            (0, h)
        ])

        # 3. Calcular as matrizes de transformação de perspectiva.
        self.transform_matrix = cv2.getPerspectiveTransform(self.roi_points, dst_points)
        self.inverse_transform_matrix = cv2.getPerspectiveTransform(dst_points, self.roi_points)

    def _thresholding(self, image):
        """
        Aplica filtros de cor e gradiente para isolar os pixels das faixas.
        Isso cria uma imagem binária (preto e branco).
        """
        # Converter para o espaço de cor HLS, que é mais robusto a mudanças de iluminação.
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        s_channel = hls[:, :, 2]  # Canal de Saturação (bom para faixas coloridas)
        l_channel = hls[:, :, 1]  # Canal de Luminosidade (bom para faixas brancas/pretas)

        # Filtro de Gradiente (Sobel no eixo x) para detectar bordas verticais.
        # Aplicado no canal de luminosidade para pegar faixas brancas.
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        sobel_binary = np.zeros_like(scaled_sobel)
        sobel_binary[(scaled_sobel >= 25) & (scaled_sobel <= 100)] = 1

        # Filtro de Cor no canal de Saturação para pegar faixas amarelas.
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= 90) & (s_channel <= 255)] = 1

        # Combinar os dois filtros para uma detecção mais robusta.
        combined_binary = np.zeros_like(sobel_binary)
        combined_binary[(s_binary == 1) | (sobel_binary == 1)] = 1

        return combined_binary

    def _perspective_warp(self, image):
        """Aplica a transformação de perspectiva para obter a visão de pássaro."""
        h, w = image.shape[:2]
        return cv2.warpPerspective(image, self.transform_matrix, (w, h), flags=cv2.INTER_LINEAR)

    def _sliding_window_search(self, binary_warped):
        """
        Encontra os pixels das faixas usando janelas deslizantes e ajusta um polinômio.
        """
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        midpoint = int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 9
        window_height = int(binary_warped.shape[0] / nwindows)
        margin, minpix = 100, 50

        nonzero = binary_warped.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
        leftx_current, rightx_current = leftx_base, rightx_base
        left_lane_inds, right_lane_inds = [], []

        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low, win_xleft_high = leftx_current - margin, leftx_current + margin
            win_xright_low, win_xright_high = rightx_current - margin, rightx_current + margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                        nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                        nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
        rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

        left_fit, right_fit = None, None
        if len(leftx) > 0:
            left_fit = np.polyfit(lefty, leftx, 2)
        if len(rightx) > 0:
            right_fit = np.polyfit(righty, rightx, 2)

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]  # Rosa para esquerda
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]  # Laranja para direita

        return left_fit, right_fit, out_img

    def _draw_lane_overlay(self, image, left_fit, right_fit):
        """
        Desenha a área da faixa e as bordas na imagem original.
        """
        h, w = image.shape[:2]
        ploty = np.linspace(0, h - 1, h)

        try:
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        except TypeError:
            return image

        # Criar uma imagem para desenhar a sobreposição
        overlay_wrap = np.zeros_like(image).astype(np.uint8)

        # Pontos para o polígono que representa a faixa
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Desenhar a área da faixa em verde
        cv2.fillPoly(overlay_wrap, np.int_([pts]), (0, 200, 50))

        # Desenhar as BORDAS em vermelho, como solicitado
        cv2.polylines(overlay_wrap, np.int_([pts_left]), isClosed=False, color=(0, 0, 255), thickness=15)
        cv2.polylines(overlay_wrap, np.int_([pts_right]), isClosed=False, color=(0, 0, 255), thickness=15)

        # Desfazer a transformação de perspectiva
        unwarped_overlay = cv2.warpPerspective(overlay_wrap, self.inverse_transform_matrix, (w, h))

        # Combinar a imagem original com a sobreposição com transparência
        return cv2.addWeighted(image, 1, unwarped_overlay, 0.4, 0)

    def process_frame(self, frame):
        """
        Executa o pipeline completo de detecção de faixas para um único frame.
        """
        if self.image_shape is None:
            self._initialize_geometry(frame)

        # 1. Pipeline de pré-processamento para obter a visão de pássaro binária
        binary_thresholded = self._thresholding(frame)
        binary_warped = self._perspective_warp(binary_thresholded)

        # 2. Encontrar faixas e obter os coeficientes do polinômio
        left_fit, right_fit, debug_image = self._sliding_window_search(binary_warped)

        # 3. Suavização temporal dos resultados
        # Se uma nova detecção for válida, adicione-a ao histórico
        if left_fit is not None and right_fit is not None:
            # Validação simples: verificar se as faixas estão a uma distância razoável
            y_eval = self.image_shape[0] - 1  # Avaliar na base da imagem
            left_x = np.polyval(left_fit, y_eval)
            right_x = np.polyval(right_fit, y_eval)
            if 0.3 * self.image_shape[1] < (right_x - left_x) < 0.9 * self.image_shape[1]:
                self.left_fit_history.append(left_fit)
                self.right_fit_history.append(right_fit)

        # Calcular a média dos últimos 'n' ajustes válidos
        if self.left_fit_history:
            self.left_fit_avg = np.average(self.left_fit_history, axis=0)
        if self.right_fit_history:
            self.right_fit_avg = np.average(self.right_fit_history, axis=0)

        # 4. Desenhar a faixa final na imagem original usando os coeficientes médios
        final_image = frame
        if self.left_fit_avg is not None and self.right_fit_avg is not None:
            final_image = self._draw_lane_overlay(frame, self.left_fit_avg, self.right_fit_avg)

        # Adicionar a imagem de debug no canto para visualização
        h, w = final_image.shape[:2]
        debug_small = cv2.resize(debug_image, (w // 4, h // 4))
        final_image[10:10 + h // 4, w - 10 - w // 4:w - 10] = debug_small

        return final_image


# --- Bloco de Execução Principal ---
if __name__ == "__main__":

    # =======================================================
    # <<<   ALTERE O CAMINHO DO VÍDEO AQUI   >>>
    video_input_path = "Video3.mp4"
    # =======================================================

    cap = cv2.VideoCapture(video_input_path)

    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir o vídeo em '{video_input_path}'")
        exit()

    # Instanciar o detector de faixas
    detector = AdvancedLaneDetector(smooth_factor=15)

    window_name = "Advanced Lane Detection - Pressione 'q' para sair"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Fim do vídeo ou erro na leitura.")
            break

        # Processar o frame
        processed_frame = detector.process_frame(frame)

        # Exibir o resultado
        cv2.imshow(window_name, processed_frame)

        # Checar se a tecla 'q' foi pressionada para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos ao final
    cap.release()
    cv2.destroyAllWindows()
    print("Janelas fechadas. Programa encerrado.")