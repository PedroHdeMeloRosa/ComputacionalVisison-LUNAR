import cv2
import numpy as np
import collections


class DefinitiveLaneDetector:
    """
    Versão definitiva e otimizada do detector de faixas, combinando as melhores características
    de várias abordagens.

    Características Principais:
    - Pipeline de Visão de Pássaro (Bird's-Eye View): Para um ajuste polinomial mais preciso.
    - Busca Adaptativa: Alterna entre uma busca completa (sliding window) e uma busca
      otimizada a partir da detecção anterior, melhorando performance e estabilidade.
    - Filtragem de Imagem Robusta: Usa CLAHE, canais de cor (HLS) e gradiente Sobel.
    - Validação de Sanidade: Verifica a plausibilidade das faixas detectadas (largura, etc.).
    - Suavização Temporal: Usa um histórico de detecções para estabilizar as linhas.
    - Interface de Debug Completa e Interativa:
        - ROI totalmente arrastável.
        - Múltiplos esquemas de cores.
        - Controle de transparência da sobreposição.
        - HUD com informações úteis.
    """

    def __init__(self, video_path, poly_degree=2):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Erro ao abrir vídeo {video_path}")

        # Obter dimensões e configurar pontos da ROI/Perspectiva
        self.im_h, self.im_w = self._get_video_dims()
        self.roi_points = self._initialize_roi_points()

        # Grau do polinômio a ser ajustado (2 é recomendado para estabilidade)
        self.poly_degree = poly_degree

        # Atributos de detecção e estado
        self.lanes_detected = False
        self.left_fit, self.right_fit = None, None
        self.history_length = 7  # Aumentar um pouco para mais suavidade
        self.left_fit_history = collections.deque(maxlen=self.history_length)
        self.right_fit_history = collections.deque(maxlen=self.history_length)

        # Matrizes de transformação (recalculadas dinamicamente)
        self.M, self.Minv = self._calculate_perspective_transform()

        # Pré-processador de imagem
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Atributos de Debug e UI
        self.debug_mode = False
        self.alpha = 0.4  # Transparência
        self.color_schemes = [
            {'name': 'Verde Classico', 'fill': (0, 255, 100), 'left_px': [255, 0, 0], 'right_px': [0, 0, 255]},
            {'name': 'Alta Visibilidade', 'fill': (255, 255, 0), 'left_px': [255, 0, 255], 'right_px': [0, 255, 255]},
            {'name': 'Branco Sutil', 'fill': (220, 220, 220), 'left_px': [180, 180, 180], 'right_px': [120, 120, 120]}
        ]
        self.color_scheme_index = 0
        self.selected_point_index = None

        self.window_name = "Detector Linha(d:debug, c:cor, +/-:transp, q:sair)"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    # --- Funções de Setup e Geometria ---
    def _get_video_dims(self):
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return h, w

    def _initialize_roi_points(self):
        """Define os pontos da Região de Interesse (ROI)."""
        return np.array([
            [int(self.im_w * 0.45), int(self.im_h * 0.65)],  # Ponto superior esquerdo
            [int(self.im_w * 0.55), int(self.im_h * 0.65)],  # Ponto superior direito
            [self.im_w, self.im_h - 1],  # Ponto inferior direito
            [0, self.im_h - 1]  # Ponto inferior esquerdo
        ], dtype=np.float32)

    def _calculate_perspective_transform(self):
        """Calcula as matrizes de transformação de perspectiva e sua inversa."""
        src = self.roi_points
        # O destino é um retângulo que cobre toda a imagem transformada
        dst = np.float32([
            [0, 0], [self.im_w, 0],
            [self.im_w, self.im_h], [0, self.im_h]
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        return M, Minv

    def _mouse_callback(self, event, x, y, flags, param):
        """Gerencia o arraste dos pontos da ROI."""
        if not self.debug_mode: return
        if event == cv2.EVENT_LBUTTONDOWN:
            for i in range(4):
                if np.linalg.norm(np.array([x, y]) - self.roi_points[i]) < 15:
                    self.selected_point_index = i
                    break
        elif event == cv2.EVENT_MOUSEMOVE and self.selected_point_index is not None:
            self.roi_points[self.selected_point_index] = [x, y]
            self.M, self.Minv = self._calculate_perspective_transform()
        elif event == cv2.EVENT_LBUTTONUP:
            self.selected_point_index = None

    # --- Pipeline de Processamento ---
    def _create_binary_warp(self, image):
        """Aplica filtros de cor/borda e a transformação de perspectiva."""
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        l_channel, s_channel = hls[:, :, 1], hls[:, :, 2]

        # Filtro de Saturação para faixas amarelas
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel > 120) & (s_channel <= 255)] = 1

        # Filtro de Gradiente (Sobel) no canal de Luminosidade para faixas brancas
        enhanced_l = self.clahe.apply(l_channel)
        sobelx = cv2.Sobel(enhanced_l, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        l_binary = np.zeros_like(scaled_sobel)
        l_binary[(scaled_sobel >= 40) & (scaled_sobel <= 100)] = 1

        # Combina os dois filtros
        combined_binary = np.zeros_like(s_binary)
        combined_binary[(s_binary == 1) | (l_binary == 1)] = 1

        # Aplica a transformação de perspectiva ("bird's-eye view")
        return cv2.warpPerspective(combined_binary, self.M, (self.im_w, self.im_h))

    def _find_lane_fits(self, binary_warped, out_img_for_debug):
        """
        Função principal de busca: decide entre busca cega ou a partir da anterior.
        """
        if not self.lanes_detected:  # Ou se self.left_fit is None
            # Busca Cega (Sliding Window): Quando não temos referência
            return self._find_lanes_sliding_window(binary_warped, out_img_for_debug)
        else:
            # Busca Otimizada: Usa os ajustes anteriores como guia
            return self._find_lanes_from_prior(binary_warped, out_img_for_debug)

    def _find_lanes_sliding_window(self, binary_warped, out_img):
        """Busca as faixas do zero usando a técnica de janelas deslizantes."""
        histogram = np.sum(binary_warped[self.im_h // 2:, :], axis=0)
        midpoint = self.im_w // 2
        leftx_base, rightx_base = np.argmax(histogram[:midpoint]), np.argmax(histogram[midpoint:]) + midpoint

        nwindows, margin, minpix = 9, 100, 50
        window_height = self.im_h // nwindows
        nonzero = binary_warped.nonzero()
        nonzeroy, nonzerox = nonzero[0], nonzero[1]

        leftx_current, rightx_current = leftx_base, rightx_base
        left_lane_inds, right_lane_inds = [], []

        for window in range(nwindows):
            win_y_low, win_y_high = self.im_h - (window + 1) * window_height, self.im_h - window * window_height

            # Janela Esquerda
            win_xleft_low, win_xleft_high = leftx_current - margin, leftx_current + margin
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            left_lane_inds.append(good_left_inds)
            if len(good_left_inds) > minpix: leftx_current = int(np.mean(nonzerox[good_left_inds]))

            # Janela Direita
            win_xright_low, win_xright_high = rightx_current - margin, rightx_current + margin
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            right_lane_inds.append(good_right_inds)
            if len(good_right_inds) > minpix: rightx_current = int(np.mean(nonzerox[good_right_inds]))

            if self.debug_mode:
                cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        left_lane_inds, right_lane_inds = np.concatenate(left_lane_inds), np.concatenate(right_lane_inds)
        return self._fit_and_validate(left_lane_inds, right_lane_inds, nonzerox, nonzeroy, out_img)

    def _find_lanes_from_prior(self, binary_warped, out_img):
        """Busca faixas em uma área definida pelos ajustes polinomiais anteriores."""
        margin = 80
        nonzero = binary_warped.nonzero()
        nonzeroy, nonzerox = nonzero[0], nonzero[1]

        # Gera as curvas do frame anterior
        left_fit_poly = np.polyval(self.left_fit, nonzeroy)
        right_fit_poly = np.polyval(self.right_fit, nonzeroy)

        # Seleciona os pixels dentro da margem
        left_lane_inds = (nonzerox > left_fit_poly - margin) & (nonzerox < left_fit_poly + margin)
        right_lane_inds = (nonzerox > right_fit_poly - margin) & (nonzerox < right_fit_poly + margin)

        return self._fit_and_validate(left_lane_inds, right_lane_inds, nonzerox, nonzeroy, out_img)

    def _fit_and_validate(self, left_inds, right_inds, nonzerox, nonzeroy, out_img):
        """Ajusta polinômios, valida e colore a imagem de debug."""
        colors = self.color_schemes[self.color_scheme_index]
        leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
        rightx, righty = nonzerox[right_inds], nonzeroy[right_inds]

        if self.debug_mode:
            out_img[nonzeroy[left_inds], nonzerox[left_inds]] = colors['left_px']
            out_img[nonzeroy[right_inds], nonzerox[right_inds]] = colors['right_px']

        # Condição mínima para tentar um ajuste
        if len(leftx) < 100 or len(rightx) < 100:
            return None, None

        left_fit = np.polyfit(lefty, leftx, self.poly_degree)
        right_fit = np.polyfit(righty, rightx, self.poly_degree)

        # Validação de sanidade: Checa a distância entre as faixas no meio da imagem
        y_eval = self.im_h / 2
        left_x_eval = np.polyval(left_fit, y_eval)
        right_x_eval = np.polyval(right_fit, y_eval)
        lane_width = abs(right_x_eval - left_x_eval)

        # A largura da faixa em pixels na visão de pássaro deve ser relativamente constante.
        # Esses valores podem precisar de ajuste dependendo do seu setup de câmera/ROI.
        if not (self.im_w * 0.4 < lane_width < self.im_w * 0.9):
            return None, None

        return left_fit, right_fit

    # --- Funções de Desenho e Visualização ---
    def _draw_final_lanes(self, original_image, left_fit, right_fit):
        """Desenha a área da faixa e a projeta de volta na perspectiva original."""
        colors = self.color_schemes[self.color_scheme_index]
        overlay = np.zeros_like(original_image)
        ploty = np.linspace(0, self.im_h - 1, self.im_h)

        try:
            left_fitx = np.polyval(left_fit, ploty)
            right_fitx = np.polyval(right_fit, ploty)

            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))

            cv2.fillPoly(overlay, np.int_([pts]), colors['fill'])

            # Transforma a sobreposição de volta para a perspectiva original
            unwarped_overlay = cv2.warpPerspective(overlay, self.Minv, (self.im_w, self.im_h))
            return cv2.addWeighted(original_image, 1, unwarped_overlay, self.alpha, 0)
        except (TypeError, np.linalg.LinAlgError):
            # Em caso de falha (fit é None ou problema numérico), retorna a imagem original
            return original_image

    def _draw_hud(self, image):
        """Desenha o HUD (Heads-Up Display) com informações de debug."""
        if self.debug_mode:
            text_alpha = f"Alpha: {self.alpha:.1f}"
            text_color = f"Cor: {self.color_schemes[self.color_scheme_index]['name']}"
            cv2.putText(image, text_alpha, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, text_color, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def _draw_debug_info(self, final_image, debug_img_search):
        """Adiciona a ROI e a janela de debug na imagem final."""
        cv2.polylines(final_image, [np.int32(self.roi_points)], isClosed=True, color=(255, 0, 0), thickness=3)
        for i, point in enumerate(self.roi_points):
            color = (0, 0, 255) if self.selected_point_index == i else (0, 255, 255)
            cv2.circle(final_image, tuple(np.int32(point)), 10, color, -1)

        # Adiciona a pequena janela com a visão de busca
        debug_view = cv2.resize(debug_img_search, (self.im_w // 4, self.im_h // 4))
        final_image[10:10 + debug_view.shape[0], self.im_w - 10 - debug_view.shape[1]:self.im_w - 10] = debug_view
        cv2.putText(final_image, "Debug Search View",
                    (self.im_w - 10 - debug_view.shape[1], 10 + debug_view.shape[0] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # --- Loop Principal ---
    def process_frame(self, frame):
        # 1. Cria a imagem binária transformada (visão de pássaro)
        warped_binary = self._create_binary_warp(frame)

        # Imagem para desenhar os retângulos/pixels de busca
        debug_img = np.dstack((warped_binary, warped_binary, warped_binary)) * 255 if self.debug_mode else None

        # 2. Encontra as faixas com a estratégia adaptativa
        current_left_fit, current_right_fit = self._find_lane_fits(warped_binary, debug_img)

        # 3. Atualiza o estado e suaviza os resultados
        if current_left_fit is not None:
            self.lanes_detected = True
            self.left_fit_history.append(current_left_fit)
            self.right_fit_history.append(current_right_fit)
            self.left_fit = np.average(self.left_fit_history, axis=0)
            self.right_fit = np.average(self.right_fit_history, axis=0)
        else:
            self.lanes_detected = False
            # Opcional: manter o último `fit` por alguns frames, ou limpar o histórico
            if not self.left_fit_history:
                self.left_fit, self.right_fit = None, None

        # 4. Desenha o resultado
        if self.left_fit is not None:
            final_image = self._draw_final_lanes(frame, self.left_fit, self.right_fit)
        else:
            final_image = frame

        # 5. Adiciona informações de debug
        self._draw_hud(final_image)
        if self.debug_mode:
            self._draw_debug_info(final_image, debug_img)

        return final_image

    def run(self):
        """Inicia o loop principal de processamento do vídeo."""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Fim do vídeo. Reiniciando...")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            final_image = self.process_frame(frame)
            cv2.imshow(self.window_name, final_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
            elif key == ord('c'):
                self.color_scheme_index = (self.color_scheme_index + 1) % len(self.color_schemes)
            elif key in [ord('+'), ord('=')]:
                self.alpha = min(1.0, self.alpha + 0.1)
            elif key in [ord('-'), ord('_')]:
                self.alpha = max(0.0, self.alpha - 0.1)

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_file = "Video3.mp4"  # Coloque o caminho do seu vídeo aqui
    detector = DefinitiveLaneDetector(video_file)
    detector.run()