import cv2
import numpy as np
import collections


class LaneDetector:
    """
    Classe para detectar faixas de rolagem em um vídeo.
    Implementa ajuste de curva polinomial, normalização de iluminação
    e um modo de depuração com ROI totalmente ajustável.
    """

    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Erro: Não foi possível abrir o vídeo em {video_path}")

        # Memória para suavização dos coeficientes do polinômio [a, b, c]
        self.history_length = 10
        self.left_fit_history = collections.deque(maxlen=self.history_length)
        self.right_fit_history = collections.deque(maxlen=self.history_length)

        # Último ajuste estável para fallback
        self.last_stable_left_fit = None
        self.last_stable_right_fit = None

        self.debug_mode = False
        self.roi_points = self._initialize_roi_points()
        self.selected_point_index = None

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.window_name = "Detector de Faixas (Pressione 'd' para debug, 'q' para sair)"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _initialize_roi_points(self):
        """Define os pontos iniciais da ROI."""
        ret, frame = self.cap.read()
        height, width = (frame.shape[0], frame.shape[1]) if ret else (720, 1280)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Pontos do polígono: [inferior_esquerdo, superior_esquerdo, superior_direito, inferior_direito]
        return np.array([
            [int(width * 0.1), height],
            [int(width * 0.45), int(height * 0.6)],
            [int(width * 0.55), int(height * 0.6)],
            [int(width * 0.95), height]
        ], dtype=np.int32)

    # MODIFICADO: Agora todos os 4 pontos são arrastáveis
    def _mouse_callback(self, event, x, y, flags, param):
        """Gerencia os eventos do mouse para ajustar a ROI."""
        if not self.debug_mode: return

        # Todos os pontos (0, 1, 2, 3) agora são arrastáveis
        if event == cv2.EVENT_LBUTTONDOWN:
            for i in range(4):
                dist = np.sqrt((x - self.roi_points[i][0]) ** 2 + (y - self.roi_points[i][1]) ** 2)
                if dist < 15:
                    self.selected_point_index = i
                    break
        elif event == cv2.EVENT_MOUSEMOVE and self.selected_point_index is not None:
            self.roi_points[self.selected_point_index] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            self.selected_point_index = None

    def _apply_image_filters(self, image):
        """Aplica CLAHE, filtros de cor e Canny."""
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        _, l, _ = cv2.split(hls)
        enhanced_l = self.clahe.apply(l)
        gray_enhanced = cv2.cvtColor(cv2.merge([_, enhanced_l, _]), cv2.COLOR_HLS2BGR)
        gray_enhanced = cv2.cvtColor(gray_enhanced, cv2.COLOR_BGR2GRAY)

        lower_white = np.array([0, 190, 0], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hls, lower_white, upper_white)

        lower_yellow = np.array([10, 0, 90], dtype=np.uint8)
        upper_yellow = np.array([50, 255, 255], dtype=np.uint8)
        yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

        color_mask = cv2.bitwise_or(white_mask, yellow_mask)
        canny_edges = cv2.Canny(gray_enhanced, 100, 100)

        final_mask = cv2.bitwise_or(canny_edges, color_mask)
        return final_mask

    def _apply_roi_mask(self, image):
        """Aplica a máscara da Região de Interesse."""
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [self.roi_points], 255)
        return cv2.bitwise_and(image, mask)

    # NOVO E FUNDAMENTAL: Encontra pixels e ajusta um polinômio
    def _fit_polynomial(self, masked_image):
        """Encontra pixels de faixas e ajusta um polinômio de 2º grau."""
        # Encontra as coordenadas de todos os pixels não-zero (brancos) na máscara
        all_nonzero = masked_image.nonzero()
        all_nonzero_y = np.array(all_nonzero[0])
        all_nonzero_x = np.array(all_nonzero[1])

        # Define um ponto central para separar faixas esquerda/direita
        midpoint = int(masked_image.shape[1] / 2)

        # Separa os pixels em esquerdo e direito
        left_lane_inds = (all_nonzero_x < midpoint)
        right_lane_inds = (all_nonzero_x >= midpoint)

        left_x, left_y = all_nonzero_x[left_lane_inds], all_nonzero_y[left_lane_inds]
        right_x, right_y = all_nonzero_x[right_lane_inds], all_nonzero_y[right_lane_inds]

        left_fit, right_fit = None, None

        # Ajusta um polinômio de 2º grau se houver pixels suficientes
        # Usamos x = f(y) = ay^2 + by + c, pois as linhas são majoritariamente verticais
        if len(left_x) > 100:
            left_fit = np.polyfit(left_y, left_x, 2)
        if len(right_x) > 100:
            right_fit = np.polyfit(right_y, right_x, 2)

        return left_fit, right_fit

    def _smooth_and_validate_fits(self, left_fit, right_fit, img_height):
        """Suaviza os ajustes polinomiais usando o histórico e valida-os."""
        # Validação do ajuste esquerdo
        if left_fit is not None:
            # Sanity check: se o desvio for muito grande, descarte
            if self.last_stable_left_fit is not None:
                # Calcula x na base da imagem para o fit atual e o último estável
                y_eval = img_height
                current_x = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
                last_x = self.last_stable_left_fit[0] * y_eval ** 2 + self.last_stable_left_fit[1] * y_eval + \
                         self.last_stable_left_fit[2]
                if abs(current_x - last_x) < 200:  # Limiar de 200 pixels na base
                    self.left_fit_history.append(left_fit)
            else:
                self.left_fit_history.append(left_fit)

        # Validação do ajuste direito
        if right_fit is not None:
            if self.last_stable_right_fit is not None:
                y_eval = img_height
                current_x = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]
                last_x = self.last_stable_right_fit[0] * y_eval ** 2 + self.last_stable_right_fit[1] * y_eval + \
                         self.last_stable_right_fit[2]
                if abs(current_x - last_x) < 200:
                    self.right_fit_history.append(right_fit)
            else:
                self.right_fit_history.append(right_fit)

        # Calcula a média dos ajustes no histórico
        final_left_fit = np.average(self.left_fit_history,
                                    axis=0) if self.left_fit_history else self.last_stable_left_fit
        final_right_fit = np.average(self.right_fit_history,
                                     axis=0) if self.right_fit_history else self.last_stable_right_fit

        # Atualiza o último ajuste estável
        if final_left_fit is not None: self.last_stable_left_fit = final_left_fit
        if final_right_fit is not None: self.last_stable_right_fit = final_right_fit

        return final_left_fit, final_right_fit

    # MODIFICADO: Desenha as curvas polinomiais
    def _draw_polynomial_curves(self, image, left_fit, right_fit):
        """Desenha as curvas das faixas e a área entre elas na imagem."""
        h, w, _ = image.shape
        # Cria uma imagem em branco para desenhar
        overlay = np.zeros_like(image)

        # Gera os pontos para a curva (de y_min a y_max)
        plot_y = np.linspace(int(h * 0.6), h - 1, h)

        try:
            if left_fit is not None:
                # Calcula os pontos x para a curva esquerda
                left_fit_x = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
                pts_left = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])
                cv2.polylines(overlay, np.int_([pts_left]), isClosed=False, color=(0, 255, 0), thickness=10)

            if right_fit is not None:
                # Calcula os pontos x para a curva direita
                right_fit_x = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]
                pts_right = np.array([np.transpose(np.vstack([right_fit_x, plot_y]))])
                cv2.polylines(overlay, np.int_([pts_right]), isClosed=False, color=(0, 255, 0), thickness=10)

        except (TypeError, ValueError):
            # Lida com casos em que o fit é None no início
            pass

        return overlay

    # MODIFICADO: Desenha todos os 4 pontos da ROI
    def _draw_debug_info(self, image, final_mask):
        """Desenha informações de depuração na imagem final."""
        cv2.polylines(image, [self.roi_points], isClosed=True, color=(255, 0, 0), thickness=3)
        # Desenha os 4 pontos arrastáveis
        for i in range(4):
            color = (0, 0, 255) if self.selected_point_index == i else (0, 255, 255)
            cv2.circle(image, tuple(self.roi_points[i]), 10, color, -1)

        h, w, _ = image.shape
        debug_view = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
        debug_view = cv2.resize(debug_view, (w // 4, h // 4))
        image[10:10 + h // 4, w - 10 - w // 4: w - 10] = debug_view
        cv2.putText(image, "Debug Mask", (w - 10 - w // 4, 10 + h // 4 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # MODIFICADO: Pipeline de processamento atualizado para polinômios
    def process_frame(self, frame):
        """Executa o pipeline completo de detecção."""
        # 1. Aplicar filtros e obter máscara de bordas
        processed_mask = self._apply_image_filters(frame)
        masked_edges = self._apply_roi_mask(processed_mask)

        # 2. Encontrar pixels e ajustar polinômios
        left_fit, right_fit = self._fit_polynomial(masked_edges)

        # 3. Validar e suavizar os ajustes
        smooth_left_fit, smooth_right_fit = self._smooth_and_validate_fits(left_fit, right_fit, frame.shape[0])

        # 4. Desenhar as curvas polinomiais
        line_overlay = self._draw_polynomial_curves(frame, smooth_left_fit, smooth_right_fit)

        # 5. Combinar imagem original com o desenho das faixas
        final_image = cv2.addWeighted(frame, 0.8, line_overlay, 1, 1)

        # 6. Desenhar informações de depuração se ativo
        if self.debug_mode:
            self._draw_debug_info(final_image, masked_edges)
        else:
            cv2.polylines(final_image, [self.roi_points], isClosed=True, color=(255, 0, 0), thickness=2)

        return final_image

    def run(self):
        """Inicia o loop principal de processamento do vídeo."""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            final_image = self.process_frame(frame)
            cv2.imshow(self.window_name, final_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
                print(f"Modo de Depuração {'Ativado' if self.debug_mode else 'Desativado'}.")

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_file = "Video3.mp4"  # Certifique-se de que o caminho está correto
    detector = LaneDetector(video_file)
    detector.run()