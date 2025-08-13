import cv2
import numpy as np
import collections


class AdvancedPolynomialDetector:
    """
    Detector de Faixas com ajuste polinomial direto, debug avançado e
    ajuste de parâmetros em tempo real.
    """

    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Erro: Não foi possível abrir o vídeo em {video_path}")

        # --- Estado da Detecção e Memória ---
        self.history_length = 15
        self.left_fit_history = collections.deque(maxlen=self.history_length)
        self.right_fit_history = collections.deque(maxlen=self.history_length)
        self.last_stable_left_fit = None
        self.last_stable_right_fit = None

        # --- Configurações de Debug e Parâmetros Ajustáveis ---
        self.debug_mode = False
        self.roi_points = self._initialize_roi_points()
        self.selected_point_index = None

        # Parâmetros de thresholding que podem ser ajustados em tempo real
        self.white_threshold = 200  # Limite inferior para o canal Vermelho (faixas brancas)
        self.yellow_threshold = 120  # Limite inferior para o canal Saturação (faixas amarelas)
        self.selected_param_for_tuning = 'white'  # Qual parâmetro está sendo ajustado

        # --- Setup da Janela ---
        self.window_name = "Detector Polinomial Avançado (d: debug, q: sair, 1/2: selecionar, +/-: ajustar)"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _initialize_roi_points(self):
        ret, frame = self.cap.read()
        h, w = (frame.shape[:2]) if ret else (720, 1280)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return np.array([
            [int(w * 0.1), h - 1], [int(w * 0.45), int(h * 0.6)],
            [int(w * 0.55), int(h * 0.6)], [int(w * 0.95), h - 1]
        ], dtype=np.int32)

    def _mouse_callback(self, event, x, y, flags, param):
        if not self.debug_mode: return
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, point in enumerate(self.roi_points):
                if np.linalg.norm(np.array([x, y]) - point) < 20:
                    self.selected_point_index = i
                    break
        elif event == cv2.EVENT_MOUSEMOVE and self.selected_point_index is not None:
            self.roi_points[self.selected_point_index] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            self.selected_point_index = None

    def _preprocess_image(self, image):
        """Aplica filtros usando os parâmetros de classe ajustáveis."""
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        R = blurred[:, :, 2]
        hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)
        S = hls[:, :, 2]

        # Usa os atributos da classe para os thresholds
        white_mask = cv2.inRange(R, self.white_threshold, 255)
        yellow_mask = cv2.inRange(S, self.yellow_threshold, 255)

        return cv2.bitwise_or(white_mask, yellow_mask)

    def _apply_roi(self, image):
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [self.roi_points], 255)
        return cv2.bitwise_and(image, mask)

    def _find_and_fit_lanes(self, masked_image):
        h, w = masked_image.shape
        histogram = np.sum(masked_image[h // 2:, :], axis=0)
        midpoint = w // 2
        left_base = np.argmax(histogram[:midpoint])
        right_base = np.argmax(histogram[midpoint:]) + midpoint

        nonzero = masked_image.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])

        left_lane_inds = (nonzerox < (left_base + right_base) // 2)
        right_lane_inds = (nonzerox >= (left_base + right_base) // 2)

        leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
        rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

        left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 200 else None
        right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 200 else None

        return left_fit, right_fit

    def _validate_and_smooth(self, left_fit, right_fit, img_height):
        # (Esta função permanece a mesma da versão anterior, garantindo estabilidade)
        is_left_sane, is_right_sane = False, False

        if left_fit is not None:
            if self.last_stable_left_fit is None:
                is_left_sane = True
            else:
                y_eval = img_height - 1
                current_x = np.polyval(left_fit, y_eval)
                last_x = np.polyval(self.last_stable_left_fit, y_eval)
                if abs(current_x - last_x) < 150: is_left_sane = True

        if right_fit is not None:
            if self.last_stable_right_fit is None:
                is_right_sane = True
            else:
                y_eval = img_height - 1
                current_x = np.polyval(right_fit, y_eval)
                last_x = np.polyval(self.last_stable_right_fit, y_eval)
                if abs(current_x - last_x) < 150: is_right_sane = True

        if is_left_sane: self.left_fit_history.append(left_fit)
        if is_right_sane: self.right_fit_history.append(right_fit)

        final_left_fit = np.average(self.left_fit_history,
                                    axis=0) if self.left_fit_history else self.last_stable_left_fit
        final_right_fit = np.average(self.right_fit_history,
                                     axis=0) if self.right_fit_history else self.last_stable_right_fit

        if final_left_fit is not None: self.last_stable_left_fit = final_left_fit
        if final_right_fit is not None: self.last_stable_right_fit = final_right_fit

        return final_left_fit, final_right_fit

    def _draw_lanes_on_image(self, image, left_fit, right_fit):
        """MODIFICADO: Desenha apenas as linhas das faixas, sem o preenchimento verde."""
        h, w, _ = image.shape
        overlay = np.zeros_like(image)
        plot_y = np.linspace(self.roi_points[1][1], h - 1, int(h - self.roi_points[1][1]))

        try:
            # Desenha a linha esquerda
            if left_fit is not None:
                left_fit_x = np.polyval(left_fit, plot_y)
                pts_left = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])
                cv2.polylines(overlay, np.int_([pts_left]), isClosed=False, color=(0, 0, 255), thickness=12)

            # Desenha a linha direita
            if right_fit is not None:
                right_fit_x = np.polyval(right_fit, plot_y)
                pts_right = np.array([np.transpose(np.vstack([right_fit_x, plot_y]))])
                cv2.polylines(overlay, np.int_([pts_right]), isClosed=False, color=(0, 0, 255), thickness=12)

        except (TypeError, ValueError):
            pass

            # Combina o overlay (agora só com as linhas) com a imagem original
        return cv2.addWeighted(image, 1.0, overlay, 0.8, 0)  # Alpha pode ser maior agora

    def _draw_debug_hud(self, image):
        """Desenha o HUD completo com informações de ROI e ajuste de parâmetros."""
        # --- HUD para ajuste de ROI ---
        cv2.polylines(image, [self.roi_points], isClosed=True, color=(0, 255, 255), thickness=3)
        for i, point in enumerate(self.roi_points):
            color = (0, 0, 255) if i == self.selected_point_index else (0, 255, 255)
            cv2.circle(image, tuple(point), 10, color, -1)

        # --- HUD para ajuste de parâmetros ---
        cv2.putText(image, "DEBUG MODE ACTIVE", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Define as cores do texto do HUD (branco para normal, amarelo para selecionado)
        white_text_color = (0, 255, 255) if self.selected_param_for_tuning == 'white' else (255, 255, 255)
        yellow_text_color = (0, 255, 255) if self.selected_param_for_tuning == 'yellow' else (255, 255, 255)

        cv2.putText(image, f"[1] White Threshold: {self.white_threshold}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    white_text_color, 2)
        cv2.putText(image, f"[2] Yellow Threshold: {self.yellow_threshold}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    yellow_text_color, 2)
        cv2.putText(image, "Use +/- to adjust selected", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def process_frame(self, frame):
        color_mask = self._preprocess_image(frame)
        masked_image = self._apply_roi(color_mask)
        left_fit, right_fit = self._find_and_fit_lanes(masked_image)
        smooth_left, smooth_right = self._validate_and_smooth(left_fit, right_fit, frame.shape[0])
        final_image = self._draw_lanes_on_image(frame, smooth_left, smooth_right)

        if self.debug_mode:
            self._draw_debug_hud(final_image)

        return final_image

    def run(self):
        """Inicia o loop principal e gerencia a entrada do teclado."""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Fim do vídeo. Reiniciando...")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            processed_frame = self.process_frame(frame)
            cv2.imshow(self.window_name, processed_frame)

            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
                status = "ATIVADO" if self.debug_mode else "DESATIVADO"
                print(f"Modo de Depuração: {status}")

            # --- Gerenciamento de Teclas para Ajuste de Parâmetros ---
            if self.debug_mode:
                if key == ord('1'):
                    self.selected_param_for_tuning = 'white'
                    print("Selecionado: White Threshold")
                elif key == ord('2'):
                    self.selected_param_for_tuning = 'yellow'
                    print("Selecionado: Yellow Threshold")
                elif key == ord('+') or key == ord('='):
                    if self.selected_param_for_tuning == 'white':
                        self.white_threshold = min(255, self.white_threshold + 5)
                        print(f"White Threshold: {self.white_threshold}")
                    else:
                        self.yellow_threshold = min(255, self.yellow_threshold + 5)
                        print(f"Yellow Threshold: {self.yellow_threshold}")
                elif key == ord('-') or key == ord('_'):
                    if self.selected_param_for_tuning == 'white':
                        self.white_threshold = max(0, self.white_threshold - 5)
                        print(f"White Threshold: {self.white_threshold}")
                    else:
                        self.yellow_threshold = max(0, self.yellow_threshold - 5)
                        print(f"Yellow Threshold: {self.yellow_threshold}")

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_file = "Video3.mp4"
    try:
        detector = AdvancedPolynomialDetector(video_file)
        detector.run()
    except ValueError as e:
        print(e)