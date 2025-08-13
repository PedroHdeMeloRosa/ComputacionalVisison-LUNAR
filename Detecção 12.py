import cv2
import numpy as np
import collections


class RobustPolynomialDetector:
    """
    Versão robusta e aprimorada do LaneDetector.
    Implementa ajuste polinomial direto com pré-processamento, memória inteligente
    e um modo de depuração avançado com ajuste de ROI, parâmetros e minitela de visualização.
    """

    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Erro: Não foi possível abrir o vídeo em {video_path}")

        # Estado da Detecção e Memória
        self.history_length = 15
        self.left_fit_history = collections.deque(maxlen=self.history_length)
        self.right_fit_history = collections.deque(maxlen=self.history_length)
        self.last_stable_left_fit = None
        self.last_stable_right_fit = None

        # Configurações de Debug e Parâmetros Ajustáveis
        self.debug_mode = False
        self.roi_points = self._initialize_roi_points()
        self.selected_point_index = None
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.canny_low = 50
        self.canny_high = 150
        self.white_thresh_L = 200
        self.yellow_thresh_S = 150
        self.selected_param_for_tuning = 1

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
                    self.selected_point_index = i;
                    break
        elif event == cv2.EVENT_MOUSEMOVE and self.selected_point_index is not None:
            self.roi_points[self.selected_point_index] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            self.selected_point_index = None

    def _preprocess_image(self, image):
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        L, S = hls[:, :, 1], hls[:, :, 2]
        enhanced_L = self.clahe.apply(L)
        canny_edges = cv2.Canny(enhanced_L, self.canny_low, self.canny_high)
        white_mask = cv2.inRange(enhanced_L, self.white_thresh_L, 255)
        yellow_mask = cv2.inRange(S, self.yellow_thresh_S, 255)
        combined_color_mask = cv2.bitwise_or(white_mask, yellow_mask)
        final_mask = cv2.bitwise_or(canny_edges, combined_color_mask)
        return final_mask

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

        split_point = (left_base + right_base) // 2
        left_lane_inds = (nonzerox < split_point)
        right_lane_inds = (nonzerox >= split_point)

        leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
        rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

        left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 200 else None
        right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 200 else None

        return left_fit, right_fit

    def _validate_and_smooth(self, left_fit, right_fit, img_height):
        is_left_sane, is_right_sane = False, False
        if left_fit is not None:
            if self.last_stable_left_fit is None:
                is_left_sane = True
            else:
                y_eval = img_height - 1
                current_x, last_x = np.polyval(left_fit, y_eval), np.polyval(self.last_stable_left_fit, y_eval)
                if abs(current_x - last_x) < 150: is_left_sane = True

        if right_fit is not None:
            if self.last_stable_right_fit is None:
                is_right_sane = True
            else:
                y_eval = img_height - 1
                current_x, last_x = np.polyval(right_fit, y_eval), np.polyval(self.last_stable_right_fit, y_eval)
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

    def _draw_lanes(self, image, left_fit, right_fit):
        h, w, _ = image.shape
        overlay = np.zeros_like(image)
        plot_y = np.linspace(self.roi_points[1][1], h - 1, int(h - self.roi_points[1][1]))
        try:
            if left_fit is not None:
                left_fit_x = np.polyval(left_fit, plot_y)
                pts_left = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])
                cv2.polylines(overlay, np.int_([pts_left]), isClosed=False, color=(0, 0, 255), thickness=12)
            if right_fit is not None:
                right_fit_x = np.polyval(right_fit, plot_y)
                pts_right = np.array([np.transpose(np.vstack([right_fit_x, plot_y]))])
                cv2.polylines(overlay, np.int_([pts_right]), isClosed=False, color=(0, 0, 255), thickness=12)
        except (TypeError, ValueError):
            pass
        return cv2.addWeighted(image, 1.0, overlay, 0.9, 0)

    def _draw_debug_hud(self, image, processed_mask):
        """MODIFICADO: Agora inclui a minitela de debug no canto superior direito."""
        # --- HUD de ROI e Parâmetros (canto esquerdo) ---
        cv2.polylines(image, [self.roi_points], isClosed=True, color=(0, 255, 255), thickness=3)
        for i, point in enumerate(self.roi_points):
            color = (0, 0, 255) if i == self.selected_point_index else (0, 255, 255)
            cv2.circle(image, tuple(point), 10, color, -1)

        cv2.putText(image, "DEBUG MODE", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        params = {1: f"Canny Low: {self.canny_low}", 2: f"Canny High: {self.canny_high}",
                  3: f"White Thresh (L): {self.white_thresh_L}", 4: f"Yellow Thresh (S): {self.yellow_thresh_S}"}
        y0, dy = 80, 30
        for i, text in params.items():
            color = (0, 255, 255) if self.selected_param_for_tuning == i else (255, 255, 255)
            cv2.putText(image, f"[{i}] {text}", (10, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(image, "Use +/- to adjust", (10, y0 + (len(params) + 1) * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        # --- Minitela de Debug (canto direito) ---
        h, w, _ = image.shape
        thumb_h, thumb_w = h // 4, w // 4
        # Converte a máscara binária (1 canal) para BGR (3 canais) para poder sobrepor
        debug_view = cv2.cvtColor(processed_mask, cv2.COLOR_GRAY2BGR)
        debug_thumbnail = cv2.resize(debug_view, (thumb_w, thumb_h))

        margin = 10
        # Define a região na imagem final onde a minitela será "colada"
        image[margin:margin + thumb_h, w - margin - thumb_w: w - margin] = debug_thumbnail
        cv2.putText(image, "Processed Mask", (w - margin - thumb_w, margin + thumb_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def process_frame(self, frame):
        """Executa o pipeline completo de detecção."""
        color_mask = self._preprocess_image(frame)
        masked_image = self._apply_roi(color_mask)
        left_fit, right_fit = self._find_and_fit_lanes(masked_image)
        smooth_left, smooth_right = self._validate_and_smooth(left_fit, right_fit, frame.shape[0])
        final_image = self._draw_lanes(frame, smooth_left, smooth_right)

        if self.debug_mode:
            # Passa a máscara processada para a função de desenho do HUD
            self._draw_debug_hud(final_image, masked_image)
        return final_image

    def handle_keyboard_input(self, key):
        if key == ord('q'): return False
        if key == ord('d'): self.debug_mode = not self.debug_mode

        if self.debug_mode:
            if ord('1') <= key <= ord('4'):
                self.selected_param_for_tuning = int(chr(key))
            elif key in [ord('+'), ord('=')]:
                if self.selected_param_for_tuning == 1:
                    self.canny_low = min(500, self.canny_low + 5)
                elif self.selected_param_for_tuning == 2:
                    self.canny_high = min(500, self.canny_high + 5)
                elif self.selected_param_for_tuning == 3:
                    self.white_thresh_L = min(255, self.white_thresh_L + 5)
                elif self.selected_param_for_tuning == 4:
                    self.yellow_thresh_S = min(255, self.yellow_thresh_S + 5)
            elif key in [ord('-'), ord('_')]:
                if self.selected_param_for_tuning == 1:
                    self.canny_low = max(0, self.canny_low - 5)
                elif self.selected_param_for_tuning == 2:
                    self.canny_high = max(0, self.canny_high - 5)
                elif self.selected_param_for_tuning == 3:
                    self.white_thresh_L = max(0, self.white_thresh_L - 5)
                elif self.selected_param_for_tuning == 4:
                    self.yellow_thresh_S = max(0, self.yellow_thresh_S - 5)
        return True

    def run(self):
        window_name = "Robust Detector (d: debug, q: quit, 1-4: select, +/-: adjust)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        is_running = True
        while is_running:
            if not self.cap.isOpened(): break
            ret, frame = self.cap.read()
            if not ret:
                print("Fim do vídeo. Reiniciando...")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            processed_frame = self.process_frame(frame)
            cv2.imshow(window_name, processed_frame)

            key = cv2.waitKey(20) & 0xFF
            is_running = self.handle_keyboard_input(key)

        self.cap.release()
        cv2.destroyAllWindows()
        print("Programa encerrado.")


if __name__ == "__main__":
    video_file = "Video3.mp4"
    try:
        detector = RobustPolynomialDetector(video_file)
        detector.run()
    except Exception as e:
        print(f"Ocorreu um erro fatal: {e}")