import cv2
import numpy as np
import collections


class LaneDetector:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened(): raise ValueError(f"Erro ao abrir vídeo {video_path}")

        # Obter dimensões e configurar pontos da ROI/Perspectiva
        self.im_h, self.im_w = self._get_video_dims()
        self.roi_points = self._initialize_roi_points()
        self.M, self.Minv = self._calculate_perspective_transform()

        # Atributos de detecção
        self.left_fit, self.right_fit = None, None
        self.lanes_detected = False
        self.history_length = 5
        self.left_fit_history = collections.deque(maxlen=self.history_length)
        self.right_fit_history = collections.deque(maxlen=self.history_length)

        # Atributos de debug
        self.debug_mode = False
        self.alpha = 0.3
        self.color_schemes = [
            {'name': 'Classic Green', 'fill': (0, 255, 100), 'lines': (0, 255, 0), 'left_px': [255, 0, 0],
             'right_px': [0, 0, 255]},
            {'name': 'High-Vis', 'fill': (255, 255, 0), 'lines': (255, 255, 0), 'left_px': [255, 0, 255],
             'right_px': [0, 255, 255]}]
        self.color_scheme_index = 0
        self.selected_point_index = None

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.window_name = "Detector de Faixas - Final (d:debug, c:cor, +/-:transp, q:sair)"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    # --- Funções de Setup e Geometria ---
    def _get_video_dims(self):
        ret, frame = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return (frame.shape[0], frame.shape[1]) if ret else (720, 1280)

    def _initialize_roi_points(self):
        return np.array([
            [int(self.im_w * 0.45), int(self.im_h * 0.65)],  # Ponto superior esquerdo
            [int(self.im_w * 0.55), int(self.im_h * 0.65)],  # Ponto superior direito
            [int(self.im_w * 0.98), self.im_h - 1],  # Ponto inferior direito
            [int(self.im_w * 0.02), self.im_h - 1]  # Ponto inferior esquerdo
        ], dtype=np.int32)

    def _calculate_perspective_transform(self):
        # Transforma os pontos da ROI em uma imagem "bird's-eye"
        src = np.float32(self.roi_points)
        dst = np.float32([
            [0, 0], [self.im_w, 0],
            [self.im_w, self.im_h], [0, self.im_h]
        ])
        M = cv2.getPerspectiveTransform(src, dst)  # Matriz de transformação
        Minv = cv2.getPerspectiveTransform(dst, src)  # Matriz inversa (para desenhar de volta)
        return M, Minv

    def _mouse_callback(self, event, x, y, flags, param):
        if not self.debug_mode: return
        if event == cv2.EVENT_LBUTTONDOWN:
            for i in range(4):
                if np.sqrt((x - self.roi_points[i][0]) ** 2 + (y - self.roi_points[i][1]) ** 2) < 15:
                    self.selected_point_index = i;
                    break
        elif event == cv2.EVENT_MOUSEMOVE and self.selected_point_index is not None:
            self.roi_points[self.selected_point_index][:] = x, y
            self.M, self.Minv = self._calculate_perspective_transform()  # Recalcula a matriz
        elif event == cv2.EVENT_LBUTTONUP:
            self.selected_point_index = None

    # --- Pipeline de Processamento de Imagem ---
    def _process_image(self, image):
        # Aplica filtros de cor/borda e depois a transformação de perspectiva
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        # Filtro de Saturação para faixas amarelas
        s_thresh = (120, 255)
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        # Filtro de Borda (Sobel) no canal L para faixas brancas
        enhanced_l = self.clahe.apply(l_channel)
        sobelx = cv2.Sobel(enhanced_l, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        l_thresh = (40, 100)
        l_binary = np.zeros_like(scaled_sobel)
        l_binary[(scaled_sobel >= l_thresh[0]) & (scaled_sobel <= l_thresh[1])] = 1

        # Combina os filtros
        combined_binary = np.zeros_like(s_binary)
        combined_binary[(s_binary == 1) | (l_binary == 1)] = 1

        # Aplica a transformação de perspectiva ("bird's-eye view")
        warped_binary = cv2.warpPerspective(combined_binary, self.M, (self.im_w, self.im_h))
        return warped_binary

    def _find_lane_pixels_sliding_window(self, binary_warped):
        histogram = np.sum(binary_warped[self.im_h // 2:, :], axis=0)
        midpoint = int(histogram.shape[0] / 2)
        leftx_base, rightx_base = np.argmax(histogram[:midpoint]), np.argmax(histogram[midpoint:]) + midpoint
        nwindows, margin, minpix = 9, 100, 50
        window_height = self.im_h // nwindows

        nonzero = binary_warped.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
        leftx_current, rightx_current = leftx_base, rightx_base
        left_lane_inds, right_lane_inds = [], []
        debug_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255 if self.debug_mode else None

        for window in range(nwindows):
            win_y_low, win_y_high = self.im_h - (window + 1) * window_height, self.im_h - window * window_height

            def process_window(x_current, lane_inds, color):
                win_x_low, win_x_high = x_current - margin, x_current + margin
                if self.debug_mode: cv2.rectangle(debug_img, (win_x_low, win_y_low), (win_x_high, win_y_high), color, 2)
                good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (
                            nonzerox < win_x_high)).nonzero()[0]
                lane_inds.append(good_inds)
                return int(np.mean(nonzerox[good_inds])) if len(good_inds) > minpix else x_current

            leftx_current = process_window(leftx_current, left_lane_inds, (0, 255, 0))
            rightx_current = process_window(rightx_current, right_lane_inds, (0, 255, 0))

        left_lane_inds, right_lane_inds = np.concatenate(left_lane_inds), np.concatenate(right_lane_inds)
        return left_lane_inds, right_lane_inds, nonzerox, nonzeroy, debug_img

    def _sanity_check_and_fit(self, left_inds, right_inds, nonzerox, nonzeroy):
        leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
        rightx, righty = nonzerox[right_inds], nonzeroy[right_inds]

        if len(leftx) < 6 or len(rightx) < 6: return None, None  # Precisa de pontos para um ajuste cúbico

        left_fit, right_fit = np.polyfit(lefty, leftx, 2), np.polyfit(righty, rightx, 2)  # Grau 2 é mais estável aqui

        # Validação: Checa se a distância entre as faixas é razoável
        y_eval = self.im_h // 2
        left_x_eval, right_x_eval = np.polyval(left_fit, y_eval), np.polyval(right_fit, y_eval)
        lane_width = abs(right_x_eval - left_x_eval)

        # Assumindo que a largura da pista é de ~3.7m e cada pixel ~3.7/700 m
        # A largura em pixels deve ser ~700. Checamos se está numa faixa plausível.
        if lane_width < 400 or lane_width > 1000: return None, None

        return left_fit, right_fit

    def _draw_lanes(self, original_image, warped_image, left_fit, right_fit, debug_img):
        colors = self.color_schemes[self.color_scheme_index]
        ploty = np.linspace(0, self.im_h - 1, self.im_h)
        left_fitx = np.polyval(left_fit, ploty)
        right_fitx = np.polyval(right_fit, ploty)

        # Cria a sobreposição
        warp_zero = np.zeros_like(warped_image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_warp, np.int_([pts]), colors['fill'])

        # Transforma a sobreposição de volta para a perspectiva original
        new_warp = cv2.warpPerspective(color_warp, self.Minv, (self.im_w, self.im_h))
        result = cv2.addWeighted(original_image, 1, new_warp, self.alpha, 0)

        # Adiciona o HUD e as infos de debug
        if self.debug_mode:
            # Pinta os pixels usados no debug
            nonzero = warped_image.nonzero()
            debug_img[nonzero[0], nonzero[1]] = colors['left_px']
            self._draw_debug_info(result, debug_img)
        self._draw_hud(result)
        return result

    def _draw_hud(self, image):  # etc.
        if self.debug_mode:
            text_alpha = f"Alpha: {self.alpha:.1f}"
            text_color = f"Color: {self.color_schemes[self.color_scheme_index]['name']}"
            cv2.putText(image, text_alpha, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, text_color, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def _draw_debug_info(self, image, debug_img_search):  # etc.
        cv2.polylines(image, [self.roi_points], isClosed=True, color=(255, 0, 0), thickness=3)
        for i in range(4):
            color = (0, 0, 255) if self.selected_point_index == i else (0, 255, 255)
            cv2.circle(image, tuple(self.roi_points[i]), 10, color, -1)

        debug_view = cv2.resize(debug_img_search, (self.im_w // 4, self.im_h // 4))
        image[10:10 + self.im_h // 4, self.im_w - 10 - self.im_w // 4: self.im_w - 10] = debug_view
        cv2.putText(image, "Debug Search", (self.im_w - 10 - self.im_w // 4, 10 + self.im_h // 4 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def process_frame(self, frame):
        warped_binary = self._process_image(frame)
        left_inds, right_inds, nonzerox, nonzeroy, debug_img = self._find_lane_pixels_sliding_window(warped_binary)

        current_left_fit, current_right_fit = self._sanity_check_and_fit(left_inds, right_inds, nonzerox, nonzeroy)

        if current_left_fit is not None:
            self.left_fit_history.append(current_left_fit)
            self.right_fit_history.append(current_right_fit)

        if self.left_fit_history:
            self.left_fit = np.average(self.left_fit_history, axis=0)
            self.right_fit = np.average(self.right_fit_history, axis=0)

        if self.left_fit is not None:
            return self._draw_lanes(frame, warped_binary, self.left_fit, self.right_fit, debug_img)
        else:
            if self.debug_mode: self._draw_debug_info(frame,
                                                      debug_img if debug_img is not None else np.zeros_like(frame))
            self._draw_hud(frame)
            return frame

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue

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
    video_file = "Video3.mp4"
    detector = LaneDetector(video_file)
    detector.run()