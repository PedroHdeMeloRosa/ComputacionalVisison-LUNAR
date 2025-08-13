import cv2
import numpy as np
import collections


class LaneDetector:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened(): raise ValueError(f"Erro ao abrir vídeo {video_path}")

        self.left_fit, self.right_fit = None, None
        self.lanes_detected = False

        self.history_length = 5
        self.left_fit_history = collections.deque(maxlen=self.history_length)
        self.right_fit_history = collections.deque(maxlen=self.history_length)

        self.debug_mode = False
        self.alpha = 0.4
        self.color_schemes = [
            {'name': 'Classic Green', 'fill': (0, 255, 100), 'lines': (0, 255, 0), 'left_px': [255, 0, 0],
             'right_px': [0, 0, 255]},
            {'name': 'High-Vis', 'fill': (255, 255, 0), 'lines': (255, 255, 0), 'left_px': [255, 0, 255],
             'right_px': [0, 255, 255]},
            {'name': 'Subtle White', 'fill': (220, 220, 220), 'lines': (255, 255, 255), 'left_px': [180, 180, 180],
             'right_px': [120, 120, 120]}
        ]
        self.color_scheme_index = 0
        self.num_color_schemes = len(self.color_schemes)

        self.roi_points = self._initialize_roi_points()
        self.selected_point_index = None

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.window_name = "Detector de Faixas (d:debug, c:cor, +/-:transp, q:sair)"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    # ... Funções _initialize_roi_points, _mouse_callback, _apply_image_filters_and_roi não mudam ...
    def _initialize_roi_points(self):
        ret, frame = self.cap.read()
        h, w = (frame.shape[0], frame.shape[1]) if ret else (720, 1280)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return np.array(
            [[int(w * 0.1), h], [int(w * 0.45), int(h * 0.6)], [int(w * 0.55), int(h * 0.6)], [int(w * 0.95), h]],
            dtype=np.int32)

    def _mouse_callback(self, event, x, y, flags, param):
        if not self.debug_mode: return
        if event == cv2.EVENT_LBUTTONDOWN:
            for i in range(4):
                if np.sqrt((x - self.roi_points[i][0]) ** 2 + (y - self.roi_points[i][1]) ** 2) < 15:
                    self.selected_point_index = i;
                    break
        elif event == cv2.EVENT_MOUSEMOVE and self.selected_point_index is not None:
            self.roi_points[self.selected_point_index] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            self.selected_point_index = None

    def _apply_image_filters_and_roi(self, image):
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        enhanced_l = self.clahe.apply(l_channel)
        gray_enhanced = cv2.cvtColor(cv2.merge([hls[:, :, 0], enhanced_l, hls[:, :, 2]]), cv2.COLOR_HLS2BGR)
        gray_enhanced = cv2.cvtColor(gray_enhanced, cv2.COLOR_BGR2GRAY)

        lower_white = np.array([0, 200, 0], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hls, lower_white, upper_white)

        lower_yellow = np.array([15, 30, 115], dtype=np.uint8)
        upper_yellow = np.array([35, 204, 255], dtype=np.uint8)
        yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

        color_mask = cv2.bitwise_or(white_mask, yellow_mask)
        canny_edges = cv2.Canny(gray_enhanced, 75, 150)
        combined_mask = cv2.bitwise_or(canny_edges, color_mask)

        roi_mask = np.zeros_like(combined_mask)
        cv2.fillPoly(roi_mask, [self.roi_points], 255)
        return cv2.bitwise_and(combined_mask, roi_mask)

    # MODIFICADO: Esta função agora usa polyfit de 3º grau
    def _find_lane_pixels(self, binary_warped, out_img, is_blind_search):
        colors = self.color_schemes[self.color_scheme_index]

        if is_blind_search:  # Sliding Window
            h, w = binary_warped.shape
            histogram = np.sum(binary_warped[h // 2:, :], axis=0)
            midpoint = int(histogram.shape[0] / 2)
            leftx_base, rightx_base = np.argmax(histogram[:midpoint]), np.argmax(histogram[midpoint:]) + midpoint
            nwindows, margin, minpix = 9, 100, 50
            window_height = int(h / nwindows)
            nonzero = binary_warped.nonzero()
            nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
            leftx_current, rightx_current = leftx_base, rightx_base
            left_lane_inds, right_lane_inds = [], []

            for window in range(nwindows):
                win_y_low, win_y_high = h - (window + 1) * window_height, h - window * window_height
                win_xleft_low, win_xleft_high = leftx_current - margin, leftx_current + margin
                win_xright_low, win_xright_high = rightx_current - margin, rightx_current + margin

                if self.debug_mode:
                    cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
                    cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

                good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                            nonzerox < win_xleft_high)).nonzero()[0]
                good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                            nonzerox < win_xright_high)).nonzero()[0]
                left_lane_inds.append(good_left)
                right_lane_inds.append(good_right)
                if len(good_left) > minpix: leftx_current = int(np.mean(nonzerox[good_left]))
                if len(good_right) > minpix: rightx_current = int(np.mean(nonzerox[good_right]))

            left_lane_inds, right_lane_inds = np.concatenate(left_lane_inds), np.concatenate(right_lane_inds)
        else:  # Search from prior
            margin = 80
            nonzero = binary_warped.nonzero()
            nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
            # MODIFICADO: usa a fórmula cúbica para a busca a partir do anterior
            left_poly = self.left_fit[0] * (nonzeroy ** 3) + self.left_fit[1] * (nonzeroy ** 2) + self.left_fit[
                2] * nonzeroy + self.left_fit[3]
            left_lane_inds = ((nonzerox > (left_poly - margin)) & (nonzerox < (left_poly + margin)))
            right_poly = self.right_fit[0] * (nonzeroy ** 3) + self.right_fit[1] * (nonzeroy ** 2) + self.right_fit[
                2] * nonzeroy + self.right_fit[3]
            right_lane_inds = ((nonzerox > (right_poly - margin)) & (nonzerox < (right_poly + margin)))

        leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
        rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

        # MODIFICADO: O '2' virou '3' para um ajuste de 3º grau
        left_fit = np.polyfit(lefty, leftx, 3) if len(lefty) > 150 else None
        right_fit = np.polyfit(righty, rightx, 3) if len(righty) > 150 else None

        if self.debug_mode:
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = colors['left_px']
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = colors['right_px']

        return left_fit, right_fit, out_img

    # ... A função process_frame continua a mesma ...
    def process_frame(self, frame):
        masked_image = self._apply_image_filters_and_roi(frame)
        debug_img_base = np.dstack((masked_image, masked_image, masked_image)) * 255

        if not self.lanes_detected or self.left_fit is None or self.right_fit is None:
            current_left_fit, current_right_fit, debug_img = self._find_lane_pixels(masked_image, debug_img_base, True)
        else:
            current_left_fit, current_right_fit, debug_img = self._find_lane_pixels(masked_image, debug_img_base, False)

        if current_left_fit is not None and current_right_fit is not None:
            self.lanes_detected = True
            self.left_fit, self.right_fit = current_left_fit, current_right_fit
            self.left_fit_history.append(current_left_fit)
            self.right_fit_history.append(current_right_fit)
        else:
            self.lanes_detected = False
            self.left_fit, self.right_fit = None, None

        smooth_left_fit = np.average(self.left_fit_history, axis=0) if self.left_fit_history else None
        smooth_right_fit = np.average(self.right_fit_history, axis=0) if self.right_fit_history else None

        line_overlay = self._draw_polynomial_curves(frame, smooth_left_fit, smooth_right_fit)
        final_image = cv2.addWeighted(frame, 1, line_overlay, self.alpha, 0)

        if self.debug_mode:
            self._draw_debug_info(final_image, debug_img)
        else:
            cv2.polylines(final_image, [self.roi_points], isClosed=True, color=(255, 0, 0), thickness=2)

        self._draw_hud(final_image)
        return final_image

    # MODIFICADO: Esta função agora calcula os pontos usando a fórmula cúbica
    def _draw_polynomial_curves(self, image, left_fit, right_fit):
        colors = self.color_schemes[self.color_scheme_index]
        h = image.shape[0]
        overlay = np.zeros_like(image)
        plot_y = np.linspace(int(h * 0.6), h - 1, int(h * 0.4))

        try:
            if left_fit is not None and right_fit is not None:
                # Usa a fórmula cúbica: ax³ + bx² + cx + d
                left_x = left_fit[0] * plot_y ** 3 + left_fit[1] * plot_y ** 2 + left_fit[2] * plot_y + left_fit[3]
                right_x = right_fit[0] * plot_y ** 3 + right_fit[1] * plot_y ** 2 + right_fit[2] * plot_y + right_fit[3]

                pts_left = np.array([np.transpose(np.vstack([left_x, plot_y]))])
                pts_right = np.array([np.flipud(np.transpose(np.vstack([right_x, plot_y])))])
                pts = np.hstack((pts_left, pts_right))

                cv2.fillPoly(overlay, np.int_([pts]), colors['fill'])
                cv2.polylines(overlay, np.int_([pts_left]), isClosed=False, color=colors['lines'], thickness=15)
                cv2.polylines(overlay, np.int_([pts_right]), isClosed=False, color=colors['lines'], thickness=15)
        except Exception:
            pass
        return overlay

    # ... Funções de desenho do HUD e debug info continuam as mesmas ...
    def _draw_hud(self, image):
        if self.debug_mode:
            color_name = self.color_schemes[self.color_scheme_index]['name']
            text_alpha = f"Alpha: {self.alpha:.1f}"
            text_color = f"Color: {color_name}"
            cv2.putText(image, text_alpha, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, text_color, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def _draw_debug_info(self, image, debug_img_from_search):
        h, w = image.shape[:2]
        cv2.polylines(image, [self.roi_points], isClosed=True, color=(255, 0, 0), thickness=3)
        for i in range(4):
            color = (0, 0, 255) if self.selected_point_index == i else (0, 255, 255)
            cv2.circle(image, tuple(self.roi_points[i]), 10, color, -1)

        debug_view = cv2.resize(debug_img_from_search, (w // 4, h // 4))
        image[10:10 + h // 4, w - 10 - w // 4: w - 10] = debug_view
        cv2.putText(image, "Debug Search", (w - 10 - w // 4, 10 + h // 4 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

    # ... A função run continua a mesma, com os controles de teclado ...
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
                self.color_scheme_index = (self.color_scheme_index + 1) % self.num_color_schemes
            elif key == ord('+') or key == ord('='):
                self.alpha = min(1.0, self.alpha + 0.1)
            elif key == ord('-') or key == ord('_'):
                self.alpha = max(0.0, self.alpha - 0.1)

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_file = "Video3.mp4"
    detector = LaneDetector(video_file)
    detector.run()