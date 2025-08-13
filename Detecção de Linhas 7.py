import cv2
import numpy as np
import time
import os
import imageio


# --- As classes KalmanLaneTracker e LaneLine são mantidas como estão ---
class KalmanLaneTracker:
    def __init__(self, n_coeffs=3):
        self.kf = cv2.KalmanFilter(n_coeffs, n_coeffs)
        self.kf.transitionMatrix = np.eye(n_coeffs, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(n_coeffs, dtype=np.float32) * 1e-5
        self.kf.measurementNoiseCov = np.eye(n_coeffs, dtype=np.float32) * 1e-2
        self.kf.measurementMatrix = np.eye(n_coeffs, dtype=np.float32)
        self.kf.errorCovPost = np.eye(n_coeffs, dtype=np.float32)

    def predict(self): return self.kf.predict()

    def update(self, measurement): return self.kf.correct(np.array(measurement, dtype=np.float32))


class LaneLine:
    def __init__(self, poly_degree=2):
        self.poly_degree = poly_degree
        self.tracker = KalmanLaneTracker(poly_degree + 1)
        self.fit_coeffs = None
        self.is_detected = False
        self.frames_since_detected = 0


# --- CLASSE PRINCIPAL TOTALMENTE INTEGRADA ---
class ProfessionalLaneDetector:
    def __init__(self, video_path):
        try:
            self.reader = imageio.get_reader(video_path, 'ffmpeg')
            meta = self.reader.get_meta_data()
            self.im_h, self.im_w = meta['size'][::-1]
        except Exception as e:
            raise IOError(f"Erro ao abrir vídeo com imageio: {e}")

        self.poly_degree = 2
        self.left_lane = LaneLine(self.poly_degree)
        self.right_lane = LaneLine(self.poly_degree)

        self.roi_points = self._initialize_roi_points()
        self.M, self.Minv = self._calculate_perspective_matrices()

        self.YM_PER_PIX, self.XM_PER_PIX = 30 / 720, 3.7 / 700

        # UI e Debug com todos os controles
        self.debug_mode = False
        self.alpha = 0.4
        self.color_schemes = [
            {'name': 'Verde Clássico', 'fill': (0, 255, 100), 'lines': (0, 255, 0)},
            {'name': 'Ciano-Vis', 'fill': (0, 255, 255), 'lines': (0, 180, 180)},
            {'name': 'Magenta Rave', 'fill': (255, 0, 255), 'lines': (180, 0, 180)}
        ]
        self.color_scheme_index = 0
        self.selected_point_index = None

        self.fps = 0
        self.window_name = "Detector Profissional (d:debug, c:cor, +/-:alpha, q:sair)"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _initialize_roi_points(self):
        return np.float32([
            (int(self.im_w * 0.2), self.im_h), (int(self.im_w * 0.45), int(self.im_h * 0.6)),
            (int(self.im_w * 0.55), int(self.im_h * 0.6)), (int(self.im_w * 0.8), self.im_h)
        ])

    def _calculate_perspective_matrices(self):
        src, dst = self.roi_points, np.float32([[0, self.im_h], [0, 0], [self.im_w, 0], [self.im_w, self.im_h]])
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        return M, Minv

    def _mouse_callback(self, event, x, y, flags, param):
        if not self.debug_mode: return
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, point in enumerate(self.roi_points):
                if np.linalg.norm(np.array([x, y]) - point) < 20:
                    self.selected_point_index = i;
                    break
        elif event == cv2.EVENT_MOUSEMOVE and self.selected_point_index is not None:
            self.roi_points[self.selected_point_index] = [x, y]
            self.M, self.Minv = self._calculate_perspective_matrices()  # Recalcula em tempo real
        elif event == cv2.EVENT_LBUTTONUP:
            self.selected_point_index = None

    def _create_binary_image(self, image):
        # A lógica de criar a imagem binária 'warped' permanece a mesma
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        l_channel, s_channel = hls[:, :, 1], hls[:, :, 2]
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        sxbinary = cv2.inRange(scaled_sobel, 20, 100)
        s_binary = cv2.inRange(s_channel, 170, 255)
        binary = cv2.bitwise_or(sxbinary, s_binary)
        return cv2.warpPerspective(binary, self.M, (self.im_w, self.im_h))

    def _find_lane_fits(self, warped_binary_image):
        # A lógica de busca cega / busca a partir do anterior permanece a mesma
        is_blind_search = not (self.left_lane.is_detected and self.right_lane.is_detected)
        left_fit, right_fit = self._blind_search(warped_binary_image) if is_blind_search else self._search_from_prior(
            warped_binary_image)
        self._update_lane_tracker(self.left_lane, left_fit)
        self._update_lane_tracker(self.right_lane, right_fit)

    def _blind_search(self, warped_binary_image):
        # ... (sem alterações)
        histogram = np.sum(warped_binary_image[warped_binary_image.shape[0] // 2:, :], axis=0)
        midpoint = np.int32(histogram.shape[0] / 2)
        leftx_base, rightx_base = np.argmax(histogram[:midpoint]), np.argmax(histogram[midpoint:]) + midpoint
        nwindows, margin, minpix = 9, 100, 50
        window_height = np.int32(warped_binary_image.shape[0] / nwindows)
        nonzero = warped_binary_image.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
        left_lane_inds, right_lane_inds = [], []

        for window in range(nwindows):
            def process_window(x_current, lane_inds):
                win_y_low = warped_binary_image.shape[0] - (window + 1) * window_height
                win_y_high = warped_binary_image.shape[0] - window * window_height
                win_x_low, win_x_high = x_current - margin, x_current + margin
                good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (
                            nonzerox < win_x_high)).nonzero()[0]
                lane_inds.append(good_inds)
                return np.int32(np.mean(nonzerox[good_inds])) if len(good_inds) > minpix else x_current

            leftx_base, rightx_base = process_window(leftx_base, left_lane_inds), process_window(rightx_base,
                                                                                                 right_lane_inds)

        leftx, lefty = nonzerox[np.concatenate(left_lane_inds)], nonzeroy[np.concatenate(left_lane_inds)]
        rightx, righty = nonzerox[np.concatenate(right_lane_inds)], nonzeroy[np.concatenate(right_lane_inds)]
        left_fit = np.polyfit(lefty, leftx, self.poly_degree) if len(lefty) > minpix else None
        right_fit = np.polyfit(righty, rightx, self.poly_degree) if len(righty) > minpix else None
        return left_fit, right_fit

    def _search_from_prior(self, warped_binary_image):
        # ... (sem alterações)
        left_fit_predicted = self.left_lane.tracker.predict().flatten()
        right_fit_predicted = self.right_lane.tracker.predict().flatten()
        nonzero, margin = warped_binary_image.nonzero(), 80
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
        left_poly, right_poly = np.polyval(left_fit_predicted, nonzeroy), np.polyval(right_fit_predicted, nonzeroy)
        left_lane_inds = (nonzerox > (left_poly - margin)) & (nonzerox < (left_poly + margin))
        right_lane_inds = (nonzerox > (right_poly - margin)) & (nonzerox < (right_poly + margin))
        leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
        rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]
        left_fit = np.polyfit(lefty, leftx, self.poly_degree) if len(lefty) > 50 else None
        right_fit = np.polyfit(righty, rightx, self.poly_degree) if len(righty) > 50 else None
        return left_fit, right_fit

    def _update_lane_tracker(self, lane, fit):
        # ... (sem alterações)
        if fit is not None:
            lane.fit_coeffs, lane.is_detected, lane.frames_since_detected = lane.tracker.update(fit).flatten(), True, 0
        else:
            lane.is_detected = False
            if lane.frames_since_detected < 10:
                lane.fit_coeffs, lane.frames_since_detected = lane.tracker.predict().flatten(), lane.frames_since_detected + 1
            else:
                lane.fit_coeffs = None

    def _calculate_real_world_metrics(self):
        # A lógica de cálculo de métricas do mundo real permanece a mesma
        # ... (sem alterações)
        if self.left_lane.fit_coeffs is None or self.right_lane.fit_coeffs is None:
            return "N/A", "N/A"
        ploty = np.linspace(0, self.im_h - 1, self.im_h)
        y_eval = np.max(ploty)
        left_fit_cr = np.polyfit(ploty * self.YM_PER_PIX,
                                 np.polyval(self.left_lane.fit_coeffs, ploty) * self.XM_PER_PIX, 2)
        right_fit_cr = np.polyfit(ploty * self.YM_PER_PIX,
                                  np.polyval(self.right_lane.fit_coeffs, ploty) * self.XM_PER_PIX, 2)
        epsilon = 1e-6
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * self.YM_PER_PIX + left_fit_cr[1]) ** 2) ** 1.5) / (
                    np.absolute(2 * left_fit_cr[0]) + epsilon)
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * self.YM_PER_PIX + right_fit_cr[1]) ** 2) ** 1.5) / (
                    np.absolute(2 * right_fit_cr[0]) + epsilon)
        avg_curverad = min((left_curverad + right_curverad) / 2, 5000)
        lane_center_px = (np.polyval(self.left_lane.fit_coeffs, y_eval) + np.polyval(self.right_lane.fit_coeffs,
                                                                                     y_eval)) / 2
        offset_m = (self.im_w / 2 - lane_center_px) * self.XM_PER_PIX
        curve_str = f"{avg_curverad:.0f}m" if avg_curverad < 4900 else "Estrada Reta"
        return curve_str, f"{offset_m:.2f}m"

    # --- FUNÇÃO DE DESENHO TOTALMENTE RESTAURADA E INTEGRADA ---
    def _draw_lanes_and_hud(self, frame, warped_binary):
        colors = self.color_schemes[self.color_scheme_index]

        # Cria a sobreposição de faixas na visão de pássaro
        overlay_warp = np.zeros_like(warped_binary).astype(np.uint8)
        overlay_warp = cv2.cvtColor(overlay_warp, cv2.COLOR_GRAY2BGR)

        if self.left_lane.fit_coeffs is not None and self.right_lane.fit_coeffs is not None:
            ploty = np.linspace(0, self.im_h - 1, self.im_h)
            left_fitx = np.polyval(self.left_lane.fit_coeffs, ploty)
            right_fitx = np.polyval(self.right_lane.fit_coeffs, ploty)

            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))

            # Desenha a área preenchida e as linhas na visão de topo
            cv2.fillPoly(overlay_warp, np.int_([pts]), colors['fill'])
            cv2.polylines(overlay_warp, np.int_([pts_left]), False, colors['lines'], 15)
            cv2.polylines(overlay_warp, np.int_([pts_right]), False, colors['lines'], 15)

        # Projeta a sobreposição de volta para a perspectiva da câmera
        projected_overlay = cv2.warpPerspective(overlay_warp, self.Minv, (self.im_w, self.im_h))
        result = cv2.addWeighted(frame, 1, projected_overlay, self.alpha, 0)

        # Desenha o HUD
        self._draw_hud_panel(result, warped_binary)

        return result

    def _draw_hud_panel(self, image, warped_binary_mask):
        """Desenha o HUD completo com métricas e controles de debug."""
        # Modo Padrão
        if not self.debug_mode:
            curve_rad, offset = self._calculate_real_world_metrics()
            cv2.putText(image, f"Raio de Curvatura: {curve_rad}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2)
            cv2.putText(image, f"Desvio do Centro: {offset}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        2)
            cv2.putText(image, f"FPS: {self.fps:.2f}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return

        # --- Modo de Debug Ativo ---
        # Desenha ROI interativa
        cv2.polylines(image, [self.roi_points.astype(np.int32)], True, (0, 0, 255), 3)
        for i, point in enumerate(self.roi_points):
            color = (0, 0, 255) if self.selected_point_index == i else (0, 255, 255)
            cv2.circle(image, tuple(point.astype(int)), 12, color, -1)

        # Desenha janela com a máscara de busca
        h_sm, w_sm = self.im_h // 4, self.im_w // 4
        debug_view = cv2.cvtColor(warped_binary_mask, cv2.COLOR_GRAY2BGR)
        debug_view = cv2.resize(debug_view, (w_sm, h_sm))
        image[10:10 + h_sm, self.im_w - 10 - w_sm:self.im_w - 10] = debug_view
        cv2.putText(image, "Debug Search Mask", (self.im_w - 10 - w_sm, 10 + h_sm + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

        # Desenha HUD de debug
        color_name = self.color_schemes[self.color_scheme_index]['name']
        cv2.putText(image, f"FPS: {self.fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(image, f"Alpha: {self.alpha:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(image, f"Cor: {color_name}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def run(self):
        last_time = time.time()
        for frame_rgb in self.reader:
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            warped_binary = self._create_binary_image(frame)
            self._find_lane_fits(warped_binary)
            final_image = self._draw_lanes_and_hud(frame, warped_binary)

            cv2.imshow(self.window_name, final_image)

            self.fps = 1.0 / (time.time() - last_time) if (time.time() - last_time) > 0 else 0
            last_time = time.time()

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

        self.reader.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        video_file_path = os.path.join(script_dir, "Video3.mp4")
        if not os.path.exists(video_file_path):
            raise FileNotFoundError(f"Arquivo de vídeo não encontrado: {video_file_path}")
        detector = ProfessionalLaneDetector(video_path=video_file_path)
        detector.run()
    except Exception as e:
        print(f"\nOcorreu um erro: {e}")
        import traceback

        traceback.print_exc()
        input("Pressione Enter para fechar.")