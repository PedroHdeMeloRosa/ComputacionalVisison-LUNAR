import cv2
import numpy as np
import collections
import os
import time


class HighPerformanceLaneDetector:
    """
    Versão final com pipeline de imagem otimizado para performance e robustez,
    focando em gradientes direcionais e reforço de cor.
    """

    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Erro ao abrir vídeo {video_path}")

        self.im_h, self.im_w = self._get_video_dims()
        self.roi_points = self._initialize_roi_points()

        self.lanes_detected = False
        self.left_fit, self.right_fit = None, None
        self.history_length = 7
        self.left_fit_history = collections.deque(maxlen=self.history_length)
        self.right_fit_history = collections.deque(maxlen=self.history_length)

        self.debug_mode = False  # Inicia em modo normal
        self.alpha = 0.4
        self.color_schemes = [
            {'name': 'Verde Classico', 'fill': (0, 255, 100), 'lines': (0, 200, 0)},
            {'name': 'Alta Visibilidade', 'fill': (255, 255, 0), 'lines': (200, 200, 0)},
            {'name': 'Magenta Rave', 'fill': (255, 0, 255), 'lines': (200, 0, 200)}
        ]
        self.color_scheme_index = 0
        self.selected_point_index = None
        self.fps = 0
        self.window_name = "Detector Final (d:debug, c:cor, +/-:transp, q:sair)"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _get_video_dims(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def _initialize_roi_points(self):
        return np.array([
            [int(self.im_w * 0.1), self.im_h], [int(self.im_w * 0.45), int(self.im_h * 0.6)],
            [int(self.im_w * 0.55), int(self.im_h * 0.6)], [int(self.im_w * 0.95), self.im_h]
        ], dtype=np.int32)

    def _mouse_callback(self, event, x, y, flags, param):
        if not self.debug_mode: return
        if event == cv2.EVENT_LBUTTONDOWN:
            for i in range(4):
                if np.linalg.norm(np.array([x, y]) - self.roi_points[i]) < 20: self.selected_point_index = i; break
        elif event == cv2.EVENT_MOUSEMOVE and self.selected_point_index is not None:
            self.roi_points[self.selected_point_index] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            self.selected_point_index = None

    # --- NOVO PIPELINE DE IMAGEM: FOCO EM GRADIENTES E VELOCIDADE ---
    def _create_binary_image(self, image):
        # 1. Recorta a sub-região da ROI para processamento rápido
        x_min, y_min = np.min(self.roi_points[:, 0]), np.min(self.roi_points[:, 1])
        x_max, y_max = np.max(self.roi_points[:, 0]), np.max(self.roi_points[:, 1])
        sub_image = image[y_min:y_max, x_min:x_max]
        if sub_image.size == 0: return np.zeros((self.im_h, self.im_w), dtype=np.uint8)

        # 2. Pipeline de gradiente
        gray = cv2.cvtColor(sub_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        grad_binary = cv2.inRange(scaled_sobel, 30, 100)

        # 3. Pipeline de cor (foco na saturação)
        hls = cv2.cvtColor(sub_image, cv2.COLOR_BGR2HLS)
        s_channel = hls[:, :, 2]
        sat_binary = cv2.inRange(s_channel, 120, 255)

        # 4. Combina os dois pipelines
        processed_sub_mask = cv2.bitwise_or(grad_binary, sat_binary)

        # 5. Reconstrói a máscara e aplica a ROI poligonal
        full_mask = np.zeros((self.im_h, self.im_w), dtype=np.uint8)
        full_mask[y_min:y_max, x_min:x_max] = processed_sub_mask
        roi_poly_mask = np.zeros_like(full_mask)
        cv2.fillPoly(roi_poly_mask, [self.roi_points], 255)

        return cv2.bitwise_and(full_mask, roi_poly_mask)

    def _find_lane_fits(self, b_img, is_blind):
        return self._find_lanes_sliding_window(b_img) if is_blind else self._find_lanes_from_prior(b_img)

    def _find_lanes_sliding_window(self, b_img):
        hist = np.sum(b_img[self.im_h // 2:, :], axis=0);
        mid = self.im_w // 2;
        l_base, r_base = np.argmax(hist[:mid]), np.argmax(hist[mid:]) + mid;
        n, m, min_p = 9, 100, 50;
        h_w = self.im_h // n;
        nz = b_img.nonzero();
        nzy, nzx = nz[0], nz[1];
        l_curr, r_curr = l_base, r_base;
        l_inds, r_inds = [], []
        for w in range(n):
            y_l, y_h = self.im_h - (w + 1) * h_w, self.im_h - w * h_w;
            xl_l, xl_h = l_curr - m, l_curr + m;
            xr_l, xr_h = r_curr - m, r_curr + m
            good_l = ((nzy >= y_l) & (nzy < y_h) & (nzx >= xl_l) & (nzx < xl_h)).nonzero()[0];
            good_r = ((nzy >= y_l) & (nzy < y_h) & (nzx >= xr_l) & (nzx < xr_h)).nonzero()[0]
            l_inds.append(good_l);
            r_inds.append(good_r)
            if len(good_l) > min_p: l_curr = int(np.mean(nzx[good_l]))
            if len(good_r) > min_p: r_curr = int(np.mean(nzx[good_r]))
        return self._fit_and_validate(np.concatenate(l_inds), np.concatenate(r_inds), nzx, nzy)

    def _find_lanes_from_prior(self, b_img):
        m = 80;
        nz = b_img.nonzero();
        nzy, nzx = nz[0], nz[1];
        l_poly, r_poly = np.polyval(self.left_fit, nzy), np.polyval(self.right_fit, nzy)
        l_inds = (nzx > l_poly - m) & (nzx < l_poly + m);
        r_inds = (nzx > r_poly - m) & (nzx < r_poly + m)
        return self._fit_and_validate(l_inds, r_inds, nzx, nzy)

    def _fit_and_validate(self, l_inds, r_inds, nzx, nzy):
        lx, ly = nzx[l_inds], nzy[l_inds];
        rx, ry = nzx[r_inds], nzy[r_inds]
        if len(lx) < 100 or len(rx) < 100: return None, None
        l_fit, r_fit = np.polyfit(ly, lx, 2), np.polyfit(ry, rx, 2)
        if abs(l_fit[0] - r_fit[0]) > 0.002 or abs(l_fit[1] - r_fit[1]) > 1.5: return None, None
        dist = np.polyval(r_fit, self.im_h) - np.polyval(l_fit, self.im_h)
        if not (self.im_w * 0.3 < dist < self.im_w * 0.95): return None, None
        return l_fit, r_fit

    def _draw_lanes_and_hud(self, frame, l_fit, r_fit, d_mask):
        f_img = frame.copy();
        overlay = np.zeros_like(frame);
        colors = self.color_schemes[self.color_scheme_index]
        if l_fit is not None and r_fit is not None:
            ploty = np.linspace(self.roi_points[1][1], self.im_h - 1, 100);
            l_fitx, r_fitx = np.polyval(l_fit, ploty), np.polyval(r_fit, ploty)
            pts_l = np.array([np.transpose(np.vstack([l_fitx, ploty]))]);
            pts_r = np.array([np.flipud(np.transpose(np.vstack([r_fitx, ploty])))]);
            pts = np.hstack((pts_l, pts_r))
            cv2.fillPoly(overlay, np.int_([pts]), colors['fill']);
            cv2.polylines(overlay, np.int_([pts_l]), False, colors['lines'], 20);
            cv2.polylines(overlay, np.int_([pts_r]), False, colors['lines'], 20)
        cv2.addWeighted(overlay, self.alpha, f_img, 1 - self.alpha, 0, f_img);
        self._draw_hud_panel(f_img, d_mask);
        return f_img

    def _draw_hud_panel(self, image, debug_mask):
        if self.debug_mode:
            cv2.polylines(image, [self.roi_points], True, (0, 0, 255), 3)
            for i, p in enumerate(self.roi_points): cv2.circle(image, tuple(p), 12,
                                                               (0, 0, 255) if i == self.selected_point_index else (
                                                               0, 255, 255), -1)
            h, w = self.im_h // 4, self.im_w // 4;
            debug_view = cv2.resize(cv2.cvtColor(debug_mask, cv2.COLOR_GRAY2BGR), (w, h))
            image[10:10 + h, self.im_w - 10 - w:self.im_w - 10] = debug_view;
            cv2.putText(image, "Debug Mask", (self.im_w - 10 - w, 10 + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
            cv2.putText(image, f"Alpha:{self.alpha:.1f}|Cor:{self.color_schemes[self.color_scheme_index]['name']}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.polylines(image, [self.roi_points], True, (255, 0, 0), 2)
        cv2.putText(image, f"FPS:{self.fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    def run(self):
        last_time = time.time()
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0);continue

            b_img = self._create_binary_image(frame)
            l_fit, r_fit = self._find_lane_fits(b_img, is_blind=(not self.lanes_detected))

            if l_fit is not None and r_fit is not None:
                self.lanes_detected = True;
                self.left_fit_history.append(l_fit);
                self.right_fit_history.append(r_fit)
                self.left_fit = np.average(self.left_fit_history, axis=0);
                self.right_fit = np.average(self.right_fit_history, axis=0)
            else:
                self.lanes_detected = False
                if self.left_fit_history:
                    self.left_fit_history.clear();
                    self.right_fit_history.clear()
                self.left_fit, self.right_fit = None, None

            f_img = self._draw_lanes_and_hud(frame, self.left_fit, self.right_fit, b_img);
            cv2.imshow(self.window_name, f_img)

            self.fps = 1.0 / (time.time() - last_time) if time.time() > last_time else 0;
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
        self.cap.release();
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__));
        video_file_path = os.path.join(script_dir, "Video3.mp4")
        if not os.path.exists(video_file_path): raise FileNotFoundError(
            f"Arquivo de vídeo não encontrado! Verifique o caminho: {video_file_path}")
        detector = HighPerformanceLaneDetector(video_path=video_file_path)
        detector.run()
    except Exception as e:
        import traceback;

        print(f"\nOcorreu um erro: {e}");
        traceback.print_exc();
        input("Pressione Enter para fechar.")