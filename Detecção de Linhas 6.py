import cv2
import numpy as np
import time
import os
import imageio
import collections


class UltimateLaneDetector:
    """
    Versão definitiva que combina:
    - Leitor de vídeo robusto 'imageio'.
    - Pipeline de processamento avançado (CLAHE, HLS, Canny).
    - Busca de faixas adaptativa (Sliding Window / Search from Prior).
    - HUD informativo com múltiplos esquemas de cores e controles.
    - ROI interativa.
    """

    def __init__(self, video_path):
        # 1. Leitor de vídeo robusto com imageio
        try:
            self.reader = imageio.get_reader(video_path, 'ffmpeg')
            meta = self.reader.get_meta_data()
            self.im_h, self.im_w = meta['size'][::-1]
        except Exception as e:
            raise IOError(f"Erro ao abrir vídeo com imageio: {e}\nVerifique o caminho e se o arquivo está corrompido.")

        # 2. Estado da detecção e suavização
        self.left_fit, self.right_fit = None, None
        self.lanes_detected = False
        self.history_length = 7
        self.left_fit_history = collections.deque(maxlen=self.history_length)
        self.right_fit_history = collections.deque(maxlen=self.history_length)

        # 3. Pipeline de processamento e ROI
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.roi_points = self._initialize_roi_points()

        # 4. Controles de UI e Debug
        self.debug_mode = False
        self.alpha = 0.4
        self.color_schemes = [
            {'name': 'Verde Classico', 'fill': (0, 255, 100), 'lines': (0, 255, 0), 'px': (255, 0, 255)},
            {'name': 'Alta Visibilidade', 'fill': (255, 255, 0), 'lines': (255, 255, 0), 'px': (0, 255, 255)},
            {'name': 'Branco Sutil', 'fill': (220, 220, 220), 'lines': (255, 255, 255), 'px': (150, 150, 150)}
        ]
        self.color_scheme_index = 0
        self.selected_point_index = None

        # 5. Utilitários e Janela
        self.fps = 0
        self.window_name = "Detector Definitivo (d:debug, c:cor, +/-:alpha, q:sair)"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _initialize_roi_points(self):
        """Define os pontos iniciais da ROI."""
        return np.array([
            [int(self.im_w * 0.1), self.im_h], [int(self.im_w * 0.45), int(self.im_h * 0.6)],
            [int(self.im_w * 0.55), int(self.im_h * 0.6)], [int(self.im_w * 0.95), self.im_h]
        ], dtype=np.int32)

    def _mouse_callback(self, event, x, y, flags, param):
        if not self.debug_mode: return
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, point in enumerate(self.roi_points):
                if np.linalg.norm(np.array([x, y]) - point) < 15: self.selected_point_index = i; break
        elif event == cv2.EVENT_MOUSEMOVE and self.selected_point_index is not None:
            self.roi_points[self.selected_point_index] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            self.selected_point_index = None

    def _create_binary_image(self, image):
        """Pipeline de imagem com CLAHE, HLS e Canny."""
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]

        enhanced_l = self.clahe.apply(l_channel)
        gray_enhanced = cv2.cvtColor(cv2.merge([hls[:, :, 0], enhanced_l, hls[:, :, 2]]), cv2.COLOR_HLS2BGR)
        gray_enhanced = cv2.cvtColor(gray_enhanced, cv2.COLOR_BGR2GRAY)

        # Filtros de Cor
        lower_white = np.array([0, 200, 0], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hls, lower_white, upper_white)

        lower_yellow = np.array([15, 30, 115], dtype=np.uint8)
        upper_yellow = np.array([35, 204, 255], dtype=np.uint8)
        yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

        color_mask = cv2.bitwise_or(white_mask, yellow_mask)
        canny_edges = cv2.Canny(gray_enhanced, 50, 150)

        combined_mask = cv2.bitwise_or(color_mask, canny_edges)

        roi_mask = np.zeros_like(combined_mask)
        cv2.fillPoly(roi_mask, [self.roi_points], 255)
        return cv2.bitwise_and(combined_mask, roi_mask)

    def _find_lane_pixels(self, binary_image, is_blind_search):
        """Encontra os pixels da faixa usando busca adaptativa."""
        out_img = None
        if self.debug_mode:
            out_img = np.dstack((binary_image, binary_image, binary_image))

        if is_blind_search:  # Busca Cega com Sliding Window
            h, w = binary_image.shape
            histogram = np.sum(binary_image[h // 2:, :], axis=0)
            midpoint = w // 2
            leftx_base, rightx_base = np.argmax(histogram[:midpoint]), np.argmax(histogram[midpoint:]) + midpoint
            nwindows, margin, minpix = 9, 100, 50
            window_height = h // nwindows
            nonzero = binary_image.nonzero()
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

        else:  # Busca Otimizada a Partir da Posição Anterior
            margin = 80
            nonzero = binary_image.nonzero()
            nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
            left_poly = np.polyval(self.left_fit, nonzeroy)
            left_lane_inds = (nonzerox > (left_poly - margin)) & (nonzerox < (left_poly + margin))
            right_poly = np.polyval(self.right_fit, nonzeroy)
            right_lane_inds = (nonzerox > (right_poly - margin)) & (nonzerox < (right_poly + margin))

        leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
        rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

        left_fit = np.polyfit(lefty, leftx, 2) if len(lefty) > 150 else None
        right_fit = np.polyfit(righty, rightx, 2) if len(righty) > 150 else None

        if self.debug_mode:
            px_color = self.color_schemes[self.color_scheme_index]['px']
            out_img[lefty, leftx] = px_color
            out_img[righty, rightx] = px_color

        return left_fit, right_fit, out_img

    def _draw_lanes_and_hud(self, frame, left_fit, right_fit, debug_image_mask):
        """Desenha as faixas, preenchimento e HUD na imagem final."""
        overlay = np.zeros_like(frame)
        colors = self.color_schemes[self.color_scheme_index]

        if left_fit is not None and right_fit is not None:
            ploty = np.linspace(int(self.im_h * 0.6), self.im_h - 1, int(self.im_h * 0.4))
            left_fitx = np.polyval(left_fit, ploty)
            right_fitx = np.polyval(right_fit, ploty)

            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))

            cv2.fillPoly(overlay, np.int_([pts]), colors['fill'])
            cv2.polylines(overlay, np.int_([pts_left]), False, colors['lines'], 15)
            cv2.polylines(overlay, np.int_([pts_right]), False, colors['lines'], 15)

        final_image = cv2.addWeighted(frame.copy(), 1, overlay, self.alpha, 0)

        self._draw_hud_info(final_image, debug_image_mask)

        return final_image

    def _draw_hud_info(self, image, debug_mask):
        """Desenha a ROI e todas as informações de debug/HUD."""
        if not self.debug_mode:
            cv2.polylines(image, [self.roi_points], True, (255, 0, 0), 2)
            cv2.putText(image, f"FPS: {self.fps:.2f}", (self.im_w - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                        2)
            return

        # --- Modo de Debug Ativo ---
        # Desenha ROI interativa
        cv2.polylines(image, [self.roi_points], True, (0, 0, 255), 3)
        for i, point in enumerate(self.roi_points):
            color = (0, 0, 255) if self.selected_point_index == i else (0, 255, 255)
            cv2.circle(image, tuple(point), 12, color, -1)

        # Desenha janela com a máscara binária de busca
        h_sm, w_sm = self.im_h // 4, self.im_w // 4
        debug_view = cv2.resize(debug_mask, (w_sm, h_sm))
        image[10:10 + h_sm, self.im_w - 10 - w_sm:self.im_w - 10] = debug_view
        cv2.putText(image, "Debug Search Mask", (self.im_w - 10 - w_sm, 10 + h_sm + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

        # Desenha HUD informativo
        color_name = self.color_schemes[self.color_scheme_index]['name']
        cv2.putText(image, f"FPS: {self.fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(image, f"Alpha: {self.alpha:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(image, f"Cor: {color_name}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def run(self):
        """Loop principal com pipeline adaptativo."""
        last_time = time.time()
        for i, frame_rgb in enumerate(self.reader):
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # 1. Pipeline de Processamento
            binary_image = self._create_binary_image(frame)
            current_fit, right_fit, debug_img = self._find_lane_pixels(binary_image,
                                                                       is_blind_search=(not self.lanes_detected))

            # 2. Atualização de Estado e Suavização
            if current_fit is not None and right_fit is not None:
                self.lanes_detected = True
                self.left_fit, self.right_fit = current_fit, right_fit
                self.left_fit_history.append(current_fit)
                self.right_fit_history.append(right_fit)
            else:
                self.lanes_detected = False

            smooth_left = np.mean(self.left_fit_history, axis=0) if self.left_fit_history else None
            smooth_right = np.mean(self.right_fit_history, axis=0) if self.right_fit_history else None

            # 4. Desenho e exibição
            final_image = self._draw_lanes_and_hud(frame, smooth_left, smooth_right, debug_img)
            cv2.imshow(self.window_name, final_image)

            # 5. Cálculo de FPS e controle de teclado
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

        print(f"Tentando abrir vídeo em: {video_file_path}")

        if not os.path.exists(video_file_path):
            raise FileNotFoundError(f"Arquivo de vídeo não encontrado! Verifique o caminho: {video_file_path}")

        detector = UltimateLaneDetector(video_path=video_file_path)
        detector.run()

    except Exception as e:
        print(f"\nOcorreu um erro: {e}")
        import traceback

        traceback.print_exc()
        input("Pressione Enter para fechar.")