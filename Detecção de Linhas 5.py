import cv2
import numpy as np
import collections
import time


class OptimizedLaneDetector:
    """
    Detector de faixas otimizado para alto desempenho, removendo a transformação
    de perspectiva (visão de pássaro) e adicionando um HUD informativo.

    Otimizações:
    - Sem `warpPerspective`: O ajuste polinomial é feito na perspectiva original da imagem.
    - HUD Avançado: Exibe desvio, curvatura da pista, visibilidade e FPS.
    - Visual Aprimorado: Desenha as linhas da faixa e uma seta de navegação,
      inspirado em sistemas ADAS modernos.
    """

    def __init__(self, video_path, poly_degree=2):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Erro ao abrir vídeo {video_path}")

        # Configurações de dimensão e ROI
        self.im_h, self.im_w = self._get_video_dims()
        self.roi_points = self._initialize_roi_points()

        # Atributos de detecção e suavização
        self.poly_degree = poly_degree
        self.left_fit, self.right_fit = None, None
        self.history_length = 10
        self.left_fit_history = collections.deque(maxlen=self.history_length)
        self.right_fit_history = collections.deque(maxlen=self.history_length)

        # Atributos para o HUD e cálculo de FPS
        self.fps_start_time = time.time()
        self.fps = 0
        self.frame_count = 0

        # Processador de imagem e UI
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.window_name = "Detector Otimizado com HUD (d:debug ROI, q:sair)"
        cv2.namedWindow(self.window_name)
        self.debug_roi_mode = False
        self.selected_point_index = None
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    # --- Funções de Setup e Geometria (Modificadas) ---
    def _get_video_dims(self):
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return h, w

    def _initialize_roi_points(self):
        # A ROI agora é crucial para focar a detecção na área correta
        return np.array([
            [int(self.im_w * 0.1), self.im_h],  # inf-esq
            [int(self.im_w * 0.45), int(self.im_h * 0.6)],  # sup-esq
            [int(self.im_w * 0.55), int(self.im_h * 0.6)],  # sup-dir
            [int(self.im_w * 0.9), self.im_h],  # inf-dir
        ], dtype=np.int32)

    def _mouse_callback(self, event, x, y, flags, param):
        """Gerencia o arraste dos pontos da ROI (apenas se o debug_roi_mode estiver ativo)."""
        if not self.debug_roi_mode: return
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, point in enumerate(self.roi_points):
                if np.linalg.norm(np.array([x, y]) - point) < 15:
                    self.selected_point_index = i
                    break
        elif event == cv2.EVENT_MOUSEMOVE and self.selected_point_index is not None:
            self.roi_points[self.selected_point_index] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            self.selected_point_index = None

    # --- Pipeline de Processamento de Imagem (Otimizado) ---
    def _create_binary_roi(self, image):
        """Aplica filtros e MÁSCARA de ROI. Sem transformação de perspectiva."""
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        # Filtro de Saturação (faixas amarelas)
        s_binary = cv2.inRange(s_channel, 120, 255)

        # Filtro de gradiente Sobel (faixas brancas)
        enhanced_l = self.clahe.apply(l_channel)
        sobelx = cv2.Sobel(enhanced_l, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        l_binary = cv2.inRange(scaled_sobel, 40, 100)

        combined_binary = cv2.bitwise_or(s_binary, l_binary)

        # Aplica a máscara da ROI
        mask = np.zeros_like(combined_binary)
        cv2.fillPoly(mask, [self.roi_points], 255)
        return cv2.bitwise_and(combined_binary, mask)

    def _fit_polynomial(self, binary_masked_image):
        """Encontra pixels e ajusta um polinômio DIRETAMENTE na imagem mascarada."""
        nonzero = binary_masked_image.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])

        # Ponto de separação entre faixas esquerda e direita
        midpoint = self.im_w // 2
        left_lane_inds = nonzerox < midpoint
        right_lane_inds = nonzerox >= midpoint

        leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
        rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

        # Ajuste de polinômio x = f(y)
        left_fit, right_fit = None, None
        if len(lefty) > 150: left_fit = np.polyfit(lefty, leftx, self.poly_degree)
        if len(righty) > 150: right_fit = np.polyfit(righty, rightx, self.poly_degree)

        # Atualiza o histórico
        if left_fit is not None: self.left_fit_history.append(left_fit)
        if right_fit is not None: self.right_fit_history.append(right_fit)

        # Retorna o ajuste médio do histórico para suavização
        smooth_left = np.average(self.left_fit_history, axis=0) if self.left_fit_history else None
        smooth_right = np.average(self.right_fit_history, axis=0) if self.right_fit_history else None

        return smooth_left, smooth_right

    # --- Funções de Desenho e HUD (Novas) ---
    def _calculate_hud_metrics(self, left_fit, right_fit, frame_roi):
        """Calcula todas as métricas para o HUD."""
        metrics = {'deviation': 'N/A', 'curve': 'N/A', 'brightness': 'N/A'}

        if left_fit is None or right_fit is None:
            return metrics

        # 1. Desvio (Posição do Carro)
        y_eval = self.im_h - 1  # Ponto mais baixo da imagem
        left_x = np.polyval(left_fit, y_eval)
        right_x = np.polyval(right_fit, y_eval)
        lane_center = (left_x + right_x) / 2
        car_center = self.im_w / 2

        pixel_deviation = car_center - lane_center
        lane_width_pixels = right_x - left_x

        # Evita divisão por zero
        if lane_width_pixels > 0:
            deviation_percent = (pixel_deviation / (lane_width_pixels / 2)) * 100
            side = 'L' if deviation_percent > 0 else 'R'
            metrics['deviation'] = f'[{side}] {abs(deviation_percent):.1f}%'

        # 2. Curvatura da Estrada
        avg_curvature = (left_fit[0] + right_fit[0]) / 2  # O coeficiente 'a' de ax^2+bx+c
        if abs(avg_curvature) < 0.0003:
            metrics['curve'] = 'Estrada Reta'
        elif avg_curvature > 0:
            metrics['curve'] = 'Curva a Direita'
        else:
            metrics['curve'] = 'Curva a Esquerda'

        # 3. Brilho e Visibilidade
        brightness_val = np.mean(cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY))
        metrics['brightness'] = f'{brightness_val:.0f}'
        if brightness_val < 70:
            metrics['visibility'] = 'Baixa Visibilidade'
        else:
            metrics['visibility'] = 'Boa Visibilidade'

        return metrics

    def _draw_lanes_and_hud(self, frame, left_fit, right_fit):
        """Desenha as faixas, a seta de navegação e o painel do HUD."""
        # Cria uma sobreposição para desenhar
        overlay = np.zeros_like(frame)
        final_image = frame.copy()

        if left_fit is not None and right_fit is not None:
            # Pontos para desenhar as linhas das faixas
            ploty = np.linspace(self.roi_points[1][1], self.im_h - 1, 100)  # Desenha dentro da ROI
            left_fitx = np.polyval(left_fit, ploty)
            right_fitx = np.polyval(right_fit, ploty)

            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])

            # Desenha as linhas das faixas na sobreposição
            cv2.polylines(overlay, np.int_([pts_left]), isClosed=False, color=(0, 255, 0), thickness=10)
            cv2.polylines(overlay, np.int_([pts_right]), isClosed=False, color=(0, 255, 0), thickness=10)

            # Desenha a seta de navegação
            y_arrow = int(self.im_h * 0.85)
            x_arrow_base = np.polyval(left_fit, y_arrow) / 2 + np.polyval(right_fit, y_arrow) / 2

            arrow_points = np.array([
                [-15, 20], [0, -20], [15, 20], [0, 10]
            ], dtype=np.int32)
            arrow_points = arrow_points + [int(x_arrow_base), y_arrow]
            cv2.polylines(overlay, [arrow_points], isClosed=True, color=(0, 255, 150), thickness=3)

        # Mistura a sobreposição com a imagem original
        final_image = cv2.addWeighted(final_image, 1, overlay, 0.4, 0)

        # Calcula e desenha o HUD
        roi_sub_image = frame[int(self.im_h * 0.6):self.im_h, 0:self.im_w]
        metrics = self._calculate_hud_metrics(left_fit, right_fit, roi_sub_image)
        self._draw_hud_panel(final_image, metrics)

        return final_image

    def _draw_hud_panel(self, image, metrics):
        """Desenha o painel de texto do HUD."""
        # Posição do painel
        x, y, w, h = 10, 10, 320, 160
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)
        cv2.addWeighted(image[y:y + h, x:x + w], 0.3, image[y:y + h, x:x + w], 0.7, 0, image[y:y + h, x:x + w])

        # Textos do HUD
        cv2.putText(image, '<LANE DETECTION>', (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(image, f"Desvio      : {metrics['deviation']}", (x + 10, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1)
        cv2.putText(image, f"Pista       : {metrics['curve']}", (x + 10, y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1)
        cv2.putText(image, f"Brilho (ROI): {metrics['brightness']}", (x + 10, y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1)
        cv2.putText(image, f"Visibilidade: {metrics.get('visibility', 'N/A')}", (x + 10, y + 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if metrics.get('visibility') == 'Boa Visibilidade' else (0, 0, 255), 1)
        cv2.putText(image, f"FPS: {self.fps:.2f}", (self.im_w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # --- Loop Principal ---
    def process_frame(self, frame):
        """Pipeline completa de processamento para um único frame."""
        # 1. Cria a imagem binária mascarada
        binary_masked = self._create_binary_roi(frame)

        # 2. Ajusta polinômios às faixas
        left_fit, right_fit = self._fit_polynomial(binary_masked)

        # 3. Desenha as faixas e o HUD
        final_image = self._draw_lanes_and_hud(frame, left_fit, right_fit)

        # Se o modo debug da ROI estiver ativo, desenha os pontos
        if self.debug_roi_mode:
            cv2.polylines(final_image, [self.roi_points], True, (255, 0, 0), 2)
            for i, point in enumerate(self.roi_points):
                color = (0, 0, 255) if self.selected_point_index == i else (0, 255, 255)
                cv2.circle(final_image, tuple(point), 10, color, -1)

        return final_image

    def run(self):
        """Inicia o loop principal de processamento do vídeo."""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Fim do vídeo. Reiniciando...")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Cálculo de FPS
            self.frame_count += 1
            if (time.time() - self.fps_start_time) >= 1.0:
                self.fps = self.frame_count / (time.time() - self.fps_start_time)
                self.frame_count = 0
                self.fps_start_time = time.time()

            final_image = self.process_frame(frame)
            cv2.imshow(self.window_name, final_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.debug_roi_mode = not self.debug_roi_mode

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_file = "Video3.mp4"  # Coloque o caminho do seu vídeo aqui
    detector = OptimizedLaneDetector(video_file)
    detector.run()