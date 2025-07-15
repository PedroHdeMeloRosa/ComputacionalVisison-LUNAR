import cv2
import numpy as np

# ---------------------------------------------------
# 1. Carregamento das imagens
# ---------------------------------------------------

# Imagem de fundo 1
img1 = cv2.imread("3D-Matplotlib (1).png")  # Deve ter as mesmas dimensões que img2

# Imagem de fundo 2
img2 = cv2.imread("mainsvmimage (1).png")

# Logo com fundo branco (que será removido)
img3 = cv2.imread("mainlogo.png")  # Usado como sobreposição sobre a imagem somada

# ---------------------------------------------------
# 2. Soma das imagens de fundo
# ---------------------------------------------------

# Soma de img1 e img2: cada pixel é somado canal por canal (B, G, R)
# Se a soma ultrapassar 255 (valor máximo para um canal de cor), ela é truncada em 255
# Exemplo: (200, 100, 100) + (100, 200, 100) = (255, 255, 200)
add1 = cv2.add(img1, img2)

# ---------------------------------------------------
# 3. Preparar a inserção do logo (img3) sobre a imagem somada
# ---------------------------------------------------

# Captura altura (rows), largura (cols) e número de canais (3 para BGR) do logo
rows, cols, channels = img3.shape

# Define a região de interesse (ROI) na imagem somada, com o mesmo tamanho do logo
# Aqui, o logo será colocado no canto superior esquerdo da imagem
roi = add1[0:rows, 0:cols]

# Converte o logo para tons de cinza
# Isso é necessário para aplicar a limiarização (threshold), já que ela opera com imagens de 1 canal
img3_gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

# Cria uma máscara binária usando limiar (threshold):
# Pixels com valor acima de 220 (quase brancos) se tornam 0 (preto)
# Pixels com valor abaixo de 220 se tornam 255 (branco)
# Resultado: fundo branco vira preto (será removido) e o logo vira branco (preservado)
ret, mask = cv2.threshold(img3_gray, 220, 255, cv2.THRESH_BINARY_INV)

# Inverte a máscara: agora o fundo é branco (255) e o logo é preto (0)
# Isso será usado para preservar o fundo da imagem base onde o logo será colado
mask_inv = cv2.bitwise_not(mask)

# Aplica a máscara invertida à região de interesse:
# Remove do fundo (add1) a parte onde o logo será inserido
img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

# Aplica a máscara original ao logo:
# Preserva apenas a parte útil (colorida) do logo, eliminando o fundo branco
img3_fg = cv2.bitwise_and(img3, img3, mask=mask)

# Soma as duas partes: fundo limpo + logo colorido recortado
dst = cv2.add(img1_bg, img3_fg)

# Substitui a região de interesse original pelo novo conteúdo (logo sobre fundo)
add1[0:rows, 0:cols] = dst

# ---------------------------------------------------
# 4. Exibição das várias imagens com imshow
# ---------------------------------------------------

# Mostra as imagens individualmente, para comparar cada etapa do processo

cv2.imshow("Imagem 1: 3D-Matplotlib", img1)         # Primeira imagem de fundo
cv2.imshow("Imagem 2: mainsvmimage", img2)          # Segunda imagem de fundo
cv2.imshow("Imagem Somada (add1)", add1)            # Resultado da soma entre img1 e img2, já com o logo inserido
cv2.imshow("Logo Original (img3)", img3)            # O logo original (com fundo branco)
cv2.imshow("Máscara (mask): logo branco, fundo preto", mask)            # A máscara binária do logo (fundo removido)
cv2.imshow("Máscara Invertida (mask_inv): fundo branco, logo preto", mask_inv)  # Máscara para limpar a imagem base
cv2.imshow("Parte do Fundo (img1_bg)", img1_bg)     # Região de fundo 'limpa' na imagem base
cv2.imshow("Parte do Logo (img3_fg)", img3_fg)      # Logo recortado com base na máscara

# Espera o pressionamento de uma tecla
cv2.waitKey(0)

# Fecha todas as janelas abertas
cv2.destroyAllWindows()
