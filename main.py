import numpy as np
from skimage.io import imshow, imread
from skimage import color
import cv2
import os
import matplotlib.pyplot as plt


def apply_transformation(image, transformation_matrix):
    rows, columns = image.shape[:2]
    transformed_image = cv2.warpAffine(
        image, transformation_matrix, (columns*2, rows*2)) #se pone el doble del valor de filas y columnas para visualizar mejor la rotación
    return transformed_image


def cut_image(url_img: str, url_img_crop: str, x_initial: int, x_final: int, y_initial: int, y_final: int) -> None:
    try:
        # Abrir la imagen
        image = cv2.imread(url_img)

        # Obtener la imagen recortada
        image_crop = image[x_initial:x_final, y_initial:y_final]

        cv2.imwrite(ruta_img_crop, image_crop)

        print("Imagen recortada con éxito. El tamaño de la imagen es de" +
              str(image_crop.shape))
    except Exception as e:
        print("Ha ocurrido un error:", str(e))



def show_img(image, title):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rotate_image(angle, image):
    rows, columns = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D(
        (columns / 2, rows / 2), angle, 1)
    rotated_image = apply_transformation(image, rotation_matrix)

    return rotated_image


def scale_image(x, y, image):
    # Crear una matriz de transformación 2x3 para escalado

    scale_matrix = np.array([[x, 0, 0], [0, y, 0]])
    # Aplicar la transformación a la imagen
    scaled_image = apply_transformation(image, scale_matrix)

    return scaled_image

def deform_image(image, dx, dy):

    deformation_matrix = np.float32([[1, dx, 0], [dy, 1, 0]])
    deformed_image = apply_transformation(image, deformation_matrix)

    return deformed_image


def compress_image_svd(image, k):
    # Convertir la imagen a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    U, S, VT = np.linalg.svd(gray_image, full_matrices=False)

    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    VT_k = VT[:k, :]

    compressed_image = np.dot(U_k, np.dot(S_k, VT_k))

    return compressed_image





image_v2 = cv2.imread("perro.jpg")
show_img(image_v2, "imagen perro")

print(" El tamaño de la imagen 1 es de" + str(image_v2.shape))



grayscale_image = cv2.cvtColor(image_v2, cv2.COLOR_BGR2GRAY)
show_img(grayscale_image, "imagen grayscale")


# Aplicar transformaciones
rotated_image = rotate_image(45, grayscale_image)
show_img(rotated_image, "imagen rotada a 45")
rotated_image = rotate_image(-30, grayscale_image)
show_img(rotated_image, "imagen rotada a -30")


scaled_image = scale_image(0.5, 0.5, grayscale_image)
print(scaled_image)
show_img(scaled_image, "imagen escalada 1")

scaled_image = scale_image(2, 0.7, grayscale_image)
print(scaled_image)
show_img(scaled_image, "imagen escalada 2")


deformed_image = deform_image(grayscale_image,0.2,-0.1)
show_img(deformed_image, "imagen deformada 1")

deformed_image = deform_image(grayscale_image,-0.2,0.3)
show_img(deformed_image, "imagen deformada 2")

k_values = [5, 20, 50, 100]


plt.figure(figsize=(15, 5))
for i, k in enumerate(k_values, 1):
    compressed_image = compress_image_svd(image_v2, k)

    plt.subplot(1, len(k_values), i)
    plt.imshow(compressed_image, cmap='gray')
    plt.title(f'k = {k}')
    plt.axis('off')

plt.show()


