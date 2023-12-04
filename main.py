import numpy as np
from skimage.io import imshow, imread
from skimage import color
import cv2
import os
import matplotlib.pyplot as plt


def recortar_imagen_v2(ruta_img: str, ruta_img_crop: str, x_inicial: int, x_final: int, y_inicial: int, y_final: int) -> None:
    try:
        # Abrir la imagen
        image = cv2.imread(ruta_img)

        # Obtener la imagen recortada
        image_crop = image[x_inicial:x_final, y_inicial:y_final]

        cv2.imwrite(ruta_img_crop, image_crop)

        print("Imagen recortada con éxito. El tamaño de la imagen es de" +
              str(image_crop.shape))
    except Exception as e:
        print("Ha ocurrido un error:", str(e))


def image_to_matrix(image):
    numpydata = np.asarray(image)
    return numpydata


def mostrar_img(image, titulo):
    cv2.imshow(titulo, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rotate_image(image, angle):
    theta = np.radians(angle)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    rotated_image = np.dot(rotation_matrix, image.T).T
    return rotated_image


def scale_image(image, sx, sy):
    scaling_matrix = np.array([[sx, 0, 0],
                               [0, sy, 0],
                               [0, 0, 1]])  # Añadir una fila y columna para manejar imágenes en color
    scaled_image = np.dot(
        scaling_matrix, image.reshape(-1, 3).T).T.reshape(image.shape)
    return scaled_image


def deform_image(image, dx, dy):
    deformation_matrix = np.array([[1 + dx, 0, 0],
                                   [0, 1 + dy, 0],
                                   [0, 0, 1]])  # Añadir una fila y columna para manejar imágenes en color
    deformed_image = np.dot(
        deformation_matrix, image.reshape(-1, 3).T).T.reshape(image.shape)
    return deformed_image


frutilla = image_to_matrix("frutilla1.jpg")

imagen = cv2.imread("perro.jpg")
mostrar_img(imagen, "imagen perro")

print(" El tamaño de la imagen 1 es de" + str(imagen.shape))

recortar_imagen_v2("perro.jpg", "perro1.jpg",
                   0, 250, 0, 250)
imagen_recor = cv2.imread("perro1.jpg")
mostrar_img(imagen_recor, "imagen recortada pradera")

scaled_image = scale_image(imagen_recor, 1.5, 0.5)
grayscale_image = color.rgb2gray(imagen)

# Aplicar transformaciones
rotated_image = rotate_image(grayscale_image, 45)

plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(grayscale_image, cmap='gray')

plt.subplot(2, 2, 2)
plt.title("Rotated Image")
plt.imshow(rotated_image, cmap='gray')
