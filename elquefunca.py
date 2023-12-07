import numpy as np
from skimage.io import imshow, imread
from skimage import color
import cv2
import os
import matplotlib.pyplot as plt


def aplicar_transformacion(imagen, matriz_transformacion):
    filas, columnas = imagen.shape[:2]
    imagen_transformada = cv2.warpAffine(
        imagen, matriz_transformacion, (columnas, filas))
    return imagen_transformada


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


def rotate_image(angulo, matriz):
    filas, columnas = matriz.shape[:2]
    matriz_rotacion = cv2.getRotationMatrix2D(
        (columnas / 2, filas / 2), angulo, 1)
    # No se si es esta matriz que quiere que imprima pero esto rota la imagen
    print(matriz_rotacion)
    imagen_rotada = aplicar_transformacion(imagen, matriz_rotacion)

    return imagen_rotada


def scale_image(x, y, imagen):
    # Crear una matriz de transformación 2x3 para escalado
    #matriz_escalado = np.array(
     #   [[x, 0, 0], [0, y, 0], [0, 0, 1]], dtype=np.float32)

    matriz_escalado = np.array([[x, 0, 0], [0, y, 0]])
    # Aplicar la transformación a la imagen
    imagen_escalada = aplicar_transformacion(imagen, matriz_escalado)

    return imagen_escalada

def deform_image(image, dx, dy):

    deformation_matrix = np.float32([[1, dx, 0], [dy, 1, 0]])
    deformed_image = aplicar_transformacion(image, deformation_matrix)

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





imagen = cv2.imread("perro.jpg")
mostrar_img(imagen, "imagen perro")

print(" El tamaño de la imagen 1 es de" + str(imagen.shape))

recortar_imagen_v2("perro.jpg", "perro1.jpg",
                   0, 250, 0, 250)
imagen_recor = cv2.imread("perro1.jpg")
mostrar_img(imagen_recor, "imagen recortada pradera")

print(image_to_matrix(imagen_recor))


grayscale_image = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
mostrar_img(grayscale_image, "imagen grayscale")

print(image_to_matrix(grayscale_image))

# Aplicar transformaciones
rotated_image = rotate_image(45, grayscale_image)
mostrar_img(rotated_image, "imagen rotada")


scaled_image = scale_image(1, 0.5, grayscale_image)
mostrar_img(scaled_image, "imagen escalada")

deformed_image = deform_image(grayscale_image,0.2,-0.1)
mostrar_img(deformed_image, "imagen defo")

k_values = [5, 20, 50, 100]


plt.figure(figsize=(15, 5))
for i, k in enumerate(k_values, 1):
    compressed_image = compress_image_svd(imagen, k)

    plt.subplot(1, len(k_values), i)
    plt.imshow(compressed_image, cmap='gray')
    plt.title(f'k = {k}')
    plt.axis('off')

plt.show()
