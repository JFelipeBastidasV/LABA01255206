"""
Padding.py

Es un programa el cual utiliza la logica de la convolucion de imagenes con la intencion de darle un filtro y le agrega un padding a dichas imagenes.

Created by Jesus Felipe Bastidas Valenzuela A01255206
Created Date: 19 de Marzo de 2025
Modificated Date: 20 de Marzo de 2025

Citas bibliograficas:
GeeksforGeeks. (2021, 16 octubre). Image Filtering Using Convolution in OpenCV. GeeksforGeeks. https://www.geeksforgeeks.org/image-filtering-using-convolution-in-opencv/
"""

# Se importan las librerias necesarias
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def convolution(img, kernel):
    # Se obtienen las medidas de los lados del trian
    img_h, img_w = img.shape
    kernel_h, kernel_w = kernel.shape

    # Se calculan los márgenes para el padding
    pad_h = ((kernel_h - 1) // 2)
    pad_w = ((kernel_w - 1) // 2) 

    #Se aplica el padding en la imagen para mantener el tamaño de la imagen original
    pad_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    # Se inicializa la matriz resultante en ceros 0
    # La medida da la matrix sera el largo y ancho de la imagen original
    res = np.zeros((img_h, img_w))

    # Se itera la imagen original utilizando la convolucion
    for i in range(img_h):
        for j in range(img_w):
            # Se obtiene la region a utilizar
            reg = pad_img[i:i+kernel_h, j:j+kernel_w]

            # Se hace hace la multiplicacion de los elementos de la region y el kernel, y se van sumando
            res[i,j] = np.sum(reg * kernel)

    return res


# Cargar la imagen en color
def Proceso(NameImg):
    img = cv2.imread(NameImg, cv2.IMREAD_COLOR)
    
    # Convertir a float32 para precisión en cálculos
    img = img.astype(np.float32)
    
    # Convertir la imagen a escala de grises para aplicar convolución
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Definir los kernels mejorados
    Gaussian_blur_kernel = (1/256) * np.array([[1, 4, 6, 4, 1],
                                          [4, 16, 24, 16, 4],
                                          [6, 24, 36, 24, 6],
                                          [4, 16, 24, 16, 4],
                                          [1, 4, 6, 4, 1]])
    
    unsharpMask = (-1/256) * np.array([[1, 4, 6, 4, 1],
                                          [4, 16, 24, 16, 4],
                                          [6, 24, -476, 24, 6],
                                          [4, 16, 24, 16, 4],
                                          [1, 4, 6, 4, 1]])
    
    Repujado_kernel = np.array([[-2, -1, 0],
                                   [-1, 1, 1],
                                   [0, 1, 2]])

    borders_rel_kernel = np.array([[0, -1, 0],
                                   [-1, 1, 0],
                                   [0, 0, 0]])

    borders_kernel = np.array([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]])

    
    # Aplicar los filtros
    Gaussian_result = convolution(gray_img, Gaussian_blur_kernel)
    unsharpMask_result = convolution(gray_img, unsharpMask)
    Repujado_result = convolution(gray_img, Repujado_kernel)
    borders_rel_result = convolution(gray_img, borders_rel_kernel)
    borders_result = convolution(gray_img, borders_kernel)
    
    # Normalizar las imágenes resultantes para visualizarlas correctamente
    Gaussian_result = cv2.convertScaleAbs(Gaussian_result)
    unsharpMask_result = np.clip(unsharpMask_result, 0, 255).astype(np.uint8)
    Repujado_result = np.clip(Repujado_result, 0, 255).astype(np.uint8)
    borders_rel_result = np.clip(borders_rel_result, 0, 255).astype(np.uint8)
    borders_result = np.clip(borders_result, 0, 255).astype(np.uint8)
    
    # Agregar nombres a las imágenes
    labels = ["Original", "Guassiano", "unsharpMask", "Repujado", "borders1", "borders2"]
    
    # Combinar todas las imágenes en una sola ventana
    titles = [img.astype(np.uint8),
              cv2.cvtColor(Gaussian_result, cv2.COLOR_GRAY2BGR),
              cv2.cvtColor(unsharpMask_result, cv2.COLOR_GRAY2BGR),
              cv2.cvtColor(Repujado_result, cv2.COLOR_GRAY2BGR),
              cv2.cvtColor(borders_rel_result, cv2.COLOR_GRAY2BGR),
              cv2.cvtColor(borders_result, cv2.COLOR_GRAY2BGR)]
    
    # Definir el tamaño de la fuente, color y posición de los textos
    for i, img in enumerate(titles):
        cv2.putText(img, labels[i], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)  # Rojo (0, 0, 255)
    
    combined = np.hstack(titles)
    
    # Mostrar la imagen combinada con los nombres de los filtros
    cv2.imshow('Comparacion de Filtros', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

        
def main():
    # Solicita al usuario el nombre de la imagen a aplicar los filtros
    NameImg = str(input("Inserta el nombre de la imagen a aplicar filtro (incluyenfo la terminacion de la imagen, ejemplo: HW_caballerito.jpeg): "))

    # Se verifica que el archivo exista en la carpeta de images, el archivo, de lo contrario, pedira una imagen
    while (os.path.exists(NameImg) != True):
        print("\nArchivo no existe. Inserte el nombre de un archivo valido")
        NameImg = input("\nInserta el nombre de la imagen a aplicar filtro (incluyenfo la terminacion de la imagen, ejemplo: HW_caballerito.jpeg):")

    Proceso(NameImg)


main()

            
            
    