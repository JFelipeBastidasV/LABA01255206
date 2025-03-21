"""
conv.py

Es un programa el cual utiliza la logica de la convolucion de imagenes con la intencion de darle un filtro.

Created by Jesus Felipe Bastidas Valenzuela A01255206
Created Date: 19 de Marzo de 2025
"""

# Se importan las librerias necesarias
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Se crea la funcion con la que se usa la image
def useImg(NameImg):
    img = cv2.imread(NameImg)

    id_kernel = np.array([[0,0,0],
                          [0,1,0],
                          [0,0,0]])

    
    flt_img = cv2.filter2D(src=img, ddepth=-1, kernel=id_kernel) 

    cv2.imshow('Identity: {NameImg}', flt_img)

    cv2.waitKey(0) 

    cv2.destroyAllWindows()

    return output


def main():
    # Solicita al usuario el nombre de la imagen a aplicar los filtros
    NameImg = str(input("Inserta el nombre de la imagen a aplicar filtro (incluyenfo la terminacion de la imagen, ejemplo: HW_caballerito.png): "))

    # Se verifica que el archivo exista en la carpeta de images, el archivo, de lo contrario, pedira una imagen
    while (os.path.exists(NameImg) != True):
    print("\nArchivo no existe. Inserte el nombre de un archivo valido")
    file_path = input("\nInserte el nombre del archivo a analizar (tiene que agregar la terminacion del documento, ejemplo: expresiones.txt) :")



main()
        