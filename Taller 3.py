from noise import *
import os
import cv2
if __name__ == '__main__':
    path = 'C:/Users/Steffany/Documents/Javeriana/Semestre 9/Procesamiento de Imagenes/imagenes'
    image_name = 'lena.png'
    path_file = os.path.join(path, image_name)              # se crea la ruta de la imagen
    image = cv2.imread(path_file)
    imagegris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('image', imagegris)                          # se muestra la imagen lena en grises
    cv2.waitKey(0)
    ns = noise("s&p",imagegris.astype(np.float)/255)        # se llama la clase noise con la imagen lena en grises y el ruido s&n
    ns = noise("gauss",imagegris.astype(np.float)/255)      # se llama la clase noise con la imagen lena en grises y el ruido gauss
