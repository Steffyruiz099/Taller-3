import numpy as np
from time import time
import math
import cv2

def noise(noise_typ, image):

    if noise_typ == "gauss":                                                                                                    #Se agrega ruido gauss a la imagen original en grises
        row, col = image.shape
        mean = 0
        var = 0.002
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        lena_gauss_noisy = image + gauss
        cv2.imshow('lena_gauss_noisy', lena_gauss_noisy)                                                                        # se muestra la imagen con ruido gauss
        cv2.waitKey(0)

        N = 7                                                                                                                   # se define N = 7
        tiempo_gauss_gauss_inicial = time()                                                                                     # se toma el tiempo antes de hacer el filtro gauss
        image_gauss_lp_gauss = cv2.GaussianBlur((255 * lena_gauss_noisy).astype(np.uint8), (N, N), 1.5, 1.5)                    # se realiza el filtro gauss
        tiempo_gauss_gauss_final = time()                                                                                       # se toma el tiempo despues de hacer el filtro gauss
        cv2.imshow('lena_gauss_noisy_filtro_gauss', image_gauss_lp_gauss)                                                       # se muestra la imagen con ruido gauss luego de ser filtrada por un filtro gauss
        cv2.waitKey(0)
        tiempo_gauss_gauss = tiempo_gauss_gauss_final - tiempo_gauss_gauss_inicial                                              # se calcula el tiempo de ejecución del filtro gauss
        print(tiempo_gauss_gauss)                                                                                               # se imprime el tiempo de ejecución del filtro gauss
        errorestimate_gauss_gauss = abs(lena_gauss_noisy - image_gauss_lp_gauss)                                                # se calcula el error estimado entre la imagen con ruido y la filtrada
        cv2.imshow('errorestimado_gauss_gauss', errorestimate_gauss_gauss.astype(np.float)/255)                                 # se muestra la imagen del error estimado
        cv2.waitKey(0)
        errorcuadratico_gauss_gauss = math.sqrt(np.square(np.subtract(lena_gauss_noisy, image_gauss_lp_gauss)).mean())          # se calcula el sqrt del error cuadratico entre la imagen con ruido y la filtrada
        print("error cuadrático gauss_gauss", errorcuadratico_gauss_gauss)                                                      # se imprime el valor del sqrt del error cuadratico

        tiempo_gauss_median_inicial = time()                                                                                    # se toma el tiempo antes de hacer el filtro median
        image_median_gauss = cv2.medianBlur((255 * lena_gauss_noisy).astype(np.uint8), 7)                                       # se realiza el filtro median
        tiempo_gauss_median_final = time()                                                                                      # se toma el tiempo despues de hacer el filtro median
        cv2.imshow('lena_gauss_noisy_filtro_median', image_median_gauss)                                                        # se muestra la imagen con ruido gauss luego de ser filtrada por un filtro median
        cv2.waitKey(0)
        tiempo_gauss_median = tiempo_gauss_median_final - tiempo_gauss_median_inicial                                           # se calcula el tiempo de ejecución del filtro median
        print(tiempo_gauss_median)                                                                                              # se imprime el tiempo de ejecución del filtro median
        errorestimate_gauss_median = abs(lena_gauss_noisy - image_median_gauss)                                                 # se calcula el error estimado entre la imagen con ruido y la filtrada
        cv2.imshow('errorestimado_gauss_median', errorestimate_gauss_median.astype(np.float)/255)                               # se muestra la imagen del error estimado
        cv2.waitKey(0)
        errorcuadratico_gauss_median = math.sqrt(np.square(np.subtract(lena_gauss_noisy, image_median_gauss)).mean())           # se calcula el sqrt del error cuadratico entre la imagen con ruido y la filtrada
        print("error cuadrático gauss_median", errorcuadratico_gauss_median)                                                    # se imprime el valor del sqrt del error cuadratico

        tiempo_gauss_bilateral_inicial = time()                                                                                 # se toma el tiempo antes de hacer el filtro bilateral
        image_bilateral_gauss = cv2.bilateralFilter((255 * lena_gauss_noisy).astype(np.uint8), 15, 25, 25)                      # se realiza el filtro bilateral
        tiempo_gauss_bilateral_final = time()                                                                                   # se toma el tiempo despues de hacer el filtro bilateral
        cv2.imshow('lena_gauss_noisy_filtro_bilateral', image_bilateral_gauss)                                                  # se muestra la imagen con ruido gauss luego de ser filtrada por un filtro bilateral
        cv2.waitKey(0)
        tiempo_gauss_bilateral = tiempo_gauss_bilateral_final - tiempo_gauss_bilateral_inicial                                  # se calcula el tiempo de ejecución del filtro bilateral
        print(tiempo_gauss_bilateral)                                                                                           # se imprime el tiempo de ejecución del filtro bilateral
        errorestimate_gauss_bilateral = abs(lena_gauss_noisy - image_bilateral_gauss)                                           # se calcula el error estimado entre la imagen con ruido y la filtrada
        cv2.imshow('errorestimado_gauss_bilateral', errorestimate_gauss_bilateral.astype(np.float)/255)                         # se muestra la imagen del error estimado
        cv2.waitKey(0)
        errorcuadratico_gauss_bilateral = math.sqrt(np.square(np.subtract(lena_gauss_noisy, image_bilateral_gauss)).mean())     # se calcula el sqrt del error cuadratico entre la imagen con ruido y la filtrada
        print("error cuadrático gauss_bilateral", errorcuadratico_gauss_bilateral)                                              # se imprime el valor del sqrt del error cuadratico

        tiempo_gauss_nlm_inicial = time()                                                                                       # se toma el tiempo antes de hacer el filtro nlm
        image_nlm_gauss = cv2.fastNlMeansDenoising((255 * lena_gauss_noisy).astype(np.uint8), 5, 15, 25)                        # se realiza el filtro nlm
        tiempo_gauss_nlm_final = time()                                                                                         # se toma el tiempo despues de hacer el filtro nlm
        cv2.imshow('lena_gauss_noisy_filtro_nml', image_nlm_gauss)                                                              # se muestra la imagen con ruido gauss luego de ser filtrada por un filtro nlm
        cv2.waitKey(0)
        tiempo_gauss_nlm = tiempo_gauss_nlm_final - tiempo_gauss_nlm_inicial                                                    # se calcula el tiempo de ejecución del filtro nlm
        print(tiempo_gauss_nlm)                                                                                                 # se imprime el tiempo de ejecución del filtro nlm
        errorestimate_gauss_nlm = abs(lena_gauss_noisy - image_nlm_gauss)                                                       # se calcula el error estimado entre la imagen con ruido y la filtrada
        cv2.imshow('errorestimado_gauss_nlm', errorestimate_gauss_nlm.astype(np.float)/255)                                     # se muestra la imagen del error estimado
        cv2.waitKey(0)
        errorcuadratico_gauss_nlm = math.sqrt(np.square(np.subtract(lena_gauss_noisy, image_nlm_gauss)).mean())                 # se calcula el sqrt del error cuadratico entre la imagen con ruido y la filtrada
        print("error cuadrático gauss_nlm", errorcuadratico_gauss_nlm)                                                          # se imprime el valor del sqrt del error cuadratico

        if tiempo_gauss_gauss < (tiempo_gauss_median and tiempo_gauss_bilateral and tiempo_gauss_nlm):                          # se pregunta si el tiempo de ejecucion del filtro gauss fue mayor a los otros
            timeprint = tiempo_gauss_gauss * 1000                                                                               # se pasa a milisegundos el tiempo de ejecucion del filtro gauss
            print("ruido gauss, filtro gauss", timeprint,"ms")                                                                  # se imprime el tiempo de ejecucion en ms
            timegaussmedian = (tiempo_gauss_median * 100)/tiempo_gauss_gauss                                                    # se encuentra el porcentaje del tiempo de ejecucion del filtro median con respecto al filtro gauss
            print("ruido gauss, filtro median", timegaussmedian, "%")                                                           # se imprime el porcentaje
            timegaussbilateral = (tiempo_gauss_bilateral * 100) / tiempo_gauss_gauss                                            # se encuentra el porcentaje del tiempo de ejecucion del filtro bilateral con respecto al filtro gauss
            print("ruido gauss, filtro bilateral", timegaussbilateral, "%")                                                     # se imprime el porcentaje
            timegaussnlm = (tiempo_gauss_nlm * 100) / tiempo_gauss_gauss                                                        # se encuentra el porcentaje del tiempo de ejecucion del filtro nlm con respecto al filtro gauss
            print("ruido gauss, filtro nlm", timegaussnlm, "%")                                                                 # se imprime el porcentaje

        elif tiempo_gauss_median < (tiempo_gauss_gauss and tiempo_gauss_bilateral and tiempo_gauss_nlm):                        # se pregunta si el tiempo de ejecucion del filtro median fue mayor a los otros
            timeprint = tiempo_gauss_median * 1000                                                                              # se pasa a milisegundos el tiempo de ejecucion del filtro median
            print("ruido gauss, filtro median", timeprint,"ms")                                                                 # se imprime el tiempo de ejecucion en ms
            timegaussgauss = (tiempo_gauss_gauss * 100)/tiempo_gauss_median                                                     # se encuentra el porcentaje del tiempo de ejecucion del filtro gauss con respecto al filtro median
            print("ruido gauss, filtro gauss", timegaussgauss, "%")                                                             # se imprime el porcentaje
            timegaussbilateral = (tiempo_gauss_bilateral * 100) / tiempo_gauss_median                                           # se encuentra el porcentaje del tiempo de ejecucion del filtro bilateral con respecto al filtro median
            print("ruido gauss, filtro bilateral", timegaussbilateral, "%")                                                     # se imprime el porcentaje
            timegaussnlm = (tiempo_gauss_nlm * 100) / tiempo_gauss_median                                                       # se encuentra el porcentaje del tiempo de ejecucion del filtro nlm con respecto al filtro median
            print("ruido gauss, filtro nlm", timegaussnlm, "%")                                                                 # se imprime el porcentaje

        elif tiempo_gauss_bilateral < (tiempo_gauss_median and tiempo_gauss_gauss and tiempo_gauss_nlm):                        # se pregunta si el tiempo de ejecucion del filtro bilateral fue mayor a los otros
            timeprint = tiempo_gauss_bilateral * 1000                                                                           # se pasa a milisegundos el tiempo de ejecucion del filtro bilateral
            print("ruido gauss, filtro bilateral", timeprint,"ms")                                                              # se imprime el tiempo de ejecucion en ms
            timegaussmedian = (tiempo_gauss_median * 100)/tiempo_gauss_bilateral                                                # se encuentra el porcentaje del tiempo de ejecucion del filtro median con respecto al filtro bilateral
            print("ruido gauss, filtro median", timegaussmedian, "%")                                                           # se imprime el porcentaje
            timegaussgauss = (tiempo_gauss_gauss * 100) / tiempo_gauss_bilateral                                                # se encuentra el porcentaje del tiempo de ejecucion del filtro gauss con respecto al filtro bilateral
            print("ruido gauss, filtro gauss", timegaussgauss, "%")                                                             # se imprime el porcentaje
            timegaussnlm = (tiempo_gauss_nlm * 100) / tiempo_gauss_bilateral                                                    # se encuentra el porcentaje del tiempo de ejecucion del filtro nlm con respecto al filtro bilateral
            print("ruido gauss, filtro nlm", timegaussnlm, "%")                                                                 # se imprime el porcentaje

        elif tiempo_gauss_nlm < (tiempo_gauss_median and tiempo_gauss_bilateral and tiempo_gauss_gauss):                        # se pregunta si el tiempo de ejecucion del filtro nlm fue mayor a los otros
            timeprint = tiempo_gauss_nlm * 1000                                                                                 # se pasa a milisegundos el tiempo de ejecucion del filtro nlm
            print("ruido gauss, filtro nlm", timeprint,"ms")                                                                    # se imprime el tiempo de ejecucion en ms
            timegaussmedian = (tiempo_gauss_median * 100)/tiempo_gauss_nlm                                                      # se encuentra el porcentaje del tiempo de ejecucion del filtro median con respecto al filtro nlm
            print("ruido gauss, filtro median", timegaussmedian,"%")                                                            # se imprime el porcentaje
            timegaussbilateral = (tiempo_gauss_bilateral * 100) / tiempo_gauss_nlm                                              # se encuentra el porcentaje del tiempo de ejecucion del filtro bilateral con respecto al filtro nlm
            print("ruido gauss, filtro bilateral", timegaussbilateral, "%")                                                     # se imprime el porcentaje
            timegaussgauss = (tiempo_gauss_gauss * 100) / tiempo_gauss_nlm                                                      # se encuentra el porcentaje del tiempo de ejecucion del filtro gauss con respecto al filtro nlm
            print("ruido gauss, filtro gauss", timegaussgauss, "%")                                                             # se imprime el porcentaje

    elif noise_typ == "s&p":                                                                                                    # Se agrega ruido s&p a la imagen original en grises
       # row, col = image.shape
        s_vs_p = 0.5
        amount = 0.01
        lena_sip_noisy = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        lena_sip_noisy[tuple(coords)] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        lena_sip_noisy[tuple(coords)] = 0
        cv2.imshow('lena_s&p_noisy', lena_sip_noisy)                                                                            # se muestra la imagen con ruido s&p
        cv2.waitKey(0)

        N = 7                                                                                                                   # se define N = 7
        tiempo_sip_gauss_inicial = time()                                                                                       # se toma el tiempo antes de hacer el filtro gauss
        image_gauss_lp_sip = cv2.GaussianBlur((255 * lena_sip_noisy).astype(np.uint8), (N, N), 1.5, 1.5)                        # se realiza el filtro gauss
        tiempo_sip_gauss_final = time()                                                                                         # se toma el tiempo despues de hacer el filtro gauss
        cv2.imshow('lena_s&p_noisy_filtro_gauss', image_gauss_lp_sip)                                                           # se muestra la imagen con ruido s&p luego de ser filtrada por un filtro gauss
        cv2.waitKey(0)
        tiempo_sip_gauss = tiempo_sip_gauss_final - tiempo_sip_gauss_inicial                                                    # se calcula el tiempo de ejecución del filtro gauss
        print(tiempo_sip_gauss)                                                                                                 # se imprime el tiempo de ejecución del filtro gauss
        errorestimate_sip_gauss = abs(lena_sip_noisy - image_gauss_lp_sip)                                                      # se calcula el error estimado entre la imagen con ruido y la filtrada
        cv2.imshow('errorestimado_sip_gauss',errorestimate_sip_gauss.astype(np.float) / 255)                                    # se muestra la imagen del error estimado
        cv2.waitKey(0)
        errorcuadratico_sip_gauss = math.sqrt(np.square(np.subtract(lena_sip_noisy, image_gauss_lp_sip)).mean())                # se calcula el sqrt del error cuadratico entre la imagen con ruido y la filtrada
        print("error cuadrático s&p_gauss", errorcuadratico_sip_gauss)                                                          # se imprime el valor del sqrt del error cuadratico

        tiempo_sip_median_inicial = time()                                                                                      # se toma el tiempo antes de hacer el filtro median
        image_median_sip = cv2.medianBlur((255 * lena_sip_noisy).astype(np.uint8), 7)                                           # se realiza el filtro median
        tiempo_sip_median_final = time()                                                                                        # se toma el tiempo despues de hacer el filtro median
        cv2.imshow('lena_s&p_noisy_filtro_median', image_median_sip)                                                            # se muestra la imagen con ruido s&p luego de ser filtrada por un filtro median
        cv2.waitKey(0)
        tiempo_sip_median = tiempo_sip_median_final - tiempo_sip_median_inicial                                                 # se calcula el tiempo de ejecución del filtro median
        print(tiempo_sip_median)                                                                                                # se imprime el tiempo de ejecución del filtro median
        errorestimate_sip_median = abs(lena_sip_noisy - image_median_sip)                                                       # se calcula el error estimado entre la imagen con ruido y la filtrada
        cv2.imshow('errorestimado_sip_median', errorestimate_sip_median.astype(np.float) / 255)                                 # se muestra la imagen del error estimado
        cv2.waitKey(0)
        errorcuadratico_sip_median = math.sqrt(np.square(np.subtract(lena_sip_noisy, image_median_sip)).mean())                 # se calcula el sqrt del error cuadratico entre la imagen con ruido y la filtrada
        print("error cuadrático s&p_median", errorcuadratico_sip_median)                                                                                       # se imprime el valor del sqrt del error cuadratico

        tiempo_sip_bilateral_inicial = time()                                                                                   # se toma el tiempo antes de hacer el filtro bilateral
        image_bilateral_sip = cv2.bilateralFilter((255 * lena_sip_noisy).astype(np.uint8), 15, 25, 25)                          # se realiza el filtro bilateral
        tiempo_sip_bilateral_final = time()                                                                                     # se toma el tiempo despues de hacer el filtro bilateral
        cv2.imshow('lena_s&p_noisy_filtro_bilateral', image_bilateral_sip)                                                      # se muestra la imagen con ruido s&p luego de ser filtrada por un filtro bilateral
        cv2.waitKey(0)
        tiempo_sip_bilateral = tiempo_sip_bilateral_final - tiempo_sip_bilateral_inicial                                        # se calcula el tiempo de ejecución del filtro bilateral
        print(tiempo_sip_bilateral)                                                                                             # se imprime el tiempo de ejecución del filtro bilateral
        errorestimate_sip_bilateral = abs(lena_sip_noisy - image_bilateral_sip)                                                 # se calcula el error estimado entre la imagen con ruido y la filtrada
        cv2.imshow('errorestimado_sip_bilateral', errorestimate_sip_bilateral.astype(np.float) / 255)                           # se muestra la imagen del error estimado
        cv2.waitKey(0)
        errorcuadratico_sip_bilateral = math.sqrt(np.square(np.subtract(lena_sip_noisy, image_bilateral_sip)).mean())           # se calcula el sqrt del error cuadratico entre la imagen con ruido y la filtrada
        print("error cuadrático s&p_bilateral", errorcuadratico_sip_bilateral)                                                                                    # se imprime el valor del sqrt del error cuadratico

        tiempo_sip_nlm_inicial = time()                                                                                         # se toma el tiempo antes de hacer el filtro nlm
        image_nlm_sip = cv2.fastNlMeansDenoising((255 * lena_sip_noisy).astype(np.uint8), 5, 15, 25)                            # se realiza el filtro nlm
        tiempo_sip_nlm_final = time()                                                                                           # se toma el tiempo despues de hacer el filtro nlm
        cv2.imshow('lena_s&p_noisy_filtro_nml', image_nlm_sip)                                                                  # se muestra la imagen con ruido s&p luego de ser filtrada por un filtro nlm
        cv2.waitKey(0)
        tiempo_sip_nlm = tiempo_sip_nlm_final - tiempo_sip_nlm_inicial                                                          # se calcula el tiempo de ejecución del filtro nlm
        print(tiempo_sip_nlm)                                                                                                   # se imprime el tiempo de ejecución del filtro nlm
        errorestimate_sip_nlm = abs(lena_sip_noisy - image_nlm_sip)                                                             # se calcula el error estimado entre la imagen con ruido y la filtrada
        cv2.imshow('errorestimado_sip_nlm', errorestimate_sip_nlm.astype(np.float) / 255)                                       # se muestra la imagen del error estimado
        cv2.waitKey(0)
        errorcuadratico_sip_nlm = math.sqrt(np.square(np.subtract(lena_sip_noisy, image_nlm_sip)).mean())                       # se calcula el sqrt del error cuadratico entre la imagen con ruido y la filtrada
        print("error cuadrático s&p_nlm", errorcuadratico_sip_nlm)                                                                                          # se imprime el valor del sqrt del error cuadratico

       if tiempo_sip_gauss < (tiempo_sip_median and tiempo_sip_bilateral and tiempo_sip_nlm):                                  # se pregunta si el tiempo de ejecucion del filtro gauss fue mayor a los otros
            timeprint = tiempo_sip_gauss * 1000                                                                                 # se pasa a milisegundos el tiempo de ejecucion del filtro gauss
            print("ruido s&p, filtro gauss", timeprint,"ms")                                                                    # se imprime el tiempo de ejecucion en ms
            timesipmedian = (tiempo_sip_median * 100)/tiempo_sip_gauss                                                          # se encuentra el porcentaje del tiempo de ejecucion del filtro median con respecto al filtro gauss
            print("ruido s&p, filtro median", timesipmedian, "%")                                                               # se imprime el porcentaje
            timesipbilateral = (tiempo_sip_bilateral * 100) / tiempo_sip_gauss                                                  # se encuentra el porcentaje del tiempo de ejecucion del filtro bilateral con respecto al filtro gauss
            print("ruido s&p, filtro bilateral", timesipbilateral, "%")                                                         # se imprime el porcentaje
            timesipnlm = (tiempo_sip_nlm * 100) / tiempo_sip_gauss                                                              # se encuentra el porcentaje del tiempo de ejecucion del filtro nlm con respecto al filtro gauss
            print("ruido s&p, filtro nlm", timesipnlm, "%")                                                                     # se imprime el porcentaje

        elif tiempo_sip_median < (tiempo_sip_gauss and tiempo_sip_bilateral and tiempo_sip_nlm):                                # se pregunta si el tiempo de ejecucion del filtro median fue mayor a los otros
            timeprint = tiempo_sip_median * 1000                                                                                # se pasa a milisegundos el tiempo de ejecucion del filtro median
            print("ruido s&p, filtro median", timeprint,"ms")                                                                   # se imprime el tiempo de ejecucion en ms
            timesipgauss = (tiempo_sip_gauss * 100)/tiempo_sip_median                                                           # se encuentra el porcentaje del tiempo de ejecucion del filtro gauss con respecto al filtro median
            print("ruido s&p, filtro gauss", timesipgauss, "%")                                                                 # se imprime el porcentaje
            timesipbilateral = (tiempo_sip_bilateral * 100) / tiempo_sip_median                                                 # se encuentra el porcentaje del tiempo de ejecucion del filtro bilateral con respecto al filtro median
            print("ruido s&p, filtro bilateral", timesipbilateral, "%")                                                         # se imprime el porcentaje
            timesipnlm = (tiempo_sip_nlm * 100) / tiempo_sip_median                                                             # se encuentra el porcentaje del tiempo de ejecucion del filtro nlm con respecto al filtro median
            print("ruido s&p, filtro nlm", timesipnlm, "%")                                                                     # se imprime el porcentaje

        elif tiempo_sip_bilateral < (tiempo_sip_median and tiempo_sip_gauss and tiempo_sip_nlm):                                # se pregunta si el tiempo de ejecucion del filtro bilateral fue mayor a los otros
            timeprint = tiempo_sip_bilateral * 1000                                                                             # se pasa a milisegundos el tiempo de ejecucion del filtro bilateral
            print("ruido s&p, filtro bilateral", timeprint,"ms")                                                                # se imprime el tiempo de ejecucion en ms
            timesipmedian = (tiempo_sip_median * 100)/tiempo_sip_bilateral                                                      # se encuentra el porcentaje del tiempo de ejecucion del filtro median con respecto al filtro bilateral
            print("ruido s&p, filtro median", timesipmedian, "%")                                                               # se imprime el porcentaje
            timesipgauss = (tiempo_sip_gauss * 100) / tiempo_sip_bilateral                                                      # se encuentra el porcentaje del tiempo de ejecucion del filtro gauss con respecto al filtro bilateral
            print("ruido s&p, filtro gauss", timesipgauss, "%")                                                                 # se imprime el porcentaje
            timesipnlm = (tiempo_sip_nlm * 100) / tiempo_sip_bilateral                                                          # se encuentra el porcentaje del tiempo de ejecucion del filtro nlm con respecto al filtro bilateral
            print("ruido s&p, filtro nlm", timesipnlm, "%")                                                                     # se imprime el porcentaje

        elif tiempo_sip_nlm < (tiempo_sip_median and tiempo_sip_bilateral and tiempo_sip_gauss):                                # se pregunta si el tiempo de ejecucion del filtro nlm fue mayor a los otros
            timeprint = tiempo_sip_nlm * 1000                                                                                   # se pasa a milisegundos el tiempo de ejecucion del filtro nlm
            print("ruido s&p, filtro nlm", timeprint,"ms")                                                                      # se imprime el tiempo de ejecucion en ms
            timesipmedian = (tiempo_sip_median * 100)/tiempo_sip_nlm                                                            # se encuentra el porcentaje del tiempo de ejecucion del filtro median con respecto al filtro nlm
            print("ruido s&p, filtro median", timesipmedian,"%")                                                                # se imprime el porcentaje
            timesipbilateral = (tiempo_sip_bilateral * 100) / tiempo_sip_nlm                                                    # se encuentra el porcentaje del tiempo de ejecucion del filtro bilateral con respecto al filtro nlm
            print("ruido s&p, filtro bilateral", timesipbilateral, "%")                                                         # se imprime el porcentaje
            timesipgauss = (tiempo_sip_gauss * 100) / tiempo_sip_nlm                                                            # se encuentra el porcentaje del tiempo de ejecucion del filtro gauss con respecto al filtro nlm
            print("ruido s&p, filtro gauss", timesipgauss, "%")                                                                 # se imprime el porcentaje

