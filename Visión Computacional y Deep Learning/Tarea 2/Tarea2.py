import cv2
import numpy as np
import scipy.signal as sn
import scipy.misc
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import filters

def magnitud_Constante(v):
    if(v<0): return -100
    else: return 100    

def correlacion_cruzada(i, k):
    return sn.correlate2d(i,k,'same',boundary='wrap', fillvalue=0)    

def convolucion(i, k):
    return sn.convolve2d(i,k,'same',boundary='wrap', fillvalue=0)    

def rotar(a, g):    
    if g == 0:
        return a
    elif g > 0:
        return rotar(list(zip(*a[::-1])), g-90)
    else:
        return rotar(list(zip(*a)[::-1]), g+90)       

def vectores_gradientes(filename): 
    
    #leo la imagen y obtengo una en gris para aplicar el sobel
    img = Image.open(filename);
    im = np.array(Image.open(filename).convert('L'))
    # obtengo las dimensiones de las columnas y filas
    Ro = np.arange(0,im.shape[0]-1)
    Co = np.arange(0,im.shape[1]-1)

    #obtengo el filtro sobel para las lineas horizontales
    imx = np.zeros(im.shape)
    filters.sobel(im,1,imx)
    #obtengo el filtro sobel para las lineas verticales
    imy = np.zeros(im.shape)
    filters.sobel(im,0,imy)
    #obtengo las magnitudes y los angulos de todos los vectores, usando las derivadas de sobel
    mag, angle = cv2.cartToPolar(imx, imy, angleInDegrees=True)
    #obtego el promedio de las magnitudes
    alphavectors = np.mean(mag)

    #elimino vectores no relevantes que tengan una magnitud  menor al promedio
    for x in range(0, mag.shape[0]):
        for y in range(0, mag.shape[1]):
            if mag[x, y] <= alphavectors:
                   imx[x, y] = None
                   imy[x, y] = None

    x, y = np.meshgrid(Ro, Co)
    skip = (slice(None, None,4), slice(None, None, 4))
    fig, ax = plt.subplots()
    imi = ax.imshow(img)
    ax.quiver(x[skip], y[skip], imx[skip], imy[skip], angles='xy', scale_units='xy', scale=25, pivot='mid',color='g')    
    plt.show()

def contornos_sobel(filename):    
    imyo = Image.open(filename);
    imyoo = np.array(Image.open(filename))    
    
    ##sobel
    img_sobelx = cv2.Sobel(imyoo,cv2.CV_8U,1,0,None,ksize=3,scale=1,delta=0)
    img_sobely = cv2.Sobel(imyoo,cv2.CV_8U,0,1,None,ksize=3,scale=1,delta=0)
    img_total = imyoo + img_sobelx + img_sobely    
    scipy.misc.imsave("imyogray_sobelx.jpg", img_sobelx)
    scipy.misc.imsave("imyogray_sobely.jpg", img_sobely)
    scipy.misc.imsave("imyogray_sobel.jpg", img_total )    
    images = np.array([])
    images = np.append(images, imyoo)
    images = np.append(images, img_sobelx)
    images = np.append(images, img_sobely)
    images = np.append(images, img_total)    
    imyo.close()

    return images

def main():

    

    #kernel w
    w = np.array(([-10,-10, 0],
                  [-10, 0, 10], 
                  [ 0, 10, 10]),
                np.float32)

    #matriz h
    h = np.array(([10, 20, 0, 30, 30],
                  [20, 0, 10, 20,  0], 
                  [10, 0, 40, 10, 10],  
                  [20, 40, 0, 30, 10], 
                  [0, 10, 40, 20, 0]), 
                 np.float32)

    print("** Primer ejercicio ** ")
    cr = correlacion_cruzada(h,w)
    print("Correlacion cruzada:")
    print(cr)
    cv = convolucion(h,np.array(rotar(w,180),np.float32))
    print("Convolucion:")
    print(cv)
    print("** Segundo ejercicio ** ")
    # Vector q
    q = np.array(([-1, 3, -1]), np.float32)
    print("q:")
    # Traspuesta de q = qt
    qt = q.reshape((-1,1))
    print("qt:")
    #Q = qt*q
    Q= np.outer(q, qt);
    print(Q)

    crQ = convolucion(h,Q)
    print("Convolucion h * Q:")
    print(crQ)
    crq = convolucion(h, q[None,:])
    print("Convolucion h * q:")
    print(crq)
    crqt = convolucion(crq, qt)
    print("Convolucion (h * q)qt:")
    print(crqt)        

    print("** Tercer ejercicio ** ")
    contornos = contornos_sobel('yo.jpg')
    
    print("** Cuarto ejercicio ** ")
    vectores_gradientes("panda.jpg")
           
if __name__ == '__main__':
    main()
