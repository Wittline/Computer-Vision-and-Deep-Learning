import numpy as np
import cv2
from matplotlib import pyplot as plt

ImagenContornos= None

def LeerImagen(l):
    Imagen = cv2.imread(l)
    return Imagen, Imagen.copy()

def PreprocessImage(i):
    return cv2.GaussianBlur(cv2.equalizeHist(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)), (5,5),0)

def DetectarContornos(i, min, max):
    ImagenBordes = cv2.Canny(i, min, max)
    contornos,_ = cv2.findContours(ImagenBordes.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return ImagenBordes, contornos

def DibujarContornos(c):
    global ImagenContornos
    i = 0
    for cn in c:    
        cv2.drawContours(ImagenContornos, cn, -1, (57,255,20),2)    
        x,y,w,h = cv2.boundingRect(cn)  
        cv2.putText(ImagenContornos, str(i+1), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (57, 255, 20), lineType=cv2.LINE_AA)     
        i=i+1
    return i

def Plotear(t,im,s, n):
    for i in range(n):
        plt.subplot(2, 2, i+1)
        plt.title(t[i])
        if(s[i]):
            plt.imshow(im[i], cmap='gray',vmin=0,vmax=255)        
        else:
            plt.imshow(cv2.cvtColor(im[i], cv2.COLOR_BGR2RGB))        
        plt.xticks([])
        plt.yticks([])
    plt.show()  

def ContarMonedas(location):      
    global ImagenContornos
    Imagen, ImagenContornos = LeerImagen(location)
    ImagenGauss = PreprocessImage(Imagen)
    ImagenBordes,contornos= DetectarContornos(ImagenGauss, 50, 300)
    i = DibujarContornos(contornos)
    Titulos = ['Original', 'Gaussiana', 'Bordes de la imagen', 'Imagen Final: ' + str(i) + ' monedas']
    Imagenes = [Imagen, ImagenGauss, ImagenBordes, ImagenContornos]
    Types = [0, 1, 0, 0]
    Plotear(Titulos, Imagenes, Types, 4)


def main():    
    ContarMonedas('monedas.png')
if __name__ == '__main__':
    main()
