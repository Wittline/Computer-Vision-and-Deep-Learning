import math
import numpy as np
import cv2
from matplotlib import pyplot as plt

EE = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

def LeerImagen(l):    
    return cv2.imread(l,0)

def Plotear(t,im,s, n):
    for i in range(n):
        plt.subplot(1, 3, i+1)
        plt.title(t[i])
        if(s[i]):
            plt.imshow(im[i], cmap='gray',vmin=0,vmax=255)        
        else:
            plt.imshow(cv2.cvtColor(im[i], cv2.COLOR_BGR2RGB))        
        plt.xticks([])
        plt.yticks([])
    plt.show()  


def Binarize(i,inverse):
    t = 0
    if inverse:
        t= 1
    return cv2.threshold(i,200, 255, t)
    
def ObtenerEsqueleto(imagename, inverse):      
    global EE
    fin = False

    Imagen = Binarize(cv2.GaussianBlur(LeerImagen(imagename), (61,61),0),inverse)[1]
    
    #cv2.imshow("e", Imagen)
    #cv2.waitKey(0)
    tamaño = np.size(Imagen)
    esqueleto = np.zeros(Imagen.shape, np.uint8)

    while(not fin):
        e= cv2.erode(Imagen, EE)
        d= cv2.dilate(e,EE)
        r= cv2.subtract(Imagen, d)
        esqueleto= cv2.bitwise_or(esqueleto, r)
        Imagen = e.copy() 

        if ((tamaño-cv2.countNonZero(Imagen))==tamaño):
            fin= True
    return esqueleto

def RotarImagen(imagen, angulo, punto):
    i = LeerImagen(imagen)
    tamaño = np.size(i)
    rotada = np.zeros(i.shape, np.uint8)        

    R = [[math.cos(math.radians(angulo)), -math.sin(math.radians(angulo)), 0], 
         [math.sin(math.radians(angulo)), math.cos(math.radians(angulo)), 0],
         [0, 0, 1]]

    T = [[1, 0, -punto[0]], 
         [0, 1, -punto[1]],
         [0, 0,  1]]
        
    T_i =   np.linalg.inv(T)        
    A = np.dot(T_i, R)    
    A = np.dot(A, T)       

    F = np.zeros(shape=(3,1))               
    
    for x in range(0, i.shape[0]):
        for y in range(0, i.shape[1]):
            p = [[x],[y],[1]]
            F = np.dot(A, p).astype(int)                           
            try:                                
              rotada[F[[0,1]][0],F[[0,1]][1]] = i[x,y]                         
            except:
              error= True                  
    
    return rotada

                           
def main():    
    Letras= ObtenerEsqueleto('letras.png', False)
    Caballo= ObtenerEsqueleto('caballo.png', True)
    punto = [180, 270]
    Minerva = RotarImagen('minerva.png', 35, punto)

    
    Titulos = ['Letras esqueleto', "Caballo esqueleto", "Minerva"]
    Imagenes = [Letras, Caballo, Minerva ]
    Types = [1, 1, 1]
    Plotear(Titulos, Imagenes, Types, 3)

if __name__ == '__main__':
    main()
