# MCC

Aqui se describen los distintos entregables para distintas asignaturas

## Visión Computacional y Deep Learning
### Tarea 1 - Procesamiento de imagenes - Transformaciones puntuales
### Imagen original para las pruebas

![alt text](https://github.com/Wittline/ITESM/blob/master/Visi%C3%B3n%20Computacional%20y%20Deep%20Learning/Tarea%201/ejemplos/baboon.png)

#### 1. Transformar a una imagen de grises usando la transformación ponderada.
La funciòn recibe una imagen en 24 bits (RGB) y retorna una imagen en la escala de grises en 8 bits
```

def TransformacionPonderada(list24bits):
    return  [round((0.29894 * list24bits[i][0]) + 
                     (0.58704 * list24bits[i][1]) + 
                     (0.11402 * list24bits[i][2])) 
                for i in range(len(list24bits))]   
```
![alt text](https://github.com/Wittline/ITESM/blob/master/Visi%C3%B3n%20Computacional%20y%20Deep%20Learning/Tarea%201/ejemplos/baboon_TP.png)

#### 2. Transformarla a una imagen de grises con el promedio aritmético.
La funcion recibe una imagen en 24 bits (RGB) y retorna una imagen en la escala de grises en 8 bits
```

def TransformacionPromedioAritmetico(list24bits):
    return  [((list24bits[i][0]) +
              (list24bits[i][1]) +  
              (list24bits[i][2]) /3) 
             for i in range(len(list24bits))]   
   
```

![alt text](https://github.com/Wittline/ITESM/blob/master/Visi%C3%B3n%20Computacional%20y%20Deep%20Learning/Tarea%201/ejemplos/baboon_TPA.png)

#### 3. A partir de la imagen de grises ponderada, realizar las siguientes transformaciones:
##### 3.1. Aplicarle la transformación negativa.
La funcion recibe una imagen en 8 bits y retorna una imagen negativa
```

def TransformacionNegativa(list8bits):
    return  [(255 - list8bits[i]) 
             for i in range(len(list8bits))]   
   
```
![alt text](https://github.com/Wittline/ITESM/blob/master/Visi%C3%B3n%20Computacional%20y%20Deep%20Learning/Tarea%201/ejemplos/baboon_TN.png)

##### 3.2. Binarizarla con un umbral de  150 y la función: t(x) = 0, x<= 150; t(x)=255, x>150.
La funcion recibe una imagen en 8 bits y retorna una imagen binarizada usando un umbral de 150
```
def TransformacionBinarizadaUmbral(list8bits, umbral):
    return  [ (0 if list8bits[i]<= umbral else 255 ) 
              for i in range(len(list8bits))]
   
```
![alt text](https://github.com/Wittline/ITESM/blob/master/Visi%C3%B3n%20Computacional%20y%20Deep%20Learning/Tarea%201/ejemplos/baboon_TB.png)

##### 3.3 Aplicar la transformación de raíz cuadrada y escalarla en (0, 255). 
La funcion recibe una imagen en 8 bits, se calcula la raiz cuadrada del valor de cada pixel y el resultado final se escala
```
def escalar(list8bits):        
    mi = min(list8bits)
    ma = max(list8bits)    
    newlist =  [ ((255) /(ma - mi)) * (list8bits[i] - mi) 
                          for i in range(len(list8bits))]        
    return newlist


def TransformacionRaizCuadrada(list8bits):
    return  [( math.sqrt(list8bits[i])) 
               for i in range(len(list8bits))]
               
TRC = escalar(TransformacionRaizCuadrada(TP))
     
``` 
![alt text](https://github.com/Wittline/ITESM/blob/master/Visi%C3%B3n%20Computacional%20y%20Deep%20Learning/Tarea%201/ejemplos/baboon_TRC.png)

##### 3.4 Aplicar la transformación de potencia al cubo y escalarla en (0, 255).
La funcion recibe una imagen en 8 bits, se calcula la la potencia al cubo del valor de cada pixel y el resultado final se escala
```
def escalar(list8bits):        
    mi = min(list8bits)
    ma = max(list8bits)    
    newlist =  [ ((255) /(ma - mi)) * (list8bits[i] - mi) 
                          for i in range(len(list8bits))]        
    return newlist

def TransformacionPotencia(list8bits, potencia):
    return [(list8bits[i]**potencia) 
           for i in range(len(list8bits))]
               
TPC = escalar(TransformacionPotencia(TP, 3))
     
```
![alt text](https://github.com/Wittline/ITESM/blob/master/Visi%C3%B3n%20Computacional%20y%20Deep%20Learning/Tarea%201/ejemplos/baboon_TPC.png)

### Código
```
import math
from PIL import Image


def escalar(list8bits):        
    mi = min(list8bits)
    ma = max(list8bits)    
    newlist =  [ ((255) /(ma - mi)) * (list8bits[i] - mi) 
                          for i in range(len(list8bits))]        
    return newlist


def TransformacionPonderada(list24bits):
    return  [round((0.29894 * list24bits[i][0]) + 
                     (0.58704 * list24bits[i][1]) + 
                     (0.11402 * list24bits[i][2])) 
                for i in range(len(list24bits))]   

def TransformacionPromedioAritmetico(list24bits):
    return  [((list24bits[i][0]) +
              (list24bits[i][1]) +  
              (list24bits[i][2]) /3) 
             for i in range(len(list24bits))]   


def TransformacionNegativa(list8bits):
    return  [(255 - list8bits[i]) 
             for i in range(len(list8bits))] 

def TransformacionBinarizadaUmbral(list8bits, umbral):
    return  [ (0 if list8bits[i]<= umbral else 255 ) 
              for i in range(len(list8bits))]

def TransformacionRaizCuadrada(list8bits):
    return  [( math.sqrt(list8bits[i])) 
               for i in range(len(list8bits))]

def TransformacionPotencia(list8bits, potencia):
    return [(list8bits[i]**potencia) 
           for i in range(len(list8bits))]


def convertirImagen(size, list8bits, filename):
    newimage = Image.new('L', size) 
    newimage.putdata(list8bits) 
    newimage.save(filename)
    newimage.close()
    return 1

 
imagenColor = Image.open('C:/ejemplos/baboon.png')
Matrix = list(imagenColor.getdata())

TP = TransformacionPonderada(Matrix)
convertirImagen(imagenColor.size,TP,'C:/ejemplos/baboon_TP.png')
TPA = TransformacionPromedioAritmetico(Matrix)
convertirImagen(imagenColor.size,TPA,'C:/ejemplos/baboon_TPA.png')
TN = TransformacionNegativa(TP);
convertirImagen(imagenColor.size,TN,'C:/ejemplos/baboon_TN.png')
TB = TransformacionBinarizadaUmbral(TP, 150)
convertirImagen(imagenColor.size,TB,'C:/ejemplos/baboon_TB.png')
TRC = escalar(TransformacionRaizCuadrada(TP))
convertirImagen(imagenColor.size,TRC,'C:/ejemplos/baboon_TRC.png')
TPC = escalar(TransformacionPotencia(TP, 3))
convertirImagen(imagenColor.size,TPC,'C:/ejemplos/baboon_TPC.png')
imagenColor.close()

     
```
### Tarea 2 - Detector de bordes
#### 1. Obtener la correlación cruzada H ⨂ w, de la imagen con dicho kernel, y obtener la convolucion H * w
```
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
```
#### 2. Obtener las convoluciones H*Q , H*q y (H * q) qt,  q= [-1,3,-1]
```
def convolucion(i, k):
    return sn.convolve2d(i,k,'same',boundary='wrap', fillvalue=0)    

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
```
#### 3. Usando la herramienta que mejor consideres, utiliza el operador Sobel de dicha librería para hacer la práctica que se indica en la diapositiva
```
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
    
    print("** Tercer ejercicio ** ")
    contornos = contornos_sobel('yo.jpg')    
```
![alt text](https://github.com/Wittline/ITESM/blob/master/Visi%C3%B3n%20Computacional%20y%20Deep%20Learning/Tarea%202/imyogray_sobelx.jpg)
![alt text](https://github.com/Wittline/ITESM/blob/master/Visi%C3%B3n%20Computacional%20y%20Deep%20Learning/Tarea%202/imyogray_sobely.jpg)
![alt text](https://github.com/Wittline/ITESM/blob/master/Visi%C3%B3n%20Computacional%20y%20Deep%20Learning/Tarea%202/imyogray_sobel.jpg)

#### 4. Este ejercicio debe usar la imagen en tono de grises de panda.png, la cual Usando la información del gradiente con el operador derivada que creas más conveniente (ver archivo 02_Transformaciones_Varias.pdf, diapositiva 36), dibujar en la imagen los vectores gradiente ortogonales a como se muestran en las imágenes de las diapositivas 36 y 37 del archivo mencionado. Para calcular y aproximar el gradiente puedes utilizar cualquiera de los kernels derivada que desees para obtener las parciales con X y Y. Sin embargo, la magnitud de todos ellos darle un valor constante a tu elección. Además, para una mejor visualización de los vectores gradientes, puedes dibujar solamente un porcentaje de dichos vectores gradientes y solo aquellos que tengan las magnitudes más significativas. 
```
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
    
    
  print("** Cuarto ejercicio ** ")
  vectores_gradientes("panda.jpg")
```

![alt text](https://github.com/Wittline/ITESM/blob/master/Visi%C3%B3n%20Computacional%20y%20Deep%20Learning/Tarea%202/panda.jpg)

![alt text](https://github.com/Wittline/ITESM/blob/master/Visi%C3%B3n%20Computacional%20y%20Deep%20Learning/Tarea%202/panda_vector.jpg)
        
### Código
```
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
     
```

      
### Requisitos de ejecución de las tareas
Visual studio 2017
Python 3.6
OpenCv
Numpy
PIL
matplotlib
scipy

### Licencia 
Este proyecto está bajo la licencia MIT.



