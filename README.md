# MCC

Aqui se describiran los distintos entregables para distintas asignaturas

## Visión Computacional y Deep Learning
### Tarea 1 - Procesamiento de imagenes
### Imagen original para las pruebas

![alt text](https://github.com/Wittline/ITESM/blob/master/Visi%C3%B3n%20Computacional%20y%20Deep%20Learning/Tarea%201/ejemplos/baboon.png)

#### 1. Transformar a una imagen de grises usando la transformación ponderada.
La funcion recibe una imagen en 24 bits (RGB) y retorna una imagen en la escala de grises en 8 bits
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



