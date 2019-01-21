# MCC

Aqui se describiran los distintos entregables para distimtas asignaturas

## Visión Computacional y Deep Learning
### Tarea 1 - Procesamiento de imagenes
#### 1. Transformar a una imagen de grises usando la transformación ponderada.
La funcion recibe una imagen en 24 bits (RGB) y retorna una imagen en la escala de grises en 8 bits
```

def TransformacionPonderada(list24bits):
    return  [round((0.29894 * list24bits[i][0]) + 
                     (0.58704 * list24bits[i][1]) + 
                     (0.11402 * list24bits[i][2])) 
                for i in range(len(list24bits))]   
```
#### 2. Transformarla a una imagen de grises con el promedio aritmético.
La funcion recibe una imagen en 24 bits (RGB) y retorna una imagen en la escala de grises en 8 bits
```

def TransformacionPromedioAritmetico(list24bits):
    return  [((list24bits[i][0]) +
              (list24bits[i][1]) +  
              (list24bits[i][2]) /3) 
             for i in range(len(list24bits))]   
   
```
#### 3. A partir de la imagen de grises ponderada, realizar las siguientes transformaciones:
##### 3.1. Aplicarle la transformación negativa.
La funcion recibe una imagen en 8 bits y retorna una imagen negativa
```

def TransformacionNegativa(list8bits):
    return  [(255 - list8bits[i]) 
             for i in range(len(list8bits))]   
   
```

##### 3.2. Binarizarla con un umbral de  150 y la función: t(x) = 0, x<= 150; t(x)=255, x>150.
La funcion recibe una imagen en 8 bits y retorna una imagen binarizada usando un umbral de 150
```
def TransformacionBinarizadaUmbral(list8bits, umbral):
    return  [ (0 if list8bits[i]<= umbral else 255 ) 
              for i in range(len(list8bits))]
   
```
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

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
