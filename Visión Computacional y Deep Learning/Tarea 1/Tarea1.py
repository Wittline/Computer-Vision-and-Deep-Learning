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
