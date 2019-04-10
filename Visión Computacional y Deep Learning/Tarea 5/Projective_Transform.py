import numpy as np
from scipy import misc
import cv2
from matplotlib import pyplot as plt
from PIL import Image

color_point = (0,0,255)

def mouse_handler(event, x, y, flags, data) :    
    if event == cv2.EVENT_LBUTTONDOWN :
        cv2.circle(data['im'], (x,y),3, color_point, 5, 16);
        cv2.imshow("Image", data['im']);
        if len(data['points']) < 4 :
            data['points'].append([x,y])

def getXYMouse(im, color):

    global color_point
    color_point = color
        
    data = {}
    data['im'] = im.copy()
    data['points'] = []
        
    cv2.imshow("Image",im)
    cv2.setMouseCallback("Image", mouse_handler, data)
    cv2.waitKey(0)
    
    points = np.vstack(data['points']).astype(int)
    
    return points

def LeerImagen(l):    
    return cv2.imread(l)

def TransformacionProyectiva(i, o, d):
    im = misc.imread(i)
    
    if(len(o)<1 or len(d)<1):
        cv2.imshow("Image", im)
        o = obtenerCoordenada(im,(0,0,255))
        print(o)
        d = obtenerCoordenada(im,(255,0,0))
        print(d)

    MAT = np.zeros((8,9))
    for ite in range(4):
        MAT[ite*2,:] = [ o[ite][1], o[ite][0], 1, 
                    0, 0, 0, 
                    -d[ite][1]*o[ite][1], -d[ite][1]*o[ite][0], -d[ite][1] ]
        MAT[ite*2+1,:] = [0, 0, 0, 
                      o[ite][1], o[ite][0], 1, 
                      -d[ite][0]*o[ite][1], -d[ite][0]*o[ite][0], -d[ite][0] ]

    mx=[];
    my=[];    
    [Z,Y,X]=np.linalg.svd(MAT)
    mat = X[-1,:]
    matH = np.reshape(mat,(3,3))

    txy = np.array([[1],[1],[1]])
    txytemporal = np.dot(matH,txy)
    mx.append(txytemporal[0]/txytemporal[2])
    my.append(txytemporal[1]/txytemporal[2])

    txy = np.array([[im.shape[1]],[1],[1]])
    txytemporal = np.dot(matH,txy)
    mx.append(txytemporal[0]/txytemporal[2])
    my.append(txytemporal[1]/txytemporal[2])

    txy = np.array([[1],[im.shape[0]],[1]])
    txytemporal = np.dot(matH,txy)
    mx.append(txytemporal[0]/txytemporal[2])
    my.append(txytemporal[1]/txytemporal[2])

    txy = np.array([[im.shape[1]],[im.shape[0]],[1]])
    txytemporal = np.dot(matH,txy)
    mx.append(txytemporal[0]/txytemporal[2])
    my.append(txytemporal[1]/txytemporal[2])

    XX1 = int(np.min(mx))
    XX2 = int(np.max(mx))
    YY1 = int(np.min(my))
    YY2 = int(np.max(my))
    
    proyectiva = np.zeros((int(YY2-YY1),int(XX2-XX1),3))

    imX=im.shape[0]
    imY=im.shape[1]

    for i in range(imX):
            for j in range(imY):
                    txy = np.array([[j],[i],[1]])
                    txytemporal = np.dot(matH,txy)
                    px=int(txytemporal[0]/txytemporal[2])-XX1
                    py=int(txytemporal[1]/txytemporal[2])-YY1

                    if (px>0 and py>0 and py<YY2-YY1 and px<XX2-XX1):
                            proyectiva[py,px,:]=im[i,j,:]
    
    return proyectiva, matH, im, (XX1, YY1)

def obtenerCoordenada(i,c):    
    coordenadasPuntos = getXYMouse(i, color= c);
    punto= []; 
    punto.append((coordenadasPuntos[0,1], coordenadasPuntos[0,0]))
    punto.append((coordenadasPuntos[1,1], coordenadasPuntos[1,0]))
    punto.append((coordenadasPuntos[2,1], coordenadasPuntos[2,0] ))
    punto.append((coordenadasPuntos[3,1], coordenadasPuntos[3,0]))
    return punto;  
    

def Interpolar(im,imo, h, lc):
    print(lc[0])
    imX=im.shape[0]
    imY=im.shape[1]
    imxx=imo.shape[0]
    imyy=imo.shape[1]    
    h_inversa = np.linalg.inv(h)

    for i in range(imX):
            for j in range(imY):
                    if sum(im[i,j,:])==0:
                            txy = np.array([[j+lc[0]],[i+lc[1]],[1]])
                            t = np.dot(h_inversa,txy)
                            px=int(t[0]/t[2])
                            py=int(t[1]/t[2])

                            if (px>0 and py>0 and px<imyy and py<imxx):
                                    im[i,j,:] = imo[py,px,:]
   
    return im


if __name__ == '__main__':
    origen= []; 
    destino= [];
    #origen = [(220,110),(30,500),
    #          (375,95),(325,500)]
    #destino = [(150,200),(150,700),
    #        (300,200),(300,700)]

    ImagenProyectiva, H, imo, llc = TransformacionProyectiva("mural.jpg", origen, destino)
    ImagenProyetivaMejorada = Interpolar(ImagenProyectiva, imo, H, llc)
    misc.imsave("ImagenProyetivaMejorada.jpg",ImagenProyetivaMejorada)



   