import itertools as itr
import os
import psutil as pu
import argparse as argumentos
import timeit as tiempo
from heapq import heappush, heappop, heapify
from Estado import Estado

estado_objetivo = [0, 1, 2, 3, 4, 5, 6, 7, 8]
nodo_objetivo = Estado

estado_inicial = list()
longitud_tablero = 0
tablero_espacio = 0

nodos_abiertos = 0
maxima_profundidad = 0
tamaño_limite = 0

movimientos = list()
costos = set()

def A_estrella(estado_entrada):

    global tamaño_limite
    global nodo_objetivo
    global maxima_profundidad

    explorados, vector, vector_, c = set(), list(), {}, itr.count()

    nuevaraiz = heuristica(estado_entrada)

    raiz = Estado(estado_entrada, None, None, 0, 0, nuevaraiz)    

    e = (nuevaraiz, 0, raiz)

    heappush(vector, e)

    vector_[raiz.map] = e

    while vector:

        nodo = heappop(vector)

        explorados.add(nodo[2].map)

        if nodo[2].estado == estado_objetivo:
            nodo_objetivo = nodo[2]
            return vector

        vecinos = abrir_nodo(nodo[2])

        for vecino in vecinos:

            vecino.k = vecino.costo + heuristica(vecino.estado)

            e = (vecino.k, vecino.movimiento, vecino)

            if vecino.map not in explorados:

                heappush(vector, e)
                explorados.add(vecino.map)
                vector_[vecino.map] = e

                if vecino.profundidad > maxima_profundidad: maxima_profundidad += 1

            elif vecino.map in vector_ and vecino.k < vector_[vecino.map][2].k:

                i = vector.index((vector_[vecino.map][2].k,
                                     vector_[vecino.map][2].movimiento,
                                     vector_[vecino.map][2]))

                vector[int(i)] = e

                vector_[vecino.map] = e

                heapify(vector)

        if len(vector) > tamaño_limite: tamaño_limite = len(vector)


def heuristica(s): return sum(
                             abs(b % tablero_espacio - g % tablero_espacio) + 
                             abs(b//tablero_espacio - g//tablero_espacio)
               for b, g in ((s.index(i), estado_objetivo.index(i)) for i in range(1, longitud_tablero)))

def abrir_nodo(nodo):

    global nodos_abiertos
    nodos_abiertos += 1

    vecinos = list()

    vecinos.append(Estado(moverse(nodo.estado,1), 
                          nodo, 
                          1, 
                          nodo.profundidad + 1, 
                          nodo.costo + 1,0))
    vecinos.append(Estado(moverse(nodo.estado, 2), 
                          nodo, 
                          2, 
                          nodo.profundidad + 1, 
                          nodo.costo + 1,0))
    vecinos.append(Estado(moverse(nodo.estado, 3), 
                          nodo, 
                          3, 
                          nodo.profundidad + 1, 
                          nodo.costo + 1,0))
    vecinos.append(Estado(moverse(nodo.estado, 4), 
                          nodo, 
                          4, 
                          nodo.profundidad + 1, 
                          nodo.costo + 1,0))

    return [vecino for vecino in vecinos if vecino.estado]    


def moverse(Estado, posicion):

    nuevoestado = Estado[:]

    i = nuevoestado.index(0)

    if posicion == 1:
        if i not in range(0, tablero_espacio):
            auxiliar = nuevoestado[i - tablero_espacio]
            nuevoestado[i - tablero_espacio] = nuevoestado[i]
            nuevoestado[i] = auxiliar
            return nuevoestado
        else:
            return None

    if posicion == 2:
        if i not in range(longitud_tablero - tablero_espacio, longitud_tablero):
            auxiliar = nuevoestado[i + tablero_espacio]
            nuevoestado[i + tablero_espacio] = nuevoestado[i]
            nuevoestado[i] = auxiliar
            return nuevoestado
        else:
            return None

    if posicion == 3:
        if i not in range(0, longitud_tablero, tablero_espacio):
            auxiliar = nuevoestado[i - 1]
            nuevoestado[i - 1] = nuevoestado[i]
            nuevoestado[i] = auxiliar
            return nuevoestado
        else:
            return None

    if posicion == 4:

        if i not in range(tablero_espacio - 1, longitud_tablero, tablero_espacio):
            auxiliar = nuevoestado[i + 1]
            nuevoestado[i + 1] = nuevoestado[i]
            nuevoestado[i] = auxiliar
            return nuevoestado
        else:
            return None

def llamadas():

    nodoactual = nodo_objetivo

    while estado_inicial != nodoactual.estado:

        if nodoactual.movimiento == 1:
            movimiento = 'Up'
        elif nodoactual.movimiento == 2:
            movimiento = 'Down'
        elif nodoactual.movimiento == 3:
            movimiento = 'Left'
        else:
            movimiento = 'Right'

        movimientos.insert(0, movimiento)
        nodoactual = nodoactual.parent

    return movimientos

def reporte(limite, tiempo):

    global movimientos

    process = pu.Process(os.getpid())
    movimientos = llamadas()

    print("path_to_goal: " + str(movimientos))
    print("cost of the path: " + str(len(movimientos)))
    print("number of visited nodes: " + str(nodos_abiertos))    
    print("running time: " + format(tiempo, '.8f'))        
    print("Used Memory: " + format(process.memory_info()[0] / float(2 ** 20), '.8f'))                  

def main():

    global longitud_tablero
    global tablero_espacio    
    
    f = open('input.txt')    
    input = f.readline().split(",")    

    for item in input:
        estado_inicial.append(int(item))

    longitud_tablero = len(estado_inicial)
    tablero_espacio = int(longitud_tablero ** 0.5)    
    
    inicio = tiempo.default_timer()    
    run = A_estrella(estado_inicial)
    fin = tiempo.default_timer()
    reporte(run,fin-inicio)

if __name__ == '__main__':
    main()
