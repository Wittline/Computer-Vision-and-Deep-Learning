import os
import psutil as pu
import argparse as argumentos
import timeit
from collections import deque
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


def busqueda_primero_anchura(estado_entrada):

    global tamaño_limite, nodo_objetivo, maxima_profundidad

    explorados, cola = set(), deque([Estado(estado_entrada, None, None, 0, 0)])

    while cola:

        nodo = cola.popleft()

        explorados.add(nodo.map)

        if nodo.Estado == estado_objetivo:
            nodo_objetivo = nodo
            return cola

        vecinos = abrir_nodo(nodo)

        for vecino in vecinos:
            if vecino.map not in explorados:
                cola.append(vecino)
                explorados.add(vecino.map)

                if vecino.profundidad > maxima_profundidad:maxima_profundidad += 1

        if len(cola) > tamaño_limite: tamaño_limite = len(cola)


def abrir_nodo(nodo):

    global nodos_abiertos
    nodos_abiertos += 1

    vecinos = list()

    vecinos.append(Estado(moverse(nodo.Estado,1), 
                          nodo, 
                          1, 
                          nodo.profundidad + 1, 
                          nodo.costo + 1))
    vecinos.append(Estado(moverse(nodo.Estado, 2), 
                          nodo, 
                          2, 
                          nodo.profundidad + 1, 
                          nodo.costo + 1))
    vecinos.append(Estado(moverse(nodo.Estado, 3), 
                          nodo, 
                          3, 
                          nodo.profundidad + 1, 
                          nodo.costo + 1))
    vecinos.append(Estado(moverse(nodo.Estado, 4), 
                          nodo, 
                          4, 
                          nodo.profundidad + 1, 
                          nodo.costo + 1))

    return [vecino for vecino in vecinos if vecino.Estado]    


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

    while estado_inicial != nodoactual.Estado:

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

    global longitud_tablero, tablero_espacio    
    
    f = open('input.txt')    
    input = f.readline().split(",")    

    for item in input:
        estado_inicial.append(int(item))

    longitud_tablero = len(estado_inicial)
    tablero_espacio = int(longitud_tablero ** 0.5)    
    
    inicio = timeit.default_timer()    
    run = busqueda_primero_anchura(estado_inicial)
    fin = timeit.default_timer()
    reporte(run,fin-inicio)

if __name__ == '__main__':
    main()


