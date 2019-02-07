

class Estado:

    def __init__(self, Estado, parent, movimiento, profundidad, costo, k):

        self.estado = Estado

        self.parent = parent

        self.movimiento = movimiento

        self.profundidad = profundidad

        self.costo = costo   
        
        self.k = k

        if self.estado: self.map = ''.join(str(e) for e in self.estado)

    def __eq__(self, other): return self.map == other.map
    def __lt__(self, other): return self.map < other.map