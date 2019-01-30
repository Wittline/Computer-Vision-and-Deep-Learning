
class Estado:

    def __init__(self, Estado, parent, movimiento, profundidad, costo):

        self.Estado = Estado

        self.parent = parent

        self.movimiento = movimiento

        self.profundidad = profundidad

        self.costo = costo   

        if self.Estado: self.map = ''.join(str(e) for e in self.Estado)

    def __eq__(self, other): return self.map == other.map
    def __lt__(self, other): return self.map < other.map