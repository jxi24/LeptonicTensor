import lorentz_class as lc
import coupling_class as cc
import numpy as np

class Vertex:
    def __init__(self, model, particles, indices):
        self.model = model
        self.spins = [part.info.spin for part in particles]
        pids = [part.pid for part in particles]
        self.pids = sorted(pids)
        ufo_vertex = self.model.vertex_map[tuple(self.pids)]
        self.name = [ltz.name for ltz in ufo_vertex.lorentz]
        self.lorentz = [lc.Lorentz(model, self.spins, nme, indices) for nme in self.name]
        self.structure = [ltz.structure for ltz in self.lorentz]
        self.indices = [ltz.indices for ltz in self.lorentz]
        s = (len(ufo_vertex.lorentz),1)
        cp_matrix = np.zeros(s, dtype=str)
        for key, val in zip(ufo_vertex.couplings.keys(),ufo_vertex.couplings.values()):
            print(key[0],key[1])
            cp_matrix[key[0],key[1]] = str(cc.CouplingInfo(val).value)
        self.coupling = cp_matrix
        
    def __str__(self):
        return '{}: {}, {}, {}, {}, {}'.format(
            self.name, self.spins, self.pids, self.structure, self.indices, self.coupling
            )
    