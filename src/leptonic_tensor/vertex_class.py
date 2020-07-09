import lorentz_class as lc
import coupling_class as cc
import numpy as np

class Vertex:
    def __init__(self, model, particles, indices):
        self.model = model
        self.particles = particles
        self.spins = [part.info.spin for part in self.particles]
        self.pids = self._get_pids()
        self.ufo_vertex = self.model.vertex_map[tuple(self.pids)]
        self.name = [ltz.name for ltz in self.ufo_vertex.lorentz]
        self.lorentz = [lc.Lorentz(model, self.spins, nme, indices) for nme in self.name]
        self.structure = [ltz.structure for ltz in self.lorentz]
        self.indices = [ltz.indices for ltz in self.lorentz]
        self.tensor = [ltz.tensor for ltz in self.lorentz]
        self.coupling = self._coupling_matrix()
        
    def __str__(self):
        return '{}: {}, {}, {}, {}'.format(
            self.name, self.spins, self.pids, self.tensor, self.coupling
            )
    
    def _get_pids(self):
        pids = [part.pid for part in self.particles]
        return sorted(pids)
    
    def _coupling_matrix(self):
        s = (len(self.ufo_vertex.lorentz),1)
        cp_matrix = np.zeros(s, dtype=np.complex128)
        for key, coup in zip(self.ufo_vertex.couplings.keys(),self.ufo_vertex.couplings.values()):
            try:
                cp_matrix[key[0],key[1]] = cc.Coupling(self.model, coup).value
            except:
                cp_matrix[key[0],key[1]] = None
        return cp_matrix