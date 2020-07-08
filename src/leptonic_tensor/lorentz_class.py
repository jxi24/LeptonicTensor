import ufo_grammer
import numpy as np
import lorentz_tensor as lt

class LorentzInfo:
    def __init__(self, model, ufo_lorentz):
        # Spin in ufo_lorentz = 2*S + 1.
        self.name = ufo_lorentz.name
        self.model = model
        ufo = ufo_grammer.UFOParser(self.model)
        self.spins = np.subtract(ufo_lorentz.spins,1)
        self.tensor = ufo(ufo_lorentz.structure)
        self.structure = self.tensor._array
        self.indices = self.tensor._indices
        
    def __str__(self):
        return '{}: {}, {}'.format(
            self.name, self.spins, self.structure)
    
class Lorentz:
    '''
    Lorentz(indices) = replace LorentzInfo indices with new indices.
    Example: e-(p1)e+(p2)A(p5). Call vertex with (e-,e+,A) and (1,2,5);
    this would call Lorentz with 'FFV1' and (1,2,5). 
    Spin convention: 2*S. Note, indices include both Lorentz and Spin indices,
    depening on the Lorentz structure (e.g. Gamma(Lorentz, Spin, Spin)).
    '''
    def __init__(self, model, spins, name, indices):
        self.model = model
        self.name = name
        self.spins = spins
        self.info = self.model.lorentz_map[tuple(self.spins), self.name]
        self.structure = self.info[0]
        self.indices = [lt.Index(indices[idx.index-1], idx.lorentz) for idx in self.info[1]]
        self.tensor = lt.Tensor(self.structure, tuple(self.indices))
        # Gamma(3,2,1) -> self.info[1] = [Index(3,T),Index(2,T),Index(1,T)].
        # Then indices = [1,2,5] -> [5,2,1] = [indices[3],indices[2],indices[1]]
        
    def __str__(self):
        return '{}: {}, {}'.format(
            self.name, self.structure, self.indices)
    
    def set_indices(self, idxs):
        return Lorentz(self, self.model, self.spins, self.name, idxs)

    