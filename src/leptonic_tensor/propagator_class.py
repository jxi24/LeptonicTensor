import ufo_grammer
import numpy as np
import lorentz_tensor as lt

class PropagatorInfo:
    def __init__(self, model, ufo_propagator):
        self.name = ufo_propagator.name
        self.model = model
        ufo = ufo_grammer.UFOParser(self.model)
        propagator_string = ufo_propagator.numerator + '/' + ufo_propagator.denominator
        self.tensor = ufo(propagator_string)
        self.structure = self.tensor._array
        self.indices = self.tensor._indices
        
    def __str__(self):
        return '{}: {}, {}'.format(
            self.name, self.indices, self.structure)
    
class Propagator:
    def __init__(self, model, name, indices):
        self.model = model
        self.name = name
        self.info = self.model.propagator_map[self.name]
        self.structure = self.info[0]
        self.indices = [lt.Index(indices[idx.index-1], idx.lorentz) for idx in self.info[1]]
        self.tensor = lt.Tensor(self.structure, tuple(self.indices))