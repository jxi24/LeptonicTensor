import ufo_grammer
import numpy as np
import lorentz_tensor as lt

class LorentzInfo:
    def __init__(self, model, ufo_lorentz):
        # Spin in ufo_lorentz = 2*S + 1.
        self.name = ufo_lorentz.name
        self.model = model
        ufo = ufo_grammer.UFOParser(self.model)
        # self.spins = np.subtract(ufo_lorentz.spins,1)
        self.spins = ufo_lorentz.spins
        self.tensor = ufo(ufo_lorentz.structure)
        self.label = self.tensor._label
        self.structure = self.tensor._array
        self.indices = self.tensor._indices

    def __str__(self):
        return '{}: {}, {}'.format(
            self.name, self.spins, self.structure)

class Lorentz:
    '''
    For a given Lorentz structure and indices, produce the tensor with
    appropriate structure and indices.
    Example: FFV1, [1,2,0] -> Gamma(0,2,1).
    '''
    def __init__(self, model, spins, name, indices):
        self.model = model
        self.name = name
        self.spins = spins
        self.info = self.model.lorentz_map[tuple(self.spins), self.name]
        self.structure = self.info[0]
        self.indices = [lt.Index(indices[idx.index-1], idx.lorentz) for idx in self.info[1]]
        self.tensor = lt.Tensor(self.structure, tuple(self.indices))

    def __str__(self):
        return '{}: {}, {}'.format(
            self.name, self.structure, self.indices)

    def set_indices(self, idxs):
        return Lorentz(self, self.model, self.spins, self.name, idxs)
