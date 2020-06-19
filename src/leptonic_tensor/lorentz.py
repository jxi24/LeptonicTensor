from . import lorentz_tensor as lt

class LorentzInfo:
    def __init__(self, name, spins, structure):
        self.name = name
        self.spins = spins
        self.structure, self.indices = parse(structure)

    def create(self, particles):
        return self.structure(self.get_index(particles))

    def get_index(self, particles):
        input self.indices and particles 
        return particles in order of self.indices and momentum
                (particles[3], particles[2], particles[1])
                (None)
        return (particles[1], particles[2])
               (particles[1])

def Lorentz:
    def __init__(self, LorentzInfo, indices):
        self.indices = (5, 1, 3)

    def __str__(self):
        return 'Gamma(5, 1, 3)'

class Vertex:
    self.lorentz = lorentz(particles)


def parse(structure):
    return lt.GAMMA_MU, (3, 2, 1), None
