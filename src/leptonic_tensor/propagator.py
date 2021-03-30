import numpy as np
import lorentz_structures as ls

class Propagator:
    model = None

    def __init__(self, particle):
        self.particle = particle
        if self.particle.is_vector:
            if self.particle.pid == 22:
            # if self.particle.massless():
                self.denominator = lambda p: (p[:, 0]*p[:, 0]-np.sum(p[:, 1:]*p[:, 1:], axis=-1))[:, np.newaxis, np.newaxis]
                self.numerator = lambda p: -1j*ls.METRIC_TENSOR[np.newaxis, ...]
            elif self.particle.pid == 24:
                mass = 79.82435974619784 # particle.mass # 91.81
                width = 2.085 # particle.width # 2.54
                self.denominator = lambda p: (p[:, 0]*p[:, 0]-np.sum(p[:, 1:]*p[:, 1:], axis=-1)-mass**2-1j*mass*width)[:, np.newaxis, np.newaxis]
                self.numerator = lambda p: \
                    -1j*ls.METRIC_TENSOR[np.newaxis, ...] + 1j*np.einsum('bi,bj->bij', p, p)/mass**2

    def __call__(self, p):
        return self.numerator(p)/self.denominator(p)
            
    def __str__(self):
        return "Prop{}".format(self.particle)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.particle == other.particle
