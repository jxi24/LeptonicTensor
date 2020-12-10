import lorentz_structures as ls
import numpy as np


class ParticleInfo:
    def __init__(self, ufo_particle):
        self.name = ufo_particle.name
        self.antiname = ufo_particle.antiname
        self.pid = ufo_particle.pdg_code
        self.mass = ufo_particle.mass
        self.width = ufo_particle.width
        self.charge = ufo_particle.charge
        self.icharge = int(ufo_particle.charge*3)
        # self.spin = ufo_particle.spin-1
        self.spin = ufo_particle.spin
        self.propagator = self._get_propagator(ufo_particle)

    def __str__(self):
        return '{}: {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(
            self.pid, self.name, self.antiname,
            self.mass, self.width, self.spin, self.icharge, self.charge,
            self.propagator, self.wavefunction)

    def _get_propagator(self, ufo_particle):
        if not hasattr(ufo_particle, 'propagator'):
            if self.spin == 3:
                if self.mass.name == 'ZERO':
                    return 'V2'
                else:
                    return 'V1'
            elif self.spin == 2:
                return 'F'
            elif self.spin == 1:
                return 'S'
            else:
                return 'G'
        else:
            return ufo_particle.propagator


class Particle:
    # Update on Dec 3, 2020: Spin is back to UFO convention 2S+1.
    gindex = 0

    def __init__(self, model, pid, momentum=None, incoming=False, index=None):
        self.model = model
        self.pid = pid
        if index is None:
            self.index = Particle.gindex
            Particle.gindex += 1
        else:
            self.index = index
        self.incoming = incoming

        self.momentum = momentum
        self.info = model.particle_map[abs(self.pid)]
        self.ref = np.array([1, 1, 0, 0])

    def anti(self):
        return Particle(self.model, -self.pid, -self.momentum,
                        not self.incoming, self.index)

    def get_spinor(self, mom, spin):
        '''
        Return outgoing wavefunction of Particle, identified with
        particle name and momentum label number.
        '''
        if self.incoming:
            if self.pid > 0:
                return ls.SpinorU(mom, self.index, spin)
            elif self.pid < 0:
                return ls.SpinorVBar(mom, self.index, spin)
        elif not self.incoming:
            if self.pid > 0:
                return ls.SpinorUBar(mom, self.index, spin)
            elif self.pid < 0:
                return ls.SpinorV(mom, self.index, spin)

    def get_polarization(self, mom, spin):
        # Spin convention: 2S+1 (12/3/2020).
        if spin == 1:
            denom = ls.SpinorU(mom, 0, spin)*ls.SpinorU(self.ref, 0, spin)
            denom = complex(denom).conjugate()*np.sqrt(2.0)
            num = (ls.SpinorU(mom, 0, spin)*ls.Gamma(self.index, 0, 1)
                   * ls.SpinorU(self.ref, 1, spin))
            return num/denom
        if spin == 2:
            denom = ls.SpinorU(mom, 0, spin)*ls.SpinorU(self.ref, 0, spin)
            denom = complex(denom).conjugate()
            return (ls.SpinorU(mom, 0, spin)*ls.Gamma(self.index, 0, 1)
                    * ls.SpinorU(mom, 1, spin))/(np.sqrt(2.)*denom)

    def Wavefunction(self, mom, spin=None):
        if self.info.spin == 1:
            return 1
        elif self.info.spin == 2:
            return self.get_spinor(mom, spin)
        elif self.info.spin == 3:
            return self.get_polarization(mom, spin)
        else:
            raise ValueError("Invalid particle spin")

    def Propagator(self, mom):
        if self.mass == 0:
            return 1.0/mom**2
        else:
            return 1.0/(mom**2-self.mass**2-1j*self.mass*self.width)

    def __str__(self):
        return 'Particle({}, {}, {})'.format(self.momentum,
                                             self.index,
                                             self.info)
