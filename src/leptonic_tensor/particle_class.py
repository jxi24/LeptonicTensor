import lorentz_structures as ls
import numpy as np

class ParticleInfo:
    def __init__(self, ufo_particle, label=None):
        self.name = ufo_particle.name
        self.antiname = ufo_particle.antiname
        self.pid = ufo_particle.pdg_code
        self.mass = ufo_particle.mass
        self.width = ufo_particle.width
        self.charge = ufo_particle.charge
        self.icharge = int(ufo_particle.charge*3)
        self.spin = ufo_particle.spin-1
        self.propagator = self._get_propagator(ufo_particle)
        self.wavefunction = self.get_wavefunction(label)

    def __str__(self):
        return '{}: {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(
            self.pid, self.name, self.antiname,
            self.mass, self.width, self.spin, self.icharge, self.charge,
            self.propagator, self.wavefunction)

    def _get_propagator(self, ufo_particle):
        if not hasattr(ufo_particle, 'propagator'):
            if self.spin == 2:
                if self.mass.name == 'ZERO':
                    return 'V2'
                else:
                    return 'V1'
            elif self.spin == 1:
                return 'F'
            elif self.spin == 0:
                return 'S'
            else:
                return 'G'
        else:
            return ufo_particle.propagator
        
    def get_wavefunction(self, label):
        if self.spin == 0:
            wavefunction_incoming = '1'
            wavefunction_outgoing = '1'
            wavefunction = [wavefunction_incoming, wavefunction_outgoing]
            self.wavefunction = wavefunction
            return wavefunction
        if self.spin == 1:
            if self.pid > 0:
                wavefunction_incoming = 'u(p_[{},{}])'.format(self.name, label)
                wavefunction_outgoing = 'ubar(p_[{},{}])'.format(self.name, label)
                wavefunction = [wavefunction_incoming, wavefunction_outgoing]
                self.wavefunction = wavefunction
                return wavefunction
            if self.pid < 0:
                wavefunction_incoming = 'vbar(p_[{},{}])'.format(self.name, label) 
                wavefunction_outgoing = 'v(p_[{},{}])'.format(self.name, label)
                wavefunction = [wavefunction_incoming, wavefunction_outgoing]
                self.wavefunction = wavefunction
                return wavefunction
        if self.spin == 2:
            wavefunction_incoming = 'epsilon(p_[{},{}])'.format(self.name, label)
            wavefunction_outgoing = 'epsilon*(p_[{},{}])'.format(self.name, label)
            wavefunction = [wavefunction_incoming, wavefunction_outgoing]
            self.wavefunction = wavefunction
            return wavefunction
        else:
            pass

class Particle:
    def __init__(self, model, pid: int, mom_array: np.array, index: int, mom_index: int, incoming: bool):
        self.model = model
        self.pid = pid
        self.mom_array = mom_array
        self.index = index
        self.mom_index = mom_index
        self.incoming = incoming
        
        self.momentum = self.mom_array[self.mom_index]
        self.spin = 0
        try:
            self.info = model.particle_map[self.pid]
        except:
            self.info = model.particle_map[-self.pid]

    def anti(self):
        new_mom_array = self.mom_array
        new_mom_array[self.mom_index] = -self.mom_array[self.mom_index]
        return Particle(self.model, -self.pid, new_mom_array, self.index, self.mom_index, not self.incoming)

    def get_spinor(self):
        '''
        Return outgoing wavefunction of Particle, identified with
        particle name and momentum label number.
        '''
        if self.incoming:
            if self.pid > 0:
                return ls.SpinorU(self.mom_array, self.index, self.mom_index, self.spin)
            elif self.pid < 0:
                return ls.SpinorVBar(self.mom_array, self.index, self.mom_index, self.spin)
        elif not self.incoming:
            if self.pid > 0:
                return ls.SpinorUBar(self.mom_array, self.index, self.mom_index, self.spin)
            elif self.pid < 0:
                return ls.SpinorV(self.mom_array, self.index, self.mom_index, self.spin)

    def __str__(self):
        return 'Particle({}, {}, {})'.format(self.momentum, self.index, self.info)
    