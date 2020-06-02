class ParticleInfo:
    def __init__(self, ufo_particle):
        self.name = ufo_particle.name
        self.antiname = ufo_particle.antiname
        self.pid = ufo_particle.pdg_code
        self.mass = ufo_particle.mass
        self.width = ufo_particle.width
        self.charge = ufo_particle.charge
        self.icharge = int(ufo_particle.charge*3)
        self.spin = ufo_particle.spin-1
        self.propagator = self._get_propagator(ufo_particle)
        self.wavefunction = self._get_wavefunction()

    def __str__(self):
        return '{}: {}, {}, {}, {}, {}, {}, {}, {}'.format(
            self.pid, self.name, self.antiname,
            self.mass, self.width, self.spin, self.icharge, self.charge,
            self.propagator)

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
        
    def _get_wavefunction(self):
        if self.spin == 0:
            wavefunction_incoming = '1'
            wavefunction_outgoing = '1'
            wavefunction = [wavefunction_incoming, wavefunction_outgoing]
            return wavefunction
        if self.spin == 1:
            wavefunction_incoming_name = 'u(p_{})'.format(self.name)
            wavefunction_incoming_antiname = 'vbar(p_{})'.format(self.antiname)
            wavefunction_outgoing_name = 'ubar(p_{})'.format(self.name)
            wavefunction_outgoing_antiname = 'v(p_{})'.format(self.antiname)
            
            wavefunction_incoming = [wavefunction_incoming_name, wavefunction_incoming_antiname]
            wavefunction_outgoing = [wavefunction_outgoing_name, wavefunction_outgoing_antiname]
            wavefunction = [wavefunction_incoming, wavefunction_outgoing]
            return wavefunction
        else:
            pass
