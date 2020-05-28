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
