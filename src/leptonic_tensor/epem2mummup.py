import models
import itertools
import particle
import feyn_rules
import numpy as np

class Model:
    def __init__(self, name, all_models):
        self.name = name
        self.model = all_models[name]
        self.particle_map = self._particle_map()
        self.vertex_map = self._vertex_map()
        self.propagator_map = self._propagator_map()
        self.lorentz_map = self._lorentz_map()
        self.couplings = self.model.all_couplings

    def _particle_map(self):
        particles = self.model.all_particles
        part_map = {}
        for part in particles:
            if part.spin < 0:
                continue
            test = particle.ParticleInfo(part)
            part_map[test.pid] = test

        for key, value in part_map.items():
            print(key, value)
        return part_map

    def _vertex_map(self):
        vertices = self.model.all_vertices
        vert_map = {}
        for vertex in vertices:
            particles = vertex.particles
            # print(vertex.lorentz, particles)
            pids = []
            for part in particles:
                pids.append(part.pdg_code)
            pids.sort()
            # print(pids)
            vert_map[tuple(pids)] = vertex
        return vert_map

    def _propagator_map(self):
        propagators = self.model.all_propagators
        prop_map = {}
        for prop in propagators:
            prop_map[prop.name] = prop
        return prop_map

    def _lorentz_map(self):
        # spin = 2*S
        lorentzs = self.model.all_lorentz
        lorentz_map = {}
        for lorentz in lorentzs:
            ltz = LorentzInfo(lorentz)
            lorentz_map[tuple(ltz.spins), ltz.name] = [ltz.structure, ltz.indices]
            #lorentz_map[tuple(ltz.spins), ltz.name] = ltz.structure
        for key, value in lorentz_map.items():
            print(key, value)
        return lorentz_map

    @property
    def particles(self):
        return (', ').join(list(self.particle_map.keys()))


class Particle:
    def __init__(self, model, pid, momentum, incoming):
        self.momentum = momentum
        try:
            self.info = model.particle_map[pid]
        except:
            # Particle is its own antiparticle.
            self.info = model.particle_map[-pid]
        self.incoming = incoming
        self.model = model

    def anti(self):
        return Particle(self.model, -self.info.pid, -self.momentum, not self.incoming)

    def wavefunction(self):
        '''
        Return outgoing wavefunction of Particle, identified with
        particle name and momentum label number.
        '''
        if self.incoming:
            return self.anti().wavefunction()
        else:
            if self.info.spin == 0:
                return ''
            if self.info.spin == 1:
                if self.info.pid < 0:
                    return 'v(p_[{}, {}])'.format(self.info.name, self.momentum)
                if self.info.pid > 0:
                    return 'ubar(p_[{}, {}])'.format(self.info.name, self.momentum)
            if self.info.spin == 2:
                return 'epsilon*(p_[{}, {}])'.format(self.info.name, self.momentum)
            else:
                pass

    def __str__(self):
        return 'Particle({}, {})'.format(self.momentum, self.info)

class LorentzInfo:
    def __init__(self, ufo_lorentz):
        self.name = ufo_lorentz.name
        self.spins = np.subtract(ufo_lorentz.spins,1)
        self.structure, self.indices = parse(ufo_lorentz.structure)
        #self.structure = ufo_lorentz.structure
        
    def __str__(self):
        return '{}: {}, {}'.format(
            self.name, self.spins, self.structure)
    
class Lorentz:
    def __init__(self, model, spins):
        self.model = model
        self.info = model.lorentz_map[spins]
        self.indices = self.info[1]
        self.structure = self.info[0]
        
    def __str__(self):
        return parse(self.info.structure, self.indices)
    
    def transform(self, idxs, change=False):
        if not change:
            self.indices = idxs
        return np.take(self.structure, idxs)

def main():
    all_models = models.discover_models()
    model = Model('Models.SM_NLO', all_models)
    # ubar = Particle(model, -2, 3, False)
    # print(ubar.wavefunction())
    # u = ubar.anti()
    # print(u.wavefunction())


    # Setup the incoming, outgoing, and internal particles
    # particles = model.particle_map
    # incoming_particles = [particles[-11], particles[11]]
    # outgoing_particles = [particles[-13], particles[13]]
    # internal_particles = [particles[22]]
    
    elec = Particle(model, 11, 1, True)
    antielec = Particle(model, -11, 2, True)
    muon = Particle(model, 13, 3, False)
    antimuon = Particle(model, -13, 4, False)
    photon = Particle(model, 22, 0, False)
    # print(elec.wavefunction())
    # print(antielec.wavefunction())
    # print(muon.wavefunction())
    # print(antimuon.wavefunction())
    
    InP = [elec, antielec]
    OutP = [muon, antimuon]
    IntP = [photon]

    # pids1 = [particles[-11].pid,
    #          particles[11].pid,
    #          particles[22].pid]
    # pids1.sort()
    # print(pids1)
    # vertex1 = model.vertex_map[tuple(pids1)]
    # print(vertex1.lorentz[0].structure, vertex1.couplings[(0, 0)].value)
    # vertexA = vertex1.lorentz[0].structure.replace('3', 'mu').replace('2', 'e-').replace('1', 'e+')

    # pids2 = [particles[-13].pid,
    #          particles[13].pid,
    #          particles[22].pid]
    # pids2.sort()
    # print(pids2)
    # vertex2 = model.vertex_map[tuple(pids2)]
    # print(vertex2.lorentz[0].structure, vertex2.couplings[(0, 0)].value)
    # vertexB = vertex2.lorentz[0].structure.replace('3', 'nu').replace('2', 'mu-').replace('1', 'mu+')

    # propagator = model.propagator_map[particles[22].propagator]
    # propagatorA = propagator.numerator.replace('1','mu').replace('2','nu') + '/' + propagator.denominator
    # print("({})/({})".format(propagator.numerator, propagator.denominator))

    # amp = ['ubar(p2)', vertexA, 'v(p1)', propagatorA, 'vbar(p3)', vertexB, 'u(p4)']
    # print('*'.join(amp))

    # amp1 = feyn_rules.FeynRules(model, incoming_particles, outgoing_particles, internal_particles)
    # print('Incoming wavefunction for {}: {}'.format(particles['e-'].name, particles['e-'].wavefunction[0]))
    # print("\n")
    # print(amp1.amplitude())
    # print("\n")
    # print(amp1._get_vertex(outgoing_particles, internal_particles))
    # amp2 = feyn_rules.FeynRules(model, incoming_particles, outgoing_particles, [particles[23]])
    # print(amp2.amplitude())
    
    Amp1 = feyn_rules.FeynRules(model, InP, OutP, IntP)
    print(Amp1.amplitude())
    
    #Zboson = Particle(model, 23, 0, False)
    #Amp2 = feyn_rules.FeynRules(model, InP, OutP, [Zboson])
    #print(Amp2.amplitude())

if __name__ == '__main__':
    main()
