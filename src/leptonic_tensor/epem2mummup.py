import models
import itertools
import particle
import feyn_rules


class Model:
    def __init__(self, name, all_models):
        self.name = name
        self.model = all_models[name]
        self.particle_map = self._particle_map()
        self.vertex_map = self._vertex_map()
        self.propagator_map = self._propagator_map()
        self.couplings = self.model.all_couplings

    def _particle_map(self):
        particles = self.model.all_particles
        part_map = {}
        for part in particles:
            if part.spin < 0:
                continue
            test = particle.ParticleInfo(part)
            part_map[test.name] = test

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

    @property
    def particles(self):
        return (', ').join(list(self.particle_map.keys()))


def main():
    all_models = models.discover_models()
    model = Model('Models.SM_NLO', all_models)

    # Setup the incoming, outgoing, and internal particles
    particles = model.particle_map
    incoming_particles = [particles['e+'], particles['e-']]
    outgoing_particles = [particles['mu+'], particles['mu-']]
    internal_particles = [particles['a']]

    pids1 = [particles['e+'].pid,
             particles['e-'].pid,
             particles['a'].pid]
    pids1.sort()
    print(pids1)
    vertex1 = model.vertex_map[tuple(pids1)]
    print(vertex1.lorentz[0].structure, vertex1.couplings[(0, 0)].value)
    vertexA = vertex1.lorentz[0].structure.replace('3', 'mu').replace('2', 'e-').replace('1', 'e+')

    pids2 = [particles['mu+'].pid,
             particles['mu-'].pid,
             particles['a'].pid]
    pids2.sort()
    print(pids2)
    vertex2 = model.vertex_map[tuple(pids2)]
    print(vertex2.lorentz[0].structure, vertex2.couplings[(0, 0)].value)
    vertexB = vertex2.lorentz[0].structure.replace('3', 'nu').replace('2', 'mu-').replace('1', 'mu+')

    propagator = model.propagator_map[particles['a'].propagator]
    propagatorA = propagator.numerator.replace('1','mu').replace('2','nu') + '/' + propagator.denominator
    print("({})/({})".format(propagator.numerator, propagator.denominator))

    amp = ['ubar(p2)', vertexA, 'v(p1)', propagatorA, 'vbar(p3)', vertexB, 'u(p4)']
    print('*'.join(amp))

    amp1 = feyn_rules.FeynRules(model, incoming_particles, outgoing_particles, internal_particles)
    #print('Incoming wavefunction for {}: {}'.format(particles['e-'].name, particles['e-'].wavefunction[0]))
    print("\n")
    print(amp1.amplitude())
    #print(amp1._get_vertex(outgoing_particles, internal_particles))
    amp2 = feyn_rules.FeynRules(model, incoming_particles, outgoing_particles, [particles['Z']])
    print(amp2.amplitude())

if __name__ == '__main__':
    main()
