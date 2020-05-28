import models
import itertools
import particle


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
            for particle in particles:
                pids.append(particle.pdg_code)
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

    pids2 = [particles['mu+'].pid,
             particles['mu-'].pid,
             particles['a'].pid]
    pids2.sort()
    print(pids2)
    vertex2 = model.vertex_map[tuple(pids2)]
    print(vertex2.lorentz[0].structure, vertex2.couplings[(0, 0)].value)

    propagator = model.propagator_map[particles['a'].propagator]
    print("({})/({})".format(propagator.numerator, propagator.denominator))


if __name__ == '__main__':
    main()
