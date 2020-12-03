import numpy as np
import collections
import models
import model_class as mc
import lorentz_class as lc

all_models = models.discover_models()
model = mc.Model('Models.SM_NLO', all_models)

VERTICES = [[11, -11, 22], [13, -13, 22], [11, -12, 24], [-11, 12, -24],
            [11, -11, 23], [13, -13, 23], [12, -12, 23], [24, -24, 22],
            [24, -24, 23]]

PARTMAP = {}


def binary_conj(x, size):
    vals = []
    for y in bin(x)[2:].zfill(size):
        if y == '1':
            vals.append('0')
        else:
            vals.append('1')
    return int(''.join(vals), 2)


class Particle:
    max_id = 0

    def __init__(self, i, pid, direction, spin):
        self.id = i
        self.pid = pid
        self.direction = direction
        self.spin = spin

    def __str__(self):
        sid = self.get_id()
        return f'({sid}, {self.pid})'

    def conjugate(self):
        pid = self.pid
        if self.pid not in (22, 23):
            pid = -self.pid
        return Particle(self.id, pid, -self.direction, self.spin)

    def __repr__(self):
        return str(self)

    def __lt__(self, other):
        sid = self.get_id()
        oid = other.get_id()
        if sid < oid:
            return True
        elif oid < sid:
            return False
        return self.pid < other.pid

    def get_id(self):
        if self.id >= Particle.max_id:
            return PARTMAP[self.id]
        return self.id

    def __eq__(self, other):
        return (self.get_id() == other.get_id()
                and abs(self.pid) == abs(other.pid))


class Vertex:
    def __init__(self, particles):
        self.particles = particles
        pids = [particle.pid for particle in particles]
        spins = [particle.spin for particle in particles]
        indices = [particle.id for particle in particles]
        self.pids = np.sort(pids)
        self.spins = np.sort(spins)
        self.ufo_vertex = model.vertex_map[tuple(self.pids)]
        
        self.name = [ltz.name for ltz in self.ufo_vertex.lorentz]
        try:
            self.lorentz = [lc.Lorentz(model, self.spins, nme, indices) for nme in self.name]
            self.structure = [ltz.structure for ltz in self.lorentz]
            self.indices = [ltz.indices for ltz in self.lorentz]
            self.tensor = [ltz.tensor for ltz in self.lorentz]
        except:
            pass
        # print(self.structure)

    def __str__(self):
        if hasattr(self, 'indices'):
            return "V("+', '.join([str(v.get_id()) for v in self.particles])+"):{}".format(self.indices)
            # return "V("+', '.join([str(v.get_id()) for v in self.particles])+")"
        else:
            return "V("+', '.join([str(v.get_id()) for v in self.particles])+"):{}".format(self.ufo_vertex)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        sids = {}
        spids = {}
        if len(self.particles) != len(other.particles):
            return False
        for particle in self.particles:
            if particle.get_id() not in sids:
                sids[particle.get_id()] = 1
                spids[abs(particle.pid)] = 1
            else:
                sids[particle.get_id()] += 1
                spids[abs(particle.pid)] += 1

        oids = {}
        opids = {}
        for particle in other.particles:
            if particle.get_id() not in oids:
                oids[particle.get_id()] = 1
                opids[abs(particle.pid)] = 1
            else:
                oids[particle.get_id()] += 1
                opids[abs(particle.pid)] += 1

        return oids == sids and opids == spids


class Propagator:
    def __init__(self, particle):
        self.particle = particle

    def __str__(self):
        return "P{}".format(self.particle)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.particle == other.particle


class Diagram:
    def __init__(self, external):
        self.vertices = []
        self.propagators = []
        self.external = external
        self.free = external

    def copy(self):
        diagram = Diagram(self.external.copy())
        diagram.vertices = self.vertices.copy()
        diagram.propagators = self.propagators.copy()
        diagram.free = self.free.copy()
        return diagram

    def add_vertex(self, vertex):
        vert = Vertex(vertex)
        self.vertices.append(vert)

    def add_propagator(self, propagator):
        prop = Propagator(propagator)
        self.propagators.append(prop)

    def __eq__(self, other):
        vert_bool = True
        prop_bool = True
        for svert in self.vertices:
            if svert not in other.vertices:
                return False
        for sprop in self.propagators:
            if sprop not in other.propagators:
                return False
        return (vert_bool and prop_bool
                and self.external == other.external)

    def __call__(self, momentums):
        pass
        # Create external currents:
        # - J(1, mom1)
        # - J(2, mom2)
        # - J(4, mom3)
        # etc.

        # for vertex in self.vertices:
        #     V(i, j, k) => V(i, j, k, ..., l, m)
        #     Symmetry factors
        #     J(k) = V(i, j, k, ..., l, m) * J(i) * J(j) * J(k) * ... * J(l)
        #     Calculate momenutm for particle k (p(i) + p(j))
        #     if k in self.propagators:
        #         J(k) *= Prop(p(k))


def AddVertex(diagram):
    particles = diagram.free
    diagrams = []
    for i in range(len(particles)-1):
        for j in range(i+1, len(particles)):
            for v in VERTICES:
                if particles[i].pid in v and particles[j].pid in v:
                    new_particles = particles.copy()
                    pids = list(set(v)
                                - set((particles[i].pid, particles[j].pid)))
                    if len(pids) > 1:
                        continue
                    pid = pids[0]
                    new_id = particles[i].id+particles[j].id
                    new_part = model.particle_map[pid]
                    new_particles[i] = Particle(new_id, pid, -1, new_part.spin)
                    #print(new_particles[i], new_part.spin)
                    new_particles.remove(new_particles[j])
                    new_diagram = diagram.copy()
                    new_vertex = [particles[i], particles[j], new_particles[i]]
                    #print(new_vertex)
                    #print([part.spin for part in new_vertex])
                    new_diagram.add_vertex(new_vertex)
                    new_diagram.add_propagator(new_particles[i])
                    new_particles[i] = new_particles[i].conjugate()
                    new_diagram.free = new_particles
                    diagrams.append(new_diagram)

    return diagrams


def main():
    # Spin convention in UFO is 2s+1, but we're using 2s. 12/3/2020 back to 2S+1.
    # particles = [Particle(1, -11, 1, 2), Particle(2, 11, 1, 2),
    #               Particle(4, 22, 1, 3), Particle(8, 23, 1, 3),
    #               Particle(16, 12, 1, 2), Particle(32, -12, 1, 2)]
    
    # particles = [Particle(1, -11, 1), Particle(2, 11, 1),
    #              Particle(4, 22, 1), Particle(8, 22, 1),
    #              Particle(16, 23, 1)]
    
    particles = [Particle(1, -11, 1, 2), Particle(2, 11, 1, 2),
                  Particle(4, -13, 1, 2), Particle(8, 13, 1, 2)]
    
    Particle.max_id = 1 << (len(particles)-1)
    for i in range(Particle.max_id << 1):
        PARTMAP[i] = binary_conj(i, len(particles))
    print(PARTMAP)
    diagrams = [Diagram(particles)]
    final_diagrams = []
    ndiagrams = 0
    while len(diagrams) > 0:
        tmp = AddVertex(diagrams.pop(0))
        for config in tmp:
            if len(config.free) != 2:
                diagrams.append(config)
            else:
                if((config.free[0].pid in (22, 23)
                    and config.free[0].pid == config.free[1].pid)
                   or config.free[0].pid == -config.free[1].pid):
                    config.propagators.pop()
                    if config not in final_diagrams:
                        final_diagrams.append(config)
                        ndiagrams += 1
        # print(particles_set)
    print(ndiagrams)
    for diagram in final_diagrams:
        print(diagram.vertices, diagram.propagators)


if __name__ == '__main__':
    main()
