import numpy as np
import yaml
import argparse
import models
import model_class as mc
import lorentz_class as lc
import propagator_class as prop

all_models = models.discover_models()
model = None

VERTICES = [[11, -11, 22], [13, -13, 22], [11, -12, 24], [-11, 12, -24],
            [11, -11, 23], [13, -13, 23], [12, -12, 23], [24, -24, 22],
            [24, -24, 23], [24, -24, 22, 22]]

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

    def __init__(self, i, pid, spin):
        self.id = i
        self.pid = pid
        self.spin = spin

    def __str__(self):
        sid = self.get_id()
        return f'({self.id}, {self.pid})'

    def conjugate(self):
        pid = self.pid
        part = model.particle_map[pid]
        if part.name != part.antiname:
            pid = -self.pid
        return Particle(self.id, pid, self.spin)

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
            return "V("+', '.join([str(v.id) for v in self.particles])+"):{}".format(self.indices)
            # return "V("+', '.join([str(v.get_id()) for v in self.particles])+")"
        else:
            return "V("+', '.join([str(v.id) for v in self.particles])+"):{}".format(self.ufo_vertex)

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
        ufo_part = model.particle_map[self.particle.pid]
        mass = ufo_part.mass
        width = ufo_part.width
        id1 = self.particle.id
        id2 = PARTMAP[id1]
        self.denominator = f'(P({id1})^2-{mass}^2' \
                           f'-1j*{mass}*{width})'
        if ufo_part.propagator == 'S':
            self.numerator = '1'
        elif ufo_part.propagator == 'F':
            self.numerator = (f'(P({id1}, mu)*Gamma(mu, {id1}, {id2})'
                              f'+{mass}*Identity(i, j))')
        elif ufo_part.propagator == 'V1':
            self.numerator = (f'(-Metric({id1}, {id2}) '
                              f'+ P({id1}, {id1}) '
                              f'* P({id2}, {id2})/{mass}^2)')
        elif ufo_part.propagator == 'V2':
            self.numerator = f'-Metric({id1}, {id2})'

    def __str__(self):
        return "(P{}, {})".format(self.particle, self.numerator+'/'+self.denominator)

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


max_level = 4


def GenerateVertex(vertex, pid, diagram, diagrams):
    particles = diagram.free
    new_particles = particles.copy()
    idx = particles.index(vertex[0])
    print(idx, vertex, particles)
    for particle in vertex[:0:-1]:
        print(particle)
        new_particles.remove(particle)
    new_id = sum([x.id for x in vertex])
    print(new_id)
    new_particles[idx] = Particle(new_id, pid)
    print(new_particles)
    new_diagram = diagram.copy()
    vertex.append(new_particles[idx])
    print(vertex)
    new_diagram.add_vertex(vertex)
    new_diagram.add_propagator(new_particles[idx])
    new_particles[idx] = new_particles[idx].conjugate()
    new_diagram.free = new_particles
    diagrams.append(new_diagram)


def MakeVertices(final_vertex, vertex, diagram, diagrams, pos=0, level=0):
    particles = diagram.free
    if level == max_level:
        return None
    if len(vertex) == 1:
        GenerateVertex(final_vertex, vertex[0], diagram, diagrams)
        old_pid = final_vertex.pop()
        vertex.append(old_pid)
        return
    for i in range(pos+1, len(particles)):
        cvertex = vertex.copy()
        if particles[i].pid not in cvertex:
            continue
        cvertex.remove(particles[i].pid)
        final_vertex.append(particles[i])
        MakeVertices(final_vertex, cvertex, diagram, diagrams, i, level+1)
    return None


def AddVertex(diagram):
    particles = diagram.free
    diagrams = []
    for v in VERTICES:
        # print(v)
        # vertex = []
        # vertex = MakeVertices(vertex, v, diagram, diagrams)
        # if vertex is not None:
        #     print(vertex)
        for i in range(len(particles)-1):
            pids0 = v.copy()
            # print('here1', pids0)
            if particles[i].pid not in v:
                continue
            pids0.remove(particles[i].pid)
            # print('here2', pids0)
            for j in range(i+1, len(particles)):
                pids = pids0.copy()
                if particles[j].pid not in pids:
                    continue
                pids.remove(particles[j].pid)
                # print('here3', pids, particles[j].pid)
                # pids = v.copy()
                new_particles = particles.copy()
                # pids.remove(particles[i].pid)
                # print('here4', pids, len(pids))
                if len(pids) > 1:
                    # Handle 4 point
                    for k in range(j+1, len(particles)):
                        if particles[k].pid in pids:
                            pids.remove(particles[k].pid)
                            if len(pids) != 1:
                                continue
                            pid = pids[0]
                            new_id = particles[i].id+particles[j].id+particles[k].id
                            new_part = model.particle_map[pid]
                            new_particles[i] = Particle(new_id, pid, new_part.spin)
                            new_particles.remove(new_particles[k])
                            new_particles.remove(new_particles[j])
                            new_diagram = diagram.copy()
                            new_vertex = [particles[i], particles[j], particles[k], new_particles[i]]
                            new_diagram.add_vertex(new_vertex)
                            new_diagram.add_propagator(new_particles[i])
                            new_particles[i] = new_particles[i].conjugate()
                            new_diagram.free = new_particles
                            diagrams.append(new_diagram)
                    if len(pids) > 1:
                        continue
                else:
                    # print('here5', pids)
                    pid = pids[0]
                    new_id = particles[i].id+particles[j].id
                    if new_id >= 2*Particle.max_id:
                        continue
                    new_part = model.particle_map[pid]
                    new_particles[i] = Particle(new_id, pid, new_part.spin)
                    new_particles.remove(new_particles[j])
                    new_diagram = diagram.copy()
                    new_vertex = [particles[i], particles[j], new_particles[i]]
                    new_diagram.add_vertex(new_vertex)
                    new_diagram.add_propagator(new_particles[i])
                    new_particles[i] = new_particles[i].conjugate()
                    new_diagram.free = new_particles
                    diagrams.append(new_diagram)
    return diagrams

def SymmetryFactor(diagram):
    fermion_indices = []
    for vertex in diagram.vertices:
        for index in vertex.indices[0]:
            if index.lorentz == False:
                fermion_indices.append(index.index)
    S = 0
    for i in range(1, len(fermion_indices)):
        item_to_insert = fermion_indices[i]
        j = i - 1
        while j >= 0 and fermion_indices[j] > item_to_insert:
            fermion_indices[j+1] = fermion_indices[j]
            j -= 1
            S += 1
        fermion_indices[j+1] = item_to_insert
    return (-1)**S    

def main(run_card):
    with open(run_card) as stream:
        parameters = yaml.safe_load(stream)

    global model
    model_name = 'Models.{}'.format(parameters['Model'])
    model = mc.Model(model_name, all_models)

    particles_yaml = parameters['Particles']
    Particle.max_id = 1 << (len(particles_yaml)-1)

    for i in range(Particle.max_id << 1):
        PARTMAP[i] = binary_conj(i, len(particles_yaml))

    particles = []
    uid = 1
    for particle in particles_yaml:
        particle = particle['Particle']
        pid = particle[0]
        part = model.particle_map[pid]
        if particle[1] == 'in' and part.name != part.antiname:
            pid = -pid
        particles.append(Particle(uid, pid, part.spin))
        uid <<= 1

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
        print(diagram.vertices, diagram.propagators, SymmetryFactor(diagram))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_card', default='run_card.yml',
                        help='Input run card')
    args = parser.parse_args()

    main(args.run_card)
