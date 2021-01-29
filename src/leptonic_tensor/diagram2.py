import numpy as np
import yaml
import argparse
import lorentz_structures as ls
import models
import model_class as mc


all_models = models.discover_models()
model = mc.Model('Models.SM_NLO', all_models)

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


def ctz(inp):
    return (inp & -inp).bit_length() - 1


def next_permutation(inp):
    t = inp | (inp - 1)
    return (t+1) | (((~t & -~t) - 1) >> (ctz(inp) + 1))


def set_bits(inp, setbits, size):
    iset = 0
    for i in range(size):
        if(inp & (1 << i)):
            setbits[iset] = (inp & (1 << i))
            iset += 1


class Propagator:
    def __init__(self, particle):
        self.particle = particle
        if self.particle.is_vector:
            if self.particle.massless() == 0:
                self.denominator = lambda p: p[0]*p[0]-np.sum(p[1:]*p[1:])
                self.numerator = lambda p: -ls.METRIC_TENSOR
            else:
                self.denominator = lambda p: p[0]*p[0]-np.sum(p[1:]*p[1:])-91.18**2-1j*91.18*2.54
                self.numerator = lambda p: \
                    -ls.METRIC_TENSOR + np.outer(p, p)/91.18**2

    def __call__(self, p):
        return self.numerator(p)/self.denominator(p)
            
    def __str__(self):
        return "Prop{}".format(self.particle)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.particle == other.particle


class Current:
    def __init__(self, current=''):
        self.current = current

    def eps(self, p):
        self.current = 'eps({})'.format(p)
        return Current(self.current)

    def add_vertex(self, v, j1, j2):
        self.current = self.current + '+' + '*'.join([str(v), str(j1), str(j2)])
        return Current(self.current)

    def finalize(self):
        self.current = '(' + self.current + ')'
        return Current(self.current)

    def vertex(self, v, j1, j2):
        self.current = '(' + self.current + '+' + '*'.join([str(v), str(j1), str(j2)]) + ')'
        return Current(self.current)

    def prop(self, prop):
        self.current = '*'.join([prop, self.current])

    def add(self, j):
        print('Add', self.current, j)
        self.current = '+'.join([str(self.current), str(j)])
        print('Add end', self.current)
        return Current(self.current)

    def __str__(self):
        return self.current

    def __repr__(self):
        return str(self)

def type_sort(part1, part2, current_part):
    sorted_list = np.empty(3, dtype=int)
    index = {}
    for part in [part1, part2, current_part]:
        if part.is_antifermion():
            sorted_list[0] = part.id
            index[part.id] = 'i'
        elif part.is_fermion():
            sorted_list[1] = part.id
            index[part.id] = 'j'
        elif part.is_vector():
            sorted_list[2] = part.id
            index[part.id] = 'm'
            
    return sorted_list, index
            

class Diagram:
    def __init__(self, particles, mom, hel):
        self.particles = [set([]) for _ in range(Particle.max_id)]
        self.currents = [[] for _ in range(Particle.max_id)]
        self.momentum = [[] for _ in range(Particle.max_id)]
        # print(len(self.particles))
        for i in range(len(particles)):
            self.particles[(1 << i)-1].add(particles[i])
            self.momentum[(1 << i)-1] = mom[i]
            self.currents[(1 << i)-1] = [ls.Spinor(mom[i], hel[i]).u]

        # print(self.currents)

    def symmetry_factor(self, part1, part2, size):
        pi1, pi2 = np.zeros(size, dtype=np.int32), np.zeros(size, dtype=np.int32)
        set_bits(part1.id, pi1, size)
        set_bits(part2.id, pi2, size)
        pi1 = list(pi1[pi1 != 0])
        pi2 = list(pi2[pi2 != 0])
        for idx in pi1:
            particle = self.particles[idx-1]
            particle = list(particle)[0]
            assert idx == particle.id
            if particle.is_vector() or particle.is_scalar():
                pi1.remove(idx)
        for idx in pi2:
            particle = self.particles[idx-1]
            particle = list(particle)[0]
            assert idx == particle.id
            if particle.is_vector() or particle.is_scalar():
                pi2.remove(idx)
        pi = pi1 + pi2
        S = 0
        for i in range(1, len(pi)):
            item_to_insert = pi[i]
            j = i - 1
            while j >= 0 and pi[j] > item_to_insert:
                pi[j+1] = pi[j]
                j -= 1
                S += 1
            pi[j+1] = item_to_insert
        return (-1)**S

    def sub_current(self, cur, iset, nset, setbits, size):
        idx = (1 << iset) - 1
        while idx < (1 << nset - 1):
            cur1 = 0
            for i in range(size):
                cur1 += setbits[i]*((idx >> i) & 1)
            # print('\t- {:07b} {:07b}'.format(cur1, cur ^ cur1))
            cur2 = cur ^ cur1
            if(self.particles[cur1-1] is not None
                    and self.particles[cur2-1] is not None):
                icurrent = 0
                for part1 in self.particles[cur1-1]:
                    for part2 in self.particles[cur2-1]:
                        pids = [part1.pid, part2.pid]
                        for key in model.vertex_map:
                            v = list(key)
                            # ufo_vertex = model.vertex_map[key]
                            pids0 = v.copy()
                            if pids[0] not in pids0:
                                continue
                            pids0.remove(pids[0])
                            if pids[1] not in pids0:
                                continue
                            pids0.remove(pids[1])
                            if len(pids0) == 1:
                                current_part = Particle(cur, pids0[0])
                                self.particles[cur-1].add(current_part)
                                self.momentum[cur-1] = self.momentum[cur1-1] + self.momentum[cur2-1]
                                if all([part1.is_vector(), part2.is_vector(), current_part.is_vector()]):
                                    # do boson stuff.
                                    continue
                                # TODO:
                                # VVV, FFV, FFS, VVS, SSS
                                
                                S_pi = self.symmetry_factor(part1, part2, size)
                                # print("part1 id: {}, part2 id: {}".format(part1.id, part2.id))
                                # print("symmetry factor: ", S_pi)
                                sorted_list, index = type_sort(part1, part2, current_part)
                                # afermion, fermion, boson = sorted_list[0], sorted_list[1], sorted_list[2]
                                sumidx = 'ijm, {}, {} -> {}'.format(index[part1.id], index[part2.id], index[cur])
                                # vertex = ls.Gamma(afermion, fermion, boson)
                                
                                # vert_name = ufo_vertex.name + '({},{},{})'.format(afermion, fermion, boson)
                                if len(self.currents[cur1-1]) > 1:
                                    j1 = self.currents[cur1-1][icurrent]
                                else:
                                    j1 = self.currents[cur1-1][0]
                                if len(self.currents[cur2-1]) > 1:
                                    j2 = self.currents[cur2-1][icurrent]
                                else:
                                    j2 = self.currents[cur2-1][0]
                                # vertex = ls.GAMMA_MU
                                # TODO: Use proper couplings
                                current = S_pi*np.einsum(sumidx, ls.GAMMA_MU, j1, j2)
                                #print('current:',current)
                                if cur+1 != 1 << (self.nparts - 1):
                                    prop = Propagator(current_part)(self.momentum[cur-1])
                                    if current_part.is_fermion():
                                        current = np.einsum('ij,j->i', prop, current)
                                    elif current_part.is_antifermion():
                                        current = np.einsum('ji,j->i', prop, current)
                                    elif current_part.is_vector():
                                        current = np.einsum('ji,j->i', prop, current)
                                    else:
                                        current = prop*current
                                # print('current:', current, current_part)
                                self.currents[cur-1].append(current)
                        icurrent += 1

            idx = next_permutation(idx)

    def generate_currents(self, m, nparts):
        val = (1 << m) - 1
        setbits = np.zeros(nparts-1, dtype=np.int32)
        self.nparts = nparts
        while val < (1 << (nparts - 1)):
            # print('Permutation: {:07b}'.format(val))
            set_bits(val, setbits, nparts-1)
            for iset in range(1, m):
                self.sub_current(val, iset, m, setbits, nparts-1)

            val = next_permutation(val)

class Particle:
    max_id = 0

    def __init__(self, i, pid):
        self.id = i
        self.pid = pid

    def __str__(self):
        sid = self.get_id()
        return f'({self.id}, {self.pid})'
    
    def conjugate(self):
        pid = self.pid
        part = model.particle_map[pid]
        if part.name != part.antiname:
            pid = -self.pid
        return Particle(self.id, pid)

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

    def __hash__(self):
        return self.pid

    def get_id(self):
        if self.id >= Particle.max_id:
            return PARTMAP[self.id]
        return self.id

    def __eq__(self, other):
        return (self.get_id() == other.get_id()
                and abs(self.pid) == abs(other.pid))

    def is_fermion(self):
        if 0 < self.pid < 20:
            return True
        return False

    def is_antifermion(self):
        if -20 < self.pid < 0:
            return True
        return False

    def is_vector(self):
        if 20 < self.pid and self.pid != 25:
            return True
        return False

    def is_scalar(self):
        if self.pid == 25:
            return True
        return False

    def massless(self):
        return model.particle_map[self.pid].mass.name != 'ZERO'


def main(run_card):
    with open(run_card) as stream:
        parameters = yaml.safe_load(stream)

    particles_yaml = parameters['Particles']
    nparts = len(particles_yaml)
    Particle.max_id = 1 << (nparts - 1)

    for i in range(Particle.max_id << 1):
        PARTMAP[i] = binary_conj(i, nparts)

    particles = []
    uid = 1
    for particle in particles_yaml:
        particle = particle['Particle']
        pid = particle[0]
        part = model.particle_map[pid]
        if particle[1] == 'in' and part.name != part.antiname:
            pid = -pid
        particles.append(Particle(uid, pid))
        uid <<= 1

    phi = 0
    costheta = 0
    sintheta = np.sqrt(1-costheta**2)
    ecm_array = np.linspace(20, 180, 1701)

    # TODO:
    # proper phase space and integration
    # Average over initial state helicities and sum over final state
    for ecm in ecm_array:
        mom = ecm/2*np.array(
                [[1, 0, 0, 1],
                 [1, 0, 0, -1],
                 [1, sintheta*np.cos(phi),
                  sintheta*np.sin(phi), costheta],
                 [1, -sintheta*np.cos(phi),
                  -sintheta*np.sin(phi), -costheta]])

        diagram = Diagram(particles, mom, [1, -1, 1, -1])

        for i in range(2, nparts):
            diagram.generate_currents(i, nparts)

        # print("Diagram current[-2]: ", np.array(diagram.currents[-2]))
        # print(np.sum(diagram.currents[-2], axis=0), diagram.currents[-1][0])
        final_curr = np.einsum('i,i->', np.sum(diagram.currents[-2], axis=0), diagram.currents[-1][0])
        # print("Final amplitude: ", final_curr)
        print("ecm {}, Final matrix^2: ".format(ecm), np.linalg.norm(final_curr))

    amp = lambda p: diagram.currents[-2](p) # Function of momentum
    #print("Diagram momentum: ", diagram.momentum)
    #print("Diagram currents: ", diagram.currents)
    #print("Diagram particles: ", diagram.particles)
    #print("particles: ", particles)
    #print("PARTMAP:", PARTMAP)

    # Generate phase space
    # Gives a set of momentum
    # current_amp = amp(momentum)
    # lmunu = np.outer(current_amp, np.conj(current_amp))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_card', default='run_card.yml',
                        help='Input run card')
    args = parser.parse_args()

    main(args.run_card)
