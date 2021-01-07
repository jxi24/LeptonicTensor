import numpy as np
import yaml
import argparse

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


class Current:
    def __init__(self, current=''):
        self.current = current

    def eps(self, p):
        self.current = f'eps({p})'
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


class Diagram:
    def __init__(self, particles):
        self.particles = [set([]) for _ in range(Particle.max_id)]
        self.currents = [[] for _ in range(Particle.max_id)]
        print(len(self.particles))
        # for i in range(len(self.currents)):
        #     self.currents[i] = Current()
        for i in range(len(particles)):
            self.particles[(1 << i)-1].add(particles[i])
            self.currents[(1 << i)-1].append(Current().eps((1 << i)-1))

        print(self.currents)

    def sub_current(self, cur, iset, nset, setbits, size):
        idx = (1 << iset) - 1
        while idx < (1 << nset - 1):
            cur1 = 0
            for i in range(size):
                cur1 += setbits[i]*((idx >> i) & 1)
            print('\t- {:07b} {:07b}'.format(cur1, cur ^ cur1))
            cur2 = cur ^ cur1
            # print(self.particles[cur1-1], self.particles[cur2-1])
            if(self.particles[cur1-1] is not None
                    and self.particles[cur2-1] is not None):
                icurrent = 0
                for part1 in self.particles[cur1-1]:
                    for part2 in self.particles[cur2-1]:
                        pids = [part1.pid, part2.pid]
                        # print(pids)
                        for v in VERTICES:
                            pids0 = v.copy()
                            if pids[0] not in pids0:
                                continue
                            pids0.remove(pids[0])
                            if pids[1] not in pids0:
                                continue
                            pids0.remove(pids[1])
                            if len(pids0) == 1:
                                self.particles[cur-1].add(Particle(cur, pids0[0]))
                                vert_name = 'V_77' if 22 in v else 'V_117'
                                if len(self.currents[cur1-1]) > 1:
                                    j1 = self.currents[cur1-1][icurrent]
                                else:
                                    j1 = self.currents[cur1-1][0]
                                if len(self.currents[cur2-1]) > 1:
                                    j2 = self.currents[cur2-1][icurrent]
                                else:
                                    j2 = self.currents[cur2-1][0]
                                current = '*'.join([vert_name, str(j1), str(j2)])
                                if cur+1 != 1 << (self.nparts - 1):
                                    current = current + '*Prop({})'.format(cur)
                                self.currents[cur-1].append(current)
                        icurrent += 1
                        # print(self.particles[cur1-1], self.particles[cur2-1])
                        # print(self.currents)

            idx = next_permutation(idx)

    def generate_currents(self, m, nparts):
        val = (1 << m) - 1
        setbits = np.zeros(nparts-1, dtype=np.int32)
        self.nparts = nparts
        while val < (1 << (nparts - 1)):
            print('Permutation: {:07b}'.format(val))
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
        pid = -self.pid if self.pid not in (22, 23) else self.pid
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

    def __hash__(self):
        return self.pid

    def get_id(self):
        if self.id >= Particle.max_id:
            return PARTMAP[self.id]
        return self.id

    def __eq__(self, other):
        return (self.get_id() == other.get_id()
                and abs(self.pid) == abs(other.pid))


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
        if particle[1] == 'in' and pid not in (22, 23):
            pid = -pid
        particles.append(Particle(uid, pid))
        uid <<= 1

    diagram = Diagram(particles)

    for i in range(2, nparts):
        diagram.generate_currents(i, nparts)

    print(diagram.currents)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_card', default='run_card.yml',
                        help='Input run card')
    args = parser.parse_args()

    main(args.run_card)
