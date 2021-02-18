import numpy as np
import yaml
import argparse
import lorentz_structures as ls
import models
import model_class as mc
import matplotlib.pyplot as plt


all_models = models.discover_models()
model = mc.Model('Models.SM_NLO', all_models)

PARTMAP = {}


def Pt(mom):
    return np.sqrt(mom[:, 1]**2 + mom[:, 2]**2)


def CosTheta(mom):
    return np.cos(np.arctan2(Pt(mom), mom[:, 3]))


def Dot(mom1, mom2):
    return (mom1[:, 0]*mom2[:, 0])[:, None]-np.sum(mom1[:, 1:]*mom2[:, 1:],
                                                   axis=-1, keepdims=True)


class Rambo:
    def __init__(self, nin, nout):
        self.nin = nin
        self.nout = nout
        pi2log = np.log(np.pi/2.)
        Z = np.zeros(nout+1)
        Z[2] = pi2log
        for k in range(3, nout+1):
            Z[k] = Z[k-1]+pi2log-2.*np.log(k-2)
        for k in range(3, nout+1):
            Z[k] -= np.log(k-1)
        self.Z_N = Z[nout]

    def cut(self, p):
        cosThetaMax = 0.4
        return np.where(abs(CosTheta(p[:, 2, :])) < cosThetaMax,
                        np.ones(p.shape[0]),
                        np.zeros(p.shape[0]))

    def __call__(self, p, rans):
        sump = np.zeros((rans.shape[0], 4))
        for i in range(self.nin):
            sump += p[:, i, :]
        ET = np.sqrt(Dot(sump, sump))

        R = np.zeros((rans.shape[0], 4))
        for i in range(self.nin, self.nin+self.nout):
            ctheta = 2*rans[:, 4*(i-self.nin)] - 1
            stheta = np.sqrt(1-ctheta**2)
            phi = 2*np.pi*rans[:, 1 + 4*(i-self.nin)]
            Q = -np.log(rans[:, 2+4*(i-self.nin)]*rans[:, 3+4*(i-self.nin)])
            p[:, i, :] = np.array([Q, Q*stheta*np.sin(phi), Q*stheta*np.cos(phi), Q*ctheta]).T
            R += p[:, i, :]

        RMAS = np.sqrt(Dot(R, R))
        B = -R[:, 1:]/RMAS
        G = R[:, 0, None]/RMAS
        A = 1.0/(1.0+G)
        X = ET/RMAS

        for i in range(self.nin, self.nin+self.nout):
            e = p[:, i, 0, None]
            BQ = np.sum(B*p[:, i, 1:], axis=-1, keepdims=True)
            term1 = (G*e)+BQ
            term2 = B*(e+A*BQ)
            p[:, i, 0, None] = X*term1
            p[:, i, 1:] = X*(p[:, i, 1:] + term2)


        wgt = np.exp((2*self.nout-4)*np.log(ET)+self.Z_N)/(2*np.pi)**(self.nout*3 - 4)
        return self.cut(p)[:, None]*wgt


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
                self.denominator = lambda p: (p[:, 0]*p[:, 0]-np.sum(p[:, 1:]*p[:, 1:], axis=-1))[:, np.newaxis, np.newaxis]
                self.numerator = lambda p: -1j*ls.METRIC_TENSOR[np.newaxis, ...]
            else:
                mass = particle.mass
                width = particle.width
                self.denominator = lambda p: (p[:, 0]*p[:, 0]-np.sum(p[:, 1:]*p[:, 1:], axis=-1)-mass**2-1j*mass*width)[:, np.newaxis, np.newaxis]
                self.numerator = lambda p: \
                    -1j*ls.METRIC_TENSOR[np.newaxis, ...] + 1j*np.einsum('bi,bj->bij', p, p)/mass**2

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
    
class Vertex:
    def __init__(self, ufo_vertex, mom=None):
        self.lorentz_structures = ufo_vertex.lorentz
        self.couplings = ufo_vertex.couplings
        self.n_lorentz = len(self.lorentz_structures)
        self.mom = mom
        self.coupling_matrix = self._cp_matrix()
        self.lorentz_vector = self._lorentz_vector()
        self.vertex = self._get_vertex()
        # print(ufo_vertex.name, self.coupling_matrix, self.couplings)
        
    def _cp_matrix(self):
        cp_matrix = np.zeros((self.n_lorentz, self.n_lorentz), dtype=np.complex128)
        for key, coup in zip(self.couplings.keys(), self.couplings.values()):
            cp_matrix[key[0], key[1]] = model.coupling_map[coup.name]
        return cp_matrix
    
    def _lorentz_vector(self):
        lorentz_vector = np.zeros((self.n_lorentz,4,4,4), dtype=np.complex128)
        for i in range(self.n_lorentz):
            ufo_lorentz = self.lorentz_structures[i]
            if ufo_lorentz.name == 'FFV1':
                lorentz_vector[i] = ls.FFV1
            elif ufo_lorentz.name == 'FFV2':
                lorentz_vector[i] = ls.FFV2
            elif ufo_lorentz.name == 'FFV3':
                lorentz_vector[i] = ls.FFV3
            elif ufo_lorentz.name == 'VVV1':
                lorentz_vector[i] = ls.VVV('VVV1', self.mom).lorentz_tensor
            elif ufo_lorentz.name == 'VVV2':
                lorentz_vector[i] = ls.VVV('VVV2', self.mom).lorentz_tensor
            elif ufo_lorentz.name == 'VVV3':
                lorentz_vector[i] = ls.VVV('VVV3', self.mom).lorentz_tensor
            elif ufo_lorentz.name == 'VVV4':
                lorentz_vector[i] = ls.VVV('VVV4', self.mom).lorentz_tensor
            elif ufo_lorentz.name == 'VVV5':
                lorentz_vector[i] = ls.VVV('VVV5', self.mom).lorentz_tensor
            elif ufo_lorentz.name == 'VVV6':
                lorentz_vector[i] = ls.VVV('VVV6', self.mom).lorentz_tensor
            elif ufo_lorentz.name == 'VVV7':
                lorentz_vector[i] = ls.VVV('VVV7', self.mom).lorentz_tensor
            elif ufo_lorentz.name == 'VVV8':
                lorentz_vector[i] = ls.VVV('VVV8', self.mom).lorentz_tensor
            elif ufo_lorentz.name == 'VVS1':
                lorentz_vector[i] = ls.VVS1
            elif ufo_lorentz.name == 'FFS1':
                lorentz_vector[i] = ls.FFS1
            elif ufo_lorentz.name == 'FFS2':
                lorentz_vector[i] = ls.FFS2
            elif ufo_lorentz.name == 'FFS3':
                lorentz_vector[i] = ls.FFS3
            elif ufo_lorentz.name == 'FFS4':
                lorentz_vector[i] = ls.FFS4
            elif ufo_lorentz.name == 'SSS1':
                lorentz_vector[i] = ls.SSS1
        return lorentz_vector
    def _get_vertex(self):                            
        vertex = np.sum(np.einsum("ij,jklm->iklm", self.coupling_matrix, self.lorentz_vector), axis=0)
        # print(vertex)
        return vertex

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
            
def alph_sort(part1, part2, current_part, mom1, mom2, mom3):
    part_tuples = [
        (mom1, model.particle_map[part1.pid].name, part1.id),
        (mom2, model.particle_map[part2.pid].name, part2.id),
        (mom3, model.particle_map[current_part.pid].name, current_part.id)
        ]
    part_tuples.sort(key= lambda part_tuple:part_tuple[1])
    mom = [part_tuple[0] for part_tuple in part_tuples]
    vert_indices = {}
    idxs = ['a','b','m']
    for part_tuple in part_tuples:
        vert_indices[part_tuple[2]] = idxs.pop()
    return mom, vert_indices

class Diagram:
    def __init__(self, particles, mom, hel, mode):
        self.particles = [set([]) for _ in range(Particle.max_id)]
        self.currents = [[] for _ in range(Particle.max_id)]
        self.momentum = [[] for _ in range(Particle.max_id)]
        self.mode = mode
        # print(len(self.particles))
        for i in range(len(particles)):
            self.particles[(1 << i)-1].add(particles[i])
            self.momentum[(1 << i)-1] = mom[:, i, :]
            if particles[i].is_fermion():
                self.currents[(1 << i)-1] = [ls.SpinorU(mom[:, i, :], hel[i]).u]
                # print("ext current: spinor {}: {}".format(i, ls.Spinor(mom[:, i, :], hel[i]).u))
            elif particles[i].is_antifermion():
                self.currents[(1 << i)-1] = [ls.SpinorUBar(mom[:, i, :], hel[i]).u]
                # print("ext current: spinor bar {}: {}".format(i, ls.Spinor(mom[:, i, :], hel[i], bar=-1).u))
            elif particles[i].is_vector():
                self.currents[(1 << i)-1] = [ls.PolarizationVector(mom[:, i, :], hel[i]).epsilon]
                #print("ext current: epsilon")
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
                            v = list(key) # pids of vertex
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
                                sorted_list, index = type_sort(part1, part2, current_part)
                                ufo_vertex = model.vertex_map[key]
                                if 'VVV' in ufo_vertex.lorentz[0].name:
                                    mom, vert_indices = alph_sort(part1, part2, current_part, self.momentum[cur1-1], self.momentum[cur2-1], self.momentum[cur-1])
                                    vertex_info = Vertex(ufo_vertex, mom)
                                    vertex = vertex_info.vertex
                                    S_pi = 1
                                    sumidx = 'agm, b{}, b{} -> b{}'.format(vert_indices[part1.id], vert_indices[part2.id], vert_indices[current_part.id])

                                elif 'FFV' in ufo_vertex.lorentz[0].name:
                                    vertex_info = Vertex(ufo_vertex)
                                    vertex = vertex_info.vertex
                                    S_pi = self.symmetry_factor(part1, part2, size)
                                    sumidx = 'mij, b{}, b{} -> b{}'.format(index[part1.id], index[part2.id], index[cur])

                                if len(self.currents[cur1-1]) > 1:
                                    j1 = self.currents[cur1-1][icurrent]
                                else:
                                    j1 = self.currents[cur1-1][0]
                                if len(self.currents[cur2-1]) > 1:
                                    j2 = self.currents[cur2-1][icurrent]
                                else:
                                    j2 = self.currents[cur2-1][0]
 
                                current = S_pi*np.einsum(sumidx, vertex, j1, j2)
                                if (cur+1 != 1 << (self.nparts - 1) or self.mode == 'lmunu'):
                                    prop = Propagator(current_part)(self.momentum[cur-1])
                                    if current_part.is_fermion():
                                        current = np.einsum('bij,bj->bi', prop, current)
                                    elif current_part.is_antifermion():
                                        current = np.einsum('bji,bj->bi', prop, current)
                                    elif current_part.is_vector():
                                        current = np.einsum('bij,bj->bi', prop, current)
                                    else:
                                        current = prop*current
                                # print(current)
                                # if(current_part.pid == 23):
                                #     current *= 0

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
    mode = parameters['Mode']
    nparts = len(particles_yaml)
    Particle.max_id = 1 << (nparts - 1)

    # for i in range(Particle.max_id << 1):
    #     PARTMAP[i] = binary_conj(i, nparts)

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
    ecm_array = np.linspace(20, 200, 51)
    results = np.zeros_like(ecm_array)
    hbarc2 = 0.38937966e9 
    alpha = 1/127.9

    # TODO:
    # proper phase space and integration
    # Average over initial state helicities and sum over final state
    # mom = ecm_array[:,np.newaxis,np.newaxis]/2*np.array(
    #             [[[-1, 0, 0, -1],
    #              [-1, 0, 0, 1],
    #              [1, sintheta*np.cos(phi),
    #               sintheta*np.sin(phi), costheta],
    #              [1, -sintheta*np.cos(phi),
    #               -sintheta*np.sin(phi), -costheta]]])

    # for hel1 in range(2):
    #     for hel2 in range(2):
    #         for hel3 in range(2):
    #             for hel4 in range(2):
    #                 diagram = Diagram(particles, mom, [2*hel1-1, 2*hel2-1, 2*hel3-1, 2*hel4-1])
 
    #                 for i in range(2, nparts):
    #                     diagram.generate_currents(i, nparts)

    #                 final_curr = np.einsum('bi,bi->b', np.sum(np.array(diagram.currents[-2]), axis=0), diagram.currents[-1][0])
    #                 results += np.absolute(final_curr)**2/(2*ecm_array**2)/4*hbarc2

    # plt.plot(ecm_array, results)
    # plt.yscale('log')
    # plt.show()

    rambo = Rambo(2, 2)
    nevents = 1
    xsec = np.zeros_like(ecm_array)
    afb = np.zeros_like(ecm_array)
    for i, ecm in enumerate(ecm_array):
        mom = np.array([[-ecm/2, 0, 0, -ecm/2],
                        [-ecm/2, 0, 0, ecm/2],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]], dtype=np.float64)
        mom = np.tile(mom, (nevents, 1, 1))
        rans = np.random.random((nevents, 8))
        weights = rambo(mom, rans)

        results = np.zeros((nevents, 1), dtype=np.float64)
        lmunu = np.zeros((nevents, 4, 4), dtype=np.complex128)
        for hel1 in range(2):
            for hel2 in range(2):
                for hel3 in range(2):
                    for hel4 in range(2):
                        diagram = Diagram(particles, mom, [2*hel1-1, 2*hel2-1, 2*hel3-1, 2*hel4-1], mode)
 
                        for j in range(2, nparts):
                            diagram.generate_currents(j, nparts)

                        final_curr = np.sum(np.array(diagram.currents[-2]), axis=0)
                        if mode == "lmunu":
                            lmunu += np.einsum('bi, bj -> bij', final_curr, final_curr)
                        else:
                            amplitude = np.einsum('bi,bi->b', np.sum(np.array(diagram.currents[-2]), axis=0), diagram.currents[-1][0])
                            results += np.absolute(final_curr[:, None])**2
                        # print(np.einsum('bi,bi->b', diagram.currents[-2][0], diagram.currents[-1][0]))
                        # print(np.einsum('bi,bi->b', diagram.currents[-2][1], diagram.currents[-1][0]))
                        # print(hel1, hel2, hel3, hel4, final_curr, np.absolute(final_curr[:, None])**2)
                        # raise
        cos_theta = CosTheta(mom[:, 2, :])
        # s, t, u = Mandelstam(mom)
        spinavg = 4
        flux = 2*ecm**2
        results = results/flux/spinavg*hbarc2*weights
        exact = 4*np.pi*alpha**2*hbarc2/(3*ecm**2)
        # exact_moller = 8.41905*alpha**2*hbarc2*2*np.pi/(ecm**2)
        # exact_compton = 2*alpha**2*hbarc2*2 
        print(ecm, np.mean(results), np.std(results)/np.sqrt(nevents), exact)
        nfwd = np.sum(results[cos_theta > 0])/nevents
        nbck = np.sum(results[cos_theta < 0])/nevents
        afb[i] = (nfwd - nbck)/np.mean(results)
        print(nfwd, nbck, afb[i])
        nbins = 100
        cos_theta_exact = np.linspace(-0.4, 0.4, nbins+1)
        dsigma_moller = alpha**2*hbarc2*(3+cos_theta_exact**2)**2/(ecm**2)*2*np.pi/(1-cos_theta_exact**2)**2
        # dsigma = alpha**2*hbarc2*(1+cos_theta_exact**2)/(4*ecm**2)*2*np.pi
        # plt.hist(cos_theta, weights=results/nevents/(2/nbins), bins=np.linspace(-1,1,nbins+1))
        # plt.plot(cos_theta_exact, dsigma_moller)
        # plt.show()
        # raise
        xsec[i] = np.mean(results)

    plt.plot(ecm_array, afb)
    # plt.yscale('log')
    plt.show()


if __name__ == '__main__':
    np.random.seed(123456789)
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_card', default='run_card.yml',
                        help='Input run card')
    args = parser.parse_args()

    main(args.run_card)
