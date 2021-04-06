import numpy as np
import scipy.integrate as integrate
import yaml
import argparse
import lorentz_structures as ls
import models
import model_class as mc
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
plt.rcParams.update({"text.usetex": True,})

from histogram import Histogram

from particle import Particle
from propagator import Propagator
from rambo import Rambo
from utils import Pt, CosTheta, Dot, ctz, next_permutation, set_bits


all_models = models.discover_models()
model = mc.Model('Models.SM_NLO', all_models)


class Vertex:
    def __init__(self, ufo_vertex, mom=None):
        self.lorentz_structures = ufo_vertex.lorentz
        self.couplings = ufo_vertex.couplings
        self.n_lorentz = len(self.lorentz_structures)
        self.mom = mom
        self.coupling_matrix = self._cp_matrix()
        self.lorentz_vector = self._lorentz_vector()
        self.vertex = self._get_vertex()
        
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
    idxs = ['a','g','m']
    for part_tuple in part_tuples:
        vert_indices[part_tuple[2]] = idxs.pop()
    return mom, vert_indices

class Diagram:
    def __init__(self, particles, mom, hel, mode):
        self.particles = [set([]) for _ in range(Particle.max_id)]
        self.currents = [[] for _ in range(Particle.max_id)]
        self.momentum = [[] for _ in range(Particle.max_id)]
        self.mode = mode
        for i in range(len(particles)):
            self.particles[(1 << i)-1].add(particles[i])
            self.momentum[(1 << i)-1] = mom[:, i, :]
            if particles[i].is_fermion():
                self.currents[(1 << i)-1] = [ls.SpinorV(mom[:, i, :], hel[i]).u]
            elif particles[i].is_antifermion():
                self.currents[(1 << i)-1] = [ls.SpinorUBar(mom[:, i, :], hel[i]).u]
            elif particles[i].is_vector():
                self.currents[(1 << i)-1] = [np.conjugate(ls.PolarizationVector(mom[:, i, :], hel[i]).epsilon)]

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
                                # print(v, cur, pids0[0])
                                current_part = Particle(cur, pids0[0])
                                # print(current_part, cur)
                                if(current_part.pid == 23):
                                    continue
                                self.momentum[cur-1] = self.momentum[cur1-1] + self.momentum[cur2-1]
                                sorted_list, index = type_sort(part1, part2, current_part)
                                # print(index)
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
                                    # print(vertex_info.coupling_matrix)
                                    S_pi = self.symmetry_factor(part1, part2, size)
                                    sumidx = 'mij, b{}, b{} -> b{}'.format(index[part1.id], index[part2.id], index[cur])
                                    # print(sumidx)
                                if len(self.currents[cur1-1]) > 1:
                                    j1 = self.currents[cur1-1][icurrent]
                                else:
                                    j1 = self.currents[cur1-1][0]
                                if len(self.currents[cur2-1]) > 1:
                                    j2 = self.currents[cur2-1][icurrent]
                                else:
                                    j2 = self.currents[cur2-1][0]
 
    
                                current = S_pi*np.einsum(sumidx, vertex, j1, j2)
                                # if cur == 13:
                                #     print("Vertex: ", vertex)
                                #     print("Current {}: ".format(cur1), j1)
                                #     print("Current {}: ".format(cur2), j2)
                                #     print("sumidx: {}".format(sumidx))
                                #     print("New current: ", current)
                                    
                                if (cur+1 != 1 << (self.nparts - 1)): # so if cur+1 != 8.
                                    prop = Propagator(current_part)(self.momentum[cur-1])
                                    # print(prop)
                                    if current_part.is_fermion():
                                        current = np.einsum('bij,bj->bi', prop, current)
                                    elif current_part.is_antifermion():
                                        current = np.einsum('bji,bj->bi', prop, current)
                                    elif current_part.is_vector():
                                        current = np.einsum('bij,bj->bi', prop, current)
                                    else:
                                        current = prop*current
                                        
                                    # if cur == 13:
                                    #     print("Propagator: ", prop)
                                    #     print("New current with Propagator: ", current)
                                self.currents[cur-1].append(current)
                                self.particles[cur-1].add(current_part.conjugate())
                                # if current_part.pid == 23:
                                #     print(current_part.pid, cur, cur1, cur2)
                        icurrent += 1

            idx = next_permutation(idx)

    def generate_currents(self, m, nparts):
        val = (1 << m) - 1
        setbits = np.zeros(nparts-1, dtype=np.int32)
        self.nparts = nparts
        while val < (1 << (nparts - 1)):
            set_bits(val, setbits, nparts-1)
            for iset in range(1, m):
                self.sub_current(val, iset, m, setbits, nparts-1)

            val = next_permutation(val)


def HadronicTensor(p1, p2, gl2, gr2):
    symmetric = np.einsum('bi,bj->bij', p1, p2) + np.einsum('bi,bj->bij', p2, p1)
    symmetric -= np.einsum('ij, b -> bij', ls.METRIC_TENSOR, Dot(p1, p2)[: ,0])
    antisymmetric = 1j*np.einsum('ijkl, bk, bl -> bij', ls.EPS, p1, p2)
    return 2*((gl2+gr2)*symmetric+(gl2-gr2)*antisymmetric)


def main(run_card):
    with open(run_card) as stream:
        parameters = yaml.safe_load(stream)

    particles_yaml = parameters['Particles']
    mode = parameters['Mode']
    ptMin = parameters['PtCut']
    energy_range = parameters['EnergyRange']
    nparts = len(particles_yaml)
    Particle.model = model
    Propagator.model = model
    Particle.max_id = 1 << (nparts - 1)

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

    ecm_array = np.linspace(energy_range[0], energy_range[1], energy_range[2])
    results = np.zeros_like(ecm_array)
    hbarc2 = 0.38937966e9 
    alpha = 1/127.9
    MW = model.parameter_map['MW']
    MZ = model.parameter_map['MZ']
    WW = model.parameter_map['WW']
    WZ = model.parameter_map['WZ']
    sw2 = model.parameter_map['sw2']
    cw = model.parameter_map['cw']
    sw = model.parameter_map['sw']

    nout = nparts - 2
    rambo = Rambo(2, nout, ptMin)
    nevents = parameters['NEvents']
    xsec_ana = np.zeros_like(ecm_array)
    xsec_err = np.zeros_like(ecm_array)
    xsec = np.zeros_like(ecm_array)
    # afb = np.zeros_like(ecm_array)
    amplitudes = []
    cosines = []
    nbins = 100
    ecm_vals = [20.0, 60.0, 100.0, 140.0, 180.0, 200.0]
    theta_cut = 0.95
    for i, ecm in enumerate(ecm_array):
        in_mom = [[-ecm/2, 0, 0, -ecm/2],
                  [-ecm/2, 0, 0, ecm/2]]
        # in_mom = [[ecm/2, 0, 0, ecm/2],
        #           [ecm/2, 0, 0, -ecm/2]]
        out_mom = [[0]*4 for i in range(nout)]
        mom = np.array(in_mom + out_mom, dtype=np.float64)
        mom = np.tile(mom, (nevents, 1, 1))
        rans = np.random.random((nevents, 4*nout))
        weights = rambo(mom, rans)
        
        # print(mom[0,0,:] + mom[0,2,:]+mom[0,1,:] + mom[0,3,:])
        # for i in range(nevents):
        #     mom_sum = mom[i,0,1:] + mom[i,2,1:] + mom[i,1,1:] + mom[i,3,1:]
        #     assert mom_sum[0] < 1e-10, "Event: {}\n {}".format(i, (mom[i,0,1:] + mom[i,2,1:] + mom[i,1,1:] + mom[i,3,1:]))
        #     assert mom_sum[1] < 1e-10, "Event: {}\n {}".format(i, (mom[i,0,1:] + mom[i,2,1:] + mom[i,1,1:] + mom[i,3,1:]))
        #     assert mom_sum[2] < 1e-10, "Event: {}\n {}".format(i, (mom[i,0,1:] + mom[i,2,1:] + mom[i,1,1:] + mom[i,3,1:]))
            # print(mom[0,:,:])
        # raise
        
        # helicity_states = [bin(x)[2:].zfill(nparts-1) for x in range(2**(nparts-1))]
        helicity_states = [bin(x)[2:].zfill(2) for x in range(2**(2))]
        # helicity_states = [bin(x)[2:].zfill(3) for x in range(2**(3))]
        
        results = np.zeros((nevents, 1), dtype=np.complex128)
        # lmunu = np.zeros((nevents, 4, 4), dtype=np.complex128)
        hmunu = np.zeros((nevents, 4, 4), dtype=np.complex128)
        lmunu2 = np.zeros((nevents, 4, 4), dtype=np.complex128)
        
        
        p1 = mom[:,0,:]
        p2 = mom[:,1,:]
        p3 = mom[:,2,:]
        p4 = mom[:,3,:]
        # p5 = mom[:,4,:]
        
        # print(p4, np.shape(p4))
        # print("Dot: ", Dot(p4, p4))
        # print("mass2: ", p4[:,0]**2 - np.sum(p4[:,1:]**2))


        s = Dot(p1+p2,p1+p2)
        t = Dot(p1+p3,p1+p3)
        u = Dot(p1+p4,p1+p4)
        # Ensure that s + t + u = \sum_{i} m^2_i = 0 (for massless case)
        # assert (s+t+u).all() == 0.0
        
        # e+e- to mu+mu-
        # hmunu = HadronicTensor(p3, p4, 4*np.pi*alpha, 4*np.pi*alpha)
        
        # lmunu = HadronicTensor(p1, p3, 4*np.pi*alpha, 4*np.pi*alpha)/t[:,:,None]**2
        # print('here', np.einsum('ik,jl,bkl, bij -> b', ls.METRIC_TENSOR, ls.METRIC_TENSOR, hmunu, lmunu)/4)
        
        ######### HADRONIC TENSORS #########
        # e-p+ to e-p+
        # hmunu = HadronicTensor(p2, p4, 4*np.pi*alpha, 4*np.pi*alpha)
        
        # nu_e nu_mu_bar to e- mu+
        hmunu = HadronicTensor(p2, p4, 2*np.pi*alpha/sw2, 0)
        
        # nu_e p+ to nu_e p+. NOTE: Remember to turn on the Z before running.
        # hmunu = HadronicTensor(p2, p4, 4*np.pi*alpha*(sw/(2*cw)-cw/(2*sw))**2, 4*np.pi*alpha*(sw/cw)**2)
        
        # nu_e n to e-p+
        # hmunu = HadronicTensor(p2, p4, -2*np.pi*alpha/(1-MW**2/MZ**2), 0)
        # switch indices.
        # hmunu = (np.einsum('bi,bj->bij', p2, p4) + np.einsum('bi,bj->bij', p4, p2) 
        #         - np.einsum('ij, b -> bij', ls.METRIC_TENSOR, Dot(p2, p4)[: ,0])
        #         - 1j*np.einsum('ijkl, bk, bl -> bij', ls.EPS, p2, p4))
        # hmunu *= 4*np.pi*alpha/((1-MW**2/MZ**2))
        
        # lmunu = (np.einsum('bi,bj->bij', p1, p3) + np.einsum('bi,bj->bij', p3, p1) 
        #         - np.einsum('ij, b -> bij', ls.METRIC_TENSOR, Dot(p1, p3)[: ,0])
        #         + 1j*np.einsum('ijkl, bk, bl -> bij', ls.EPS, p1, p3))
        # lmunu *= 4*np.pi*alpha/((1-MW**2/MZ**2))
    
        
        # lmunu2_curr = np.zeros((nevents, 4), dtype=np.complex128) 
        for state in helicity_states:
            # helicities = [2*int(state[i])-1 for i in range(nparts-1)]
            hel1, hel2 = int(state[0]), int(state[1])
            # print(hel1, hel2)
            
            # hel1, hel2, hel3 = int(state[0]), int(state[1]), int(state[2])
            # print(hel1, hel2, hel3)
            # e-mu- to e-mu-
            diagram = Diagram(particles, mom, [2*hel1-1, 1, 2*hel2-1, 1], mode)
            # print(particles)
            # e+e- to mu+mu-
            # diagram = Diagram(particles, mom, [2*hel1-1, 2*hel2-1, 1, 1], mode)
            
            # e-mu- to e-mu- gamma
            # diagram = Diagram(particles, mom, [2*hel1-1, 1, 2*hel2-1, 2*hel3-1, 1], mode)
                
            for j in range(2, nparts):
                diagram.generate_currents(j, nparts)

            # for i in range(16):
            #     print("Current {}".format(i+1), diagram.currents[i])
            #     # print("Momentum {}".format(i+1), diagram.momentum[i])
            #     print("Particle {}".format(i+1), diagram.particles[i])
                
            # raise
            
            # print("Event 19")
            # for i in [0, 1, 3, 4, 6, 7]:
            #     print(np.shape(diagram.currents[i]))
            #     print("Current {}".format(i+1), diagram.currents[i][0][18])
            #     print("Momentum {}".format(i+1), diagram.momentum[i][18])
            
            # print(-diagram.momentum[0] - diagram.momentum[3])
            # print(diagram.momentum[1] + diagram.momentum[7])
            # # final_curr = np.sum(np.array(diagram.currents[-2]), axis=0)
            # raise
            
            
            # e-mu- to e-mu-
            lmunu2_curr = np.sum(np.array(diagram.currents[4]), axis=0)
            # e+e- to mu+mu-
            # lmunu2_curr = np.sum(np.array(diagram.currents[2]), axis=0)
            # e-p to e-p gamma
            # lmunu2_curr = np.sum(np.array(diagram.currents[12]), axis=0)
            
            
            # print('curr = ', lmunu2_curr)
            lmunu2 += np.einsum('bi, bj -> bij', lmunu2_curr, np.conj(lmunu2_curr))
            
            # print(lmunu2)
            
            # amplitude = np.einsum('bi,bi->b', np.sum(np.array(diagram.currents[-2]), axis=0), diagram.currents[-1][0])
            # print(np.shape(amplitude), np.shape(np.einsum('b,b->b', amplitude, np.conj(amplitude))))
            # results += np.einsum('b,b->b', amplitude, np.conj(amplitude))
            # if mode == "lmunu":                          
            #     lmunu += np.einsum('bi, bj -> bij', final_curr, np.conj(final_curr))
                
            #     amplitude = np.einsum('bi,bi->b', np.sum(np.array(diagram.currents[-2]), axis=0), diagram.currents[-1][0])
                    
            #     results += np.absolute(amplitude[:, None])**2
            # else:
            #     amplitude = np.einsum('bi,bi->b', np.sum(np.array(diagram.currents[-2]), axis=0), diagram.currents[-1][0])
                    
            #     results += np.absolute(amplitude[:, None])**2
        
        # raise
        # exact_Res = 2*16*alpha**2*np.pi**2*(s**2+u**2)/t**2  # This is 1/4 sum |M|^2. (i.e. already includes Ecm)
        # exact_Res comes from 1/4 sum |M|^2 = 8e^4/s^2*(p1.p3*p2.p4+p1.p4*p2.p3) 
        #                          sum |M|^2 = 32e^4/s^2*(...)
        #                          sum |M|^2 = 32*16*pi^2*alpha^2/(ecm^4)*(...)
        
        # Comparison to results.
       
        # print("by hand: ", np.einsum('ik,jl,bij->bkl',ls.METRIC_TENSOR, ls.METRIC_TENSOR, lmunu)[0])
        # print("Diagram2 Amp =\n{}".format(results))
        # print("Exact:", exact_Res)
        # print("difference: ", (exact_Res[0]+np.einsum('bij,bij->b', lmunu2, hmunu)[0]/4))
        
        # exact_Res = 2*np.pi**2*alpha**2/(1-MW**2/MZ**2)**2*(s**2)/((t-MW**2)**2 + MW**2*WW**2)
        # print("Exact:", exact_Res[0])
        
        ######### AMPLITUDES AND COSINES #########
        
        cos_theta = CosTheta(p3)
        cos_theta_exact = np.linspace(-1, 0.99, nbins+1)
        
        amp2 = np.abs(np.real(np.einsum('bij,bij->b', lmunu2, hmunu))) # correct sign for nue n, incorrect sign for e-p+.
        
        # e- p to e- p
        # spinavg = 4
        
        # nue nu_mu_bar to e- mu+
        spinavg = 1
        
        # nue p to nue p
        # spinavg = 2
        
        amp2 = amp2/spinavg
        
        # e- p to e- p
        # exact_Res = 2*16*alpha**2*np.pi**2*(s**2+u**2)/t**2
        
        # nue p to nue p
        # exact_Res = 2*np.pi**2*alpha**2/(cw**4*sw2**2)*(s**2*cw**4 - 2*s**2*cw**2*sw**2 + sw**4*(s**2 + 4*u**2))/((t-MZ**2)**2 + (MZ*WZ)**2)
        
        # for i in range(nevents):
        #     if np.divide(np.real(amp2[i]), np.real(exact_Res[i][0])) < 0.8:
        #         print("Event number: ", i+1)
        #         print("numerical: ", lmunu2[i])
        #         print("Diagram2 lmunu*hmunu/{} =\n{}".format(spinavg, amp2[i]))
        #         print("Exact:", exact_Res[i])
        #         print("difference: ", (exact_Res[i] + np.einsum('bij,bij->b', lmunu2, hmunu)[i]/spinavg))
        #         print("ratio: ", np.divide(amp2[i], exact_Res[i][0]))
        #         print("cos(theta): ", cos_theta[i])
        #         raise
        # raise
        
        def amp_eemumu(cos_theta_exact, ecm):
            amp_eemumu = 16*np.pi**2*alpha**2*(1+cos_theta_exact**2)
            return amp_eemumu
        
        def amp_ep(cos_theta_exact, ecm):
            amp_ep = 2 * (16*np.pi**2*alpha**2) * (4+(1+cos_theta_exact)**2) / (1-cos_theta_exact)**2
            return amp_ep
        
        def amp_nue_nu_mu_bar(cos_theta_exact, ecm):
            amp_nue_nu_mu_bar = 4*(np.pi**2)*(alpha**2)/(sw2)**2
            amp_nue_nu_mu_bar *= ecm**4*(1+cos_theta_exact)**2
            amp_nue_nu_mu_bar /= (ecm**2/2*(1-cos_theta_exact)+MW**2)**2 + (MW*WW)**2
            return amp_nue_nu_mu_bar
        
        def amp_nue_p(cos_theta_exact, ecm):
            amp_nue_p = 2*np.pi**2*alpha**2/(cw**4*sw2**2)
            amp_nue_p *= ecm**4*cw**4 - 2*ecm**4*sw**2*cw**2 + sw**4*(ecm**4*(1 + cos_theta_exact)**2 + ecm**4)
            amp_nue_p /= (ecm**2/2*(1-cos_theta_exact) + MZ**2)**2 + (MZ**2*WZ**2)
            return amp_nue_p
        
        # def amp_nue_n(cos_theta_exact, ecm):
        #     amp_nue_nJ = (np.pi**2)*(alpha**2)/(1-MW**2/MZ**2)**2
        #     amp_nue_nJ /= (-ecm**2/2*(1-cos_theta_exact)-MW**2)**2+(MW*WW)**2
        #     amp_nue_nJ *= ecm**4
        #     return amp_nue_n
        
        # amp_plot_noFunc(cos_theta, amp2, ecm, nevents, nbins)
        # raise
        
        # M = amp_nue_p(cos_theta_exact, ecm)
        # # # print(M)
        # ratios = np.divide(amp2, exact_Res[:][0])
        # print(min(ratios), max(ratios))
        # print(np.shape(ratios), np.shape(cos_theta_exact))
        # plt.scatter(np.linspace(1,nevents, num=nevents), ratios)
        # plt.show
        # raise
        # plt.plot(cos_theta_exact, M)
        # plt.hist(cos_theta, weights=amp2/nevents/(2/nbins), bins=nbins)
        # plt.semilogy()
        # plt.show()
        # raise
        # amp_plot(cos_theta, cos_theta_exact, amp2, amp_ep, ecm, nevents, nbins)
        # amp_plot(cos_theta, cos_theta_exact, amp2, amp_nue_nu_mu_bar, ecm, nevents, nbins)
        # amp_plot(cos_theta, cos_theta_exact, amp2, amp_nue_p, ecm, nevents, nbins)
        
        # raise
        
        
        if ecm in ecm_vals:
            # M = amp_ep(cos_theta_exact, ecm)
            M = amp_nue_nu_mu_bar(cos_theta_exact, ecm)
            # M = amp_nue_p(cos_theta_exact, ecm)
            amplitudes.append([M, amp2])
            cosines.append([cos_theta_exact, cos_theta])
            
        ######### CROSS SECTION CALCULATION #########
        amp2 = np.where(cos_theta <= theta_cut, amp2, 0)
        
        flux = 2*ecm**2
        
        results = np.einsum("b,bi->b", amp2, weights)/flux*hbarc2
        
        # xsec_ana[i] = 1/(32*np.pi*ecm**2)*integrate.quad(amp_ep, -1, theta_cut, args=(ecm,))[0]*hbarc2
        xsec_ana[i] = 1/(32*np.pi*ecm**2)*integrate.quad(amp_nue_nu_mu_bar, -1, theta_cut, args=(ecm,))[0]*hbarc2
        # xsec_ana[i] = 1/(32*np.pi*ecm**2)*integrate.quad(amp_nue_p, -1, theta_cut, args=(ecm,))[0]*hbarc2
        
        xsec_err[i] = np.std(results)/np.sqrt(nevents)
        xsec[i] = np.mean(results)
        
        # print(ecm, np.mean(results), np.std(results)/np.sqrt(nevents), exact_emu)
        # nfwd = np.sum(results[cos_theta > 0])/nevents
        # nbck = np.sum(results[cos_theta < 0])/nevents
        # afb[i] = (nfwd - nbck)/np.mean(results)
        # # print(nfwd, nbck, afb[i])
        
        # print("ana: ", xsec_ana[i])
        # print("compt: ", xsec[i])
        # print("uncertainty: ", xsec_err[i])
        
        # raise
        
        # raise
        # cos_theta = CosTheta(mom[:, 2, :])
        # # s, t, u = Mandelstam(mom)
        # # spinavg = 4
        # # flux = 2*ecm**2
        # # results = results/flux/spinavg*hbarc2*weights
        # #results2 = results2/flux/spinavg*hbarc2*weights
        # exact = 4*np.pi*alpha**2*hbarc2/(3*ecm**2) # This is the (exact) cross section.
        # # exact_moller = 8.41905*alpha**2*hbarc2*2*np.pi/(ecm**2)
        # # exact_compton = 2*alpha**2*hbarc2*2 
        # print(ecm, np.mean(results), np.std(results)/np.sqrt(nevents), exact)
        # nfwd = np.sum(results[cos_theta > 0])/nevents
        # nbck = np.sum(results[cos_theta < 0])/nevents
        # afb[i] = (nfwd - nbck)/np.mean(results)
        # print(nfwd, nbck, afb[i])
        # nbins = 100
        # cos_theta_exact = np.linspace(-0.4, 0.4, nbins+1)
        # dsigma_moller = alpha**2*hbarc2*(3+cos_theta_exact**2)**2/(ecm**2)*2*np.pi/(1-cos_theta_exact**2)**2
        # # dsigma = alpha**2*hbarc2*(1+cos_theta_exact**2)/(4*ecm**2)*2*np.pi
        # # plt.hist(cos_theta, weights=results/nevents/(2/nbins), bins=np.linspace(-1,1,nbins+1))
        # # plt.plot(cos_theta_exact, dsigma_moller)
        # # plt.show()
        # # raise
        # xsec[i] = np.mean(results)
    
    order = len(str(nevents))-1
    
    # amp_plots(cosines, amplitudes, ecm_vals, nevents, nbins, name='amps_ep_1e{}.jpeg'.format(order))
    amp_plots(cosines, amplitudes, ecm_vals, nevents, nbins, name='amps_nue_nu_mu_bar_1e{}.jpeg'.format(order))
    # amp_plots(cosines, amplitudes, ecm_vals, nevents, nbins, name='amps_nue_p_1e{}.jpeg'.format(order))
    
    # amp_plots_noFunc(cosines, amplitudes, ecm_vals, nevents, nbins, name='amps_ep_gamma_1e{}.pdf'.format(order))

    # xsec_plot(ecm_array, xsec, xsec_ana, xsec_err, r'$\sigma(e^- p^+ \rightarrow e^-p^+)$ (pb)' , theta_cut, nevents, name='xsec_ep_1e{}.jpeg'.format(order))
    xsec_plot(ecm_array, xsec, xsec_ana, xsec_err, r'$\sigma(\nu_e \bar{\nu}_\mu \rightarrow e^- \mu^+)$ (pb)' , theta_cut, nevents, name='xsec_nue_nu_mu_bar_1e{}.jpeg'.format(order))
    # xsec_plot(ecm_array, xsec, xsec_ana, xsec_err, r'$\sigma(\nu_e p^+ \rightarrow \nu_e p^+)$ (pb)' , theta_cut, nevents, name='xsec_nue_p_1e{}.jpeg'.format(order))
    # xsec_plot_noFunc(ecm_array, xsec, xsec_err, r'$\sigma(e^- p^+ \rightarrow e^- p^+ \gamma)$ (pb)' , theta_cut, nevents, name='xsec_ep_gamma_1e{}.pdf'.format(order))

def xsec_plot(ecm_array, xsec, xsec_ana, xsec_err, process, theta_cut, nevents, name='xsec.pdf'):

    fig, ax = plt.subplots(2,1, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.0}, dpi=400)
    ax = ax.flatten()
    for ax_i in ax:
        ax_i.tick_params(axis='both', direction='in', reset=True, labelsize=15,
                   which='both')
        ax_i.tick_params(which='major', length=7)
        ax_i.tick_params(which='minor', length=4)
        ax_i.xaxis.set_minor_locator(AutoMinorLocator())
        ax_i.yaxis.set_minor_locator(AutoMinorLocator())
        ax_i.set_xlim(10, 210)
        
    ax[0].set_ylabel(process, fontsize=12, labelpad=1)
    ax[0].semilogy()
    ax[0].plot(ecm_array, xsec_ana, color='mediumpurple', label='Analytic')
    ax[0].errorbar(ecm_array, xsec, yerr=xsec_err, color='seagreen', marker='|', label='Numerical')
    ax[0].legend(frameon=False, fontsize=12, borderpad=0.2, ncol=2,
          columnspacing=0.5)
    
    error_xsec = np.divide(xsec, xsec_ana)
    
    ax[1].set_xlabel(r'$\sqrt{s}$ (GeV)', fontsize=12, labelpad=1)
    ax[1].set_ylabel(r"Num./Ana.", fontsize=12, labelpad=1)
    ax[1].set_ylim(0.50,1.50)
    ax[1].plot(ecm_array, error_xsec, color='dodgerblue')
    ax[1].axhline(y=1.0, color='black', linestyle='-')
    
    textstr = '\n'.join((
    r'$\cos(\theta)<%.2f$' % (theta_cut, ),
    r'events = {}'.format(nevents) ))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='antiquewhite', alpha=0.5)

    # place a text box in upper left in axes coords
    ax[0].text(0.68, 0.75, textstr, transform=ax[0].transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
    
    # nue p to nue p placement.
    # ax[0].text(0.68, 0.60, textstr, transform=ax[0].transAxes, fontsize=14,
    #             verticalalignment='top', bbox=props)
    
    plt.savefig(name, bbox_inches='tight')

def xsec_plot_noFunc(ecm_array, xsec, xsec_err, process, theta_cut, nevents, name='xsec.pdf'):

    fig, ax = plt.subplots()
        
    ax.tick_params(axis='both', direction='in', reset=True, labelsize=15,
                     which='both')
    ax.tick_params(which='major', length=7)
    ax.tick_params(which='minor', length=4)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlim(10, 210)
        
    ax.set_ylabel(process, fontsize=12, labelpad=1)
    ax.semilogy()
    ax.errorbar(ecm_array, xsec, yerr=xsec_err, color='seagreen', marker='|', label='Data')
    ax.legend(frameon=False, fontsize=12, borderpad=0.2, ncol=2,
          columnspacing=0.5)
    
    textstr = '\n'.join((
    r'$\cos(\theta)<%.2f$' % (theta_cut, ),
    r'events = {}'.format(nevents) ))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='antiquewhite', alpha=0.5)

    # place a text box in upper left in axes coords
    # ax.text(0.68, 0.75, textstr, transform=ax[0].transAxes, fontsize=14,
    #            verticalalignment='top', bbox=props)
    
    # nue p to nue p placement.
    ax.text(0.68, 0.60, textstr, transform=ax.transAxes, fontsize=14,
               verticalalignment='top', bbox=props)
    
    plt.savefig(name, bbox_inches='tight')


def amp_plots(cosines, amplitudes, ecm_vals, nevents, nbins, name='amps.pdf'):
  
    fig, ax = plt.subplots(3, 2, tight_layout=True, figsize=(11, 11), dpi=400)
    ax = ax.flatten()
    for i, axi in enumerate(ax):
        axi.tick_params(axis='both', direction='in', reset=True, labelsize=15,
                        which='both')
        axi.tick_params(which='major', length=7)
        axi.tick_params(which='minor', length=4)
        axi.xaxis.set_minor_locator(AutoMinorLocator())
        axi.yaxis.set_minor_locator(AutoMinorLocator())
        axi.set_xlim(-1, 1)
        axi.set_xlabel(r'$\cos(\theta)$', fontsize=12, labelpad=1)
        axi.set_ylabel(r'$\sum|\mathcal{M}|^2$', fontsize=12, labelpad=1)
        axi.semilogy()
        textstr = '\n'.join((
            r'$\sqrt{s}=%.2f$ GeV' % (ecm_vals[i], ),
            r'events = {}'.format(nevents) ))
        histogram = Histogram([-1,1], bins=nbins)
        for j, cosine in enumerate(cosines[i][1]):
            histogram.fill(cosine, weight=amplitudes[i][1][j]/nevents)
        axi, weights_true = histogram.plot(axi, color='firebrick', label='Numerical')
    
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        # axi.text(0.05, 0.93, textstr, transform=axi.transAxes, fontsize=14,
        #           verticalalignment='top', bbox=props)
        
        # placement for nue nu_mu_bar to e-mu+, and nue p to nue p.
        axi.text(0.65, 0.2, textstr, transform=axi.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
    
        axi.plot(cosines[i][0], amplitudes[i][0], color='goldenrod', linewidth=2, label='Analytic')
        axi.legend(frameon=False, fontsize=12, borderpad=0.2, ncol=2,
                     columnspacing=0.5)
        
    plt.savefig(name, bbox_inches='tight')

    
def amp_plot(cos_theta, cos_theta_exact, amp2, amp_func, ecm, nevents, nbins=100):
    hist_ep = Histogram([-1, 0.99], bins=nbins)
    for i, cosine in enumerate(cos_theta):
        hist_ep.fill(cosine, weight=amp2[i]/nevents)

    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1], 'hspace':0.0})
    ax.flatten()
    for axi in ax:
        axi.tick_params(axis='both', direction='in', reset=True, labelsize=15,
                        which='both')
        axi.tick_params(which='major', length=7)
        axi.tick_params(which='minor', length=4)
        axi.xaxis.set_minor_locator(AutoMinorLocator())
        axi.yaxis.set_minor_locator(AutoMinorLocator())
        axi.set_xlim(-1, 1)
    ax[1].set_xlabel(r'$\cos(\theta)$', fontsize=12, labelpad=1)
    ax[1].set_ylabel(r'Data/Ana.', fontsize=12, labelpad=1)
    ax[0].set_ylabel(r'$\frac{1}{2}\sum|\mathcal{M}|^2$', fontsize=12, labelpad=1)
    ax[0].semilogy()
    ax[0], weights_true = hist_ep.plot(ax[0], color='firebrick', label='Data')
    print(np.shape(weights_true), weights_true[0])

    
    textstr = '\n'.join((
    # r'$e^-p^+ \rightarrow e^-p^+$',
        r'$\sqrt{s}=%.2f$ GeV' % (ecm, ),
        r'events = {}'.format(nevents) ))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    # ax[0].text(0.05, 0.95, textstr, transform=ax[0].transAxes, fontsize=14,
    #            verticalalignment='top', bbox=props)
    
    # placement for nue n.
    ax[0].text(0.60, 0.60, textstr, transform=axi.transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)
    
    # e-p+ to e-p+
    # M = amp_func(cos_theta_exact)
        
    # nue n to e-p+
    M = amp_func(cos_theta_exact, ecm)
    
    ax[0].plot(cos_theta_exact, M, color='goldenrod', linewidth=2, label='Analytic')
    ax[0].legend(frameon=False, fontsize=12, borderpad=0.2, ncol=2,
                 columnspacing=0.5)
    
    # ax[1].semilogy()
    # ax[1].plot(cos_theta_exact, np.where(cos_theta_exact<0.90, np.where(cos_theta_exact>-0.90, np.divide(weights_true, M), 1),1))
    ax[1].plot(cos_theta_exact, np.divide(weights_true, M))
    # ax[1].set_ylim(0.75,1.25)
    ax[1].axhline(y=1.0, color='black', linestyle='-')
        
    plt.savefig('amp.pdf', bbox_inches='tight')
    
def amp_plots_noFunc(cosines, amplitudes, ecm_vals, nevents, nbins, name='amps.pdf'):
  
    fig, ax = plt.subplots(3, 2, tight_layout=True, figsize=(11, 11))
    ax = ax.flatten()
    for i, axi in enumerate(ax):
        axi.tick_params(axis='both', direction='in', reset=True, labelsize=15,
                        which='both')
        axi.tick_params(which='major', length=7)
        axi.tick_params(which='minor', length=4)
        axi.xaxis.set_minor_locator(AutoMinorLocator())
        axi.yaxis.set_minor_locator(AutoMinorLocator())
        axi.set_xlim(-1, 1)
        axi.set_xlabel(r'$\cos(\theta)$', fontsize=12, labelpad=1)
        axi.set_ylabel(r'$\frac{1}{4}\sum|\mathcal{M}|^2$', fontsize=12, labelpad=1)
        axi.semilogy()
        textstr = '\n'.join((
            r'$\sqrt{s}=%.2f$ GeV' % (ecm_vals[i], ),
            r'events = {}'.format(nevents) ))
        histogram = Histogram([-1,1], bins=nbins)
        for j, cosine in enumerate(cosines[i][1]):
            histogram.fill(cosine, weight=amplitudes[i][1][j]/nevents)
        axi, weights_true = histogram.plot(axi, color='firebrick', label='Data')
    
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        axi.text(0.05, 0.93, textstr, transform=axi.transAxes, fontsize=14,
                  verticalalignment='top', bbox=props)
        
        # placement for nue nu_mu_bar to e-mu+.
        # axi.text(0.65, 0.2, textstr, transform=axi.transAxes, fontsize=14,
        #         verticalalignment='top', bbox=props)
    
        axi.legend(frameon=False, fontsize=12, borderpad=0.2, ncol=2,
                     columnspacing=0.5)
        
    plt.savefig(name, bbox_inches='tight')
    
def amp_plot_noFunc(cos_theta, amp2, ecm, nevents, nbins=100):
    hist_ep = Histogram([-1, 0.99], bins=nbins)
    for i, cosine in enumerate(cos_theta):
        hist_ep.fill(cosine, weight=amp2[i]/nevents)

    fig, ax = plt.subplots()
    ax.tick_params(axis='both', direction='in', reset=True, labelsize=15,
                    which='both')
    ax.tick_params(which='major', length=7)
    ax.tick_params(which='minor', length=4)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlim(-1, 1)
    ax.set_xlabel(r'$\cos(\theta)$', fontsize=12, labelpad=1)
    ax.set_ylabel(r'$\frac{1}{4}\sum|\mathcal{M}|^2$', fontsize=12, labelpad=1)
    ax.semilogy()
    ax, weights_true = hist_ep.plot(ax, color='firebrick', label='Data')
    # print(np.shape(weights_true), weights_true[0])

    
    textstr = '\n'.join((
    # r'$e^-p^+ \rightarrow e^-p^+$',
        r'$\sqrt{s}=%.2f$ GeV' % (ecm, ),
        r'events = {}'.format(nevents) ))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
    
    # placement for nue n.
    # ax[0].text(0.60, 0.60, textstr, transform=axi.transAxes, fontsize=14,
    #                 verticalalignment='top', bbox=props)
    
    ax.legend(frameon=False, fontsize=12, borderpad=0.2, ncol=2,
                 columnspacing=0.5)
        
    plt.savefig('amp_noFunc.pdf', bbox_inches='tight')

if __name__ == '__main__':
    np.random.seed(123456789)
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_card', default='run_card.yml',
                        help='Input run card')
    args = parser.parse_args()

    main(args.run_card)
