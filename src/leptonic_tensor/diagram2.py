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
                self.currents[(1 << i)-1] = [ls.PolarizationVector(mom[:, i, :], hel[i]).epsilon]

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
                                current_part = Particle(cur, pids0[0])
                                if(current_part.pid == 23):
                                    continue
                                self.particles[cur-1].add(current_part)
                                self.momentum[cur-1] = self.momentum[cur1-1] + self.momentum[cur2-1]
                                sorted_list, index = type_sort(part1, part2, current_part)
                                ufo_vertex = model.vertex_map[key]
                                batch = np.size(self.momentum[cur-1], axis=0)
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
                                if (cur+1 != 1 << (self.nparts - 1)): # so if cur+1 != 8.
                                    prop = Propagator(current_part)(self.momentum[cur-1])
                                    if current_part.is_fermion():
                                        current = np.einsum('bij,bj->bi', prop, current)
                                    elif current_part.is_antifermion():
                                        current = np.einsum('bji,bj->bi', prop, current)
                                    elif current_part.is_vector():
                                        current = np.einsum('bij,bj->bi', prop, current)
                                    else:
                                        current = prop*current
                                self.currents[cur-1].append(current)
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

    # phi = 0
    # costheta = 0
    # sintheta = np.sqrt(1-costheta**2)
    ecm_array = np.linspace(energy_range[0], energy_range[1], energy_range[2])
    results = np.zeros_like(ecm_array)
    hbarc2 = 0.38937966e9 
    alpha = 1/127.9
    # Me = 0.000511
    # Mmu = 0.105658 
    MW = model.parameter_map['MW']
    MZ = model.parameter_map['MZ']
    WW = model.parameter_map['WW']

    nout = nparts - 2
    rambo = Rambo(2, nout, ptMin)
    nevents = parameters['NEvents']
    xsec_ana = np.zeros_like(ecm_array)
    xsec_err = np.zeros_like(ecm_array)
    xsec = np.zeros_like(ecm_array)
    afb = np.zeros_like(ecm_array)
    amplitudes = []
    cosines = []
    theta_cut = 0.95
    for i, ecm in enumerate(ecm_array):
        in_mom = [[-ecm/2, 0, 0, -ecm/2],
                  [-ecm/2, 0, 0, ecm/2]]
        out_mom = [[0]*4 for i in range(nout)]
        mom = np.array(in_mom + out_mom, dtype=np.float64)
        mom = np.tile(mom, (nevents, 1, 1))
        rans = np.random.random((nevents, 4*nout))
        weights = rambo(mom, rans)
    
        
        # helicity_states = [bin(x)[2:].zfill(nparts-1) for x in range(2**(nparts-1))]
        helicity_states = [bin(x)[2:].zfill(2) for x in range(2**(2))]
        
        results = np.zeros((nevents, 1), dtype=np.float64)
        lmunu = np.zeros((nevents, 4, 4), dtype=np.complex128)
        hmunu = np.zeros((nevents, 4, 4), dtype=np.complex128)
        lmunu2 = np.zeros((nevents, 4, 4), dtype=np.complex128)
        
        
        p1 = mom[:,0,:]
        p2 = mom[:,1,:]
        p3 = mom[:,2,:]
        p4 = mom[:,3,:]

        
        s = Dot(p1+p2,p1+p2)
        t = Dot(p1+p3,p1+p3)
        u = Dot(p1+p4,p1+p4)
        # Ensure that s + t + u = \sum_{i} m^2_i = 0 (for massless case)
        # print(s, t, u, s+t+u)
        
        # e-mu- to e-mu-. This only includes photon, no Z.
        hmunu = HadronicTensor(p2, p4, 4*np.pi*alpha, 4*np.pi*alpha)
        
        # e+e- to mu+mu-
        # hmunu = HadronicTensor(p3, p4, 4*np.pi*alpha, 4*np.pi*alpha)
        
        lmunu = HadronicTensor(p1, p3, 4*np.pi*alpha, 4*np.pi*alpha)/t[:,:,None]**2
        # print('here', np.einsum('ik,jl,bkl, bij -> b', ls.METRIC_TENSOR, ls.METRIC_TENSOR, hmunu, lmunu)/4)
        
        
        # lmunu2_curr = np.zeros((nevents, 4), dtype=np.complex128) 
        for state in helicity_states:
            # helicities = [2*int(state[i])-1 for i in range(nparts-1)]
            hel1, hel2 = int(state[0]), int(state[1])
            
            # e-mu- to e-mu-
            diagram = Diagram(particles, mom, [2*hel1-1, 1, 2*hel2-1, 1], mode)
            
            # e+e- to mu+mu-
            # diagram = Diagram(particles, mom, [2*hel1-1, 2*hel2-1, 1, 1], mode)
                
            for j in range(2, nparts):
                diagram.generate_currents(j, nparts)

            final_curr = np.sum(np.array(diagram.currents[-2]), axis=0)
            
            # e-mu- to e-mu-
            lmunu2_curr = np.sum(np.array(diagram.currents[4]), axis=0)
            
            # e+e- to mu+mu-
            # lmunu2_curr = np.sum(np.array(diagram.currents[2]), axis=0)
            
            # print('curr = ', lmunu2_curr)
            lmunu2 += np.einsum('bi, bj -> bij', lmunu2_curr, np.conj(lmunu2_curr))
            if mode == "lmunu":                          
                lmunu += np.einsum('bi, bj -> bij', final_curr, np.conj(final_curr))
                
                amplitude = np.einsum('bi,bi->b', np.sum(np.array(diagram.currents[-2]), axis=0), diagram.currents[-1][0])
                    
                results += np.absolute(amplitude[:, None])**2
            # else:
            #     amplitude = np.einsum('bi,bi->b', np.sum(np.array(diagram.currents[-2]), axis=0), diagram.currents[-1][0])
                    
            #     results += np.absolute(amplitude[:, None])**2
        
        # exact_Res = 2*16*alpha**2*np.pi**2*(s**2+u**2)/t**2  # This is 1/4 sum |M|^2. (i.e. already includes Ecm)
        # exact_Res comes from 1/4 sum |M|^2 = 8e^4/s^2*(p1.p3*p2.p4+p1.p4*p2.p3) 
        #                          sum |M|^2 = 32e^4/s^2*(...)
        #                          sum |M|^2 = 32*16*pi^2*alpha^2/(ecm^4)*(...)
        
        # Comparison to results.
       
        # print("by hand: ", np.einsum('ik,jl,bij->bkl',ls.METRIC_TENSOR, ls.METRIC_TENSOR, lmunu))
        # print("numerical: ", lmunu2)
        # print("Diagram2 lmunu*hmunu/4 =\n{}".format(np.einsum('bij,bij->b', lmunu2, hmunu)/4))
        # print("Diagram2 Amp =\n{}".format(results))
        # print("Exact:", exact_Res)
        # print("difference: ", (exact_Res[0]+np.einsum('bij,bij->b', lmunu2, hmunu)[0]/4))
        
        # raise
        
        # e-mu- to e-mu-
        cos_theta = CosTheta(p3)
        
        # e+e- to mu+mu-
        # cos_theta = CosTheta(mom[:, 2, :])
        
        amp2 = -np.real(np.einsum('bij,bij->b', lmunu2, hmunu)) # These are diag2 results.
        spinavg = 4
        flux = 2*ecm**2
        amp2 = amp2/spinavg  # When plotting amp vs costheta, do not include flux only spinavg.
        
        nbins = 200
        
        # # e-mu- to e-mu-
        cos_theta_exact = np.linspace(-1, 0.99, nbins+1)
        
        # # # e+e- to mu+mu-
        # # # cos_theta_exact = np.linspace(-1, 1, nbins+1)
        
        M_emu = 2 * (16*np.pi**2*alpha**2) * (4+(1+cos_theta_exact)**2) / (1-cos_theta_exact)**2
        # # # M_eemumu = 16*np.pi**2*alpha**2*(1+cos_theta_exact**2)
        # # # # print(np.shape(cos_theta), np.shape(amp2[0]/nevents/(2/nbins)))
        
        # # plot_M_cos(M_emu, amp2, cos_theta, cos_theta_exact, nevents, ecm, nbins)
    
        print(np.shape(cos_theta), np.shape(amp2))
    
        hist_ep = Histogram([-1, 1], bins=nbins)
        for i, cosine in enumerate(cos_theta):
            hist_ep.fill(cosine, weight=amp2[i]/nevents)
    
        hist_ep = hist_ep.normalize()
    
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
        ax[0].set_ylabel(r'$\frac{1}{4}\sum|\mathcal{M}|^2$', fontsize=12, labelpad=1)
        ax[0].semilogy()
        ax[0], weights_true = hist_ep.plot(ax[0], color='firebrick', label='Data')
        # ax.show()
    
        textstr = '\n'.join((
        # r'$e^-p^+ \rightarrow e^-p^+$',
        r'$\sqrt{s}=%.2f$ GeV' % (ecm, ),
        r'events = {}'.format(nevents) ))

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        ax[0].text(0.05, 0.95, textstr, transform=ax[0].transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        
        ax[0].plot(cos_theta_exact, M_emu, color='goldenrod', linewidth=2, label='Analytic')
        ax[0].legend(frameon=False, fontsize=12, borderpad=0.2, ncol=2,
                      columnspacing=0.5)
        
        ax[1].semilogy()
        ax[1].plot(cos_theta_exact, np.divide(weights_true, M_emu))
        ax[1].axhline(y=1.0, color='black', linestyle='-')
        
        plt.savefig('amp.pdf', bbox_inches='tight')
        # plt.show()
        
        raise
        
        # nue n to e- p+
        # M_nue_n = (np.pi**2)*(alpha**2)/(1-MW**2/MZ**2)**2
        # M_nue_n /= 1/4*(1-cos_theta_exact)**2 + (MW/ecm)**2*(1-cos_theta_exact) + (MW/ecm)**4 + ((MW/ecm)**2)*((WW/ecm)**2)
        # M_nue_n *= (1+cos_theta_exact)**2 - 1/4*(ecm/MW)**2*(1-cos_theta_exact)**2*(3+cos_theta_exact) - ecm**4/(8*MW**2)*(1+cos_theta_exact)**2*(5+cos_theta_exact)*(3-cos_theta_exact)
        
        
        # if ecm in [20.0, 60.0, 100.0, 140.0, 180.0, 200.0]:
        #     nbins = 100
        #     cos_theta_exact = np.linspace(-0.99, 0.99, nbins+1)
        #     M_emu = 2*(16*np.pi**2*alpha**2)*(4+(1+cos_theta_exact)**2)/(1-cos_theta_exact)**2
        #     amplitudes.append([M_emu, amp2])
        #     cosines.append([cos_theta_exact, cos_theta])
            
        ######### CROSS SECTION CALCULATION #########
        # Cut on cos(theta).
        amp2 = np.where(cos_theta < theta_cut, amp2, 0)
        
        
        results = np.einsum("b,bi->b", amp2, weights)/flux*hbarc2
        
        xsec_ana[i] = 1/(32*np.pi*ecm**2)*integrate.quad(lambda ctheta:2*(16*np.pi**2*alpha**2)*(4+(1+ctheta)**2)/(1-ctheta)**2, -1, theta_cut)[0]*hbarc2
        
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

    # Plot of cross sections.
    
    # exact_eemumu = 4*np.pi*alpha**2*hbarc2/(3*ecm**2) # This is the (exact, total) cross section of e+e- to mu+mu-.

    # figM, ((ax11, ax12), (ax21, ax22), (ax31, ax32)) = plt.subplots(3,2, figsize=(11.0, 11.0), dpi=200)
    
    # ax11 = plot_M_six(cosines, amplitudes, nevents, 100, 0, 20.0, ax11)
    # ax12 = plot_M_six(cosines, amplitudes, nevents, 100, 1, 60.0, ax12)
    # ax21 = plot_M_six(cosines, amplitudes, nevents, 100, 2, 100.0, ax21)
    # ax22 = plot_M_six(cosines, amplitudes, nevents, 100, 3, 140.0, ax22)
    # ax31 = plot_M_six(cosines, amplitudes, nevents, 100, 4, 180.0, ax31)
    # ax32 = plot_M_six(cosines, amplitudes, nevents, 100, 5, 200.0, ax32)
    

    xsec_plot(ecm_array, xsec, xsec_ana, xsec_err, r'$\sigma(e^-p^+ \rightarrow e^-p^+)$ (pb)' , theta_cut, nevents)
    


def xsec_plot(ecm_array, xsec, xsec_ana, xsec_err, process, theta_cut, nevents, name='xsec.pdf'):

    fig, ax = plt.subplots(2,1, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.0})
    ax.flatten()
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
    ax[0].errorbar(ecm_array, xsec, yerr=xsec_err, color='seagreen', marker='|', label='Data')
    ax[0].legend(frameon=False, fontsize=12, borderpad=0.2, ncol=2,
          columnspacing=0.5)
    
    error_xsec = np.divide(xsec, xsec_ana)
    
    ax[1].set_xlabel(r'$\sqrt{s}$ (GeV)', fontsize=12, labelpad=1)
    ax[1].set_ylabel(r"Data/Ana.", fontsize=12, labelpad=1)
    ax[1].set_ylim(0.75,1.25)
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
    
    plt.savefig(name, bbox_inches='tight')

def plot_M_six(cosines, amplitudes, nevents, nbins, i, ecm, ax):
    ax.hist(cosines[i][1], weights=amplitudes[i][1]/nevents/(2/nbins), bins=np.linspace(-1,1,nbins+1), color='firebrick')
        
    ax.semilogy()
    ax.set_ylabel(r'$\displaystyle \frac{1}{4}\sum|\mathcal{M}|^2$')
    ax.set_xlabel(r'$\displaystyle \cos(\theta)$')
        
            
    textstr = '\n'.join((
        # r'$e^-p^+\rightarrow e^-p^+$',
        r'$\sqrt{s}=%.2f$ GeV' % (ecm, ),
        r'events = {}'.format(nevents) ))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
    verticalalignment='top', bbox=props)
        
    ax.plot(cosines[i][0], amplitudes[i][0], color='goldenrod', linewidth=3)
    
    ax.legend(['Data', 'Analytic'], loc=(0.65,0.75))
    
    return ax

def plot_M_cos(M, amp2, cos_theta, cos_theta_exact, nevents, ecm, nbins):
    fig, ax = plt.subplots()
        
    ax.hist(cos_theta, weights=amp2/nevents/(2/nbins), bins=np.linspace(-1,1,nbins+1), color='firebrick')
        
    ax.semilogy()
    ax.set_ylabel(r'$\displaystyle \frac{1}{4}\sum|\mathcal{M}|^2$')
    ax.set_xlabel(r'$\displaystyle \cos(\theta)$')
        
            
    textstr = '\n'.join((
        r'$e^-p^+ \rightarrow e^-p^+$',
        r'$\sqrt{s}=%.2f$ GeV' % (ecm, ),
        r'events = {}'.format(nevents) ))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
    verticalalignment='top', bbox=props)
        
    ax.plot(cos_theta_exact, M, color='goldenrod', linewidth=3)
    # # plt.plot(cos_theta_exact, M_eemumu)
    plt.show()
    # return fig, ax

if __name__ == '__main__':
    np.random.seed(123456789)
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_card', default='run_card.yml',
                        help='Input run card')
    args = parser.parse_args()

    main(args.run_card)
