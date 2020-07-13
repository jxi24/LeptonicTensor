import vertex_class as vc
import lorentz_structures as ls
import ufo_grammer
import numpy as np

class FeynRules:
    def __init__(self, model, incoming_particles, outgoing_particles, internal_particles):
        '''
        Calculate spin-summed/averaged amplitude for incoming to outgoing particles
        with internal particles.
        '''
        self.model = model
        # All particles considered outgoing
        self.incoming = [part.anti() for part in incoming_particles]
        self.outgoing = outgoing_particles
        self.internal = internal_particles
        
        # Update momentum array of particles.
        part_list = self.incoming + self.outgoing
        mom = np.empty_like(self.outgoing[0].mom_array[0:4])
        for part in part_list:
            mom[part.mom_index] = part.mom_array[part.mom_index]
        mom = np.append(mom, [mom[0] + mom[1]], axis=0) # s-channel momentum
        mom = np.append(mom, [mom[0] - mom[2]], axis=0) # t-channel momentum
        mom = np.append(mom, [mom[0] - mom[3]], axis=0) # u-channel momentum
        for part in part_list:
            setattr(part,'mom_array',mom)
        
        self.pids = self._get_pids()
        # self.vertices = self._get_vertices()
        # self.propagator = self._get_propagator()
        self.amplitude = self._get_amplitude()
        
    def _get_pids(self):
        pids = []
        for part in self.incoming:
            pids.append(part.pid)
        for part in self.outgoing:
            pids.append(part.pid)
        for part in self.internal:
            pids.append(part.pid)
        return pids
        
    def _get_vertices(self, part_list1: list, part_list2: list):
        '''
        Get the two vertices involved in a 2->2 tree-level process. Particle lists should
        include external and internal particles.
        '''
        part_idx1 = []
        part_idx2 = []
        for part in part_list1:
            part_idx1.append(part.index)
        for part in part_list2:
            if part in self.internal:
                part_idx2.append(part.index+1)
            else:
                part_idx2.append(part.index)
        V1 = vc.Vertex(self.model, part_list1, part_idx1)
        V2 = vc.Vertex(self.model, part_list2, part_idx2)
        return [V1, V2]
    
    # def _get_propagator(self):
        
    def _s_channel_amplitude(self, spinor1, spinor2, spinor3, spinor4):
        part_list1 = self.incoming + self.internal
        part_list2 = self.outgoing + self.internal
        
        vertices = self._get_vertices(part_list1, part_list2)
        vert1 = vertices[0].tensor[0] # Gamma(0,2,1)
        vert2 = vertices[1].tensor[0] # Gamma(1,4,3)
        
        coup1 = vertices[0].coupling[0][0] # -0.313451j
        coup2 = vertices[1].coupling[0][0] # -0.313451j
        
        ph = self.internal[0]
        propagator = -1j*ls.Metric(0,1)
        propagator /= ls.Momentum(ph.mom_array, 10, ph.mom_index)*ls.Metric(10, 11)*ls.Momentum(ph.mom_array, 11, ph.mom_index)
        # Propagator(0,1)
        
        M_s_channel = coup1*coup2*vert1*vert2*spinor1*spinor2*spinor3*spinor4*propagator
        
        return M_s_channel
        
    def _t_channel_amplitude(self, spinor1, spinor2, spinor3, spinor4):
        part_list1 = [self.incoming[0], self.outgoing[0]] + self.internal
        part_list2 = [self.incoming[1], self.outgoing[1]] + self.internal
        
        vertices = self._get_vertices(part_list1, part_list2)
        vert1 = vertices[0].tensor[0] # Gamma(0,3,1)
        #vert2 = ls.Gamma(1,2,4)
        vert2 = vertices[1].tensor[0] # Gamma(1,4,2) should be 1,2,4.
        #print(vert1, vert2)
        
        coup1 = vertices[0].coupling[0][0] # -0.313451j
        coup2 = vertices[1].coupling[0][0] # -0.313451j
        
        ph = self.internal[0]
        propagator = -1j*ls.Metric(0,1)
        #print(ls.Momentum(ph.mom_array, 10, ph.mom_index+1)*ls.Metric(10, 11)*ls.Momentum(ph.mom_array, 11, ph.mom_index+1))
        propagator /= ls.Momentum(ph.mom_array, 10, ph.mom_index+1)*ls.Metric(10, 11)*ls.Momentum(ph.mom_array, 11, ph.mom_index+1)
        # Propagator(0,1)
        
        M_t_channel = coup1*coup2*vert1*vert2*spinor1*spinor2*spinor3*spinor4*propagator
    
        return M_t_channel
    
    def _u_channel_amplitude(self, spinor1, spinor2, spinor3, spinor4):
        part_list1 = [self.incoming[0], self.outgoing[1]] + self.internal
        part_list2 = [self.incoming[1], self.outgoing[0]] + self.internal
        
        vertices = self._get_vertices(part_list1, part_list2)
        vert1 = vertices[0].tensor[0] # Gamma(0,4,1)
        vert2 = vertices[1].tensor[0] # Gamma(1,3,2)
        
        coup1 = vertices[0].coupling[0][0] # -0.313451j
        coup2 = vertices[1].coupling[0][0] # -0.313451j
        
        ph = self.internal[0]
        propagator = -1j*ls.Metric(0,1)
        propagator /= ls.Momentum(ph.mom_array, 10, ph.mom_index+2)*ls.Metric(10, 11)*ls.Momentum(ph.mom_array, 11, ph.mom_index+2)
        # Propagator(0,1)
        
        M_u_channel = coup1*coup2*vert1*vert2*spinor1*spinor2*spinor3*spinor4*propagator
        
        return M_u_channel
        
    def _get_amplitude(self):
        total = 0
        for spin1 in range(2):
            for spin2 in range(2):
                for spin3 in range(2):
                    for spin4 in range(2):
                        setattr(self.incoming[0],'spin',spin1)
                        setattr(self.incoming[1],'spin',spin2)
                        setattr(self.outgoing[0],'spin',spin3)
                        setattr(self.outgoing[1],'spin',spin4)
                        
                        spinor1 = self.incoming[0].get_spinor() # e-  SpinorU(1)
                        spinor2 = self.incoming[1].get_spinor() # e+  SpinorVBar(2)
                        spinor3 = self.outgoing[0].get_spinor() # mu- SpinorUBar(3)
                        spinor4 = self.outgoing[1].get_spinor() # mu+ SpinorV(4)
                        
                        if (all(pid > 0 for pid in self.pids)) or (all(pid < 0 for pid in self.pids)):
                            M_tot = 0
                            M_tot += self._t_channel_amplitude(spinor1, spinor2, spinor3, spinor4)
                            if self.outgoing[0].pid == self.outgoing[1].pid:
                                M_tot += self._u_channel_amplitude(spinor1, spinor2, spinor3, spinor4)
                            total += M_tot*M_tot.conjugate()
                        
                        elif (any(pid > 0 for pid in self.pids)) and (any(pid < 0 for pid in self.pids)):
                            scatter = False
                            scatter1, scatter2 = False, False
                            
                            annihilation = False
                            annihilation1, annihilation2 = False, False
                            
                            # Check whether scatter vertex exists
                            for vertex in self.model.vertices:
                                particle_list = vertex.particles
                                particle_pids = [part.pdg_code for part in particle_list]
                                particle_pids = sorted(particle_pids)
                                #print(particle_pids)
                                
                                part_list1 = [self.incoming[0], self.outgoing[0]] + self.internal
                                part_list2 = [self.incoming[1], self.outgoing[1]] + self.internal
                                part_list1_pids = [part.pid for part in part_list1]
                                part_list2_pids = [part.pid for part in part_list2]
                                part_list1_pids = sorted(part_list1_pids)
                                part_list2_pids = sorted(part_list2_pids)
                                #print(part_list1_pids, part_list2_pids)
                                
                                if particle_pids == part_list1_pids:
                                    scatter1 = True
                                if particle_pids == part_list2_pids:
                                    scatter2 = True
                                
                                if scatter1 and scatter2:
                                    # print(scatter1, scatter2)
                                    # print(particle_pids)
                                    # print(part_list1_pids, part_list2_pids)
                                    # print(spin1, spin2, spin3, spin4)
                                    scatter = True
                                    break
                                
                            # Check whether annihilation vertex exists
                            for vertex in self.model.vertices:
                                particle_list = vertex.particles
                                particle_pids = [part.pdg_code for part in particle_list]
                                particle_pids = sorted(particle_pids)
                                
                                part_list1 = self.incoming + self.internal
                                part_list2 = self.outgoing + self.internal
                                part_list1_pids = [part.pid for part in part_list1]
                                part_list2_pids = [part.pid for part in part_list2]
                                part_list1_pids = sorted(part_list1_pids)
                                part_list2_pids = sorted(part_list2_pids)
                                
                                if particle_pids == part_list1_pids:
                                    annihilation1 = True
                                if particle_pids == part_list2_pids:
                                    annihilation2 = True
                                if annihilation1 and annihilation2:
                                    annihilation = True
                                    break
                            
                            # print(scatter, annihilation)
                            M_tot = 0
                            if scatter:
                                M_tot += self._t_channel_amplitude(spinor1, spinor2, spinor3, spinor4)
                                if self.outgoing[0].pid == self.outgoing[1].pid:
                                    M_tot += self._u_channel_amplitude(spinor1, spinor2, spinor3, spinor4)
                            if annihilation:
                                M_tot += self._s_channel_amplitude(spinor1, spinor2, spinor3, spinor4)
                                
                            total += M_tot*M_tot.conjugate()
                        
                        # print("M: {}".format(M_tot))
                        # print("M*: {}".format(M_tot.conjugate()))
                        # print("MM*: {}".format(M_tot*M_tot.conjugate()))
        
        #total /= 4 # Spin averaged.
        
        #print("Total amplitude: {}".format(total))
        
        if total._scalar:
            return complex(total._array)
        else:
            return total
    