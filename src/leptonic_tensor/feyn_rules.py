import vertex_class as vc
import lorentz_structures as ls
import ufo_grammer

class FeynRules:
    def __init__(self, model, incoming_particles, outgoing_particles, internal_particles):
        '''
        Calculate spin-summed/averaged amplitude for incoming to outgoing particles
        with internal particles.
        '''
        self.model = model
        self.incoming = incoming_particles
        self.outgoing = outgoing_particles
        self.internal = internal_particles
        self.vertices = self._get_vertices()
        # self.propagator = self._get_propagator()
        self.amplitude = self._get_amplitude()
        
    def _get_vertices(self):
        part_list1, part_idx1 = [], []
        part_list2, part_idx2 = [], []
        for part in self.incoming:
            part_list1.append(part)
            part_idx1.append(part.index)
        for part in self.outgoing:
            part_list2.append(part)
            part_idx2.append(part.index)
        for part in self.internal:
            part_list1.append(part)
            part_idx1.append(part.index)
            part_list2.append(part)
            part_idx2.append(part.index + 1)
        vert1 = vc.Vertex(self.model, part_list1, part_idx1)
        vert2 = vc.Vertex(self.model, part_list2, part_idx2)
        return [vert1, vert2]
    
    # def _get_propagator(self):
        
        
    def _get_amplitude(self):
        vert1 = self.vertices[0].tensor[0] # Gamma(0,2,1)
        vert2 = self.vertices[1].tensor[0] # Gamma(1,4,3)
        
        coup1 = self.vertices[0].coupling[0][0] # -0.313451j
        coup2 = self.vertices[1].coupling[0][0] # -0.313451j
        
        ufo = ufo_grammer.UFOParser(self.model.model)
        ph = self.internal[0]
        propagator = ufo("complex(0,-1) * Metric(0, 1)")
        propagator /= ls.Momentum(ph.mom_array, 10, ph.mom_index)*ls.Metric(10, 11)*ls.Momentum(ph.mom_array, 11, ph.mom_index)
        # Propagator(0,1)
        
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
                        
                        M = coup1*coup2*vert1*vert2*spinor1*spinor2*spinor3*spinor4*propagator
                        # print("M: {}".format(M))
                        # print("M*: {}".format(M.conjugate()))
                        # print("MM*: {}".format(M*M.conjugate()))
                        total += M*M.conjugate()
        
        #total /= 4 # Spin averaged.
        
        #print("Total amplitude: {}".format(total))
        
        if total._scalar:
            return complex(total._array)
        else:
            return total
    