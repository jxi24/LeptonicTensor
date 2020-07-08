import vertex_class as vc
import lorentz_structures as ls
import ufo_grammer

class FeynRules:
    def __init__(self, model, incoming_particles, outgoing_particles, internal_particles):
        '''
        Initiate object with Model instance and Particle instances.
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
            part_idx1.append(part.spin_index)
        for part in self.outgoing:
            part_list2.append(part)
            part_idx2.append(part.spin_index)
        for part in self.internal:
            part_list1.append(part)
            part_idx1.append(part.spin_index)
            part_list2.append(part)
            part_idx2.append(part.spin_index + 1)
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
                        #print(self.outgoing[1].spin)
                        
                        spinor1 = self.incoming[0].get_spinor() # e-  SpinorU(1)
                        spinor2 = self.incoming[1].get_spinor() # e+  SpinorVBar(2)
                        spinor3 = self.outgoing[0].get_spinor() # mu- SpinorUBar(3)
                        spinor4 = self.outgoing[1].get_spinor() # mu+ SpinorV(4)
                        #print(self.outgoing[1].spin)
                        #print(spinor4)
                        
                        M = coup1*coup2*vert1*vert2*spinor1*spinor2*spinor3*spinor4*propagator
                        # print("M: {}".format(M))
                        # print("M*: {}".format(M.conjugate()))
                        # print("MM*: {}".format(M*M.conjugate()))
                        total += M*M.conjugate()
        
        # print(vert1, vert2)
        # print(coup1, coup2)
        # print(propagator)
        # print(spinor1, spinor2, spinor3, spinor4)
        
        #total /= 4 # Spin averaged.
        
        #print("Total amplitude: {}".format(total))
        
        if total._scalar:
            return complex(total._array)
        
        # return coup1*coup2*vert1*vert2*spinor1*spinor2*spinor3*spinor4
        
        # wavefunctions = []
        # for part in self.incoming:
        #     wf = part.wavefunction()
        #     wavefunctions.append(wf)
        # for part in self.outgoing:
        #     wf = part.wavefunction()
        #     wavefunctions.append(wf)
            
        # propagators = []
        # for part in self.internal:
        #     propagators.append(self._get_propagator(part))
            
        # vertex_in = self._get_vertex(self.incoming, self.internal)
        # vertex_out = self._get_vertex(self.outgoing, self.internal)
        
        # V_in = ''
        # for i in range(len(vertex_in[0])):
        #     V_in += '(' + vertex_in[1][i] + ')' + '*' + vertex_in[2][0] + '*' + '(' + vertex_in[0][i] + ')'
        #     if i != (len(vertex_in[0])-1):
        #         V_in += '+'
        # if len(vertex_in[0]) > 1:
        #     V_in = '(' + V_in + ')'
            
        # V_out = ''
        # for i in range(len(vertex_out[0])):
        #     V_out += '(' + vertex_out[1][i] + ')' + '*' + vertex_out[2][0] + '*' + '(' + vertex_out[0][i] + ')'
        #     if i != (len(vertex_out[0])-1):
        #         V_out += '+'
        # if len(vertex_out[0]) > 1:
        #     V_out = '(' + V_out + ')'        
    
        # inNames = [part.info.name for part in self.incoming]
        # outNames = [part.info.name for part in self.outgoing]
        # propNames = [part.info.name for part in self.internal]
        # amp = wavefunctions + propagators + [V_in, V_out]
        # amp = '*'.join(amp)
        # result = 'Amplitude for {} -> {} via {}. Amplitude format (wavefunctions, propagators, vertices):\n%s'.format(inNames, outNames, propNames) %amp
        # return result

    
    # def _get_propagator(self, internal):
    #     propagator = self.model.propagator_map[internal.info.propagator]
    #     if 'Metric' in propagator.numerator:
    #         idx = propagator.numerator.index('Metric')
    #         l = [s for s in propagator.numerator]
    #         # Metric(1, 2)
    #         l[idx+7] = 'mu'
    #         l[idx+10] = 'nu'
    #         num = ''.join(l)
    #     num = num.replace('Mass(id)',str(internal.info.mass)).replace('Width(id)',str(internal.info.width)).replace('id',internal.info.name)
    #     denom = propagator.denominator.replace('Mass(id)',str(internal.info.mass)).replace('Width(id)',str(internal.info.width)).replace('id',internal.info.name).replace("'mu'", str(internal.momentum))
    #     return '(({}) / ({}))'.format(num, denom)
    
    # def amplitude(self):
    #     # All particles will be considered outgoing.
    #     incoming_wavefunctions = []
    #     outgoing_wavefunctions = []
    #     propagators = []
    #     labels = self._get_labels()
    #     i = 0
    #     for part in self.incoming:
    #         part.get_wavefunction(labels[i])
    #         incoming_wavefunctions.append(part.wavefunction[0])
    #         i += 1
    #     for part in self.outgoing:
    #         part.get_wavefunction(labels[i])
    #         outgoing_wavefunctions.append(part.wavefunction[1])
    #         i += 1
    #     for part in self.internal:
    #         propagators.append(self._get_propagator(part))
        
    #     vertex_in = self._get_vertex(self.incoming, self.internal)
    #     vertex_out = self._get_vertex(self.outgoing, self.internal)
    #     #prefact = vertex_in[1] + vertex_out[1] + vertex_in[2] + vertex_out[2]
    #     #lorentz = vertex_in[0] + vertex_out[0]
    #     V_in = ''
    #     if len(vertex_in[0]) > 1:
    #             V_in += '('
    #     for i in range(len(vertex_in[0])):
    #         V_in += '(' + vertex_in[1][i] + ')' + '*' + vertex_in[2][0] + '*' + '(' + vertex_in[0][i] + ')'
    #         if i != (len(vertex_in[0])-1):
    #             V_in += '+'
    #     if len(vertex_in[0]) > 1:
    #         V_in += ')'
    #     V_out = ''
    #     if len(vertex_out[0]) > 1:
    #             V_out += '('
    #     for i in range(len(vertex_out[0])):
    #         V_out += '(' + vertex_out[1][i] + ')' + '*' + vertex_out[2][0] + '*' + '(' + vertex_out[0][i] + ')'
    #         if i != (len(vertex_out[0])-1):
    #             V_out += '+'
    #     if len(vertex_out[0]) > 1:
    #             V_out += ')'
        
    #     amp = [outgoing_wavefunctions[1], V_out, outgoing_wavefunctions[0], propagators[0], incoming_wavefunctions[0], V_in, incoming_wavefunctions[1]]
    #     Amp = '*'.join(amp)
    #     return Amp
    