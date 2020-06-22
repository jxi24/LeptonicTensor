class FeynRules:
    def __init__(self, model, incoming_particles, outgoing_particles, internal_particles):
        '''
        Initiate object with Model instance and Particle instances.
        '''
        self.model = model
        self.incoming = incoming_particles
        self.outgoing = outgoing_particles
        self.internal = internal_particles
        
    # def _get_labels(self):
    #     labels = [i for i in range(1,len(self.incoming) + len(self.outgoing) + 1)]
    #     return labels
        
    def _get_vertex(self, real_particles, virtual_particles):
        particles_list = []
        for part in real_particles:
            particles_list.append(part)
        for part in virtual_particles:
            particles_list.append(part)
        pids = [part.info.pid for part in particles_list]
        pids.sort()
        vertex = self.model.vertex_map[tuple(pids)]
        if len(vertex.color) == 1:
            lorentz = [] 
            for ltz in vertex.lorentz:
                if ('Gamma' in ltz.structure) & ('Gamma5' != ltz.structure):
                    idx = ltz.structure.index('Gamma')
                    l = [s for s in ltz.structure]
                    # Gamma(1,2,3)
                    l[idx+6] = str(virtual_particles[0].momentum)
                    l[idx+8] = str(real_particles[0].momentum)
                    l[idx+10] = str(real_particles[1].momentum)
                    newl = ''.join(l)
                    lorentz.append(newl)
            color = vertex.color
            coupling = [vertex.couplings[(0,i)].value for i in range(len(lorentz))]
            return [lorentz, coupling, color]
        else:
            pass
    
    def _get_propagator(self, internal):
        propagator = self.model.propagator_map[internal.info.propagator]
        if 'Metric' in propagator.numerator:
            idx = propagator.numerator.index('Metric')
            l = [s for s in propagator.numerator]
            # Metric(1, 2)
            l[idx+7] = 'mu'
            l[idx+10] = 'nu'
            num = ''.join(l)
        num = num.replace('Mass(id)',str(internal.info.mass)).replace('Width(id)',str(internal.info.width)).replace('id',internal.info.name)
        denom = propagator.denominator.replace('Mass(id)',str(internal.info.mass)).replace('Width(id)',str(internal.info.width)).replace('id',internal.info.name).replace("'mu'", str(internal.momentum))
        return '(({}) / ({}))'.format(num, denom)
    
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
    def amplitude(self):
        wavefunctions = []
        for part in self.incoming:
            wf = part.wavefunction()
            wavefunctions.append(wf)
        for part in self.outgoing:
            wf = part.wavefunction()
            wavefunctions.append(wf)
            
        propagators = []
        for part in self.internal:
            propagators.append(self._get_propagator(part))
            
        vertex_in = self._get_vertex(self.incoming, self.internal)
        vertex_out = self._get_vertex(self.outgoing, self.internal)
        
        V_in = ''
        for i in range(len(vertex_in[0])):
            V_in += '(' + vertex_in[1][i] + ')' + '*' + vertex_in[2][0] + '*' + '(' + vertex_in[0][i] + ')'
            if i != (len(vertex_in[0])-1):
                V_in += '+'
        if len(vertex_in[0]) > 1:
            V_in = '(' + V_in + ')'
            
        V_out = ''
        for i in range(len(vertex_out[0])):
            V_out += '(' + vertex_out[1][i] + ')' + '*' + vertex_out[2][0] + '*' + '(' + vertex_out[0][i] + ')'
            if i != (len(vertex_out[0])-1):
                V_out += '+'
        if len(vertex_out[0]) > 1:
            V_out = '(' + V_out + ')'        
    
        inNames = [part.info.name for part in self.incoming]
        outNames = [part.info.name for part in self.outgoing]
        propNames = [part.info.name for part in self.internal]
        amp = wavefunctions + propagators + [V_in, V_out]
        amp = '*'.join(amp)
        result = 'Amplitude for {} -> {} via {}. Amplitude format (wavefunctions, propagators, vertices):\n%s'.format(inNames, outNames, propNames) %amp
        return result
