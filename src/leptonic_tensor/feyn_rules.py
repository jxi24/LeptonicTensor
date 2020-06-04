class FeynRules:
    def __init__(self, model, incoming_particles, outgoing_particles, internal_particles):
        self.model = model
        self.incoming = incoming_particles
        self.outgoing = outgoing_particles
        self.internal = internal_particles
        
    def _get_labels(self):
        labels = [i for i in range(1,len(self.incoming) + len(self.outgoing) + 1)]
        return labels
        
    def _get_pids(self, particles_list):
        pids = []
        for part in particles_list:
            pids.append(part.pid)
        return pids
        
    def _get_vertex(self, real_particles, virtual_particles):
        particles_list = []
        for part in real_particles:
            particles_list.append(part)
        for part in virtual_particles:
            particles_list.append(part)
        pids = self._get_pids(particles_list)
        vertex = self.model.vertex_map[tuple(pids)]
        if len(vertex.color) == 1:
            lorentz = [ltz.structure for ltz in vertex.lorentz]
            color = vertex.color
            coupling = [vertex.couplings[(0,i)].value for i in range(len(lorentz))]
            return [lorentz, coupling, color]
        else:
            pass
    
    def _get_propagator(self, internal):
        propagator = self.model.propagator_map[internal.propagator]
        num = propagator.numerator.replace('Mass(id)',str(internal.mass)).replace('Width(id)',str(internal.width)).replace('id',internal.name)
        denom = propagator.denominator.replace('Mass(id)',str(internal.mass)).replace('Width(id)',str(internal.width)).replace('id',internal.name)
        return '(({}) / ({}))'.format(num, denom)
    
    def amplitude(self):
        incoming_wavefunctions = []
        outgoing_wavefunctions = []
        propagators = []
        labels = self._get_labels()
        i = 0
        for part in self.incoming:
            part.get_wavefunction(labels[i])
            incoming_wavefunctions.append(part.wavefunction[0])
            i += 1
        for part in self.outgoing:
            part.get_wavefunction(labels[i])
            outgoing_wavefunctions.append(part.wavefunction[1])
            i += 1
        for part in self.internal:
            propagators.append(self._get_propagator(part))
        
        vertex_in = self._get_vertex(self.incoming, self.internal)
        vertex_out = self._get_vertex(self.outgoing, self.internal)
        #prefact = vertex_in[1] + vertex_out[1] + vertex_in[2] + vertex_out[2]
        #lorentz = vertex_in[0] + vertex_out[0]
        V_in = ''
        for i in range(len(vertex_in[0])):
            V_in += '(' + vertex_in[1][i] + ')' + '*' + vertex_in[2][0] + '*' + '(' + vertex_in[0][i] + ')'
        V_out = ''
        for i in range(len(vertex_out[0])):
            V_out += '(' + vertex_out[1][i] + ')' + '*' + vertex_out[2][0] + '*' + '(' + vertex_out[0][i] + ')'
        
        amp = [outgoing_wavefunctions[1], V_out, outgoing_wavefunctions[0], propagators[0], incoming_wavefunctions[0], V_in, incoming_wavefunctions[1]]
        Amp = '*'.join(amp)
        return Amp
    
            