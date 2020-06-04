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
            #lorentz = vertex.lorentz[0].structure
            #coupling = vertex.couplings[(0,0)].value
            return [lorentz, coupling, color]
        else:
            pass
    
    def _get_propagator(self, internal):
        propagator = self.model.propagator_map[internal.propagator]
        
        return '(({}) / ({}))'.format(propagator.numerator, propagator.denominator)
    
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
        prefact = vertex_in[1] + vertex_out[1] + vertex_in[2] + vertex_out[2]
        lorentz = vertex_in[0] + vertex_out[0]
        
        amp = prefact + [outgoing_wavefunctions[1], lorentz[1], outgoing_wavefunctions[0], propagators[0], incoming_wavefunctions[0], lorentz[0], incoming_wavefunctions[1]]
        Amp = '*'.join(amp)
        return Amp
    
            