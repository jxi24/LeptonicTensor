class FeynRules:
    def __init__(self, model, incoming_particles, outgoing_particles, internal_particles):
        self.model = model
        self.incoming = incoming_particles
        self.outgoing = outgoing_particles
        self.internal = internal_particles
        
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
        lorentz = vertex.lorentz[0].structure
        coupling = vertex.couplings[(0,0)].value
        return [lorentz, coupling]
    
    def _get_propagator(self, internal):
        propagator = self.model.propagator_map[internal.propagator]
        return '(({}) / ({}))'.format(propagator.numerator, propagator.denominator)
    
    def _amplitude(self):
        incoming_wavefunctions = []
        outgoing_wavefunctions = []
        propagators = []
        
        for part in self.incoming:
            incoming_wavefunctions.append(part.wavefunction[0])
        for part in self.outgoing:
            outgoing_wavefunctions.append(part.wavefunction[1])
        for part in self.internal:
            propagators.append(self._get_propagator(part))
        
        vertex_in = self._get_vertex(self.incoming, self.internal)
        vertex_out = self._get_vertex(self.outgoing, self.internal)
        prefact = []
        prefact.append(vertex_in[1])
        prefact.append(vertex_out[1])
        lorentz = []
        lorentz.append(vertex_in[0])
        lorentz.append(vertex_out[0])
        
        amp = [prefact[0], prefact[1], outgoing_wavefunctions[1], lorentz[1], outgoing_wavefunctions[0], propagators[0], incoming_wavefunctions[0], lorentz[0], incoming_wavefunctions[1]]
        Amp = '*'.join(amp)
        return Amp
    
            