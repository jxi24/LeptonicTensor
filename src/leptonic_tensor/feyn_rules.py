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
        
    def _get_vertex(self, incoming, outgoing):
        particles_list = incoming + outgoing
        pids = self._get_pids(particles_list)
        vertex = self.model.vertex_map[tuple(pids)]
        lorentz = vertex.lorentz[0].structure
        coupling = vertex.couplings[(0,0)].value
        return [lorentz, coupling]
    
    def _get_propagator(self, internal):
        propagator = self.model.propagator_map[internal.propagator]
        return '(({}) / ({}))'.format(propagator.numerator, propagator.denominator)
    
    def _amplitude(self):
        amp = []
        prefact = []
        incoming_wavefunctions = []
        outgoing_wavefunctions = []
        propagators = []
        for part in self.incoming:
            incoming_wavefunctions.append(part.wavefunction[0])
        for part in self.outgoing:
            outgoing_wavefunctions.append(part.wavefunction[1])
        for part in self.internal:
            propagators.append(self._get_propagator(part))
        return incoming_wavefunctions
        
    
            