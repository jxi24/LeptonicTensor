import models
import particle_class as pc
import lorentz_class as lc
import coupling_class as cc
import propagator_class as propc
import ufo_grammer

class Model:
    def __init__(self, name, all_models):
        self.name = name
        self.model = all_models[name]
        ufo = ufo_grammer.UFOParser(self.model)
        self.couplings = self.model.all_couplings
        self.parameters = self.model.all_parameters
        self.lorentzs = self.model.all_lorentz
        self.propagators = self.model.all_propagators
        self.vertices = self.model.all_vertices
        
        self.particle_map = self._particle_map()
        self.vertex_map = self._vertex_map()
        self.propagator_map = self._propagator_map()
        self.lorentz_map = self._lorentz_map()
        parameter_coupling_maps = self._parameter_coupling_map()
        self.parameter_map = parameter_coupling_maps[0]
        self.coupling_map = parameter_coupling_maps[1]
        

    def _particle_map(self):
        particles = self.model.all_particles
        part_map = {}
        for part in particles:
            if part.spin < 0:
                continue
            test = pc.ParticleInfo(part)
            part_map[test.pid] = test
        return part_map

    def _vertex_map(self):
        vert_map = {}
        for vertex in self.vertices:
            particles = vertex.particles
            pids = []
            for part in particles:
                pids.append(part.pdg_code)
            pids.sort()
            vert_map[tuple(pids)] = vertex
        return vert_map

    def _propagator_map(self):
        prop_map = {}
        for prop in self.propagators:
            try:
                pgt = propc.PropagatorInfo(self.model, prop)
                prop_map[pgt.name] = [pgt.structure, pgt.indices, pgt.tensor]
            except:
                pass
        return prop_map

    def _lorentz_map(self):
        # Spin here = 2*S. Update on Dec 3, 2020: spin is now back to UFO convention 2S+1.
        lorentz_map = {}
        for lorentz in self.lorentzs:
            try:
                ltz = lc.LorentzInfo(self.model, lorentz)
                lorentz_map[tuple(ltz.spins), ltz.name] = [ltz.structure, ltz.indices, ltz.tensor]
            except:
                pass
        return lorentz_map
    
    def _parameter_coupling_map(self):
        parameter_map = {}
        coupling_map = {}
        ufo = ufo_grammer.UFOParser(self.model)
        for parameter in self.parameters:
            try:
                ufo("{} := {}".format(parameter.name, parameter.value))
                parameter_map[parameter.name] = ufo(parameter.name)
            except:
                parameter_map[parameter.name] = None
            
        for coupling in self.couplings:
            try:
                ufo("{} := {}".format(coupling.name, coupling.value))
                coupling_map[coupling.name] = ufo(coupling.name)
            except:
                coupling_map[coupling.name] = None
        
        return [parameter_map, coupling_map]
    
    @property
    def particles(self):
        return (', ').join(list(self.particle_map.keys()))