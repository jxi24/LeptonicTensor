import models
import particle_class as pc
import lorentz_class as lc
import coupling_class as cc
import ufo_grammer

class Model:
    def __init__(self, name, all_models):
        self.name = name
        self.model = all_models[name]
        self.couplings = self.model.all_couplings
        self.parameters = self.model.all_parameters
        self.particle_map = self._particle_map()
        self.vertex_map = self._vertex_map()
        self.propagator_map = self._propagator_map()
        self.lorentz_map = self._lorentz_map()
        self.parameter_map = self._parameter_map()
        self.coupling_map = self._coupling_map()
        

    def _particle_map(self):
        particles = self.model.all_particles
        part_map = {}
        for part in particles:
            if part.spin < 0:
                continue
            test = pc.ParticleInfo(part)
            part_map[test.pid] = test

        # for key, value in part_map.items():
        #     print(key, value)
        return part_map

    def _vertex_map(self):
        vertices = self.model.all_vertices
        vert_map = {}
        for vertex in vertices:
            particles = vertex.particles
            # print(vertex.lorentz, particles)
            pids = []
            for part in particles:
                pids.append(part.pdg_code)
            pids.sort()
            # print(pids)
            vert_map[tuple(pids)] = vertex
        return vert_map

    def _propagator_map(self):
        propagators = self.model.all_propagators
        prop_map = {}
        for prop in propagators:
            prop_map[prop.name] = prop
        return prop_map

    def _lorentz_map(self):
        # Particle class/UFO convention: spin = 2*S
        lorentzs = self.model.all_lorentz
        lorentz_map = {}
        for lorentz in lorentzs:
            try:
                ltz = lc.LorentzInfo(lorentz)
                lorentz_map[tuple(ltz.spins), ltz.name] = [ltz.structure, ltz.indices]
            except:
                pass
        return lorentz_map
    
    def _parameter_map(self):
        parameter_map = {}
        for parameter in self.parameters:
            try:
                print(parameter.name, parameter.value)
                param = ufo_grammer.ufo("{} := {}".format(parameter.name, parameter.value))
                parameter_map[parameter.name] = param
            except:
                print('failed')
            # if parameter.nature == 'external':
            #     param = ufo_grammer.ufo("{} := {}".format(parameter.name, parameter.value))
            #     parameter_map[parameter.name] = param
            # elif parameter.nature == 'internal':
            #     if parameter.name == 'ZERO':
            #         parameter_map[parameter.name] = 0.0
            #     try:
            #         param = ufo_grammer.ufo(parameter.value)
            #         parameter_map[parameter.name] = param
            #     except:
            #         pass
        return parameter_map
    
    def _coupling_map(self):
        coupling_map = {}
        for coupling in self.couplings:
            cp = cc.CouplingInfo(coupling)
            coupling_map[cp.name] = cp.value
        return coupling_map

    @property
    def particles(self):
        return (', ').join(list(self.particle_map.keys()))