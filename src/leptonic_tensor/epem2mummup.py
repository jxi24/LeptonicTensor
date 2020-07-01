import models
import itertools
import model_class as mc
import particle_class as pc
import lorentz_class as lc
import vertex_class as vc
import feyn_rules
import numpy as np
import ufo_grammer
import lorentz_tensor as lt

def main():
    all_models = models.discover_models()
    model = mc.Model('Models.SM_NLO', all_models)
    
    elec = pc.Particle(model, 11, 1, True)
    antielec = pc.Particle(model, -11, 2, True)
    muon = pc.Particle(model, 13, 3, False)
    antimuon = pc.Particle(model, -13, 4, False)
    photon = pc.Particle(model, 22, 0, False)
    
    InP = [elec, antielec]
    OutP = [muon, antimuon]
    IntP = [photon]
    
    l = lc.Lorentz(model, [1,1,2], 'FFV1', [1,2,5])
    print(l)
    
    vert1 = vc.Vertex(model, [elec, antielec, photon], [elec.momentum,antielec.momentum,photon.momentum])
    #V1 = lt.Tensor(vert1.structure, vert1.indices)
    print(vert1)
    
    gl1 = pc.Particle(model, 21, 1, True)
    gl2 = pc.Particle(model, 21, 2, False)
    gl3 = pc.Particle(model, 21, 3, False)
    
    #vert2 = vc.Vertex(model, [gl1, gl2, gl3], [1,2,3])
    #print(vert2)
    
    print(model.parameter_map)
    #print(model.coupling_map)
    print(model.parameter_map['aEWM1'])
    print(model.parameter_map['ee'])
    print(model.parameter_map['aEW'])

    # pids1 = [particles[-11].pid,
    #          particles[11].pid,
    #          particles[22].pid]
    # pids1.sort()
    # print(pids1)
    # vertex1 = model.vertex_map[tuple(pids1)]
    # print(vertex1.lorentz[0].structure, vertex1.couplings[(0, 0)].value)
    # vertexA = vertex1.lorentz[0].structure.replace('3', 'mu').replace('2', 'e-').replace('1', 'e+')

    # pids2 = [particles[-13].pid,
    #          particles[13].pid,
    #          particles[22].pid]
    # pids2.sort()
    # print(pids2)
    # vertex2 = model.vertex_map[tuple(pids2)]
    # print(vertex2.lorentz[0].structure, vertex2.couplings[(0, 0)].value)
    # vertexB = vertex2.lorentz[0].structure.replace('3', 'nu').replace('2', 'mu-').replace('1', 'mu+')

    # propagator = model.propagator_map[particles[22].propagator]
    # propagatorA = propagator.numerator.replace('1','mu').replace('2','nu') + '/' + propagator.denominator
    # print("({})/({})".format(propagator.numerator, propagator.denominator))

    # amp = ['ubar(p2)', vertexA, 'v(p1)', propagatorA, 'vbar(p3)', vertexB, 'u(p4)']
    # print('*'.join(amp))

    # amp1 = feyn_rules.FeynRules(model, incoming_particles, outgoing_particles, internal_particles)
    # print('Incoming wavefunction for {}: {}'.format(particles['e-'].name, particles['e-'].wavefunction[0]))
    # print("\n")
    # print(amp1.amplitude())
    # print("\n")
    # print(amp1._get_vertex(outgoing_particles, internal_particles))
    # amp2 = feyn_rules.FeynRules(model, incoming_particles, outgoing_particles, [particles[23]])
    # print(amp2.amplitude())
    
    Amp1 = feyn_rules.FeynRules(model, InP, OutP, IntP)
    #print(Amp1.amplitude())
    
    #Zboson = Particle(model, 23, 0, False)
    #Amp2 = feyn_rules.FeynRules(model, InP, OutP, [Zboson])
    #print(Amp2.amplitude())

if __name__ == '__main__':
    main()
