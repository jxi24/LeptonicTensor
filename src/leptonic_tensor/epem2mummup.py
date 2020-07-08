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
import lorentz_structures as ls
import matplotlib.pyplot as plt

def plot_amp():
    # Initialize variables.
    costheta_list = np.linspace(-1,1,1000)
    phi = 2*np.pi*np.random.uniform()
    comput_sol = []
    analytic_sol = []
    
    # Compute analytic and computational amplitude for each cos(theta).
    for costheta in costheta_list:
        sintheta = np.sqrt(1-costheta**2)
        mom = np.array(
            [[10, 0, 0, 10],
             [10, 0, 0, -10],
             [10, 10*sintheta*np.cos(phi),
              10*sintheta*np.sin(phi), 10*costheta],
             [10, -10*sintheta*np.cos(phi),
              -10*sintheta*np.sin(phi), -10*costheta]])
        q = mom[0, :] + mom[1, :]
        mom = np.append(mom, [q], axis=0)
        mom = np.append(mom, [mom[0] - mom[2]], axis=0)
        mom = np.append(mom, [mom[0] - mom[3]], axis=0)
        
        q2 = float((ls.Momentum(mom, 0, 4)*ls.Metric(0, 1)*ls.Momentum(mom, 1, 4)))
        t = float((ls.Momentum(mom, 0, 5)*ls.Metric(0, 1)*ls.Momentum(mom, 1, 5)))
        u = float((ls.Momentum(mom, 0, 6)*ls.Metric(0, 1)*ls.Momentum(mom, 1, 6)))
    
        elec = pc.Particle(model, 11, mom, 1, 0, True)
        antielec = pc.Particle(model, -11, mom, 2, 1, True)
        muon = pc.Particle(model, 13, mom, 3, 2, False)
        antimuon = pc.Particle(model, -13, mom, 4, 3, False)
        photon = pc.Particle(model, 22, mom, 0, 4, False)
    
        InP = [elec, antielec]
        OutP = [muon, antimuon]
        IntP = [photon]
        
        Amp1 = feyn_rules.FeynRules(model, InP, OutP, IntP)
        analytic = 8*ee**4*(t*t+u*u)/q2**2
        
        comput_sol.append(Amp1.amplitude.real)
        analytic_sol.append(analytic)
    
    # Plot amplitudes and amplitude error vs cos(theta).
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    
    axs[0,].plot(costheta_list, comput_sol, color='red')
    axs[0,].plot(costheta_list, analytic_sol, color='blue')
    axs[0,].set_xlabel("Cos(theta)")
    axs[0,].legend(["Computational","Analytic"])    
    
    axs[1,].plot(costheta_list, np.subtract(comput_sol, analytic_sol))
    axs[1,].set_xlabel("Cos(theta)")
    
    fig.suptitle("Comparison of amplitudes for phi={}".format(phi))
    fig.savefig("ee2mumu.pdf")

def main():
    # Initialize model.
    all_models = models.discover_models()
    model = mc.Model('Models.SM_NLO', all_models)
    
    # Initialize momenta.
    costheta = 2*np.random.uniform()-1
    sintheta = np.sqrt(1-costheta**2)
    phi = 2*np.pi*np.random.uniform()
    mom = np.array(
        [[10, 0, 0, 10],
         [10, 0, 0, -10],
         [10, 10*sintheta*np.cos(phi),
          10*sintheta*np.sin(phi), 10*costheta],
         [10, -10*sintheta*np.cos(phi),
          -10*sintheta*np.sin(phi), -10*costheta]])
    q = mom[0, :] + mom[1, :]
    mom = np.append(mom, [q], axis=0)
    mom = np.append(mom, [mom[0] - mom[2]], axis=0)
    mom = np.append(mom, [mom[0] - mom[3]], axis=0)
    
    # Initialize particles with spin index and momentum index.
    elec = pc.Particle(model, 11, mom, 1, 0, True)
    antielec = pc.Particle(model, -11, mom, 2, 1, True)
    muon = pc.Particle(model, 13, mom, 3, 2, False)
    antimuon = pc.Particle(model, -13, mom, 4, 3, False)
    photon = pc.Particle(model, 22, mom, 0, 4, False)
    
    InP = [elec, antielec]
    OutP = [muon, antimuon]
    IntP = [photon]

    # Compute amplitude.
    Amp1 = feyn_rules.FeynRules(model, InP, OutP, IntP)
    
    # Compare with analytic solution.
    q2 = float((ls.Momentum(mom, 0, 4)*ls.Metric(0, 1)*ls.Momentum(mom, 1, 4)))
    t = float((ls.Momentum(mom, 0, 5)*ls.Metric(0, 1)*ls.Momentum(mom, 1, 5)))
    u = float((ls.Momentum(mom, 0, 6)*ls.Metric(0, 1)*ls.Momentum(mom, 1, 6)))
    
    ee = model.parameter_map["ee"]
    analytic = 8*ee**4*(t*t+u*u)/q2**2
    
    print("Computational solution: {}".format(Amp1.amplitude))
    print("Analytic solution: {}".format(analytic))
    print("Ratio: {}".format(analytic/Amp1.amplitude))
    
if __name__ == '__main__':
    main()
