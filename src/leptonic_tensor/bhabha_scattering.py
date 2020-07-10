import models
import model_class as mc
import particle_class as pc
import feyn_rules
import numpy as np
import lorentz_structures as ls
import matplotlib.pyplot as plt

def plot_amp(model):
    # Initialize variables.
    costheta_list = np.linspace(-1,1,100,endpoint=False)
    phi = 2*np.pi*np.random.uniform()
    comput_sol = []
    analytic_sol = []
    ee = model.parameter_map["ee"]
    i = 1
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
    
        #q = mom[0, :] + mom[1, :]
        mom = np.append(mom, [mom[0] + mom[1]], axis=0) # s-channel momentum
        mom = np.append(mom, [mom[0] - mom[2]], axis=0) # t-channel momentum
        mom = np.append(mom, [mom[0] - mom[3]], axis=0) # u-channel momentum
        
        # Compare with analytic solution.
        s = float((ls.Momentum(mom, 0, 4)*ls.Metric(0, 1)*ls.Momentum(mom, 1, 4)))
        t = float((ls.Momentum(mom, 0, 5)*ls.Metric(0, 1)*ls.Momentum(mom, 1, 5)))
        u = float((ls.Momentum(mom, 0, 6)*ls.Metric(0, 1)*ls.Momentum(mom, 1, 6)))
    
        # Initialize particles with spin index and momentum index.
        elec = pc.Particle(model, 11, mom, 1, 0, True)
        antielec = pc.Particle(model, -11, mom, 2, 1, True)
        elec2 = pc.Particle(model, 11, mom, 3, 2, False)
        antielec2 = pc.Particle(model, -11, mom, 4, 3, False)
        photon = pc.Particle(model, 22, mom, 0, 4, False)
    
        InP = [elec, antielec]
        OutP = [elec2, antielec2]
        IntP = [photon]
        
        # Compute amplitude.
        try:
            Amp1 = feyn_rules.FeynRules(model, InP, OutP, IntP)
            #comput_sol.append(Amp1.amplitude.real)
        except:
            Amp1 = 0
            #comput_sol.append(Amp1)
        try:
            analytic = 2*ee**4*((u*u + s*s)/t**2 + 2*u*u/(s*t) + (u*u + t*t)/s**2)
        except:
            analytic = 0
        
        comput_sol.append(Amp1.amplitude.real)
        analytic_sol.append(analytic)
        print("Computational solution: {}".format(Amp1.amplitude.real))
        print("Analytic solution: {}".format(analytic))
        print("Run: {}".format(i))
        i += 1
    
    # Plot amplitudes and amplitude error vs cos(theta).
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12,8))
    
    axs[0,0].plot(costheta_list, comput_sol, color='red')
    axs[0,0].plot(costheta_list, analytic_sol, color='blue')
    axs[0,0].set_xlabel("Cos(theta)")
    axs[0,0].legend(["Computational","Analytic"])    
    
    axs[1,0].plot(costheta_list, np.subtract(comput_sol, analytic_sol))
    axs[1,0].set_xlabel("Cos(theta)")
    
    # Cut-off data
    epsilon = 0.5
    analytic_cut = [ana for ana in analytic_sol if ana < epsilon]
    s = len(analytic_cut)
    costheta_cut = costheta_list[0:s]
    comput_cut = comput_sol[0:s]
    axs[0,1].plot(costheta_cut, comput_cut, color='red')
    axs[0,1].plot(costheta_cut, analytic_cut, color='blue')
    axs[0,1].set_xlabel("Cos(theta)")
    axs[0,1].legend(["Computational","Analytic"])
    
    axs[1,1].plot(costheta_cut, np.subtract(comput_cut, analytic_cut))
    axs[1,1].set_xlabel("Cos(theta)")
    
    fig.suptitle("Comparison of amplitudes for phi={:.2f}, epsilon={:.2f}".format(phi, epsilon))
    fig.savefig("bhabha_scattering.pdf")

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
    
    #q = mom[0, :] + mom[1, :]
    mom = np.append(mom, [mom[0] + mom[1]], axis=0) # s-channel momentum
    mom = np.append(mom, [mom[0] - mom[2]], axis=0) # t-channel momentum
    mom = np.append(mom, [mom[0] - mom[3]], axis=0) # u-channel momentum
    
    # Initialize particles with spin index and momentum index.
    elec = pc.Particle(model, 11, mom, 1, 0, True)
    antielec = pc.Particle(model, -11, mom, 2, 1, True)
    elec2 = pc.Particle(model, 11, mom, 3, 2, False)
    antielec2 = pc.Particle(model, -11, mom, 4, 3, False)
    photon = pc.Particle(model, 22, mom, 0, 4, False)
    
    InP = [elec, antielec]
    OutP = [elec2, antielec2]
    IntP = [photon]

    # Compute amplitude.
    Amp1 = feyn_rules.FeynRules(model, InP, OutP, IntP)
    
    # Compare with analytic solution.
    s = float((ls.Momentum(mom, 0, 4)*ls.Metric(0, 1)*ls.Momentum(mom, 1, 4)))
    t = float((ls.Momentum(mom, 0, 5)*ls.Metric(0, 1)*ls.Momentum(mom, 1, 5)))
    u = float((ls.Momentum(mom, 0, 6)*ls.Metric(0, 1)*ls.Momentum(mom, 1, 6)))
    
    ee = model.parameter_map["ee"]
    analytic = 2*ee**4*((u*u + s*s)/t**2 + 2*u*u/(s*t) + (u*u + t*t)/s**2)
    
    # print("Computational solution: {}".format(Amp1.amplitude))
    # print("Analytic solution: {}".format(analytic))
    # print("Ratio: {}".format(analytic/Amp1.amplitude))
    
    plot_amp(model)
    
if __name__ == '__main__':
    main()
