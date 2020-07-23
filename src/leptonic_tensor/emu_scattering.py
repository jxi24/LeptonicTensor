import models
import model_class as mc
import particle_class as pc
import feyn_rules
import numpy as np
import lorentz_structures as ls
import matplotlib.pyplot as plt
import time


def plot_amp(model):
    # Initialize variables.
    costheta_list = np.linspace(-1, 1, 500, endpoint=False)
    phi = 2*np.pi*np.random.uniform()
    comput_sol = []
    analytic_sol = []
    ee = model.parameter_map["ee"]
    Amp1 = feyn_rules.FeynRules(model)

    # Compute analytic and computational amplitude for each cos(theta).
    start = time.time()
    for i, costheta in enumerate(costheta_list):
        pc.Particle.gindex = 0
        sintheta = np.sqrt(1-costheta**2)
        mom = np.array(
            [[10, 0, 0, 10],
             [10, 0, 0, -10],
             [10, 10*sintheta*np.cos(phi),
              10*sintheta*np.sin(phi), 10*costheta],
             [10, -10*sintheta*np.cos(phi),
              -10*sintheta*np.sin(phi), -10*costheta]])

        mom = np.append(mom, [mom[0] + mom[1]], axis=0) # s-channel momentum
        mom = np.append(mom, [mom[0] - mom[2]], axis=0) # t-channel momentum
        mom = np.append(mom, [mom[0] - mom[3]], axis=0) # u-channel momentum

        # Initialize Mandelstam variables.
        p12 = mom[0]+mom[1]
        p13 = mom[0]-mom[2]
        p14 = mom[0]-mom[3]

        s = float((ls.Momentum(p12, 0)*ls.Metric(0, 1)*ls.Momentum(p12, 1)))
        t = float((ls.Momentum(p13, 0)*ls.Metric(0, 1)*ls.Momentum(p13, 1)))
        u = float((ls.Momentum(p14, 0)*ls.Metric(0, 1)*ls.Momentum(p14, 1)))

        # Initialize particles.
        elec = pc.Particle(model, 11, mom[0], True)
        muon = pc.Particle(model, 13, mom[1], True)
        elec2 = pc.Particle(model, 11, mom[2], False)
        muon2 = pc.Particle(model, 13, mom[3], False)
        photon = pc.Particle(model, 22, mom[4], False)

        InP = [elec, muon]
        OutP = [elec2, muon2]
        IntP = [photon]

        # Compute amplitudes.
        analytic = 8*ee**4*(s*s+u*u)/t**2

        comput_sol.append(Amp1.amplitude(InP, OutP, IntP).real)
        analytic_sol.append(analytic)
    end = time.time()
    print(f"It took {end-start}s")

    # Plot amplitudes and amplitude error vs cos(theta).
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10.5,5),dpi=200,constrained_layout=True)

    axs[0].plot(costheta_list, comput_sol)
    axs[0].plot(costheta_list, analytic_sol)
    axs[0].set_xlabel(r"$\displaystyle\cos(\theta)$")
    axs[0].set_ylabel(r"$\displaystyle|\mathcal{M}|^2$")
    axs[0].semilogy()
    axs[0].legend(["Computational", "Analytic"])

    axs[1].plot(costheta_list, np.subtract(comput_sol, analytic_sol)/analytic_sol)
    axs[1].set_xlabel(r"$\displaystyle\cos(\theta)$")
    axs[1].set_ylabel(r"$\displaystyle \frac{\delta |\mathcal{M}|^2}{|\mathcal{M}|_\textsubscript{ana}}$")

    fig.suptitle(r'''$\displaystyle e^-\mu^- \rightarrow e^-\mu^-$
                 Comparison of amplitudes for $\displaystyle\phi={:.2f}$'''.format(phi))
    fig.savefig("emu_scattering_latex.jpeg")


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

    mom = np.append(mom, [mom[0] + mom[1]], axis=0) # s-channel momentum
    mom = np.append(mom, [mom[0] - mom[2]], axis=0) # t-channel momentum
    mom = np.append(mom, [mom[0] - mom[3]], axis=0) # u-channel momentum
    p12 = mom[0]+mom[1]
    p13 = mom[0]-mom[2]
    p14 = mom[0]-mom[3]

    s = float((ls.Momentum(p12, 0)*ls.Metric(0, 1)*ls.Momentum(p12, 1)))
    t = float((ls.Momentum(p13, 0)*ls.Metric(0, 1)*ls.Momentum(p13, 1)))
    u = float((ls.Momentum(p14, 0)*ls.Metric(0, 1)*ls.Momentum(p14, 1)))

    # Initialize particles with spin index and momentum index.
    elec = pc.Particle(model, 11, mom[0], True)
    muon = pc.Particle(model, 13, mom[1], True)
    elec2 = pc.Particle(model, 11, mom[2], False)
    muon2 = pc.Particle(model, 13, mom[3], False)
    photon = pc.Particle(model, 22, mom[4], False)

    InP = [elec, muon]
    OutP = [elec2, muon2]
    IntP = [photon]

    # Compute amplitude.
    Amp1 = feyn_rules.FeynRules(model)

    # Compare with analytic solution.
    ee = model.parameter_map["ee"]
    analytic = 8*ee**4*(s*s+u*u)/t**2
    numerical = Amp1.amplitude(InP, OutP, IntP).real

    print("Computational solution: {}".format(numerical))
    print("Analytic solution: {}".format(analytic))
    print("Ratio: {}".format(analytic/numerical))

    plot_amp(model)


if __name__ == '__main__':
    main()
