import vertex_class as vc
import particle_class as pc
import lorentz_structures as ls
import ufo_grammer
import numpy as np


class Amplitude:
    def __init__(self, name, vertices, couplings, propagators, fermions):
        self.name = name
        self.vertices = vertices
        self.couplings = couplings
        self.propagators = propagators
        self.parity = (-1)**self._get_inversions(fermions)

    def __call__(self, momentum):
        propagators = 1
        for propagator in self.propagators:
            propagators *= propagator(momentum)
        return self.vertices*self.couplings*propagators*self.parity

    def _get_inversions(self, fermions):
        count = 0
        n = len(fermions)
        for i in range(n):
            for j in range(i + 1, n):
                if fermions[i] > fermions[j]:
                    count += 1
        return count


class FeynRules:
    def __init__(self, model):
        '''
        Calculate spin-summed/averaged amplitude for incoming to
        outgoing particles with internal particles.
        '''
        self.model = model
        # All particles considered outgoing
        self.incoming = None  # [part.anti() for part in incoming_particles]
        self.outgoing = None  # outgoing_particles
        self.internal = None  # internal_particles

        self.first = True

        self.amplitudes = []

    def _get_pids(self):
        pids = []
        for part in self.incoming:
            pids.append(part.pid)
        for part in self.outgoing:
            pids.append(part.pid)
        for part in self.internal:
            pids.append(part.pid)
        return pids

    def _allowed_vertices(self, part1, part2):
        vert_dict = {}
        part_map = self.model.particle_map
        for vertex in self.model.vertices:
            particle_list = vertex.particles
            particle_pids = [part.pdg_code for part in particle_list]
            particle_pids = sorted(particle_pids)
            if (part1.pid in particle_pids) and (part2.pid in particle_pids):
                new_pids = []
                for pid in particle_pids:
                    if part1.pid != pid and part2.pid != pid:
                        new_pids.append(pid)
                particles = [part_map[pid] for pid in new_pids]
                for part in particles:
                    # if (part1.incoming and part2.incoming) or (not part1.incoming and not part2.incoming):
                    #     Part_mom = part1.momentum + part2.momentum
                    # elif part1.incoming and not part2.incoming:
                    #     Part_mom = part1.momentum - part2.momentum
                    # elif part2.incoming and not part2.incoming:
                    #     Part_mom = part2.momentum - part1.momentum 
                    # Part = pc.Particle(self.model, part.pid, Part_mom)
                    # P = [part1, part2, Part]
                    # indices = [part.index for part in P]
                    # print(indices)
                    # vert_dict[part.name] = vc.Vertex(self.model, P, indices)
                    vert_dict[part.name] = part
        print(vert_dict)
        return vert_dict

    def _get_vertices(self, part_list1: list, part_list2: list):
        '''
        Get the two vertices involved in a 2->2 tree-level process.
        Particle lists should include external and internal particles.
        '''
        part_idx1 = []
        part_idx2 = []
        for part in part_list1:
            part_idx1.append(part.index)
        for part in part_list2:
            if part in self.internal:
                part_idx2.append(part.index+1)
            else:
                part_idx2.append(part.index)
        V1 = vc.Vertex(self.model, part_list1, part_idx1)
        V2 = vc.Vertex(self.model, part_list2, part_idx2)
        return [V1, V2]

    def _s_channel_amplitude(self):
        part_list1 = self.incoming + self.internal
        part_list2 = self.outgoing + self.internal

        vertices = self._get_vertices(part_list1, part_list2)
        vert1 = vertices[0].tensor[0]  # Gamma(0,2,1)
        vert2 = vertices[1].tensor[0]  # Gamma(1,4,3)
        # vert1 = ls.Gamma(4, 0, 1)
        # vert2 = ls.Gamma(5, 3, 2)
        vert = vert1*vert2*ls.Metric(4, 5)*-1j

        coup1 = vertices[0].coupling[0][0]  # -0.313451j
        coup2 = vertices[1].coupling[0][0]  # -0.313451j
        coup = coup1*coup2

        def denominator(momentum):
            mom = momentum[0] + momentum[1]
            return mom[0]**2 - np.sum(mom[1:]**2)

        propagators = [lambda momentum: (1  # *ls.Metric(4, 5)
                                         / denominator(momentum))]

        amp = Amplitude('s', vert, coup, propagators, [0, 1, 3, 2])

        return amp

    def _t_channel_amplitude(self):
        part_list1 = [self.incoming[0], self.outgoing[0]] + self.internal
        part_list2 = [self.incoming[1], self.outgoing[1]] + self.internal

        vertices = self._get_vertices(part_list1, part_list2)
        vert1 = vertices[0].tensor[0]  # Gamma(0,3,1)
        vert2 = vertices[1].tensor[0]  # Gamma(1,4,2)
        # vert1 = ls.Gamma(4, 0, 2)
        # vert2 = ls.Gamma(5, 1, 3)
        vert = vert1*vert2*ls.Metric(4, 5)*-1j

        coup1 = vertices[0].coupling[0][0]  # -0.313451j
        coup2 = vertices[1].coupling[0][0]  # -0.313451j
        coup = coup1*coup2

        def denominator(momentum):
            mom = momentum[0] + momentum[2]
            return mom[0]**2 - np.sum(mom[1:]**2)

        propagators = [lambda momentum: (1  # *ls.Metric(4, 5)
                                         / denominator(momentum))]

        amp = Amplitude('t', vert, coup, propagators, [0, 2, 3, 1])

        return amp

    def _u_channel_amplitude(self):
        part_list1 = [self.incoming[0], self.outgoing[1]] + self.internal
        part_list2 = [self.incoming[1], self.outgoing[0]] + self.internal

        vertices = self._get_vertices(part_list1, part_list2)
        vert1 = vertices[0].tensor[0]  # Gamma(0,4,1)
        vert2 = vertices[1].tensor[0]  # Gamma(1,3,2)
        vert = vert1*vert2*ls.Metric(4, 5)*-1j

        coup1 = vertices[0].coupling[0][0]  # -0.313451j
        coup2 = vertices[1].coupling[0][0]  # -0.313451j
        coup = coup1*coup2

        def denominator(momentum):
            mom = momentum[0] + momentum[3]
            return mom[0]**2 - np.sum(mom[1:]**2)

        propagators = [lambda momentum: (-1j  # *ls.Metric(4, 5)
                                         / denominator(momentum))]

        amp = Amplitude(vert, coup, propagators)

        return amp

    def amplitude(self, incoming, outgoing, internal):
        # All particles considered outgoing
        self.incoming = [part.anti() for part in incoming]
        self.outgoing = outgoing
        self.internal = internal

        momentum = [part.momentum for part in self.incoming]
        momentum += [part.momentum for part in self.outgoing]

        if(self.first):
            self.pids = self._get_pids()
            if((all(pid > 0 for pid in self.pids))
                    or (all(pid < 0 for pid in self.pids))):
                self.amplitudes.append(self._t_channel_amplitude)
                if self.outgoing[0].pid == self.outgoing[1].pid:
                    self.amplitudes.append(self._u_channel_amplitude)
            elif (any(pid > 0 for pid in self.pids)) and (any(pid < 0 for pid in self.pids)):
                self.scatter = False
                scatter1, scatter2 = False, False

                self.annihilation = False
                annihilation1, annihilation2 = False, False

                # Check whether scatter vertex exists
                for vertex in self.model.vertices:
                    particle_list = vertex.particles
                    particle_pids = [part.pdg_code for part in particle_list]
                    particle_pids = sorted(particle_pids)

                    part_list1 = ([self.incoming[0], self.outgoing[0]]
                                  + self.internal)
                    part_list2 = ([self.incoming[1], self.outgoing[1]]
                                  + self.internal)
                    part_list1_pids = [part.pid for part in part_list1]
                    part_list2_pids = [part.pid for part in part_list2]
                    part_list1_pids = sorted(part_list1_pids)
                    part_list2_pids = sorted(part_list2_pids)

                    if particle_pids == part_list1_pids:
                        scatter1 = True
                    if particle_pids == part_list2_pids:
                        scatter2 = True

                    if scatter1 and scatter2:
                        self.scatter = True
                        break

                # Check whether annihilation vertex exists
                for vertex in self.model.vertices:
                    particle_list = vertex.particles
                    particle_pids = [part.pdg_code for part in particle_list]
                    particle_pids = sorted(particle_pids)

                    part_list1 = self.incoming + [self.internal[0]]
                    part_list2 = self.outgoing + [self.internal[0]]
                    part_list1_pids = [part.pid for part in part_list1]
                    part_list2_pids = [part.pid for part in part_list2]
                    part_list1_pids = sorted(part_list1_pids)
                    part_list2_pids = sorted(part_list2_pids)

                    if particle_pids == part_list1_pids:
                        annihilation1 = True
                    if particle_pids == part_list2_pids:
                        annihilation2 = True
                    if annihilation1 and annihilation2:
                        self.annihilation = True
                        break
                if self.scatter:
                    self.amplitudes.append(self._t_channel_amplitude())
                    if self.outgoing[0].pid == self.outgoing[1].pid:
                        self.amplitudes.append(self._u_channel_amplitude())
                if self.annihilation:
                    self.amplitudes.append(self._s_channel_amplitude())

            self.first = False

        total = 0
        mtot = 0
        propagators = 1
        for amplitude in self.amplitudes:
            mtot += amplitude(momentum)

        for spin1 in range(2):
            spinor1 = self.incoming[0].get_spinor(spin1)
            for spin2 in range(2):
                spinor2 = self.incoming[1].get_spinor(spin2)
                for spin3 in range(2):
                    spinor3 = self.outgoing[0].get_spinor(spin3)
                    for spin4 in range(2):
                        spinor4 = self.outgoing[1].get_spinor(spin4)
                        spinors = [spinor1, spinor2, spinor3, spinor4]
                        result = mtot*spinor1*spinor2*spinor3*spinor4
                        total += result*result.conjugate()

        if total._scalar:
            return complex(total._array)
        else:
            return total
