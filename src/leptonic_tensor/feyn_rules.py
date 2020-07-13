import vertex_class as vc
import lorentz_structures as ls
import ufo_grammer


class FeynRules:
    def __init__(self, model):
        '''
        Calculate spin-summed/averaged amplitude for incoming to
        outgoing particles with internal particles.
        '''
        self.model = model
        # self.incoming = incoming_particles
        self.incoming = []  # [part.anti() for part in incoming_particles]
        self.outgoing = []  # outgoing_particles
        self.internal = []  # internal_particles
        # All particles considered outgoing

        # self.pids = self._get_pids()
        # self.vertices = self._get_vertices()
        # self.propagator = self._get_propagator()
        # self.ufo = ufo_grammer.UFOParser(self.model.model)
        self.first = True
        self.sfirst = True
        self.tfirst = True
        self.ufirst = True
        self.svert = 0
        self.scoup = 0
        self.tvert = 0
        self.tcoup = 0
        self.uvert = 0
        self.ucoup = 0
        self.amplitudes = []
        # self.amplitude = self._get_amplitude()

    def _get_pids(self):
        pids = []
        for part in self.incoming:
            pids.append(part.pid)
        for part in self.outgoing:
            pids.append(part.pid)
        for part in self.internal:
            pids.append(part.pid)
        return pids

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

    # def _get_propagator(self):

    def _s_channel_amplitude(self):
        if self.sfirst:
            part_list1 = self.incoming + self.internal
            part_list2 = self.outgoing + self.internal

            vertices = self._get_vertices(part_list1, part_list2)
            vert1 = vertices[0].tensor[0]  # Gamma(0,2,1)
            vert2 = vertices[1].tensor[0]  # Gamma(1,4,3)
            self.svert = vert1*vert2

            coup1 = vertices[0].coupling[0][0]  # -0.313451j
            coup2 = vertices[1].coupling[0][0]  # -0.313451j
            self.scoup = coup1*coup2

            self.sfirst = False

        mom = self.internal[0].mom_array[4]
        propagator = 1j*ls.Metric(0, 1)
        propagator /= (ls.Momentum(mom, 10)
                       * ls.Metric(10, 11)
                       * ls.Momentum(mom, 11))

        return self.svert*self.scoup*propagator

    def _t_channel_amplitude(self):
        if self.tfirst:
            part_list1 = [self.incoming[0], self.outgoing[0]] + self.internal
            part_list2 = [self.incoming[1], self.outgoing[1]] + self.internal

            vertices = self._get_vertices(part_list1, part_list2)
            vert1 = vertices[0].tensor[0]  # Gamma(0,3,1)
            # vert2 = vertices[1].tensor[0] # Gamma(1,4,2)
            vert2 = ls.Gamma(1, 2, 4)
            self.tvert = vert1*vert2

            coup1 = vertices[0].coupling[0][0]  # -0.313451j
            coup2 = vertices[1].coupling[0][0]  # -0.313451j
            self.tcoup = coup1*coup2

            self.tfirst = False

        mom = self.internal[0].mom_array[5]
        propagator = 1j*ls.Metric(0, 1)
        propagator /= (ls.Momentum(mom, 10)
                       * ls.Metric(10, 11)
                       * ls.Momentum(mom, 11))
        # Propagator(0,1)

        return self.tvert*self.tcoup*propagator

    def _u_channel_amplitude(self):
        if self.ufirst:
            part_list1 = [self.incoming[0], self.outgoing[1]] + self.internal
            part_list2 = [self.incoming[1], self.outgoing[0]] + self.internal

            vertices = self._get_vertices(part_list1, part_list2)
            vert1 = vertices[0].tensor[0]  # Gamma(0,4,1)
            vert2 = vertices[1].tensor[0]  # Gamma(1,3,2)
            self.uvert = vert1*vert2

            coup1 = vertices[0].coupling[0][0]  # -0.313451j
            coup2 = vertices[1].coupling[0][0]  # -0.313451j
            self.ucoup = coup1*coup2

            self.ufrist = False

        mom = self.internal[0].mom_array[6]
        propagator = 1j*ls.Metric(0, 1)
        propagator /= (ls.Momentum(mom, 10)
                       * ls.Metric(10, 11)
                       * ls.Momentum(mom, 11))
        # Propagator(0,1)

        return self.uvert*self.ucoup*propagator

    def amplitude(self, incoming, outgoing, internal):
        self.incoming = [part.anti() for part in incoming]
        self.outgoing = outgoing
        self.internal = internal
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
                        # print(scatter1, scatter2)
                        # print(particle_pids)
                        # print(part_list1_pids, part_list2_pids)
                        # print(spin1, spin2, spin3, spin4)
                        self.scatter = True
                        break

                # Check whether annihilation vertex exists
                for vertex in self.model.vertices:
                    particle_list = vertex.particles
                    particle_pids = [part.pdg_code for part in particle_list]
                    particle_pids = sorted(particle_pids)

                    part_list1 = self.incoming + self.internal
                    part_list2 = self.outgoing + self.internal
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
                    self.amplitudes.append(self._t_channel_amplitude)
                    if self.outgoing[0].pid == self.outgoing[1].pid:
                        self.amplitudes.append(self._u_channel_amplitude)
                if self.annihilation:
                    self.amplitudes.append(self._s_channel_amplitude)
            self.first = False

        total = 0
        mtot = 0
        for amplitude in self.amplitudes:
            mtot += amplitude()

        for spin1 in range(2):
            spinor1 = self.incoming[0].get_spinor(spin1)
            for spin2 in range(2):
                spinor2 = self.incoming[1].get_spinor(spin2)
                for spin3 in range(2):
                    spinor3 = self.outgoing[0].get_spinor(spin3)
                    for spin4 in range(2):
                        spinor4 = self.outgoing[1].get_spinor(spin4)
                        result = mtot*spinor1*spinor2*spinor3*spinor4
                        total += result*result.conjugate()

        if total._scalar:
            return complex(total._array)
        else:
            return total
