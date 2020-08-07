import numpy as np
import collections
import graph as G


class Rambo:
    def __init__(self, nin, nout):
        self.nin = nin
        self.nout = nout
        pi2log = np.log(np.pi/2.)
        Z = np.zeros(nout+1)
        Z[2] = pi2log
        for k in range(3, nout+1):
            Z[k] = Z[k-1]+pi2log-2.*np.log(k-2)
        for k in range(3, nout+1):
            Z[k] -= np.log(k-1)
        self.Z_N = Z[nout]

    def __call__(self, p, rans):
        sump = np.zeros((rans.shape[0], 4))
        for i in range(self.nin):
            sump += p[:, i]
        ET = np.sqrt(Dot(sump, sump))

        R = np.zeros((rans.shape[0], 4))
        for i in range(self.nin, self.nin+self.nout):
            ctheta = 2*rans[:, 4*(i-self.nin)] - 1
            stheta = np.sqrt(1-ctheta**2)
            phi = 2*np.pi*rans[:, 1 + 4*(i-self.nin)]
            Q = -np.log(rans[:, 2+4*(i-self.nin)]*rans[:, 3+4*(i-self.nin)])
            p[:, i, 0] = Q
            p[:, i, 1] = Q*stheta*np.sin(phi)
            p[:, i, 2] = Q*stheta*np.cos(phi)
            p[:, i, 3] = Q*ctheta
            R += p[:, i]

        RMAS = np.sqrt(Dot(R, R))
        B = -R[:, 1:]/RMAS
        G = R[:, 0, None]/RMAS
        A = 1.0/(1.0+G)
        X = ET/RMAS

        for i in range(self.nin, self.nin+self.nout):
            e = p[:, i, 0, None]
            BQ = np.sum(B*p[:, i, 1:], axis=-1, keepdims=True)
            term1 = (G*e)+BQ
            term2 = B*(e+A*BQ)
            p[:, i, 0, None] = X*term1
            p[:, i, 1:] = X*(p[:, i, 1:] + term2)


def WeylSpinor(mr, p):
    rpp = np.sqrt(p[:, 0] + p[:, 3] + 0j)
    rpm = np.sqrt(p[:, 0] - p[:, 3] + 0j)
    pt = p[:, 1] + 1j*p[:, 2]
    mu = np.stack([rpp, rpm]).T
    mu[:, 1] = np.where(np.abs(pt) > 0,
                        np.divide(pt.real + 1j*mr*pt.imag, rpp,
                                  out=np.zeros_like(pt),
                                  where=rpp != 0),
                        mu[:, 1])
    return mu


def SpinorU(ph):
    sh = WeylSpinor(1, ph)
    u = np.zeros((ph.shape[0], 4), dtype=np.complex)
    u[:, 2] = sh[:, 0]
    u[:, 3] = sh[:, 1]
    return u


def SpinorV(p, ph):
    sh = np.where(p[:, 0, None] < 0, -WeylSpinor(-1, ph), WeylSpinor(-1, ph))
    u = np.zeros((ph.shape[0], 4), dtype=np.complex)
    u[:, 0] = sh[:, 1]
    u[:, 1] = -sh[:, 0]
    return u


def Spinor(p, hel):
    ph = p.copy()
    ps = np.sqrt(np.sum(p[:, 1:]**2, axis=-1))
    ph[:, 0] = np.where(p[:, 0] < 0, -ps, ps)
    return np.where(hel > 0, SpinorU(ph).T, SpinorV(p, ph).T).T


kp = WeylSpinor(1, np.array([[1, 1, 0, 0]]))
km = WeylSpinor(-1, np.array([[1, 1, 0, 0]]))


def EVec(sp, sm, batch):
    vec = np.zeros((batch, 4), dtype=np.complex)
    vec[:, 0] = sp[:, 0]*sm[:, 0] + sp[:, 1]*sm[:, 1]
    vec[:, 3] = sp[:, 0]*sm[:, 0] - sp[:, 1]*sm[:, 1]
    vec[:, 1] = sp[:, 0]*sm[:, 1] + sp[:, 1]*sm[:, 0]
    vec[:, 2] = 1j*(sp[:, 0]*sm[:, 1] - sp[:, 1]*sm[:, 0])
    return vec


def PolarizationVec(mom, spin):
    psp = np.where(spin[:, None] > 0, WeylSpinor(-1, mom), WeylSpinor(1, mom))
    vec = np.where(spin[:, None] > 0,
                   EVec(kp, psp, mom.shape[0]),
                   EVec(psp, km, mom.shape[0]))
    denom = np.where(spin > 0,
                     (km[:, 0]*psp[:, 1] - km[:, 1]*psp[:, 0]).conjugate(),
                     (psp[:, 0]*kp[:, 1] - psp[:, 1]*kp[:, 0]).conjugate())
    denom *= np.sqrt(2.0)
    return vec/denom[:, None]


def Wavefunction(part, mom, spin):
    if part.info.spin == 0:
        return 1
    elif part.info.spin == 1:
        return Spinor(mom, 1, spin)
    elif part.info.spin == 2:
        return PolarizationVec(mom, spin)


def Propagator(part_info, mom):
    sij = Dot(mom, mom)
    mass = part_info.mass.value
    if float(mass) == 0:
        return sij
    else:
        width = part_info.width
        return sij - mass**2 + 1j*mass*width


def Dot(mom1, mom2):
    return (mom1[:, 0]*mom2[:, 0])[:, None]-np.sum(mom1[:, 1:]*mom2[:, 1:],
                                                   axis=-1, keepdims=True)


class BG_Amplitude:
    def __init__(self, model, external, batch=1):
        self.model = model
        self.external = external
        self.currents = np.zeros((batch, self.n_external, self.n_external, 4),
                                 dtype=np.complex)
        print(self.currents)
        self.momentums = np.zeros((batch, self.n_external, self.n_external, 4),
                                  dtype=np.complex)
        self.pids = np.zeros((self.n_external, self.n_external), dtype=np.int)
        # self.pids = np.zeros((self.n_external, self.n_external), dtype=list)
        for d, part in enumerate(self.external):
            self.pids[d, d] = part.pid
        print("PID Matrix:\n{}".format(self.pids))

        self.vertices = self._get_vertices()

    def _get_vertices(self):
        graph = G.Graph()
        j = 0
        # Initialize 4x4 matrix 'vertices' with objects of type list.
        vertices = np.empty((self.n_external, self.n_external),
                            dtype=list)
        for d in range(1, self.n_external):
            for i in range(1, self.n_external):
                if i + d < self.n_external:
                    print("i = {}, d = {}, i+d = {}".format(i,d,i+d))
                    vertices[i, i+d] = self._get_vertex(i, i+d, graph, j)
                    j += 1
                    print("Vertices matrix:\n{}\nPID matrix: {}".format(vertices, self.pids))

    def _get_vertex(self, ei, ej, graph, j):
        vertices = []
        for i in range(ei, ej):
            pid1 = self.pids[ei, i]
            pid2 = self.pids[i+1, ej]
            node = G.Node(j, (pid1,pid2))
            print("PID 1: {}, PID 2: {}".format(pid1, pid2))
            for key in self.model.vertex_map.keys():
                if pid1 in key:
                    lstkey = list(key)
                    lstkey.remove(pid1)
                    print("List of key without PID 1: {}".format(lstkey))
                    if pid2 in lstkey:
                        if len(lstkey) == 2:
                            lstkey.remove(pid2)
                            # self.pids[ei, ej].append(lstkey[0])
                            graph.add_node(node)
                            j += 1
                            new_node = G.Node(j, (lstkey[0]))
                            edge = G.Edge([node, new_node], self.model.vertex_map[key])
                            graph.add_edge(edge)
                            self.pids[ei, ej] = lstkey[0]
                            print(self.pids)
                            print(graph)
                            if self.model.vertex_map[key] not in vertices:
                                vertices.append(self.model.vertex_map[key])
                                print(key, self.model.vertex_map[key])
        if vertices == []:
            return None
        else:
            return vertices

    @property
    def n_external(self):
        return len(self.external)

    def __call__(self, momentum, spins):
        for d, part in enumerate(self.external):
            self.currents[:, d, d] = Wavefunction(part, momentum[:, d],
                                                  spins[:, d])
            self.momentums[:, d, d] = momentum[:, d]
        for d in range(1, self.n_external):
            for i in range(1, self.n_external):
                if i + d < self.n_external:
                    self.currents[:, i, i+d] = self.JL(i, i+d)
        # return Dot(self.currents[:, 0, 0], self.currents[:, 1, -1])
        # Convert current to Tensor
        return np.outer(self.currents[:, 1, -1],
                        self.currents[:, 1, -1].conjugate()).real

    def JL(self, i, j):
        vl = self.VL(i, j)
        # if (i == 1 and j == self.n_external-1):
        #     return vl
        # else:
        return vl/Propagator(self.model.particle_map[abs(self.pids[i, j])],
                             self.momentums[:, i, j])

    def VL(self, ei, ej):
        self.momentums[:, ei, ej] = (self.momentums[:, ei, ei]
                                     + self.momentums[:, ei+1, ej])
        result = 0
        self.pids[ei, ej] = 21
        for i in range(ei, ej):
            if i > ei:
                self.momentums[:, ei, i] = (self.momentums[:, ei, ei]
                                            + self.momentums[:, ei+1, i])
            if ej < i+1:
                self.momentums[:, i+1, ej] = (self.momentums[:, i+1, i+1]
                                              + self.momentums[:, i+2, ej])

            result += self.V3L(self.momentums[:, ei, i],
                               self.momentums[:, i+1, ej],
                               self.currents[:, ei, i],
                               self.currents[:, i+1, ej])
            if i < ej - 1:
                for j in range(i+1, ej):
                    result += self.V4L(self.currents[:, ei, i],
                                       self.currents[:, i+1, j],
                                       self.currents[:, j+1, ej])

        return result

    @staticmethod
    def V3L(p1, p2, j1, j2):
        return (1.0/np.sqrt(2.0)*Dot(j1, j2)*(p1-p2)
                + np.sqrt(2.0)*(Dot(j1, p2)*j2
                                - Dot(j2, p1)*j1))

    @staticmethod
    def V4L(j1, j2, j3):
        return (Dot(j1, j3)*j2
                - 0.5*(Dot(j2, j3)*j1 + Dot(j2, j1)*j3))


if __name__ == '__main__':
    import models
    import model_class as mc
    import particle_class as pc
    import lorentz_structures as ls

    all_models = models.discover_models()
    model = mc.Model('Models.SM_NLO', all_models)

    nevents = 1
    nout = 2
    rambo = Rambo(2, nout)
    rans = np.random.random((nevents, 4*nout))
    mom = np.zeros((nevents, 2+nout, 4))
    mom[:, 0, 0] = 10
    mom[:, 0, 3] = 15
    mom[:, 1, 0] = 10
    mom[:, 1, 3] = -10

    rambo(mom, rans)

    mom[:, 0] = -mom[:, 0]
    mom[:, 1] = -mom[:, 1]

    # external = [pc.Particle(model, 21)]*(2+nout)
    external = [pc.Particle(model, 11),
                pc.Particle(model, -11),
                pc.Particle(model, 13),
                pc.Particle(model, -13)]

    amp = BG_Amplitude(model, external, nevents)
    #spins = np.array([[1, 1, -1, -1]])
    #print(spins, amp(mom, spins))
