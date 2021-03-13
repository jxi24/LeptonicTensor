import numpy as np
from utils import Pt, CosTheta, Dot

class Rambo:
    def __init__(self, nin, nout, ptMin):
        self.nin = nin
        self.nout = nout
        self.ptMin = ptMin
        pi2log = np.log(np.pi/2.)
        Z = np.zeros(nout+1)
        Z[2] = pi2log
        for k in range(3, nout+1):
            Z[k] = Z[k-1]+pi2log-2.*np.log(k-2)
        for k in range(3, nout+1):
            Z[k] -= np.log(k-1)
        self.Z_N = Z[nout]

    def cut(self, p):
        wt = np.ones(p.shape[0])
        for i in range(self.nin, self.nin+self.nout):
            wt *= np.where(Pt(p[:, i, :]) < self.ptMin,
                          np.zeros(p.shape[0]),
                          np.ones(p.shape[0]))
        return wt

    def __call__(self, p, rans):
        sump = np.zeros((rans.shape[0], 4))
        for i in range(self.nin):
            sump += p[:, i, :]
        ET = np.sqrt(Dot(sump, sump))

        R = np.zeros((rans.shape[0], 4))
        for i in range(self.nin, self.nin+self.nout):
            ctheta = 2*rans[:, 4*(i-self.nin)] - 1
            stheta = np.sqrt(1-ctheta**2)
            phi = 2*np.pi*rans[:, 1 + 4*(i-self.nin)]
            Q = -np.log(rans[:, 2+4*(i-self.nin)]*rans[:, 3+4*(i-self.nin)])
            p[:, i, :] = np.array([Q, Q*stheta*np.sin(phi), Q*stheta*np.cos(phi), Q*ctheta]).T
            R += p[:, i, :]

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


        wgt = np.exp((2*self.nout-4)*np.log(ET)+self.Z_N)/(2*np.pi)**(self.nout*3 - 4)
        return self.cut(p)[:, None]*wgt
