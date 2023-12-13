import numpy as np
import pickle
from opt_einsum import contract
from cc_utils import *

np.set_printoptions(precision=10, linewidth=200, threshold=200, suppress=True)

# load SCF data
with open("scf.pkl", "rb") as f:
    data = pickle.load(f)

# Spin orbitals: occupied before virtual, alternating alpha/beta spins
nt = 2*data['nao']
no = 2*data['ndocc']
nv = nt - no

escf = data['escf']
eps = data['eps']
C = data['C']
H = data['H'] # AO-basis core Hamiltonian

# load spatial-MO ERIs, convert them to Dirac notation, and generate their spin-orbital version
ERI_MO = np.load("MO_ERI.npy")
ERI = np.zeros((nt, nt, nt, nt))
for p in range(nt):
	P = p//2;
	for q in range(nt):
		Q = q//2;
		for r in range(nt):
			R = r//2;
			for s in range(nt):
				S = s//2;
				ERI[p,q,r,s] = ERI_MO[P,R,Q,S]*(p%2 == r%2)*(q%2 == s%2) - ERI_MO[P,S,Q,R]*(p%2 == s%2) * (q%2 == r%2)

# orbital subspaces
o = slice(0, no)
v = slice(no, nt)
a = slice(0, nt)

# build Fock matrix
H = C.transpose() @ H @ C # Spatial MO basis
h = np.zeros((nt,nt))
for p in range(nt):
	for q in range(nt):
		h[p,q] = H[p//2,q//2]*(p%2 == q%2)
F = h + contract('pmqm->pq', ERI[a,o,a,o])

eps_occ = np.diag(F)[o]
eps_vir = np.diag(F)[v]
Dia = eps_occ.reshape(-1,1) - eps_vir
Dijab = eps_occ.reshape(-1,1,1,1) + eps_occ.reshape(-1,1,1) - eps_vir.reshape(-1,1) - eps_vir

# initial guess amplitudes
t1 = np.zeros((no, nv))
t2 = ERI[o,o,v,v]/Dijab

# initial CCSD energy (= MP2 energy)
ecc = compute_ccsd_energy(o, v, F, ERI, t1, t2)

print("CCSD Iter %3d: CCSD Ecorr = %.15f  dE = % .5E  MP2" % (0, ecc, -ecc))

maxiter = 50
e_conv= 1e-12
r_conv= 1e-12
ediff = ecc
rmsd = 0.0
niter = 0

while ((abs(ediff) > e_conv) or (abs(rmsd) > r_conv)) and (niter <= maxiter):
    niter += 1
    ecc_last = ecc

    Fae = build_Fae(o, v, nv, F, ERI, t1, t2)
    Fmi = build_Fmi(o, v, no, F, ERI, t1, t2)
    Fme = build_Fme(o, v, F, ERI, t1)
    Wmnij = build_Wmnij(o, v, ERI, t1, t2)
    Wabef = build_Wabef(o, v, ERI, t1, t2)
    Wmbej = build_Wmbej(o, v, ERI, t1, t2)

    r1 = r_T1(o, v, F, ERI, t1, t2, Fae, Fme, Fmi)
    r2 = r_T2(o, v, F, ERI, t1, t2, Fae, Fme, Fmi, Wmnij, Wabef, Wmbej)

    t1 = r1/Dia
    t2 = r2/Dijab

    rms = contract('ia,ia->', r1/Dia, r1/Dia)
    rms += contract('ijab,ijab->', r2/Dijab, r2/Dijab)
    rms = np.sqrt(rms)

    ecc = compute_ccsd_energy(o, v, F, ERI, t1, t2)
    ediff = ecc - ecc_last

    print('CCSD Iter %3d: CCSD Ecorr = %.15f  dE = % .5E  rms = % .5E' % (niter, ecc, ediff, rms))

