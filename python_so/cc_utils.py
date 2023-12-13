from opt_einsum import contract
import numpy as np

def build_tau(t1, t2, fact1=1.0, fact2=1.0):
    return fact1 * t2 + fact2 * (contract('ia,jb->ijab', t1, t1) - contract('ib,ja->ijab', t1, t1))

def build_Fae(o, v, nv, F, ERI, t1, t2):
    Fae = F[v,v].copy()
    Fae -= F[v,v] * np.identity(nv)
    Fae -= 0.5 * contract('me,ma->ae', F[o,v], t1)
    Fae += contract('mf,mafe->ae', t1, ERI[o,v,v,v])
    Fae -= 0.5 * contract('mnaf,mnef->ae', build_tau(t1, t2, 1.0, 0.5), ERI[o,o,v,v])
    return Fae

def build_Fmi(o, v, no, F, ERI, t1, t2):
    Fmi = F[o,o].copy()
    Fmi -= F[o,o] * np.identity(no)
    Fmi += 0.5 * contract('ie,me->mi', t1, F[o,v])
    Fmi += contract('ne,mnie->mi', t1, ERI[o,o,o,v])
    Fmi += 0.5 * contract('inef,mnef->mi', build_tau(t1, t2, 1.0, 0.5), ERI[o,o,v,v])
    return Fmi

def build_Fme(o, v, F, ERI, t1):
    Fme = F[o,v].copy()
    Fme += contract('nf,mnef->me', t1, ERI[o,o,v,v])
    return Fme

def build_Wmnij(o, v, ERI, t1, t2):
    Wmnij = ERI[o,o,o,o].copy()
    Wmnij += contract('je,mnie->mnij', t1, ERI[o,o,o,v])
    Wmnij -= contract('ie,mnje->mnij', t1, ERI[o,o,o,v])
    Wmnij += 0.25 * contract('ijef,mnef->mnij', build_tau(t1, t2), ERI[o,o,v,v])
    return Wmnij

def build_Wabef(o, v, ERI, t1, t2):
    Wabef = ERI[v,v,v,v].copy()
    Wabef -= contract('mb,amef->abef', t1, ERI[v,o,v,v])
    Wabef += contract('ma,bmef->abef', t1, ERI[v,o,v,v])
    Wabef += 0.25 * contract('mnab,mnef->abef', build_tau(t1, t2), ERI[o,o,v,v])
    return Wabef

def build_Wmbej(o, v, ERI, t1, t2):
    Wmbej = ERI[o,v,v,o].copy()
    Wmbej += contract('jf,mbef->mbej', t1, ERI[o,v,v,v])
    Wmbej -= contract('nb,mnej->mbej', t1, ERI[o,o,v,o])
    Wmbej -= 0.5 * contract('jnfb,mnef->mbej', t2, ERI[o,o,v,v])
    Wmbej -= contract('jf,nb,mnef->mbej', t1, t1, ERI[o,o,v,v])
    return Wmbej

def r_T1(o, v, F, ERI, t1, t2, Fae, Fme, Fmi):
    r_T1 = F[o,v].copy()
    r_T1 += contract('ie,ae->ia', t1, Fae)
    r_T1 -= contract('ma,mi->ia', t1, Fmi)
    r_T1 += contract('imae,me->ia', t2, Fme)
    r_T1 -= contract('nf,naif->ia', t1, ERI[o,v,o,v])
    r_T1 -= 0.5 * contract('imef,maef->ia', t2, ERI[o,v,v,v])
    r_T1 -= 0.5 * contract('mnae,nmei->ia', t2, ERI[o,o,v,o])
    return r_T1

def r_T2(o, v, F, ERI, t1, t2, Fae, Fme, Fmi, Wmnij, Wabef, Wmbej):
    r_T2 = ERI[o,o,v,v].copy()

    tmp = Fae - 0.5 * contract('mb,me->be', t1, Fme)
    r_T2 += contract('ijae,be->ijab', t2, tmp)
    r_T2 -= contract('ijbe,ae->ijab', t2, tmp)

    tmp = Fmi + 0.5 * contract('je,me->mj', t1, Fme)
    r_T2 -= contract('imab,mj->ijab', t2, tmp)
    r_T2 += contract('jmab,mi->ijab', t2, tmp)

    r_T2 += 0.5 * contract('mnab,mnij->ijab', build_tau(t1, t2), Wmnij)
    r_T2 += 0.5 * contract('ijef,abef->ijab', build_tau(t1, t2), Wabef)

    r_T2 += contract('imae,mbej->ijab', t2, Wmbej) - contract('ie,ma,mbej->ijab', t1, t1, ERI[o,v,v,o])
    r_T2 -= contract('imbe,maej->ijab', t2, Wmbej) - contract('ie,mb,maej->ijab', t1, t1, ERI[o,v,v,o])
    r_T2 -= contract('jmae,mbei->ijab', t2, Wmbej) - contract('je,ma,mbei->ijab', t1, t1, ERI[o,v,v,o])
    r_T2 += contract('jmbe,maei->ijab', t2, Wmbej) - contract('je,mb,maei->ijab', t1, t1, ERI[o,v,v,o])

    r_T2 += contract('ie,abej->ijab', t1, ERI[v,v,v,o])
    r_T2 -= contract('je,abei->ijab', t1, ERI[v,v,v,o])

    r_T2 -= contract('ma,mbij->ijab', t1, ERI[o,v,o,o])
    r_T2 += contract('mb,maij->ijab', t1, ERI[o,v,o,o])

    return r_T2

def compute_ccsd_energy(o, v, F, ERI, t1, t2):
    ecc = contract('ia,ia->', F[o,v], t1)
    ecc += 0.25*contract('ijab,ijab->', t2, ERI[o,o,v,v])
    ecc += 0.5*contract('ia,jb,ijab->', t1, t1, ERI[o,o,v,v])
    return ecc
