import numpy as np

np.set_printoptions(precision=10, linewidth=200, threshold=200, suppress=True)

class CCSD:

    def __init__(self):
        self.atom_charges = NotImplemented
        self.atom_coords = NotImplemented
        self.natoms = NotImplemented
        self.nao = NotImplemented
        self.nocc = NotImplemented
        self.nao_so = NotImplemented
        self.nocc_so = NotImplemented
        self.charge = 0
        self.enuc = NotImplemented
        self.nuclear = NotImplemented
        self.overlap = NotImplemented
        self.kinetic = NotImplemented
        self.core = NotImplemented
        self.eri = NotImplemented
        self.fock = NotImplemented
        self.S_inv = NotImplemented
        self.F = NotImplemented
        self.F_0 = NotImplemented
        self.density_matrix = NotImplemented
        self.coulomb = NotImplemented
        self.exchange = NotImplemented
        self.eigen_vectors = NotImplemented
        self.eigen_values = NotImplemented
        self.eigenvectors = NotImplemented
        self.ERHF = NotImplemented
        self.spatialFock = NotImplemented
        self.eri_mo = NotImplemented
        self.so_int = None
        self.C = NotImplemented
        self.nv = NotImplemented
        self.nv_so = NotImplemented
        self.dim = NotImplemented
        self.spinints = NotImplemented
        self.new_F = NotImplemented
        self.v_vvvv = NotImplemented
        self.mo_energy = NotImplemented
        self.eri_mo = NotImplemented
        self.Fae = NotImplemented
        self.Fmi = NotImplemented
        self.Fme = NotImplemented
        self.Wmnij = NotImplemented
        self.Wabef = NotImplemented
        self.Wmbej = NotImplemented
        self.dim = NotImplemented
        self.spinints = NotImplemented
        self.new_F = NotImplemented
        self.v_vvvv = NotImplemented
        self.mo_energy = NotImplemented
        self.fs = NotImplemented
        self.core_mo = NotImplemented

        # Integral processing

    def read_geometry_file(self):
        # Geometry file
        with open("/Users/jattakumi/Downloads/ProgrammingProjects-master/Project#03/input/h2o/STO-3G/geom.dat",
                  'r') as geomfile:
            dat = np.array([line.split() for line in geomfile.readlines()][1:])
            self.atom_charges = np.array(dat[:, 0], dtype=float).astype(int)
            self.atom_coords = np.array(dat[:, 1:4], dtype=float)
            self.natoms = self.atom_charges.shape[0]
        return self.natoms

    def read_nuclear_repulsion_energy(self):  # E{nuc} from enuc.dat
        # Nuclear repulsion energy
        with open("/Users/jattakumi/Downloads/ProgrammingProjects-master/Project#03/input/h2o/STO-3G/enuc.dat",
                  'r') as enucfile:
            self.enuc = enucfile.readline()
        return self.enuc

    def read_overlap_integral(self):  # S from s.dat
        # Overlap integrals
        with open("/Users/jattakumi/Downloads/ProgrammingProjects-master/Project#03/input/h2o/STO-3G/s.dat",
                  'r') as overlapfile:
            dat = np.array([line.split() for line in overlapfile.readlines()][:])
            self.nao = int(dat[-1:, 0])
            self.overlap = np.zeros((self.nao, self.nao))
            for a in range(len(dat)):
                i = (dat[a][0]).astype(int) - 1
                j = (dat[a][1]).astype(int) - 1
                integral = (dat[a][2]).astype(float)
                self.overlap[i, j] = self.overlap[j, i] = integral
        return self.overlap

    def read_one_electron_kinetic_energy(self):  # T from t.dat
        # Kinetic integrals
        with open("/Users/jattakumi/Downloads/ProgrammingProjects-master/Project#03/input/h2o/STO-3G/t.dat",
                  'r') as kineticfile:
            dat = np.array([line.split() for line in kineticfile.readlines()][:])
            self.nao = int(dat[-1:, 0])
            self.kinetic = np.zeros((self.nao, self.nao))
            for a in range(len(dat)):
                i = (dat[a][0]).astype(int) - 1
                j = (dat[a][1]).astype(int) - 1
                integral = (dat[a][2]).astype(float)
                self.kinetic[i, j] = self.kinetic[j, i] = integral
        return self.kinetic

    def read_one_electron_nuclear_attraction_integrals(self):  # V from v.dat
        # Nuclear Attraction Integral
        with open("/Users/jattakumi/Downloads/ProgrammingProjects-master/Project#03/input/h2o/STO-3G/v.dat",
                  'r') as nuclearfile:
            dat = np.array([line.split() for line in nuclearfile.readlines()][:])
            self.nao = int(dat[-1:, 0])
            self.nuclear = np.zeros((self.nao, self.nao))
            for a in range(len(dat)):
                i = (dat[a][0]).astype(int) - 1
                j = (dat[a][1]).astype(int) - 1
                integral = (dat[a][2]).astype(float)
                self.nuclear[i, j] = self.nuclear[j, i] = integral
        return self.nuclear

    def form_core_Hamiltonian_matrix(self):
        self.core = self.kinetic + self.nuclear
        return self.core

        # Two-electron Integral

    def read_two_electron_repulsion_integral(self):  # from eriout.dat
        with open("/Users/jattakumi/Downloads/ProgrammingProjects-master/Project#03/input/h2o/STO-3G/eri.dat",
                  'r') as erifile:
            dat = np.array([line.split() for line in erifile.readlines()][:])
            self.nao = int(dat[-1:, 0])
            self.eri = (np.zeros((self.nao, self.nao, self.nao, self.nao)))
            for a in range(len(dat)):
                i = (dat[a][0]).astype(int) - 1
                j = (dat[a][1]).astype(int) - 1
                k = (dat[a][2]).astype(int) - 1
                l = (dat[a][3]).astype(int) - 1
                integral = (dat[a][4]).astype(float)
                self.eri[i, j, k, l] = self.eri[i, j, l, k] = self.eri[j, i, k, l] = self.eri[j, i, l, k] = self.eri[
                    k, l, i, j] = self.eri[l, k, i, j] = self.eri[k, l, j, i] = self.eri[l, k, j, i] = integral
        return self.eri

        # Orthogonalization of the Basis Set: The S^{1/2} Matrix

    def diagonalize_overlap_matrix(self):  # Diagonalize the overlap matrix S from the s.dat file
        self.eigen_values, self.eigen_vectors = np.linalg.eigh(self.overlap)
        return self.eigen_vectors, self.eigen_values

    def build_symmetric_orthogonaliztion_matrix(self):
        eigen_values, eigen_vectors = self.diagonalize_overlap_matrix()
        # Check if eigen_values contain valid numerical values
        if not np.all(np.isfinite(eigen_values)):
            raise ValueError("Invalid eigenvalues encountered.")
        Lambda_sqrt_inv = np.diag(1.0 / np.sqrt(self.eigen_values))
        self.S_inv = self.eigen_vectors @ Lambda_sqrt_inv @ self.eigen_vectors.T
        return self.S_inv

        # The Initial (Guess) Density Matrix

    def form_initial_Fock_matrix(self):
        self.F_0 = self.S_inv.T @ self.core @ self.S_inv
        return self.F_0

    def diagonalize_initial_Fock_matrix(self):
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.F_0)
        return self.eigenvectors

    def transform_eigenvector_to_original_basis(self):
        # Transform the resulting eigenvectors into original (non-orthonormal) basis,
        self.C_0 = self.S_inv @ self.diagonalize_initial_Fock_matrix()
        return self.C_0

    def obtain_nocc(self):
        self.nocc = self.atom_charges.sum() // 2
        return self.nocc

    def obtain_nv(self):
        self.nv = self.nao - self.nocc
        return self.nv

    def construct_initial_guess_density_matrix(self):  # from the original basis
        self.density_matrix = np.dot(self.transform_eigenvector_to_original_basis()[:, :self.obtain_nocc()],
                                     (self.transform_eigenvector_to_original_basis()[:, :self.obtain_nocc()]).T)
        return self.density_matrix

    def exchange_integral(self):
        self.exchange = np.einsum("ukvl, kl -> uv", self.read_two_electron_repulsion_integral(), self.density_matrix)
        return self.exchange

    def coulomb_integral(self):
        self.coulomb = np.einsum("uvkl, kl -> uv", self.read_two_electron_repulsion_integral(), self.density_matrix)
        return self.coulomb

    def compute_elect_n_total_energies(self):
        first_term = np.sum(self.density_matrix * (self.core + self.core))  # self.F = self.core
        second_term = float(self.enuc)
        E_total_0 = first_term + second_term
        return E_total_0

    def self_consistent_field_iteration(self):
        max_iter = 40
        thresh_eng, thresh_dm = 1e-12, 1e-12
        old_E, dm = self.compute_elect_n_total_energies(), np.zeros((self.nao, self.nao)) if self.density_matrix is None else np.copy(self.density_matrix)
        new_E, dm_next = 0, 0
        print("{:>5} {:>20} {:>20} {:>20} {:>20}".format("Epoch", "E(elect)", "E(tot)", "Delta(E)", "RMS(D)"))
        for epoch in range(max_iter):
            new_F = self.core + np.einsum("kl,ijkl-> ij", self.density_matrix,
                                          2 * self.read_two_electron_repulsion_integral() - self.read_two_electron_repulsion_integral().transpose(
                                              0, 2, 1, 3))
            F_1 = self.S_inv.T @ new_F @ self.S_inv
            e_values, e_vectors = np.linalg.eigh(F_1)
            C = self.S_inv @ e_vectors
            nocc = self.atom_charges.sum() // 2
            dm_next = np.einsum("ij,jk->ik", C[:, :nocc], (C[:, :nocc]).T)
            new_E = (dm_next * (self.core + new_F)).sum()
            E_total = new_E + float(self.enuc)
            self.density_matrix = dm_next
            rms = np.linalg.norm(dm_next - dm)
            dm = dm_next
            print("{:5d} {:20.12f} {:20.12f} {:20.12f} {:20.12f}".format(epoch, float(new_E), float(E_total),
                                                                         float((E_total - old_E)), rms))
            diff = E_total - old_E
            if abs(diff) <= thresh_eng and abs(rms) <= thresh_dm:
                break
            old_E = E_total
        self.ERHF = E_total
        self.C = C
        self.mo_energy = e_values
        self.spatialFock = new_F

        # print("MO-Energies", self.mo_energy)
        # print("Fock matrix for converged", self.spatialFock)

        # # MP2 Code
        # def get_mo_eri_naive(self):
        #     self.eri_mo = np.zeros((self.nao, self.nao,self.nao, self.nao))
        #     new_F = self.core + np.einsum("kl,ijkl-> ij", self.density_matrix,\
        #     2 * self.read_two_electron_repulsion_integral() - self.read_two_electron_repulsion_integral().transpose(0, 2, 1,
        #                                                                                                             3))
        #     F_1 = self.S_inv.T @ new_F @ self.S_inv
        #     e_values, e_vectors = np.linalg.eigh(F_1)
        #     C = self.S_inv @ e_vectors
        #     # self.nao =self.nocc
        #     for p in range(self.nao):
        #         for q in range(self.nao):
        #             for r in range(self.nao):
        #                 for s in range(self.nao):
        #                     for u in range(self.nao):
        #                         for v in range(self.nao):
        #                             for k in range(self.nao):
        #                                 for l in range(self.nao):
        #                                     self.eri_mo[p,q,r,s] += self.eri[u,v,k,l] * C[u,p] * C[v,q] * C[k,r] * C[l,s]
        #     return self.eri_mo

    def get_mo_eri_einsum(self):
        new_F = self.core + np.einsum("kl,ijkl-> ij", self.density_matrix,
                                      2 * self.read_two_electron_repulsion_integral() - self.read_two_electron_repulsion_integral().transpose(
                                          0, 2, 1,
                                          3))
        F_1 = self.S_inv.T @ new_F @ self.S_inv
        e_values, e_vectors = np.linalg.eigh(F_1)
        C = self.S_inv @ e_vectors
        self.eri_mo = np.einsum("uvkl, up, vq, kr, ls -> pqrs", self.eri, C, C, C, C)
        return self.eri_mo

    def get_mo_core(self):
        self.core_mo = self.C.T @ self.core @ self.C
        return self.core_mo
    def spin_nao(self):
        self.nao_so = 2 * self.nao
        return self.nao_so

    def spin_nocc(self):
        self.nocc_so = 2 * self.obtain_nocc()
        return self.nocc_so

    def spin_nv(self):
        self.nv_so = self.nao_so - self.nocc_so
        return self.nv_so


    def compute_MP2_energy(self):
        new_F = self.core + np.einsum("kl,ijkl-> ij", self.density_matrix,
                                      2 * self.read_two_electron_repulsion_integral() - self.read_two_electron_repulsion_integral().transpose(
                                          0, 2, 1, 3))
        F_1 = self.S_inv.T @ new_F @ self.S_inv
        e_values, e_vectors = np.linalg.eigh(F_1)
        C = self.S_inv @ e_vectors
        nocc = self.atom_charges.sum() // 2
        eri = self.eri_mo[:nocc, nocc:, :nocc, nocc:]
        D_iajb = e_values[:nocc, None, None, None] - e_values[None, nocc:, None, None] \
                 + e_values[None, None, :nocc, None] - e_values[None, None, None, nocc:]
        temp = eri * (2 * eri - eri.swapaxes(1, 3))
        mp2_corr = (temp / D_iajb).sum()
        print("MP2 Correlation Energy", mp2_corr)
        E_mp2 = self.ERHF + mp2_corr
        print("E MP2 = ", E_mp2)
        pass


    # def get_mo_eri_einsum(self):
    #     C = self.C
    #     self.eri_mo = np.einsum("uvkl, up, vq, kr, ls -> pqrs", self.eri_mo, C, C, C, C)
    #     return self.eri_mo

    # CCSD Code
    # Step 1: Preparing the Spin-Orbital Basis Integrals
    def translate_integrals_to_spin(self):
        # self.nocc = 2 * self.obtain_nocc()
        # self.nao = 2 * self.nao
        # self.nv = self.nao - self.nocc
        # print("Number of virtual orbitals", self.nv)
        self.so_int = np.zeros((self.nao_so, self.nao_so, self.nao_so, self.nao_so))
        TEI = self.get_mo_eri_einsum()
        for p in range(self.nao_so):
            for q in range(self.nao_so):
                for r in range(self.nao_so):
                    for s in range(self.nao_so):
                        value1 = TEI[p // 2][r // 2][q // 2][s // 2] * (p % 2 == r % 2) * (q % 2 == s % 2)
                        value2 = TEI[p // 2][s // 2][q // 2][r // 2] * (p % 2 == s % 2) * (q % 2 == r % 2)
                        self.so_int[p][q][r][s] = value1 - value2
                        # print("p, q, r, s", p, q, r, s,  self.so_int[p][q][r][s])
        return self.so_int

    # Build Fock Matrix
    def spin_basis_fock_matrix_eigenvalues(self):
        h_mo = self.get_mo_core()
        pqrs = self.translate_integrals_to_spin()
        self.fs = np.zeros((self.nao_so, self.nao_so))
        for p in range(self.nao_so):
            for q in range(self.nao_so):
                if p % 2 == q % 2:
                    self.fs[p, q] = h_mo[p // 2, q // 2]
                for m in range(self.nocc_so):
                    self.fs[p, q] += pqrs[p][m][q][m]
        return self.fs

    # def spin_basis_fock_matrix_eigenvalues(self):
    #     self.fs = np.zeros(self.nao_so)
    #     for i in range(0,self.nao_so):
    #          self.fs[i] = self.mo_energy[int(i/2)]
    #     self.fs = np.diag(self.fs)
    #     return self.fs

    def int_t1(self):
        t1 = np.zeros((self.nocc_so, self.nv_so))
        return t1

    def MP2(self):
        pqrs = self.translate_integrals_to_spin()
        nocc = self.nocc_so
        oovv = pqrs[:nocc, :nocc, nocc:, nocc:]
        t2 = self.int_t2()
        E_mp2 = 0.25 * np.einsum("ijab, ijab -> ", oovv, t2)
        return E_mp2

    def Dia(self):
        Dia = np.zeros((self.nocc_so, self.nv_so))
        f = self.fs
        for i in range(0, self.nocc_so):
            for a in range(0, self.nv_so):
                Dia[i, a] = f[i, i] - f[a+self.nocc_so,a+self.nocc_so]
        return Dia

    def Dijab(self):
        Dijab = np.zeros((self.nocc_so, self.nocc_so, self.nv_so, self.nv_so))
        f = self.fs
        # print("self.nocc_so", self.nocc_so)
        # print("self.nv_so", self.nv_so)
        for i in range(0, self.nocc_so):
            for j in range(0, self.nocc_so):
                for a in range(0, self.nv_so):
                    for b in range(0, self.nv_so):
                        Dijab[i, j, a, b] = f[i, i] + f[j, j] - f[a + self.nocc_so, a+self.nocc_so] - f[b+self.nocc_so, b+self.nocc_so]
        return Dijab

    def int_t2(self):
        pqrs = self.translate_integrals_to_spin()
        nocc = self.nocc_so
        ijab = pqrs[:nocc, :nocc, nocc:, nocc:]
        dijab = self.Dijab()
        t2 = ijab / dijab
        return t2

    def build_tau(self, t1, t2, fac):
        tau1 = t2.copy()
        t1t1_1 = np.einsum('ia,jb->ijab', t1, t1)
        t1t1_2 = np.einsum('ib,ja->ijab', t1, t1)
        tau1 += fac * (t1t1_1 - t1t1_2)
        # t1t1 = t1t1 - t1t1.transpose(1, 0, 2, 3)
        # tau1 += t1t1 - t1t1.transpose(0, 1, 3, 2)
        return tau1

    def cc_Fvv(self, t1, t2):
        # Eq 3 in Stanton et al.: Direct product decomposition I
        pqrs = self.so_int
        f = self.fs
        nocc = self.nocc_so
        fov = f[:nocc, nocc:]
        eris_ovvv = pqrs[:nocc,nocc:, nocc:, nocc:]
        eris_oovv = pqrs[:nocc, :nocc, nocc:, nocc:]
        tau1_tilde = self.build_tau(t1, t2, 0.5)
        # fvv_temp = fvv.copy()
        # for a in range(self.nv):
        #     for b in range(self.nv):
        #         if a == b:
        #             fvv_temp[a,b] = 0.0
        Fae = - 0.5 * np.einsum("me, ma -> ae", fov.copy(), t1)
        Fae += np.einsum("mf, mafe -> ae", t1, eris_ovvv.copy())
        Fae -= 0.5 * np.einsum("mnaf, mnef -> ae", tau1_tilde, eris_oovv.copy())      # -
        return Fae

    def cc_Foo(self, t1, t2):
        # Eq 4 in Stanton et al.: Direct product decomposition I
        pqrs = self.so_int
        f = self.fs
        nocc = self.nocc_so
        fov = f[:nocc, nocc:]
        eris_oovv = pqrs[:nocc, :nocc, nocc:, nocc:]
        eris_ooov = pqrs[:nocc, :nocc, :nocc, nocc:]
        tau1_tilde = self.build_tau(t1, t2, 0.5)
        Fmi = 0.5*np.einsum("me, ie -> mi", fov.copy(), t1)
        Fmi += np.einsum("ne, mnie -> mi", t1, eris_ooov.copy())
        Fmi += 0.5*np.einsum("inef, mnef -> mi", tau1_tilde, eris_oovv.copy())      # +
        return Fmi

    def cc_Fov(self, t1, t2):
        # Eq 5 in Stanton et al.: Direct product decomposition I
        pqrs = self.so_int
        f = self.fs
        nocc = self.nocc_so
        fov = f[:nocc, nocc:]
        eris_oovv = pqrs[:nocc, :nocc, nocc:, nocc:]
        Fme = fov.copy() + np.einsum("nf, mnef -> me", t1, eris_oovv.copy())
        return Fme

    def cc_Woooo(self, t1, t2):
        # Eq 6 in Stanton et al.: Direct product decomposition I
        pqrs = self.so_int
        nocc = self.nocc_so
        eris_oooo = pqrs[:nocc, :nocc, :nocc, :nocc]
        eris_ooov = pqrs[:nocc, :nocc, :nocc, nocc:]
        eris_oovv = pqrs[:nocc, :nocc, nocc:, nocc:]
        tau1 = self.build_tau(t1, t2, 1)
        Wmnij = eris_oooo.copy()
        tmp = np.einsum("je, mnie -> mnij", t1, eris_ooov.copy())
        Wmnij += tmp - tmp.transpose(0,1,3,2)
        Wmnij += 0.25 * np.einsum("ijef, mnef -> mnij", tau1, eris_oovv.copy())
        return Wmnij

    def cc_Wvvvv(self, t1, t2):
        # Eq 7 in Stanton et al.: Direct product decomposition I
        pqrs = self.so_int
        nocc = self.nocc_so
        eris_vvvv = pqrs[nocc:, nocc:, nocc:, nocc:]
        eris_vovv = pqrs[nocc:, :nocc, nocc:, nocc:]
        eris_oovv = pqrs[:nocc, :nocc, nocc:, nocc:]
        tau1 = self.build_tau(t1, t2, 1)
        Wabef = eris_vvvv.copy()
        tmp = np.einsum("mb, amef -> abef", t1, eris_vovv.copy())
        Wabef -= tmp - tmp.transpose(1, 0, 2, 3)
        Wabef += 0.25 * np.einsum("mnab, mnef -> abef", tau1, eris_oovv.copy())
        return Wabef

    def cc_Wovvo(self, t1, t2):
        # Eq 8 in Stanton et al.: Direct product decomposition I
        pqrs = self.so_int
        nocc = self.nocc_so
        eris_oovv = pqrs[:nocc, :nocc, nocc:, nocc:]
        eris_oovo = pqrs[:nocc, :nocc, nocc:, :nocc]
        eris_ovvv = pqrs[:nocc, nocc:, nocc:, nocc:]
        eris_ovvo = pqrs[:nocc, nocc:, nocc:, :nocc]
        Wmbej = eris_ovvo.copy()
        Wmbej += np.einsum("jf, mbef -> mbej", t1, eris_ovvv.copy())
        Wmbej -= np.einsum("nb, mnej -> mbej", t1, eris_oovo.copy())
        Wmbej -= 0.5 * np.einsum("jnfb, mnef -> mbej", t2, eris_oovv.copy())
        Wmbej -= np.einsum("jf, nb, mnef -> mbej", t1, t1, eris_oovv.copy())


        return Wmbej

    def update_t1(self, t1, t2, Fae, Fmi, Fme):
        # Eq 1 in Stanton et al.: Direct product decomposition I
        pqrs = self.so_int
        f = self.fs
        nocc = self.nocc_so
        fov = f[:nocc, nocc:]
        eris_ovov = pqrs[:nocc, nocc:, :nocc, nocc:]
        eris_ovvv = pqrs[:nocc, nocc:, nocc:, nocc:]
        eris_oovo = pqrs[:nocc, :nocc, nocc:, :nocc]
        t1new = fov.copy()
        t1new += np.einsum("ie, ae -> ia", t1, Fae)
        t1new -= np.einsum("ma, mi -> ia", t1, Fmi)
        t1new += np.einsum("imae, me -> ia", t2, Fme)
        t1new -= np.einsum("nf, naif -> ia", t1, eris_ovov.copy())
        t1new -= 0.5 * np.einsum("imef, maef -> ia", t2, eris_ovvv.copy())
        t1new -= 0.5 * np.einsum("mnae, nmei -> ia", t2, eris_oovo.copy())
        t1new /= self.Dia()
        return t1new

    def update_t2(self, t1, t2, Fae, Fmi, Fme, Wmbej, Wabef, Wmnij):    # Fme,
        # Eq 2 in Stanton et al.: Direct product decomposition I
        pqrs = self.so_int
        nocc = self.nocc_so
        tau1 = self.build_tau(t1, t2, 1)
        eris_oovv = pqrs[:nocc, :nocc, nocc:, nocc:]
        eris_ovoo = pqrs[:nocc, nocc:, :nocc, :nocc]
        eris_ovvo = pqrs[:nocc, nocc:, nocc:, :nocc]
        eris_vvvo = pqrs[nocc:, nocc:, nocc:, :nocc]

        t2new = eris_oovv.copy()
        Ftmp = Fae - (0.5 * np.einsum("mb, me -> be", t1, Fme))
        tmp = np.einsum("ijae, be -> ijab", t2, Ftmp)
        t2new += tmp - tmp.transpose(0, 1, 3, 2)
        Ftmp = Fmi + (0.5 * np.einsum("je, me -> mj", t1, Fme))
        tmp = np.einsum("imab, mj -> ijab", t2, Ftmp)
        t2new -= tmp - tmp.transpose(1, 0, 2, 3)

        t2new += 0.5*np.einsum("mnab, mnij -> ijab", tau1, Wmnij)
        t2new += 0.5*np.einsum("ijef, abef -> ijab", tau1, Wabef)
        tmp = np.einsum("imae, mbej -> ijab", t2, Wmbej)
        tmp -= np.einsum("ie, ma, mbej -> ijab", t1, t1, eris_ovvo.copy())
        tmp = tmp - tmp.transpose(0, 1, 3, 2)
        tmp = tmp - tmp.transpose(1, 0, 2, 3)
        #
        t2new += tmp
        tmp = np.einsum("ie, abej -> ijab", t1, eris_vvvo.copy())
        t2new += (tmp - tmp.transpose(1, 0, 2, 3))
        tmp = np.einsum("ma, mbij -> ijab", t1, eris_ovoo.copy())
        t2new -= (tmp - tmp.transpose(0, 1, 3, 2))
        t2new /= self.Dijab()
        return t2new

    def compute_CC_corr_energy(self, t1, t2):
        pqrs = self.translate_integrals_to_spin()
        f = self.spin_basis_fock_matrix_eigenvalues()
        nocc = self.nocc_so
        fov = f[:nocc, nocc:]
        eris_oovv = pqrs[:nocc, :nocc, nocc:, nocc:]
        Ecc = np.einsum("ia, ia -> ", fov, t1)
        Ecc += 0.25 * np.einsum("ijab, ijab -> ", eris_oovv.copy(), t2)
        Ecc += 0.50 * np.einsum("ijab, ia, jb -> ", eris_oovv.copy(), t1, t1)
        return Ecc

    def update_int(self, t1, t2):
        Fae = self.cc_Fvv(t1, t2)
        Fmi = self.cc_Foo(t1, t2)
        Fme = self.cc_Fov(t1, t2)
        Wabef = self.cc_Wvvvv(t1, t2)
        Wmnij = self.cc_Woooo(t1, t2)
        Wmbej = self.cc_Wovvo(t1, t2)
        # print("Fme", Fme)
        # t1new = self.update_t1(t1, t2, Fae, Fmi, Fme)
        # t2new = self.update_t2(t1, t2, Fae, Fmi, Fme, Wmbej, Wabef, Wmnij)  # Fme,
        return Fae, Fmi, Fme, Wabef, Wmnij, Wmbej   # t1new, t2new

    def Dijkabc(self):
        Dijkabc = np.zeros((self.nocc_so, self.nocc_so, self.nocc_so, self.nv_so, self.nv_so, self.nv_so))
        f = self.fs
        # print("self.nocc_so", self.nocc_so)
        # print("self.nv_so", self.nv_so)
        for i in range(0, self.nocc_so):
            for j in range(0, self.nocc_so):
                for k in range(0, self.nocc_so):
                    for a in range(0, self.nv_so):
                        for b in range(0, self.nv_so):
                            for c in range(0, self.nv_so):
                                Dijkabc[i, j, k, a, b, c] = f[i, i] + f[j, j] + f[k, k] - f[
                                    a + self.nocc_so, a + self.nocc_so] - f[b + self.nocc_so, b + self.nocc_so] - f[
                                                                c + self.nocc_so, c + self.nocc_so]
        return Dijkabc

    def disconnected_triples(self, t1, t2):
        Dijkabc = self.Dijkabc()
        pqrs = self.so_int
        nocc = self.nocc_so
        f = self.fs
        eris_oovv = pqrs[:nocc, :nocc, nocc:, nocc:]
        t3_d = np.einsum("ia, jkbc -> ijkabc", t1, eris_oovv.copy())
        t3_d = (t3_d - t3_d.transpose(0, 1, 2, 4, 3, 5) - t3_d.transpose(0, 1, 2, 5, 4, 3)).copy()
        t3_d = (t3_d - t3_d.transpose(1, 0, 2, 3, 4, 5) - t3_d.transpose(2, 1, 0, 3, 4, 5)).copy()
        t3_d /= Dijkabc
        return t3_d

    def connected_triples(self, t2):
        Dijkabc = self.Dijkabc()
        pqrs = self.so_int
        nocc = self.nocc_so
        eris_vovv = pqrs[nocc:, :nocc, nocc:, nocc:]
        eris_ovoo = pqrs[:nocc, nocc:, :nocc, :nocc]
        t3_c = np.einsum("jkae, eibc -> ijkabc", t2, eris_vovv.copy())
        t3_c -= np.einsum("imbc, majk -> ijkabc", t2, eris_ovoo.copy())
        t3_c = t3_c - t3_c.transpose(0, 1, 2, 4, 3, 5) - t3_c.transpose(0, 1, 2, 5, 4, 3)
        t3_c = t3_c - t3_c.transpose(1, 0, 2, 3, 4, 5) - t3_c.transpose(2, 1, 0, 3, 4, 5)
        t3_c /= Dijkabc
        return t3_c

    def ccsdt_energy(self, t1, t2):
        d_3 = self.Dijkabc()
        t3_c = self.connected_triples(t2)
        t3_d = self.disconnected_triples(t1, t2)
        E_T = np.einsum('ijkabc,ijkabc,ijkabc', d_3, t3_c, (t3_c + t3_d)) / 36
        # E_T = (1/36) * (t3_c * d_3 * (t3_c + t3_d)).sum()
        return E_T

    def ccsd_iteration(self):
        # self.initialise_ccsd()
        t1, t2 = self.int_t1(), self.int_t2()
        ECCSD = 0.0
        OLDCC = self.compute_CC_corr_energy(t1, t2)
        for iter in range(1, 50):
            Fae, Fmi, Fme, Wabef, Wmnij, Wmbej = self.update_int(t1, t2)
            t1new = self.update_t1(t1, t2, Fae, Fmi, Fme)
            t2new = self.update_t2(t1, t2, Fae, Fmi, Fme, Wmbej, Wabef, Wmnij)
            t1 = t1new.copy()
            t2 = t2new.copy()
            ECCSD = self.compute_CC_corr_energy(t1, t2)
            print(f"Iter = {iter} ECC = {ECCSD}")
            # t1 = t1new.copy()
            # t2 = t2new.copy()
            DECC = abs(ECCSD - OLDCC)
            OLDCC = ECCSD
            if DECC < 1e-12:
                print("A big win for you, your CCSD iteration converged")
                break

        print("E(corr,CCSD) = ", ECCSD)
        print("E(CCSD)_total = ", ECCSD + self.ERHF)
        # print("E_total: ", self.compute_CC_corr_energy(t1, t2) + self.ERHF)
        t1 = self.update_t1(t1, t2, Fae, Fmi, Fme)
        t2 = self.update_t2(t1, t2, Fae, Fmi, Fme, Wmbej, Wabef, Wmnij)
        ECCSD_T = self.ccsdt_energy(t1, t2)
        print("E_(T) = ", ECCSD_T)
        print("E_total = ", self.ERHF + ECCSD + ECCSD_T)
        pass


    def print_solution_04(self):
        print("========================= Total number of atoms =========================")
        print(self.read_geometry_file(), '\n')
        print("============================== Atom Charges ==============================")
        print(self.atom_charges, '\n')
        print("========================== Atom Coordinates ==========================")
        print(self.atom_coords, '\n')
        print("========================== Nuclear Repulsion Energy ==========================")
        print(self.read_nuclear_repulsion_energy(), '\n')
        print("==============================  Overlap Integral ============================== ")
        print(self.read_overlap_integral(), '\n')
        print("==============================  Kinetic Integral ============================== ")
        print(self.read_one_electron_kinetic_energy(), '\n')
        print("============================= Nuclear Attraction Integral =============================")
        print(self.read_one_electron_nuclear_attraction_integrals(), '\n')
        print("=================================== Core Hamiltonian ==================================")
        print(self.form_core_Hamiltonian_matrix(), '\n')
        print("======================== Electron Repulsion Integral ==========================")
        print(self.read_two_electron_repulsion_integral(), '\n')
        print("======================= Build Orthogonalization Matrix =========================")
        print(self.build_symmetric_orthogonaliztion_matrix(), '\n')
        print("======================= Initial Fock Matrix =========================")
        print(self.form_initial_Fock_matrix(), '\n')
        # print("======================= Initial Fock Matrix =========================")
        # print(self.nocc, '\n')
        print("======================= Initial MO Coefficients =========================")
        print(self.transform_eigenvector_to_original_basis(), '\n')
        print("======================= Initial Density Matrix =========================")
        print(self.construct_initial_guess_density_matrix(), '\n')
        print("=================================== Total Energy ==================================")
        print(self.compute_elect_n_total_energies(), '\n')
        print("=================================== SCF ==================================")
        print(self.self_consistent_field_iteration(), '\n')
        print("=================================== Transform the Two-Electron Integral ==================================")
        # print("===================================  Noddy Transformation ==================================")
        # print(self.get_mo_eri_naive()[0,3], '\n')
        print("=================================== Smart Algorithm ==================================")
        print(self.get_mo_eri_einsum(), '\n')
        print("=================================== MP2 Energy ==================================")
        print(self.compute_MP2_energy(), '\n')
        print("=================================== CCSD Code Output ==================================")
        print("=================================== Spin Orbitals ==================================")
        print(self.spin_nao(), '\n')
        print("=================================== Spin NOCC ==================================")
        print(self.spin_nocc(), '\n')
        print("=================================== Spin Nv ==================================")
        print(self.spin_nv(), '\n')
        print("=================================== Translate Spatial Orbital to Spin ==================================")
        print(self.translate_integrals_to_spin()[0,:, 0, :], '\n')
        print("=================================== Spin Fock ==================================")
        print(self.spin_basis_fock_matrix_eigenvalues(), '\n')
        print("=================================== Initial T1 amplitudes ==================================")
        # print(self.int_t1(), '\n')
        # print("=================================== Dia ==================================")
        # print(self.Dia(), '\n')
        # print("=================================== Dijab ==================================")
        # print(self.Dijab(), '\n')
        # print("=================================== Initial T2 amplitudes ==================================")
        # print(self.int_t2(), '\n')
        # print("=================================== MP2 Energy using the Initial T2 amplitudes ==================================")
        print(self.MP2(), '\n')
        print("=================================== CCSD Iteration ==================================")
        print(self.ccsd_iteration(), '\n')






cc = CCSD()
cc.print_solution_04()
# print("Emp2:", cc.MP2())
# # cc.compute_CC_corr_energy()
# cc.ccsd_iteration()


