import numpy as np


class Molecule:

    def __init__(self):
        self.atom_charges = NotImplemented
        self.atom_coords = NotImplemented
        self.natoms = NotImplemented
        self.nao = NotImplemented
        self.nocc = NotImplemented
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
        self.eri_mo = NotImplemented
        self.so_int = None
        self.C = NotImplemented
        # self.scf = HF()
        # self.scf.do_scf()
        # self.nocc = 2 * self.scf.nocc
        # self.nao = 2 * self.scf.nao
        self.nv = NotImplemented
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

    def read_nuclear_repulsion_energy(self): # E{nuc} from enuc.dat
        # Nuclear repulsion energy
        with open("/Users/jattakumi/Downloads/ProgrammingProjects-master/Project#03/input/h2o/STO-3G/enuc.dat",
                  'r') as enucfile:
            self.enuc = enucfile.readline()
        return self.enuc

    def read_overlap_integral(self): # S from s.dat
        # Overlap integrals
        with open("/Users/jattakumi/Downloads/ProgrammingProjects-master/Project#03/input/h2o/STO-3G/s.dat", 'r') as overlapfile:
            dat = np.array([line.split() for line in overlapfile.readlines()][:])
            self.nao = int(dat[-1:,0])
            self.overlap = np.zeros((self.nao,self.nao))
            for a in range(len(dat)):
                i = (dat[a][0]).astype(int) - 1
                j = (dat[a][1]).astype(int) - 1
                integral = (dat[a][2]).astype(float)
                self.overlap[i,j] = self.overlap[j,i] = integral
        return self.overlap

    def read_one_electron_kinetic_energy(self): # T from t.dat
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

    def read_one_electron_nuclear_attraction_integrals(self):   # V from v.dat
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
    def read_two_electron_repulsion_integral(self): # from eriout.dat
        with open("/Users/jattakumi/Downloads/ProgrammingProjects-master/Project#03/input/h2o/STO-3G/eri.dat", 'r') as erifile:
            dat = np.array([line.split() for line in erifile.readlines()][:])
            self.nao = int(dat[-1:,0])
            self.eri = (np.zeros((self.nao,self.nao,self.nao, self.nao)))
            for a in range(len(dat)):
                i = (dat[a][0]).astype(int) - 1
                j = (dat[a][1]).astype(int) - 1
                k = (dat[a][2]).astype(int) - 1
                l = (dat[a][3]).astype(int) - 1
                integral = (dat[a][4]).astype(float)
                self.eri[i,j,k,l] = self.eri[i,j,l,k] = self.eri[j,i,k,l] = self.eri[j,i,l,k] = self.eri[k,l,i,j] = self.eri[l,k,i,j] = self.eri[k,l,j,i] = self.eri[l,k,j,i]= integral
        return self.eri

    # Orthogonalization of the Basis Set: The S^{1/2} Matrix
    def diagonalize_overlap_matrix(self):   # Diagonalize the overlap matrix S from the s.dat file
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


    def construct_initial_guess_density_matrix(self):   # from the original basis
        self.density_matrix = np.dot(self.transform_eigenvector_to_original_basis()[:, :self.obtain_nocc()], (self.transform_eigenvector_to_original_basis()[:, :self.obtain_nocc()]).T)
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
        max_iter = 64
        thresh_eng, thresh_dm = 1e-7, 1e-8
        old_E, dm = self.compute_elect_n_total_energies(), np.zeros((self.nao,self.nao)) if self.density_matrix is None else np.copy(self.density_matrix)
        new_E, dm_next = NotImplemented, NotImplemented
        print("{:>5} {:>20} {:>20} {:>20} {:>20}".format("Epoch", "E(elect)", "E(tot)", "Delta(E)", "RMS(D)"))
        for epoch in range(max_iter):
            new_F = self.core + np.einsum("kl,ijkl-> ij", self.density_matrix, 2 * self.read_two_electron_repulsion_integral() - self.read_two_electron_repulsion_integral().transpose(0,2,1,3))
            F_1 = self.S_inv.T @ new_F @ self.S_inv
            e_values, e_vectors = np.linalg.eigh(F_1)
            C = self.S_inv @ e_vectors
            nocc = self.atom_charges.sum() // 2
            dm_next = np.einsum("ij,jk->ik" ,C[:, :nocc], (C[:, :nocc]).T)
            new_E = (dm_next * (self.core + new_F)).sum()
            E_total = new_E + float(self.enuc)
            self.density_matrix = dm_next
            rms = np.linalg.norm(dm_next - dm)
            dm = dm_next
            print("{:5d} {:20.12f} {:20.12f} {:20.12f} {:20.12f}".format(epoch, float(new_E), float(E_total), float((E_total - old_E)), rms))
            if abs(new_E - old_E) <= thresh_eng and rms <= thresh_dm:
                break
            old_E = E_total
            self.ERHF = E_total
            self.C = C
            self.mo_energy = e_values

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
                                      2 * self.read_two_electron_repulsion_integral() - self.read_two_electron_repulsion_integral().transpose(0, 2, 1,
                                                                                                                3))
        F_1 = self.S_inv.T @ new_F @ self.S_inv
        e_values, e_vectors = np.linalg.eigh(F_1)
        C = self.S_inv @ e_vectors
        self.eri_mo = np.einsum("uvkl, up, vq, kr, ls -> pqrs", self.eri, C, C, C, C)
        return self.eri_mo


    def compute_MP2_energy(self):
        new_F = self.core + np.einsum("kl,ijkl-> ij", self.density_matrix,
                                      2 * self.read_two_electron_repulsion_integral() - self.read_two_electron_repulsion_integral().transpose(0, 2, 1,3))
        F_1 = self.S_inv.T @ new_F @ self.S_inv
        e_values, e_vectors = np.linalg.eigh(F_1)
        C = self.S_inv @ e_vectors
        nocc = self.atom_charges.sum() // 2
        eri = self.eri_mo[:nocc, nocc:, :nocc, nocc:]
        D_iajb = e_values[:nocc, None, None, None] - e_values[None, nocc:, None, None]\
                 + e_values[None, None, :nocc, None] - e_values[None, None, None, nocc:]
        temp = eri * (2*eri - eri.swapaxes(1,3))
        mp2_corr = (temp/D_iajb).sum()
        print("MP2 Correlation Energy", mp2_corr)
        E_mp2 = self.ERHF + mp2_corr
        print("E MP2 = ", E_mp2)
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
        print(self.read_two_electron_repulsion_integral()[0, 3], '\n')
        print("======================= Build Orthogonalization Matrix =========================")
        print(self.build_symmetric_orthogonaliztion_matrix(), '\n')
        print("======================= Initial Fock Matrix =========================")
        print(self.form_initial_Fock_matrix(), '\n')
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
        print(self.get_mo_eri_einsum()[0, 3], '\n')
        print("=================================== MP2 Energy ==================================")
        print(self.compute_MP2_energy(), '\n')




mol = Molecule()
mol.print_solution_04()