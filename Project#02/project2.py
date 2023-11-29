import numpy as np
from molmass import ELEMENTS
from scipy import constants
from scipy.constants import physical_constants

E_h = physical_constants["Hartree energy"][0]
a_0 = physical_constants["Bohr radius"][0]
amu = physical_constants["atomic mass constant"][0]
c_0 = physical_constants["speed of light in vacuum"][0]
h = physical_constants["Planck constant"][0]
kilo = constants.kilo
centi = constants.centi
angstrom = constants.angstrom
mega = constants.mega


class Molecule:

    def __init__(self):
        self.hessian_matrix = None
        self.hessian_data = None
        self.natoms = None
        self.atom_coordinates = None
        self.atom_charges = None

    def read_coordinate_data(self):
        # Open and close the coordinate file
        with open("/Users/jattakumi/Downloads/ProgrammingProjects-master/Project#02/input/h2o_geom.txt",
                  "r") as geomfile:
            data = np.array([line.split() for line in geomfile.readlines()][1:])
            self.atom_charges = np.array(data[:, 0], dtype=float).astype(int)
            self.natoms = self.atom_charges.shape[0]
            self.atom_coordinates = np.array(data[:, 1:4], dtype=float)

    def read_hessian_data(self):
        try:
            with open("/Users/jattakumi/Downloads/ProgrammingProjects-master/Project#02/input/h2o_hessian.txt",
                      'r') as hessfile:
                # Input: Read Hessian file from filepath
                # Read the number of atoms (N) from the first line
                self.natoms = int(hessfile.readline().strip())
                # Read the rest of the file into a flattened 1D array
                self.hessian_data = np.fromfile(hessfile, sep=' ', dtype=float)
                # Reshape the flattened array into a (3N) x (3N) matrix
                self.hessian_matrix = self.hessian_data.reshape((3 * self.natoms, 3 * self.natoms))
                return self.hessian_matrix

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None

    def mass_weighted_hess(self) -> np.ndarray:
        mass = []
        for i in self.atom_charges:
            mass.append(ELEMENTS[i].mass)
        tmp = np.repeat(mass, 3)
        temp = np.sqrt(tmp)
        hess = self.hessian_matrix / (temp[:, None] * temp[None, :])
        return hess

    def eig_mass_weight_hess(self) -> np.ndarray or list:
        # Output: Eigenvalue of mass-weighted Hessian matrix (in unit Eh/(amu*a0^2))
        return np.linalg.eigvalsh(self.mass_weighted_hess())

    def harmonic_vib_freq(self) -> np.ndarray or list:
        # Output: Harmonic vibrational frequencies (in unit cm^-1)
        scalar_constants = np.sqrt(E_h / (amu * a_0 ** 2)) * centi / (2 * np.pi * c_0)
        eigen_values = np.sign(self.eig_mass_weight_hess()) * np.sqrt(np.abs(self.eig_mass_weight_hess()))
        frequency = eigen_values * scalar_constants
        print(frequency)

    def print_solution_02(self):
        print("======= Total number of atoms =======")
        print(self.natoms)
        print("======= Atom Charges =======")
        print(self.atom_charges)
        print("========== Atom Coordinates ==========")
        print(self.atom_coordinates)
        print("========= Hessian Coordinates ==========")
        print(self.read_hessian_data())
        print("========== Mass Weighted Hess ==========")
        print(self.mass_weighted_hess())
        print("========= Eigenvalues of Mass-Weighted Hessian =========")
        print(self.eig_mass_weight_hess())
        print(self.harmonic_vib_freq())


mole = Molecule()
mole.read_coordinate_data()
mole.print_solution_02()
