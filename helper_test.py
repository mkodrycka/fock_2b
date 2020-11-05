# A SAPT helper object
import numpy as np
import psi4
import time
from numpy import linalg as LA

class helper_SAPT(object):

    def __init__(self,dimer,memory=18,algorithm='MO',reference='RHF',ri_basis=None,jk_basis=None,gamma=1.0):
        """
        Initializes the helper_SAPT object.

        Parameters:
        ----------
        dimer : psi4.core.Molecule or str
            The molecule to be used for the given helper object.
        memory : {int, 2}, optional
            The amount of memory (in GB) to use.
        algorithm : {AO, MO},optional
        reference : {"RHF", "ROHF", "UHF"}, optional
            The type of reference wavefunction.
        ri_basis: string, optional
            The basis set to be used for RI.
        jk_basis: string, optional
            The basis set to be used for JKFIT.
        gamma : {float, 2}, optional
            Exponent for Slater-type frozen geminal.
            
        Returns:
        ------
        ret : helper_SAPT
        """
        
        if ri_basis == None:
            raise Exception("You did not specify RI basis!!")

        self.ri_basis_spec = ri_basis
        self.jk_basis_spec = jk_basis

        print("\nInitializing SAPT object...\n")
        tinit_start = time.time()

        # Set a few crucial attributes
        self.alg = algorithm.upper()
        self.reference = reference.upper()
        dimer.reset_point_group('c1')
        dimer.fix_orientation(True)
        dimer.fix_com(True)
        dimer.update_geometry()
        nfrags = dimer.nfragments()
        if nfrags != 2:
            psi4.core.clean()
            raise Exception("Found %d fragments, must be 2." % nfrags)

        # Grab monomers in DCBS
        self.monomerA = dimer.extract_subsets(1, 2)
        self.monomerA.set_name('monomerA')
        self.monomerB = dimer.extract_subsets(2, 1)
        self.monomerB.set_name('monomerB')

        # Setup a few variables
        self.memory = memory
        self.rhfA, self.wfnA = psi4.energy('SCF', return_wfn=True, molecule=self.monomerA)
        print("RHF for monomer A finished in %.2f seconds." % (time.time() - tinit_start))
        self.rhfB, self.wfnB = psi4.energy('SCF', return_wfn=True, molecule=self.monomerB)
        print("RHF for monomer B finished in %.2f seconds." % (time.time() - tinit_start))
        self.dimer_wfn = psi4.core.Wavefunction.build(dimer, psi4.core.get_global_option('BASIS'))
        mints = psi4.core.MintsHelper(self.dimer_wfn.basisset())
        self.mints = mints
        self.nmo = self.wfnA.nmo()
        
        # Monomer A
        self.nuc_rep_A = self.monomerA.nuclear_repulsion_energy()
        self.ndocc_A = self.wfnA.doccpi()[0]
        self.nvirt_A = self.nmo - self.ndocc_A
        if reference == 'ROHF':
          self.idx_A = ['i', 'a', 'r']
          self.nsocc_A = self.wfnA.soccpi()[0]
          occA = self.ndocc_A + self.nsocc_A
        else:
          self.idx_A = ['a', 'r']
          self.nsocc_A = 0
          occA = self.ndocc_A 

        self.C_A = np.asarray(self.wfnA.Ca())
        self.Co_A = self.C_A[:, :self.ndocc_A]
        self.Ca_A = self.C_A[:, self.ndocc_A:occA]
        self.Cv_A = self.C_A[:, occA:]
        self.eps_A = np.asarray(self.wfnA.epsilon_a())

        # Monomer B
        self.nuc_rep_B = self.monomerB.nuclear_repulsion_energy()
        self.ndocc_B = self.wfnB.doccpi()[0]
        self.nvirt_B = self.nmo - self.ndocc_B
        if reference == 'ROHF':
          self.idx_B = ['j', 'b', 's']
          self.nsocc_B = self.wfnB.soccpi()[0]
          occB = self.ndocc_B + self.nsocc_B
        else:
          self.idx_B = ['b', 's']
          self.nsocc_B = 0
          occB = self.ndocc_B 

        self.C_B = np.asarray(self.wfnB.Ca())
        self.Co_B = self.C_B[:, :self.ndocc_B]
        self.Ca_B = self.C_B[:, self.ndocc_B:occB]
        self.Cv_B = self.C_B[:, occB:]
        self.eps_B = np.asarray(self.wfnB.epsilon_a())

        # Build basis sets      
        self.conv = psi4.core.BasisSet.build(dimer,'BASIS',psi4.core.get_global_option('BASIS'))
        self.mp2fit_basis = psi4.core.BasisSet.build(dimer,'DF_BASIS_MP2',"","RIFIT",\
                            psi4.core.get_global_option('DF_BASIS_MP2'))
        self.ri_basis = psi4.core.BasisSet.build(dimer,'BASIS',ri_basis)
        self.aori_basis = psi4.core.BasisSet.build(dimer,'BASIS',psi4.core.get_global_option('BASIS')+'_'+ri_basis)
        self.zero_basis = psi4.core.BasisSet.zero_ao_basis_set()

        self.A_ao_basis = psi4.core.BasisSet.build(self.monomerA, 'BASIS', psi4.core.get_global_option('BASIS'))
        self.B_ao_basis = psi4.core.BasisSet.build(self.monomerB, 'BASIS', psi4.core.get_global_option('BASIS'))

        if jk_basis != None:
            self.jk_basis = psi4.core.BasisSet.build(dimer,'BASIS',jk_basis)
            self.jk_aori = psi4.core.JK.build(self.aori_basis,self.jk_basis)
        else:
            self.jk_aori = psi4.core.JK.build(self.aori_basis,self.mp2fit_basis)

        # This section of the code is to instantiate the new 2bjk object
        if jk_basis != None:
            print("jk_2b? first")
            self.jk_basis = psi4.core.BasisSet.build(dimer, 'BASIS', jk_basis)
            self.jk_2b_A = psi4.core.JK.build(self.A_ao_basis, self.jk_basis, "mem_2b_df", "False", \
                           self.memory, self.aori_basis, self.A_ao_basis )
        else:
            print("jk_2b? second")
            self.jk_2b_A = psi4.core.JK.build( self.A_ao_basis, self.mp2fit_basis, "mem_2b_df", "False", \
                           self.memory, self.aori_basis, self.A_ao_basis )
            self.jk_2b_B = psi4.core.JK.build( self.A_ao_basis, self.mp2fit_basis, "mem_2b_df", "False", \
                           self.memory, self.aori_basis, self.A_ao_basis )

        self.nri = psi4.core.BasisSet.nbf(self.ri_basis)
        # AO + RI fuctions
        self.naori = psi4.core.BasisSet.nbf(self.aori_basis)

        # Dimer
        self.nuc_rep = dimer.nuclear_repulsion_energy() - self.nuc_rep_A - self.nuc_rep_B
        self.vt_nuc_rep = self.nuc_rep / ((2 * self.ndocc_A + self.nsocc_A)
                                           * (2 * self.ndocc_B + self.nsocc_B))

        # Make slice, orbital, and size dictionaries
        if reference == 'ROHF':
          self.slices = {
                       'i': slice(0, self.ndocc_A), 
                       'a': slice(self.ndocc_A, occA), 
                       'r': slice(occA, self.ncabs),  
                       'j': slice(0, self.ndocc_B),
                       'b': slice(self.ndocc_B, occB),
                       's': slice(occB, None)
                      }

          self.orbitals = {'i': self.Co_A,
                           'a': self.Ca_A,
                           'r': self.Cv_A,
                           'j': self.Co_B,
                           'b': self.Ca_B,
                           's': self.Cv_B,
                        }

          self.sizes = {'i': self.ndocc_A,
                        'a': self.nsocc_A,
                        'r': self.nvirt_A,
                        'j': self.ndocc_B,
                        'b': self.nsocc_B,
                        's': self.nvirt_B,
                        'ri': self.nri
                       }

        else:
          self.slices = {
                       'a': slice(0, self.ndocc_A), 
                       'r': slice(occA, self.nmo),  
                       'b': slice(0, self.ndocc_B),
                       's': slice(occB, self.nmo),
                      }

          self.orbitals = {'a': self.Co_A,
                           'r': self.Cv_A,
                           'b': self.Co_B,
                           's': self.Cv_B,
                           'p': self.C_A,
                           'q': self.C_B,

                        }

          self.sizes = {'a': self.ndocc_A,
                        'r': self.nvirt_A,
                        'b': self.ndocc_B,
                        's': self.nvirt_B,
                        'p': self.nmo,
                        'q': self.nmo,
                        'ri': self.nri
                       }

          self.basis = {'a': self.conv,
                        'r': self.conv,
                        'b': self.conv,
                        's': self.conv,
                        'p': self.conv,
                        'q': self.conv,
                       }


    def get_size(self):
        """
        Obtains orbital sizes.
        
        Returns
        -------
        sizes: Dictionary
           Obrbital sizes.
        """

        return self.sizes


    def get_Cxi_A_Cxj_B(self, dimer):
        """
        This is the hack for copmute_jk.

        The goal is to create a Cxi_A and Cxj_B matrices.
        Therefore, the Co_A and Co_B are extended by the number
        of RI basis and zeroed out.

        Psi4 groups all basis functions on each atom seperately,
        so they need to be sorted.

        Parameters:
        ----------
        dimer : psi4.core.Molecule or str
            The molecule to be used for the given helper object.
            
        Returns:
        -------
        Cxi_A and Cxj_B : ndarray
            The Cxi_A coefficent matrix.   
        Cxj_B : ndarray       
            The Cxi_B coefficent matrix.   
        """
        
        self.natoms, self.nbf_ao_list, self.nbf_ri_list = get_bf_for_all_atoms(dimer, self.ri_basis_spec)

        Cxi_A = np.zeros((self.naori, self.ndocc_A))
        Cxj_B = np.zeros((self.naori, self.ndocc_B))

        stop = 0
        start_i = self.nbf_ao_list[0]
        for index, (nbf_ao, nbf_ri) in enumerate(zip(self.nbf_ao_list, self.nbf_ri_list)):
            if index == 0:
                Cxi_A[:nbf_ao,:] = self.Co_A[:nbf_ao,:]
                Cxj_B[:nbf_ao,:] = self.Co_B[:nbf_ao,:]
                stop = nbf_ao + nbf_ri
                stop_i = nbf_ao
            else:
                start = stop
                stop = start + nbf_ao

                start_i = stop_i
                stop_i = start_i + nbf_ao

                Cxi_A[start:stop,:] = self.Co_A[start_i:stop_i,:]
                Cxj_B[start:stop,:] = self.Co_B[start_i:stop_i,:]

                stop += nbf_ri

        return Cxi_A, Cxj_B


    def sort_bf_in_Jxx_Kxx(self, Jxx, Kxx):
        """
        This is the hack for copmute_jk.
        Psi4 groups all basis functions on each atom seperately.
        Therefore, they need to be sorted for each subsystem
        to obtain AO+RI (AO then RI).

        Parameters:
        ----------
        Jxx: ndarray
            J matrix in the AO+RI basis.
        Jxx: ndarray
            K matrix in the AO+RI basis.
                       
        Returns:
        ------
        Jxx_sorted2: ndarray
            J matrix in the AO+RI basis with sorted functions. 
        Kxx_sorted2 : ndarray
            K matrix in the AO+RI basis with sorted functions. 
        """

        Jxx_sorted = np.zeros((self.naori, self.naori))
        Kxx_sorted = np.zeros((self.naori, self.naori))

        for index, (nbf_ao, nbf_ri) in enumerate(zip(self.nbf_ao_list, self.nbf_ri_list)):
            if index == 0:
                Jxx_sorted[:,:nbf_ao] = Jxx[:,:nbf_ao]
                Kxx_sorted[:,:nbf_ao] = Kxx[:,:nbf_ao]
                stop = nbf_ao + nbf_ri
                stop_i = nbf_ao

                Jxx_sorted[:,self.nmo:self.nmo+nbf_ri] = Jxx[:,nbf_ao:nbf_ao+nbf_ri]
                Kxx_sorted[:,self.nmo:self.nmo+nbf_ri] = Kxx[:,nbf_ao:nbf_ao+nbf_ri]
                stop_ii = self.nmo + nbf_ri

                stop_ri = nbf_ao + nbf_ri
            else:
                start = stop
                stop = start + nbf_ao

                start_i = stop_i
                stop_i = start_i + nbf_ao

                start_ri = stop_ri + nbf_ao
                stop_ri = start_ri + nbf_ri

                start_ii = stop_ii
                stop_ii = start_ii + nbf_ri

                Jxx_sorted[:,start_i:stop_i] = Jxx[:,start:stop]
                Kxx_sorted[:,start_i:stop_i] = Kxx[:,start:stop]

                Jxx_sorted[:,start_ii:stop_ii] = Jxx[:,start_ri:stop_ri]
                Kxx_sorted[:,start_ii:stop_ii] = Kxx[:,start_ri:stop_ri]

                stop += nbf_ri

        Jxx_sorted2 = Jxx_sorted.copy()
        Kxx_sorted2 = Kxx_sorted.copy()

        for index, (nbf_ao, nbf_ri) in enumerate(zip(self.nbf_ao_list, self.nbf_ri_list)):
            if index == 0:
                Jxx_sorted2[:nbf_ao,:] = Jxx_sorted[:nbf_ao,:]
                Kxx_sorted2[:nbf_ao,:] = Kxx_sorted[:nbf_ao,:]
                stop = nbf_ao + nbf_ri
                stop_i = nbf_ao

                Jxx_sorted2[self.nmo:self.nmo+nbf_ri,:] = Jxx_sorted[nbf_ao:nbf_ao+nbf_ri,:]
                Kxx_sorted2[self.nmo:self.nmo+nbf_ri,:] = Kxx_sorted[nbf_ao:nbf_ao+nbf_ri,:]
                stop_ii = self.nmo+nbf_ri

                stop_ri = nbf_ao+nbf_ri
            else:
                start = stop
                stop = start + nbf_ao

                start_i = stop_i
                stop_i = start_i + nbf_ao

                start_ri = stop_ri + nbf_ao
                stop_ri = start_ri + nbf_ri

                start_ii = stop_ii
                stop_ii = start_ii + nbf_ri

                Jxx_sorted2[start_i:stop_i,:] = Jxx_sorted[start:stop,:]
                Kxx_sorted2[start_i:stop_i,:] = Kxx_sorted[start:stop,:]

                Jxx_sorted2[start_ii:stop_ii,:] = Jxx_sorted[start_ri:stop_ri,:]
                Kxx_sorted2[start_ii:stop_ii,:] = Kxx_sorted[start_ri:stop_ri,:]

                stop += nbf_ri

        return Jxx_sorted2, Kxx_sorted2

    def print_basis_sets(self):
        """
        Prints basis stes utilised for F12 calculations.
        """

        print("\nBasis sets used for F12 calculations")
        print('------------------------------------')
        print('AO: %s' % (str(psi4.core.get_global_option('BASIS'))))
        print('RI: %s' % (self.ri_basis_spec).upper())
        print('MP2-FIT: %s' % (str(psi4.core.get_global_option('DF_BASIS_MP2'))))
        print('JK-FIT: %s\n' % (self.jk_basis_spec))

    def build_jk(self, C_left, C_right=None):
        """
        A wrapper to compute the J and K objects.

        Parameters
        ----------
        C_left : list of array_like or a array_like object
                 Orbitals used to compute the JK object with
        C_right : list of array_like (optional, None)
                 Optional C_right orbitals, otherwise it is assumed C_right == C_left
        Returns
        -------
        JK : tuple of ndarray
        Returns the J and K objects
        """

        self.jk_aori.set_memory(int(self.memory * 1e9))
        self.jk_aori.initialize()


        return compute_jk(self.jk_aori, C_left, C_right)

    def build_2b_jk(self, C_left, C_right=None):
        """
        build_jk but with the new machinery

        Parameters
        ----------
        C_left : list of array_like or a array_like object
                 Orbitals used to compute the JK object with
        C_right : list of array_like (optional, None)
                 Optional C_right orbitals, otherwise it is assumed C_right == C_left
        Returns
        -------
        JK : tuple of ndarray
        Returns the J and K objects
        """

        self.jk_2b_A.set_memory(int(self.memory * 1e9))
        self.jk_2b_A.initialize()

        return compute_2b_jk(self.jk_2b_A, C_left, C_right)

def compute_jk(jk, C_left, C_right=None):
    """
    A python wrapper for a Psi4 JK object to consume and produce NumPy arrays.
    Computes the following matrices:
    D = C_left C_right.T
    J_pq = (pq|rs) D_rs
    K_pq = (pr|qs) D_rs
    Parameters
    ----------
    jk : psi4.core.JK
        A initialized Psi4 JK object
    C_left : list of array_like or a array_like object
        Orbitals used to compute the JK object with
    C_right : list of array_like (optional, None)
        Optional C_right orbitals, otherwise it is assumed C_right == C_left
    Returns
    -------
    JK : tuple of ndarray
        Returns the J and K objects
    Notes
    -----
    This function uses the Psi4 JK object and will compute the initialized JK type (DF, PK, OUT_OF_CORE, etc)
    Examples
    --------
    ndocc = 5
    nbf = 15
    Cocc = np.random.rand(nbf, ndocc)
    jk = psi4.core.JK.build(wfn.basisset())
    jk.set_memory(int(1.25e8))  # 1GB
    jk.initialize()
    jk.print_header()
    J, K = compute_jk(jk, Cocc)
    J_list, K_list = compute_jk(jk, [Cocc, Cocc])
    """

    # Clear out the matrices
    jk.C_clear()

    list_input = True
    if not isinstance(C_left, (list, tuple)):
        C_left = [C_left]
        list_input = False

    for c in C_left:
        mat = psi4.core.Matrix.from_array(c)
        jk.C_left_add(mat)

    # Do we have C_right?
    if C_right is not None:
        if not isinstance(C_right, (list, tuple)):
            C_right = [C_right]

        if len(C_left) != len(C_right):
            raise ValueError("JK: length of left and right matrices is not equal")

        if not isinstance(C_right, (list, tuple)):
            C_right = [C_right]

        for c in C_right:
            mat = psi4.core.Matrix.from_array(c)
            jk.C_right_add(mat)

    # Compute the JK
    jk.compute()

    # Unpack
    J = []
    K = []
    for n in range(len(C_left)):
        J.append(np.array(jk.J()[n]))
        K.append(np.array(jk.K()[n]))

    jk.C_clear()
    jk.finalize()
    del jk

    # Duck type the return
    if list_input:
        return (J, K)
    else:
        return (J[0], K[0])

#initially, this file did the same as the function above: compute_jk
#I will now use it to return multiple JK objects.
def compute_2b_jk(jk, C_left, C_right=None):
    list_input = True
    if not isinstance(C_left, (list, tuple)):
        C_left = [C_left]
        list_input = False

    for c in C_left:
        mat = psi4.core.Matrix.from_array(c)
        jk.C_left_add(mat)

    # Do we have C_right?
    if C_right is not None:
        if not isinstance(C_right, (list, tuple)):
            C_right = [C_right]

        if len(C_left) != len(C_right):
            raise ValueError("JK: length of left and right matrices is not equal")

        if not isinstance(C_right, (list, tuple)):
            C_right = [C_right]

        for c in C_right:
            mat = psi4.core.Matrix.from_array(c)
            jk.C_right_add(mat)

    # Compute the JK
    jk.compute_2B_JK()

    # Unpack
    J = []
    K = []
    J_ot = []
    K_ot = []

    for n in range(len(C_left)):
        J.append(np.array(jk.J_oo_ao()[n]))
        K.append(np.array(jk.K_oo_ao()[n]))
        J_ot.append(np.array(jk.J_ot_ao()[n]))
        K_ot.append(np.array(jk.K_ot_ao()[n]))

    jk.C_clear()
    jk.finalize()
    del jk

    if list_input:
        return (J, K, J_ot, K_ot)
    else:
        return (J[0], K[0], J_ot[0], K_ot[0])

def get_bf_for_all_atoms(dimer,ri_basis_spec):
    """
    This is the hack for copmute_jk.
    Obtains the number of AO and RI functions for each atom.

    Parameters
    ----------
        dimer: psi4.core.Molecule or str
            The molecule to be used for the given helper object.
        ri_basis_spec: string 
            The name of RI basis set.     
    Returns
    -------
        natoms: int
            Numer of atoms.   
        nbf_ao: int
            Number of AO functions.
        nbf_ri: int
            Number of RI functions.    
    """

    natoms = psi4.core.Molecule.natom(dimer)
    nbf_ao = []
    nbf_ri = []
    for i in range(natoms):
        atom_symbol = psi4.core.Molecule.symbol(dimer,i)
        # I build the Molecule object for each atom
        atom_spec = psi4.geometry(atom_symbol, name='default')
        ao_basis = psi4.core.BasisSet.build(atom_spec,'BASIS', psi4.core.get_global_option('BASIS'))
        ri_basis = psi4.core.BasisSet.build(atom_spec,'BASIS', ri_basis_spec)
        atom_nbf_ao = psi4.core.BasisSet.nbf(ao_basis)
        atom_nbf_ri = psi4.core.BasisSet.nbf(ri_basis)

        nbf_ao.append(atom_nbf_ao)
        nbf_ri.append(atom_nbf_ri)

    return natoms, nbf_ao, nbf_ri

# End SAPT helper

class sapt_timer(object):
    def __init__(self, name):
        self.name = name
        self.start = time.time()
        print('\nStarting %s...' % name)

    def stop(self):
        t = time.time() - self.start
        print('...%s took a total of % .2f seconds.' % (self.name, t))


def sapt_printer(line, value):
    spacer = ' ' * (20 - len(line))
    print(line + spacer + '% 16.8f mH  % 16.8f kcal/mol' % (value * 1000, value * 627.509))
# End SAPT helper
