"""
A python script to compute JK objects in mixed bases.
The purpose of the script is to test a new functionality,
which computes J/K objects in the mixed bases (primary1, primary2).

__authors__ = "Monika Kodrycka"
__credits__ = ["Monika Kodrycka"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2018-02-20"
"""

#!/apps/anaconda/bin/python
import numpy as np
import os
import sys
import psi4
sys.path.append('/home/mok0005/Git/psi4_build_jk_last_version/jk_test/jk_test/fock_2b')
from helper_test import *
sys.path.append('/home/mok0005/Git/psi4_build_jk_last_version/jk_test/jk_test/fock_2b/basis')
# Memory for Psi4 in GB
psi4.set_memory('20 GB')
psi4.core.set_output_file('output.dat', True)

# Memory for numpy in GB
numpy_memory = 20

dimer = psi4.geometry("""
0 1
O 0.0000000000 -0.0578657100 -1.4797930300
H 0.0000000000 0.8229338400 -1.8554147400
H 0.0000000000 0.0794956700 -0.5193425300
--
0 1
N 0.0000000000 0.0143639400 1.4645462800
H 0.0000000000 -0.9810485700 1.6534477900
H -0.8134835100 0.3987677600 1.9293404900
H 0.8134835100 0.3987677600 1.9293404900

units angstrom
""")

psi4.set_options({'basis':'aug-cc-pvdz',
                  'df_basis_mp2':'aug-cc-pvdz-ri',
                  'scf_type': 'mem_df',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8,
                })

tstart_begining = time.time()
# Specify RI and JKFIT(optionally) basis
ri_basis = 'aug-cc-pvdz-ri'
jk_basis = 'aug-cc-pvdz-jkfit'


# Use the JKFIT basis?
use_jk_basis = False

if use_jk_basis:
        sapt = helper_SAPT(dimer,memory=20,ri_basis=ri_basis,jk_basis=jk_basis)
else:
        sapt = helper_SAPT(dimer,memory=20,ri_basis=ri_basis)


sapt.print_basis_sets()

obs = sapt.get_size()
print("TYPE: ", type(obs))
nocc_A = obs['a']
nocc_B = obs['b']
nvir_A = obs['r']
nvir_B = obs['s']
nobs = obs['p']
nri =  obs['ri'] 

# Orbital spaces
print('\nOrbital Spaces:')
print('-----------------')
print('  nobs  : %d' % nobs)
print('  nocc_A: %d' % nocc_A)
print('  nocc_B: %d' % nocc_B)
print('  nvir_A: %d' % nvir_A)
print('  nvir_B: %d' % nvir_B)
print('  nri   : %d' % nri)

testC_A = sapt.Co_A
testC_B = sapt.Co_B

print("testC_A has "+str(testC_A.shape[0])+" rows")

Cxi_A, Cxj_B = sapt.get_Cxi_A_Cxj_B(dimer)

# J and K for mon A
Jxx_A, Kxx_A = sapt.build_jk(Cxi_A,Cxi_A)

# J and K for mon A using new feature
Jxx_A_2B, Kxx_A_2B, Jxy_A_2B, Kxy_A_2B = sapt.build_2b_jk(testC_A, testC_A)

print("Jxx_A")
print(Jxx_A)
print("Kxx_A")
print(Kxx_A)

print("Jxx_A_2B")
print(Jxx_A_2B)
print("Kxx_A_2B")
print(Kxx_A_2B)

mat_a = Jxx_A - Jxx_A_2B 
mat_b = Kxx_A - Kxx_A_2B
print("mat_a")
print(mat_a)
print(np.linalg.norm(mat_a, 2))

print("mat_b")
print(mat_b)
print(np.linalg.norm(mat_b, 2))

Jxx_A, Kxx_A = sapt.sort_bf_in_Jxx_Kxx(Jxx_A,Kxx_A)
