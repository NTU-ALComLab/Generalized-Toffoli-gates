import os
# Avoid oversubscribing CPU threads inside each worker process (common with NumPy/BLAS).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import qutip as qt
import multiprocessing as mp

# -----------------------------
# Time-step parameter noise
# -----------------------------
# Noise is sampled in the *parent* process before forking the worker pool, so results are reproducible.
SEED = 0
rng = np.random.default_rng(SEED)

enable_time_step_noise = True  # set False to recover the original (noise-free) behavior

# Multiprocessing strategy:
# The original version parallelized the operator-basis loop inside every Monte-Carlo trial,
# which creates many process pools and is slow.  This version parallelizes over Monte-Carlo
# trials instead.  Each worker builds one complete noisy sequential-pulse trajectory and
# computes its average-fidelity summation serially.

N=5  #num of qubits
n1=2   #-4
n2=-2
n3=4
w=np.array([1, 1, 1, 1])  # small indx first (alternating one first)
plateform="cir" #cir, ion
deco="on" #on, off
number_drive=2 #1, 2, 3
n_range=9
n_min=8
dis_n=np.arange(n_min, n_range, 4)
#dis_n=np.array([2, 4, 6, 8, 7, 13, np.e*7])
# N_RESAMPLE is the physical number of independent parameter samples per single pulse.
# It is NOT the ODE solver step count.  Numerical integration accuracy is controlled
# separately by MAX_INTERNAL_STEPS, tolerances, and max_step below.
# N_RESAMPLE_LIST = [10, 20, 30, 40, 50, 60, 70, 80, 100, 1000]
N_RESAMPLE_LIST = [10000]

# Relative one-sigma Gaussian control error.
# Each nonzero J_{ij}, each active drive amplitude Omega_l, and each active drive
# frequency omega_l gets its own independent noise trajectory in each pulse.
sigma_J_rel     = 0.001
sigma_Omaga_rel = 0.001
sigma_omaga_rel = 0.001

# Optional additive residual detuning when the nominal omega0 is exactly zero.
# Strict relative-noise model: leave this False, because relative noise on zero is zero.
# Hardware residual-detuning model: set True to use std = sigma_omaga0_rel * |J|.
ADD_OMEGA0_DETUNING_NOISE_WHEN_ZERO = False
sigma_omaga0_rel = sigma_omaga_rel

# Optional additive residual coupling for designed-zero J_{ij} terms, e.g. control-control
# couplings that are nominally set to zero.  Strict relative-noise model: keep False.
ADD_SPURIOUS_ZERO_COUPLING_NOISE = False
sigma_zero_J_rel = sigma_J_rel

N_TRIALS  = 100          # number of random trials per (N_RESAMPLE, dis_n) point
N_WORKERS = None         # set to an int (e.g., 8) to cap CPU cores; None uses all cores
POOL_CHUNKS_PER_WORKER = 2
MAX_TASKS_PER_CHILD = 10

# QuTiP solver settings.
MAX_INTERNAL_STEPS = 200000
SOLVER_ATOL = 1e-8
SOLVER_RTOL = 1e-6
USE_MAX_STEP = True
POINTS_PER_FAST_OSCILLATION = 20

CHECK_OPERATOR_BASIS = False

J = 0.0
T1 = np.inf
T2 = np.inf
rt_gamma_amp = 0.0
rt_gamma_phase = 0.0

if plateform == "cir":
    J = 2*np.pi*40*10**6
    T1 = 30*10**(-6)
    T2 = 30*10**(-6)
elif plateform == "ion":
    J = 2*np.pi*2*10**3
    T1 = np.inf
    T2 = 50
else:
    raise ValueError('plateform must be either "cir" or "ion".')

# Independent per-qubit T1/T2 decoherence rates.
# For Pauli-Z as collapse operator, (gamma_phi/2) D[Z] is implemented by
# collapse operator sqrt(gamma_phi/2) * Z.
gamma_amp = 0.0 if np.isinf(T1) else 1.0/T1
gamma_phi = 1.0/T2 - 0.5*gamma_amp
if gamma_phi < -1e-15:
    raise ValueError('Invalid T1/T2: pure-dephasing rate 1/T2 - 1/(2*T1) is negative.')
gamma_phi = max(0.0, gamma_phi)

rt_gamma_amp = np.sqrt(gamma_amp)
rt_gamma_phase = np.sqrt(gamma_phi/2.0)

J = J/np.max(np.abs(w))
omaga0=0#*np.e/2
#omaga=(-2*n+(N-1))*J-omaga0
omaga1=-n1*J-omaga0#(-2*n1+np.sum(np.abs(w)))*J-omaga0
omaga2=-n2*J-omaga0#(-2*n2+np.sum(np.abs(w)))*J-omaga0
omaga3=-n3*J-omaga0#(-2*n3+np.sum(np.abs(w)))*J-omaga0

if number_drive not in (1, 2, 3):
    raise ValueError("number_drive must be 1, 2, or 3.")
# Coupling strength and external magnetic field   #(0000 0001 0010 0011 0100 0101 0110 0111 1000 1001 1010 1011 1100 1101 1110 1111)


# Construct the Pauli matrices
# Build pairwise ZZ coupling operators (matching your original tensor ordering).
# We keep the same coupling topology as your original code: i is fixed to 0, and pairs are (0, j).
H_J_pairs = []  # list of Qobj, one per coupled pair
# Create the Hamiltonian
for i in range(1):
    for j in range(i + 1, N):
        if i == 0:
            pH = qt.sigmaz() * w[j - 1]
        else:
            pH = qt.qeye(2)
        for k in range(1, N):
            if k == i or k == j:
                pH = qt.tensor(qt.sigmaz(), pH)
            else:
                pH = qt.tensor(qt.qeye(2), pH)
        H_J_pairs.append(pH)

# For reference / backward-compatibility (not used in QobjEvo anymore):
H_J = 0
for _pH in H_J_pairs:
    H_J = H_J + _pH


def build_ZZ_pair(q1, q2, n_qubits):
    """Unweighted Z_q1 Z_q2 operator using the tensor ordering of this script."""
    if q1 == 0 or q2 == 0:
        out = qt.sigmaz()
    else:
        out = qt.qeye(2)
    for q in range(1, n_qubits):
        if q == q1 or q == q2:
            out = qt.tensor(qt.sigmaz(), out)
        else:
            out = qt.tensor(qt.qeye(2), out)
    return out


# Nominally-zero ZZ pairs. In the main model these are absent. They are used only
# if ADD_SPURIOUS_ZERO_COUPLING_NOISE=True to model residual unwanted couplings.
H_J_zero_pairs = []
for a in range(N):
    for b in range(a + 1, N):
        # The nominal nonzero couplings in this script are the target-control pairs (0, b).
        if a == 0:
            continue
        H_J_zero_pairs.append(build_ZZ_pair(a, b, N))

H_E0=0
H_Z_list=[]  # per-qubit Z operators (same tensor ordering as your original H_E0 build)
for i in range(N):
    if i==0:
        pH=qt.sigmaz()
    else:
        pH=qt.qeye(2)
    for j in range(1, N):
        if j==i:
            pH=qt.tensor(qt.sigmaz(), pH)
        else:
            pH=qt.tensor(qt.qeye(2), pH)
    H_Z_list.append(pH)
    H_E0=H_E0+pH

H_dx=qt.sigmax()
for i in range(N-1):
    H_dx=qt.tensor(qt.qeye(2), H_dx)
    #H_dx=qt.tensor(H_dx, qt.qeye(2))

def osc_cos1(t, args):
    return np.cos(omaga1*t)

def osc_cos2(t, args):
    return np.cos(omaga2*t)

def osc_cos3(t, args):
    return np.cos(omaga3*t)

H_dy=qt.sigmay()
for i in range(N-1):
    H_dy=qt.tensor(qt.qeye(2), H_dy)
    #H_dy=qt.tensor(H_dy, qt.qeye(2))


def osc_sin1(t, args):
    return np.sin(omaga1*t)

def osc_sin2(t, args):
    return np.sin(omaga2*t)

def osc_sin3(t, args):
    return np.sin(omaga3*t)


def single_qubit_operator(op, target, n_qubits):
    """Return op acting on qubit `target` using the tensor ordering of this script.

    Qubit 0 is the rightmost/least-significant qubit, matching `basis_state`.
    """
    if target == 0:
        out = op
    else:
        out = qt.qeye(2)
    for q in range(1, n_qubits):
        if q == target:
            out = qt.tensor(op, out)
        else:
            out = qt.tensor(qt.qeye(2), out)
    return out


# Independent per-qubit collapse operators. Do NOT sum these before passing to
# qt.liouvillian; summing would implement collective damping/dephasing.
sigma_minus_ops = [
    single_qubit_operator(qt.basis(2, 0)*qt.basis(2, 1).dag(), q, N)
    for q in range(N)
]
sigma_z_ops = [single_qubit_operator(qt.sigmaz(), q, N) for q in range(N)]

c_ops = []
if rt_gamma_amp > 0:
    c_ops.extend([rt_gamma_amp * op for op in sigma_minus_ops])
if rt_gamma_phase > 0:
    c_ops.extend([rt_gamma_phase * op for op in sigma_z_ops])


#making unitary bases

def basis_state(dim, label):    #2**3 deim
    state=0

    pstate=qt.basis(2, label%2)
    label=label//2
    for k in range(dim-1):
        pstate=qt.tensor(qt.basis(2, label%2), pstate)
        label=label//2
    state=state+pstate

    state=state.unit() 
    return state 

prefactor_basis_3= np.array([
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, -1, -1, -1, -1],
    [1, 1, -1, -1, 1, 1, -1, -1],
    [1, -1, 1, -1, 1, -1, 1, -1],
    [1, 1, -1, -1, -1, -1, 1, 1],
    [-1, 1, 1, -1, -1, 1, 1, -1],
    [1, -1, 1, -1, -1, 1, -1, 1],
    [-1, 1, 1, -1, 1, -1, -1, 1], 
    ])  

prefactor_basis_4= np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1],
    [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1],
    [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
    [1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1],
    [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
    [1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1],
    [1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1],
    [1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1],
    [-1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1],
    [-1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1],
    [1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1],
    [1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1],
    [-1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1], 
    [-1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1]
    ])      

prefactor_basis_5= np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1],
    [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1],
    [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
    [1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1],
    [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
    [1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1],
    [1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1],
    [1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1],
    [-1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1],
    [-1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1],
    [1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1],
    [1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1],
    [-1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1],
    [-1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1],
    [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1],
    [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1],
    [1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1],
    [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
    [1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1],
    [1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1],
    [1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1],
    [-1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
    [-1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1],
    [1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1],
    [1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1],
    [-1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1],
    [-1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1]
    ])      

Ubases=np.empty((2**(2*N)), dtype=object)
for i in range(2**N): #|0><i|
    for j in range(2**N): #prefactor
        if N==3:
            U=basis_state(N, 0)*basis_state(N, i).dag()*prefactor_basis_3[j, 0]
        elif N==4:
            U=basis_state(N, 0)*basis_state(N, i).dag()*prefactor_basis_4[j, 0]
        elif N==5:
            U=basis_state(N, 0)*basis_state(N, i).dag()*prefactor_basis_5[j, 0]

        for k in range(1, 2**N): # make each element
            if N==3:
                U=U+prefactor_basis_3[j][k]*basis_state(N, k)*basis_state(N, (k+i)%(2**N)).dag()
            elif N==4:
                U=U+prefactor_basis_4[j][k]*basis_state(N, k)*basis_state(N, (k+i)%(2**N)).dag()   
            elif N==5:
                U=U+prefactor_basis_5[j][k]*basis_state(N, k)*basis_state(N, (k+i)%(2**N)).dag()      
        Ubases[2**N*i+j]=U#/((U.dag()*U).tr())**0.5


def check_operator_basis(Ubases, n_qubits, tol=1e-9):
    d = 2**n_qubits
    ident = qt.qeye([2]*n_qubits)
    for a, Ua in enumerate(Ubases):
        err = (Ua.dag()*Ua - ident).norm()
        if err > tol:
            raise ValueError(f"Ubases[{a}] is not unitary: error={err}")
    for a, Ua in enumerate(Ubases):
        Uadag = Ua.dag()
        for b, Ub in enumerate(Ubases):
            inner = (Uadag*Ub).tr()
            expected = d if a == b else 0
            if abs(inner - expected) > tol:
                raise ValueError(
                    f"Operator basis is not orthogonal at ({a}, {b}): "
                    f"inner={inner}, expected={expected}"
                )
    print("Operator basis check passed.")


if CHECK_OPERATOR_BASIS:
    check_operator_basis(Ubases, N)

#######
#Toffoli
Toffoli=0
if number_drive==1:
    for i in range(2**(N-1)):
        total_w=0
        temp_i=i
        for j in range(N-1):
            if temp_i%2!=0:
                total_w=total_w+w[j]
            else:
                total_w=total_w-w[j]
            temp_i=temp_i//2
        if total_w==n1:# 
            print(i)
            Toffoli=Toffoli+qt.tensor(basis_state(N-1, i)*basis_state(N-1, i).dag(), -1j*qt.sigmax())
        else:
            Toffoli=Toffoli+qt.tensor(basis_state(N-1, i)*basis_state(N-1, i).dag(), qt.qeye(2))
elif number_drive==2:
    for i in range(2**(N-1)):
        total_w=0
        temp_i=i
        for j in range(N-1):
            if temp_i%2!=0:
                total_w=total_w+w[j]
            else:
                total_w=total_w-w[j]
            temp_i=temp_i//2
        if total_w==n1 or total_w==n2:# 
            print(i)
            Toffoli=Toffoli+qt.tensor(basis_state(N-1, i)*basis_state(N-1, i).dag(), -1j*qt.sigmax())
        else:
            Toffoli=Toffoli+qt.tensor(basis_state(N-1, i)*basis_state(N-1, i).dag(), qt.qeye(2))
elif number_drive==3:
    for i in range(2**(N-1)):
        total_w=0
        temp_i=i
        for j in range(N-1):
            if temp_i%2!=0:
                total_w=total_w+w[j]
            else:
                total_w=total_w-w[j]
            temp_i=temp_i//2
        if total_w==n1 or total_w==n2 or total_w==n3:# 
            print(i)
            Toffoli=Toffoli+qt.tensor(basis_state(N-1, i)*basis_state(N-1, i).dag(), -1j*qt.sigmax())
        else:
            Toffoli=Toffoli+qt.tensor(basis_state(N-1, i)*basis_state(N-1, i).dag(), qt.qeye(2))


def make_solver_options(max_step_value=None):
    kwargs = dict(
        store_final_state=True,
        store_states=False,
        nsteps=MAX_INTERNAL_STEPS,
        atol=SOLVER_ATOL,
        rtol=SOLVER_RTOL,
    )
    if max_step_value is not None:
        kwargs["max_step"] = max_step_value
    return qt.Options(**kwargs)


def interval_index(t, boundaries):
    """Return k such that t is in [boundaries[k], boundaries[k+1])."""
    k = int(np.searchsorted(boundaries, t, side="right") - 1)
    if k < 0:
        return 0
    last = len(boundaries) - 2
    if k > last:
        return last
    return k


def make_piecewise_coeff(values, boundaries):
    values = np.asarray(values, dtype=float)
    boundaries = np.asarray(boundaries, dtype=float)

    def coeff(t, args=None):
        return values[interval_index(t, boundaries)]

    return coeff


def make_drive_coeff(Omega_values, omega_values, boundaries, trig):
    """Drive coefficient with piecewise-constant Omega/omega and continuous phase.

    In interval k, Omega(t)=Omega_k and omega(t)=omega_k.  The phase is
        phi(t)=phi(t_k)+omega_k*(t-t_k),
    so frequency noise accumulates as phase noise without making the carrier
    waveform piecewise constant.
    """
    Omega_values = np.asarray(Omega_values, dtype=float)
    omega_values = np.asarray(omega_values, dtype=float)
    boundaries = np.asarray(boundaries, dtype=float)
    interval_dt = np.diff(boundaries)
    phase_start = np.zeros_like(omega_values, dtype=float)
    if len(omega_values) > 1:
        phase_start[1:] = np.cumsum(omega_values[:-1] * interval_dt[:-1])

    if trig == "cos":
        trig_fn = np.cos
    elif trig == "sin":
        trig_fn = np.sin
    else:
        raise ValueError('trig must be "cos" or "sin"')

    def coeff(t, args=None):
        k = interval_index(t, boundaries)
        phi = phase_start[k] + omega_values[k] * (t - boundaries[k])
        return Omega_values[k] * trig_fn(phi)

    return coeff


def sample_relative_or_constant(rng_local, nominal, sigma_rel, size, noisy=True):
    if noisy:
        return nominal * (1.0 + rng_local.normal(0.0, sigma_rel, size=size))
    return nominal * np.ones(size, dtype=float)


def sample_omega0_values(rng_local, n_resample, noisy=True):
    if not noisy:
        return omaga0 * np.ones((N, n_resample), dtype=float)
    if omaga0 != 0:
        return omaga0 * (1.0 + rng_local.normal(0.0, sigma_omaga0_rel, size=(N, n_resample)))
    if ADD_OMEGA0_DETUNING_NOISE_WHEN_ZERO:
        return rng_local.normal(0.0, sigma_omaga0_rel * abs(J), size=(N, n_resample))
    return np.zeros((N, n_resample), dtype=float)


def build_sequential_pulse_liouvillians(Omaga, T, n_resample, trial_seed):
    """Build the sequential single-tone pulses for one Monte-Carlo trial.

    Each pulse has independent noise in every nonzero J_ij, the drive amplitude Omega,
    the drive frequency omega, and per-qubit omega0 detuning.
    """
    rng_local = np.random.default_rng(trial_seed)
    boundaries = np.linspace(0.0, T, n_resample + 1)
    n_pairs = len(H_J_pairs)

    active_omegas = []
    if number_drive >= 1:
        active_omegas.append(omaga1)
    if number_drive >= 2:
        active_omegas.append(omaga2)
    if number_drive >= 3:
        active_omegas.append(omaga3)

    S_list = []
    for omega_nom in active_omegas:
        # Independent static-Hamiltonian control errors for this pulse.
        J_nonzero_values = sample_relative_or_constant(
            rng_local, J, sigma_J_rel, size=(n_pairs, n_resample), noisy=enable_time_step_noise
        )
        omega0_q_values = sample_omega0_values(
            rng_local, n_resample, noisy=enable_time_step_noise
        )

        terms = []
        for q in range(N):
            terms.append([H_Z_list[q], make_piecewise_coeff(-omega0_q_values[q] / 2.0, boundaries)])

        for p in range(n_pairs):
            # H_J_pairs[p] already contains the intended signed/weighted ZZ operator.
            terms.append([H_J_pairs[p], make_piecewise_coeff(J_nonzero_values[p] / 2.0, boundaries)])

        if ADD_SPURIOUS_ZERO_COUPLING_NOISE and enable_time_step_noise:
            zero_J_values = rng_local.normal(
                0.0, sigma_zero_J_rel * abs(J), size=(len(H_J_zero_pairs), n_resample)
            )
            for p, H_zero in enumerate(H_J_zero_pairs):
                terms.append([H_zero, make_piecewise_coeff(zero_J_values[p] / 2.0, boundaries)])

        Omega_values = sample_relative_or_constant(
            rng_local, Omaga, sigma_Omaga_rel, size=n_resample, noisy=enable_time_step_noise
        )
        omega_values = sample_relative_or_constant(
            rng_local, omega_nom, sigma_omaga_rel, size=n_resample, noisy=enable_time_step_noise
        )

        terms += [
            [H_dx, make_drive_coeff(Omega_values, omega_values, boundaries, "cos")],
            [H_dy, make_drive_coeff(Omega_values, omega_values, boundaries, "sin")],
        ]

        S_p = qt.QobjEvo(terms, tlist=boundaries)
        if deco == "on":
            S_p = qt.liouvillian(S_p, c_ops)
        S_list.append(S_p)

    return S_list, boundaries


def fidelity_for_trial(args):
    n_resample, dis_value, trial_seed, max_step_value = args
    Omaga = J / dis_value
    T = np.pi / (2 * Omaga)
    S_list, times = build_sequential_pulse_liouvillians(Omaga, T, n_resample, trial_seed)
    options = make_solver_options(max_step_value)

    total = 0
    n_terms = 2 ** (2 * N)
    for j in range(n_terms):
        state = Ubases[j]
        for S in S_list:
            result = qt.mesolve(S, state, times, options=options)
            state = result.final_state
        total += (Toffoli * Ubases[j].dag() * Toffoli.dag() * state).tr()

    return ((total + 2 ** (2 * N)) / (2 ** (2 * N) * (2 ** N + 1))).real


def run_trials_for_point(n_resample, dis_value, trial_seeds, max_step_value, n_workers=None):
    tasks = [(n_resample, dis_value, int(seed), max_step_value) for seed in trial_seeds]

    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = int(n_workers)

    if n_workers <= 1 or len(tasks) <= 1:
        return np.array([fidelity_for_trial(task) for task in tasks], dtype=float)

    if "fork" not in mp.get_all_start_methods():
        print("Warning: multiprocessing start method 'fork' is unavailable; using serial trials.")
        return np.array([fidelity_for_trial(task) for task in tasks], dtype=float)

    n_workers = min(n_workers, len(tasks))
    chunksize = max(1, len(tasks) // (n_workers * POOL_CHUNKS_PER_WORKER))
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=n_workers, maxtasksperchild=MAX_TASKS_PER_CHILD) as pool:
        return np.fromiter(pool.imap_unordered(fidelity_for_trial, tasks, chunksize=chunksize), dtype=float, count=len(tasks))


def max_step_for_drive_frequencies():
    active = []
    if number_drive >= 1:
        active.append(abs(omaga1))
    if number_drive >= 2:
        active.append(abs(omaga2))
    if number_drive >= 3:
        active.append(abs(omaga3))
    omega_max = max(active) if active else 0.0
    if USE_MAX_STEP and omega_max > 0:
        return (2*np.pi/omega_max) / POINTS_PER_FAST_OSCILLATION
    return None


# -----------------------------
# Monte-Carlo trials
# -----------------------------

if N not in (3, 4, 5):
    raise ValueError("This script only defines prefactor_basis tables for N = 3, 4, or 5.")
if len(w) < N-1:
    raise ValueError("The weight array w must contain at least N-1 entries for the control qubits.")

solver_max_step = max_step_for_drive_frequencies()
seed_sequence = np.random.SeedSequence(SEED)

for n_resample in N_RESAMPLE_LIST:
    print(f"\n===== N_resample={n_resample} =====")

    for i in range(len(dis_n)):
        trial_seeds = seed_sequence.spawn(N_TRIALS)
        trial_seed_ints = [s.generate_state(1, dtype=np.uint32)[0] for s in trial_seeds]

        fid_trials = run_trials_for_point(
            n_resample=n_resample,
            dis_value=float(dis_n[i]),
            trial_seeds=trial_seed_ints,
            max_step_value=solver_max_step,
            n_workers=N_WORKERS,
        )

        mean_fid = float(np.mean(fid_trials))
        std_fid = float(np.std(fid_trials, ddof=1)) if N_TRIALS > 1 else 0.0

        print(
            f"N_resample={n_resample}  dis_n={dis_n[i]}  "
            f"mean_fidelity={mean_fid:.12f}  std={std_fid:.12f}  "
            f"trials={N_TRIALS}  workers={N_WORKERS if N_WORKERS is not None else os.cpu_count()}"
        )
