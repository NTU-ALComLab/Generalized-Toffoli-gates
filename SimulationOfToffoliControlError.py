import os
# Avoid oversubscribing CPU threads inside each worker process (common with NumPy/BLAS).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import multiprocessing as mp
import numpy as np
import qutip as qt

# -----------------------------
# Reproducibility and noise model
# -----------------------------
SEED = 0
enable_time_step_noise = True  # False recovers the nominal/no-control-error behavior.

# The values in this list are physical resampling counts per gate, not ODE solver steps.
# For a value N_resample, each control parameter is held constant on each interval
# [t_k, t_{k+1}), with t_k = k*T/N_resample.
# N_RESAMPLE_LIST = [10, 20, 30, 40, 50, 60, 70, 80, 100, 1000]
N_RESAMPLE_LIST = [10000]

# Relative Gaussian control error for nonzero designed parameters.
sigma_J_rel = 0.001       # nonzero target-control couplings J_ij
sigma_Omaga_rel = 0.001   # each drive amplitude Omega_l
sigma_omaga_rel = 0.001   # each drive angular frequency omega_l

# Nominal omega0 is zero in this script. Relative noise around zero gives exactly zero.
# Setting this True models residual/additive per-qubit detuning noise with std sigma_omaga0_rel*|J|.
# Set False for a strict relative-error-only model.
ADD_OMEGA0_DETUNING_NOISE_WHEN_ZERO = False
sigma_omaga0_rel = 0.01

# Designed-zero Jij terms are normally not included. Turning this on models spurious residual
# control-control couplings, which is an additional hardware-error assumption beyond relative error.
ADD_SPURIOUS_ZERO_COUPLING_NOISE = False
sigma_zero_J_rel = 0.01

N_TRIALS = 100
N_WORKERS = None  # None uses all CPU cores for trial-level parallelism; set 1 for serial.
MAX_TASKS_PER_CHILD = 20

# -----------------------------
# Solver settings
# -----------------------------
MAX_INTERNAL_STEPS = 200000
SOLVER_ATOL = 1e-8
SOLVER_RTOL = 1e-6
USE_MAX_STEP = True
POINTS_PER_FAST_OSCILLATION = 20
CHECK_OPERATOR_BASIS = False

# -----------------------------
# Gate / platform settings
# -----------------------------
N = 5  # total qubits: N-1 controls plus 1 target
n1 = -2
n2 = 2
n3 = -2
w = np.array([1, 1, 1, 1])  # weights of the N-1 controls; small index first
plateform = "cir"  # "cir" or "ion"
deco = "on"        # "on" or "off"
number_drive = 2   # 1, 2, or 3
n_range = 9
n_min = 8
dis_n = np.arange(n_min, n_range, 4)
# dis_n = np.array([2, 4, 6, 8, 7, 13, np.e*7])

if N not in (3, 4, 5):
    raise ValueError("This script only defines prefactor_basis tables for N = 3, 4, or 5.")
if len(w) < N - 1:
    raise ValueError("The weight array w must contain at least N-1 entries for the control qubits.")
if number_drive not in (1, 2, 3):
    raise ValueError("number_drive must be 1, 2, or 3.")

# -----------------------------
# Platform parameters and independent T1/T2 decoherence
# -----------------------------
if plateform == "cir":
    J = 2 * np.pi * 40 * 10**6
    T1 = 30 * 10**(-6)
    T2 = 30 * 10**(-6)
elif plateform == "ion":
    J = 2 * np.pi * 2 * 10**3
    T1 = np.inf
    T2 = 50
else:
    raise ValueError('plateform must be either "cir" or "ion".')

# Normalize J exactly as in the original script.
J = J / np.max(np.abs(w))

# Independent per-qubit T1/T2 decoherence rates.
# With the Lindblad dissipator D[L](rho) = L rho L^dag - 1/2 {L^dag L, rho},
# the pure-dephasing term (gamma_phi/2) D[Z_i] is implemented by
# collapse operator sqrt(gamma_phi/2) * Z_i.
gamma_amp = 0.0 if np.isinf(T1) else 1.0 / T1
gamma_phi = 1.0 / T2 - 0.5 * gamma_amp
if gamma_phi < -1e-15:
    raise ValueError("Invalid T1/T2 values: 1/T2 - 1/(2*T1) is negative.")
gamma_phi = max(0.0, gamma_phi)
rt_gamma_amp = np.sqrt(gamma_amp)
rt_gamma_phase = np.sqrt(gamma_phi / 2.0)

omaga0 = 0.0
omaga1 = -n1 * J - omaga0
omaga2 = -n2 * J - omaga0
omaga3 = -n3 * J - omaga0
omaga_nominal_all = [omaga1, omaga2, omaga3]

active_omagas = [abs(omaga_nominal_all[k]) for k in range(number_drive)]
omega_max = max(active_omagas) if active_omagas else 0.0
solver_max_step = None
if USE_MAX_STEP and omega_max > 0:
    solver_max_step = (2 * np.pi / omega_max) / POINTS_PER_FAST_OSCILLATION

# -----------------------------
# Operator builders
# -----------------------------
def basis_state(dim, label):
    """Computational basis state using the tensor ordering of the original script."""
    pstate = qt.basis(2, label % 2)
    label //= 2
    for _ in range(dim - 1):
        pstate = qt.tensor(qt.basis(2, label % 2), pstate)
        label //= 2
    return pstate.unit()


def single_qubit_operator(op, target, n_qubits):
    """Return op acting on qubit `target`; qubit 0 is rightmost/least-significant."""
    out = op if target == 0 else qt.qeye(2)
    for q in range(1, n_qubits):
        out = qt.tensor(op if q == target else qt.qeye(2), out)
    return out


def two_qubit_zz_operator(q1, q2, n_qubits, weight=1.0):
    """Return weight * Z_q1 Z_q2 with the same tensor ordering as the original script."""
    out = qt.sigmaz() if (q1 == 0 or q2 == 0) else qt.qeye(2)
    if q1 == 0 or q2 == 0:
        out = weight * out
    for q in range(1, n_qubits):
        out = qt.tensor(qt.sigmaz() if (q == q1 or q == q2) else qt.qeye(2), out)
    return out


# Nonzero designed couplings: target qubit 0 coupled to each control qubit j=1..N-1.
# The weight w[j-1] is included in the operator, so the noisy scalar below represents J(t).
H_J_pairs = []
J_pair_labels = []
for j in range(1, N):
    if abs(w[j - 1]) > 0:
        H_J_pairs.append(two_qubit_zz_operator(0, j, N, weight=w[j - 1]))
        J_pair_labels.append((0, j, w[j - 1]))

# Optional spurious designed-zero couplings among control qubits.
H_J_zero_pairs = []
J_zero_pair_labels = []
if ADD_SPURIOUS_ZERO_COUPLING_NOISE:
    for i in range(1, N):
        for j in range(i + 1, N):
            H_J_zero_pairs.append(two_qubit_zz_operator(i, j, N, weight=1.0))
            J_zero_pair_labels.append((i, j, 0.0))

H_J = sum(H_J_pairs, 0)
H_Z_list = [single_qubit_operator(qt.sigmaz(), q, N) for q in range(N)]
H_E0 = sum(H_Z_list, 0)

H_dx = single_qubit_operator(qt.sigmax(), 0, N)
H_dy = single_qubit_operator(qt.sigmay(), 0, N)

# Independent per-qubit collapse operators.
sigma_minus_ops = [single_qubit_operator(qt.basis(2, 0) * qt.basis(2, 1).dag(), q, N) for q in range(N)]
sigma_z_ops = H_Z_list
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



# -----------------------------
# Solver and piecewise-noise helpers
# -----------------------------
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


options = make_solver_options(solver_max_step)


def interval_index(t, boundaries):
    k = np.searchsorted(boundaries, t, side="right") - 1
    if k < 0:
        return 0
    last = len(boundaries) - 2
    if k > last:
        return last
    return k


def make_piecewise_constant_coeff(values, boundaries):
    values = np.asarray(values, dtype=float)
    def coeff(t, args=None):
        return values[interval_index(t, boundaries)]
    return coeff


def make_drive_coeff(Omega_values, omega_values, boundaries, trig):
    """Coefficient Omega_k*cos/sin(phi_k(t)) with piecewise-constant Omega and omega."""
    Omega_values = np.asarray(Omega_values, dtype=float)
    omega_values = np.asarray(omega_values, dtype=float)
    dt = np.diff(boundaries)
    phase_start = np.zeros_like(omega_values, dtype=float)
    if len(omega_values) > 1:
        phase_start[1:] = np.cumsum(omega_values[:-1] * dt[:-1])

    if trig == "cos":
        func = np.cos
    elif trig == "sin":
        func = np.sin
    else:
        raise ValueError('trig must be "cos" or "sin".')

    def coeff(t, args=None):
        k = interval_index(t, boundaries)
        phase = phase_start[k] + omega_values[k] * (t - boundaries[k])
        return Omega_values[k] * func(phase)
    return coeff


def build_qobjevo_for_trial(Omaga, T, N_resample, local_rng):
    """Build the noisy Hamiltonian/Liouvillian for one Monte-Carlo trial."""
    boundaries = np.linspace(0.0, T, N_resample + 1)
    n_intervals = N_resample
    terms = []

    if enable_time_step_noise:
        J_nonzero_values = J * (1.0 + local_rng.normal(0.0, sigma_J_rel, size=(len(H_J_pairs), n_intervals)))

        if ADD_SPURIOUS_ZERO_COUPLING_NOISE:
            J_zero_values = local_rng.normal(
                0.0, sigma_zero_J_rel * abs(J), size=(len(H_J_zero_pairs), n_intervals)
            )
        else:
            J_zero_values = np.zeros((len(H_J_zero_pairs), n_intervals))

        if omaga0 != 0:
            omaga0_q_values = omaga0 * (1.0 + local_rng.normal(0.0, sigma_omaga0_rel, size=(N, n_intervals)))
        elif ADD_OMEGA0_DETUNING_NOISE_WHEN_ZERO:
            omaga0_q_values = local_rng.normal(0.0, sigma_omaga0_rel * abs(J), size=(N, n_intervals))
        else:
            omaga0_q_values = np.zeros((N, n_intervals))

        Omega_values = []
        omega_values = []
        for drive_idx in range(number_drive):
            Omega_values.append(Omaga * (1.0 + local_rng.normal(0.0, sigma_Omaga_rel, size=n_intervals)))
            omega_nom = omaga_nominal_all[drive_idx]
            omega_values.append(omega_nom * (1.0 + local_rng.normal(0.0, sigma_omaga_rel, size=n_intervals)))
    else:
        J_nonzero_values = J * np.ones((len(H_J_pairs), n_intervals))
        J_zero_values = np.zeros((len(H_J_zero_pairs), n_intervals))
        omaga0_q_values = omaga0 * np.ones((N, n_intervals))
        Omega_values = [Omaga * np.ones(n_intervals) for _ in range(number_drive)]
        omega_values = [omaga_nominal_all[d] * np.ones(n_intervals) for d in range(number_drive)]

    for q in range(N):
        terms.append([H_Z_list[q], make_piecewise_constant_coeff(-omaga0_q_values[q] / 2.0, boundaries)])

    for p, H_pair in enumerate(H_J_pairs):
        terms.append([H_pair, make_piecewise_constant_coeff(J_nonzero_values[p] / 2.0, boundaries)])

    for p, H_pair in enumerate(H_J_zero_pairs):
        terms.append([H_pair, make_piecewise_constant_coeff(J_zero_values[p] / 2.0, boundaries)])

    for d in range(number_drive):
        terms.append([H_dx, make_drive_coeff(Omega_values[d], omega_values[d], boundaries, "cos")])
        terms.append([H_dy, make_drive_coeff(Omega_values[d], omega_values[d], boundaries, "sin")])

    S = qt.QobjEvo(terms)
    if deco == "on":
        S = qt.liouvillian(S, c_ops)
    return S, boundaries


def fidelity_sum_serial(S, times):
    total = 0
    for j in range(2 ** (2 * N)):
        result = qt.mesolve(S, Ubases[j], times, options=options)
        total += (Toffoli * Ubases[j].dag() * Toffoli.dag() * result.final_state).tr()
    return total


def trial_worker(task):
    N_resample, dis_n_value, trial_seed = task
    local_rng = np.random.default_rng(int(trial_seed))
    Omaga = J / dis_n_value
    T = np.pi / (2 * Omaga)
    S, times = build_qobjevo_for_trial(Omaga, T, N_resample, local_rng)
    Fidelity = fidelity_sum_serial(S, times)
    return ((Fidelity + 2 ** (2 * N)) / (2 ** (2 * N) * (2 ** N + 1))).real


def run_trials_for_point(N_resample, dis_n_value, trial_seeds, n_workers=None):
    tasks = [(N_resample, dis_n_value, int(seed)) for seed in trial_seeds]

    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = int(n_workers)

    if n_workers <= 1 or len(tasks) <= 1:
        return np.array([trial_worker(task) for task in tasks], dtype=float)

    if "fork" not in mp.get_all_start_methods():
        print("Warning: multiprocessing start method 'fork' is unavailable; using serial trials.")
        return np.array([trial_worker(task) for task in tasks], dtype=float)

    n_workers = min(n_workers, len(tasks))
    chunksize = max(1, len(tasks) // (n_workers * 4))
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=n_workers, maxtasksperchild=MAX_TASKS_PER_CHILD) as pool:
        return np.fromiter(pool.imap_unordered(trial_worker, tasks, chunksize=chunksize), dtype=float, count=len(tasks))


# -----------------------------
# Optional operator-basis sanity check
# -----------------------------
def check_operator_basis(Ubases, n_qubits, tol=1e-9):
    d = 2 ** n_qubits
    ident = qt.qeye([2] * n_qubits)
    for a, Ua in enumerate(Ubases):
        err = (Ua.dag() * Ua - ident).norm()
        if err > tol:
            raise ValueError(f"Ubases[{a}] is not unitary: error={err}")
    for a, Ua in enumerate(Ubases):
        Uadag = Ua.dag()
        for b, Ub in enumerate(Ubases):
            inner = (Uadag * Ub).tr()
            expected = d if a == b else 0
            if abs(inner - expected) > tol:
                raise ValueError(f"Operator basis is not orthogonal at ({a}, {b}): inner={inner}, expected={expected}")
    print("Operator basis check passed.")


if CHECK_OPERATOR_BASIS:
    check_operator_basis(Ubases, N)


# -----------------------------
# Monte-Carlo sweep
# -----------------------------
seed_sequence = np.random.SeedSequence(SEED)
all_trial_seeds = seed_sequence.generate_state(len(N_RESAMPLE_LIST) * len(dis_n) * N_TRIALS, dtype=np.uint32)
seed_cursor = 0

for N_resample in N_RESAMPLE_LIST:
    print(f"\n===== N_resample={N_resample} =====")
    for dis_n_value in dis_n:
        trial_seeds = all_trial_seeds[seed_cursor:seed_cursor + N_TRIALS]
        seed_cursor += N_TRIALS

        fid_trials = run_trials_for_point(int(N_resample), float(dis_n_value), trial_seeds, n_workers=N_WORKERS)
        mean_fid = float(np.mean(fid_trials))
        std_fid = float(np.std(fid_trials, ddof=1)) if N_TRIALS > 1 else 0.0
        stderr = std_fid / np.sqrt(N_TRIALS) if N_TRIALS > 1 else 0.0

        print(
            f"N_resample={N_resample}  J/Omega={dis_n_value}  "
            f"mean_fidelity={mean_fid:.12f}  std={std_fid:.12f}  stderr={stderr:.12f}  "
            f"trials={N_TRIALS}"
        )
