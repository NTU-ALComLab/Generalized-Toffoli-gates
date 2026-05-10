import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import multiprocessing as mp
import numpy as np
import qutip as qt
N=5  #num of qubits
n1=0
n2=-2
n3=2
w=np.array([1, 1, 1, 1])  
plateform="ion" #cir, ion
deco="on" #on, off
number_drive=1 #1, 2, 3
n_range=33
n_min=4
dis_n=np.arange(n_min, n_range, 4)


N_OUTPUT_POINTS = 2
MAX_INTERNAL_STEPS = 200000
SOLVER_ATOL = 1e-8
SOLVER_RTOL = 1e-6
USE_MAX_STEP = True
POINTS_PER_FAST_OSCILLATION = 20
CHECK_OPERATOR_BASIS = False 

N_WORKERS = None
POOL_CHUNKS_PER_WORKER = 4
MAX_TASKS_PER_CHILD = 200

if N not in (3, 4, 5):
    raise ValueError("This script only defines prefactor_basis tables for N = 3, 4, or 5.")
if len(w) < N-1:
    raise ValueError("The weight array w must contain at least N-1 entries for the control qubits.")

J=0
T1=np.inf
T2=np.inf
rt_gamma_amp=0.0
rt_gamma_phase=0.0

if plateform=="cir":
    J=2*np.pi*40*10**6
    T1=30*10**(-6)
    T2=30*10**(-6)
elif plateform=="ion":
    J=2*np.pi*2*10**3
    T1=np.inf
    T2=50
else:
    raise ValueError('plateform must be either "cir" or "ion".')


gamma_amp = 0.0 if np.isinf(T1) else 1.0/T1
gamma_phi = 1.0/T2 - 0.5*gamma_amp
if gamma_phi < -1e-15:
    raise ValueError(
        'Invalid T1/T2 values: pure-dephasing rate 1/T2 - 1/(2*T1) is negative.'
    )
gamma_phi = max(0.0, gamma_phi)

rt_gamma_amp = np.sqrt(gamma_amp)
rt_gamma_phase = np.sqrt(gamma_phi/2.0)
# rt_gamma_phase = np.sqrt(gamma_phi)


J=J/np.max(np.abs(w))
omaga0=0

omaga1=-n1*J-omaga0
omaga2=-n2*J-omaga0
omaga3=-n3*J-omaga0

active_omagas = []
if number_drive >= 1:
    active_omagas.append(abs(omaga1))
if number_drive >= 2:
    active_omagas.append(abs(omaga2))
if number_drive >= 3:
    active_omagas.append(abs(omaga3))
omega_max = max(active_omagas) if active_omagas else 0.0
solver_max_step = None
if USE_MAX_STEP and omega_max > 0:
    solver_max_step = (2*np.pi/omega_max) / POINTS_PER_FAST_OSCILLATION


H_J=0
for i in range(1):
    for j in range(i + 1, N):
        if i==0:
            pH=qt.sigmaz()*w[j-1]
        else:
            pH=qt.qeye(2)
        for k in range(1, N):
            if k==i or k==j:
                pH=qt.tensor(qt.sigmaz(), pH)
            else:
                pH=qt.tensor(qt.qeye(2), pH)   
        H_J=H_J+pH



H_E0=0
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


sigma_minus_ops = [
    single_qubit_operator(qt.basis(2, 0)*qt.basis(2, 1).dag(), i, N)
    for i in range(N)
]
sigma_z_ops = [single_qubit_operator(qt.sigmaz(), i, N) for i in range(N)]

c_ops = []
if rt_gamma_amp > 0:
    c_ops.extend([rt_gamma_amp * op for op in sigma_minus_ops])
if rt_gamma_phase > 0:
    c_ops.extend([rt_gamma_phase * op for op in sigma_z_ops])



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
for i in range(2**N): 
    for j in range(2**N): 
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
    """Check unitarity and Hilbert-Schmidt orthogonality of the operator basis."""
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



_gS = None
_gTimes = None
_gOptions = None


def _fidelity_term_for_basis_index(j):
    """Return one term in the average-fidelity summation.

    This is exactly the same calculation as one iteration of the original serial loop.
    """
    result = qt.mesolve(_gS, Ubases[j], _gTimes, options=_gOptions)
    return (Toffoli * Ubases[j].dag() * Toffoli.dag() * result.final_state).tr()


def _serial_fidelity_sum(S, times, options, n_terms):
    """Reference serial implementation of the fidelity summation."""
    total = 0
    for j in range(n_terms):
        result = qt.mesolve(S, Ubases[j], times, options=options)
        total += (Toffoli * Ubases[j].dag() * Toffoli.dag() * result.final_state).tr()
    return total


def parallel_fidelity_sum(S, times, options, n_terms, n_workers=None):
    """Compute the fidelity summation in parallel over basis index j.

    Mathematical result:
        sum_j Tr(Toffoli * Ubases[j]^dag * Toffoli^dag * E(Ubases[j]))

    The summation is order-independent, so imap_unordered is safe.
    """
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = int(n_workers)

    if n_workers <= 1 or n_terms <= 1:
        return _serial_fidelity_sum(S, times, options, n_terms)

    if "fork" not in mp.get_all_start_methods():
        print("Warning: multiprocessing start method 'fork' is unavailable; using serial loop.")
        return _serial_fidelity_sum(S, times, options, n_terms)

    n_workers = min(n_workers, n_terms)

    global _gS, _gTimes, _gOptions
    _gS = S
    _gTimes = times
    _gOptions = options

    chunksize = max(1, n_terms // (n_workers * POOL_CHUNKS_PER_WORKER))
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=n_workers, maxtasksperchild=MAX_TASKS_PER_CHILD) as pool:
        return sum(pool.imap_unordered(_fidelity_term_for_basis_index, range(n_terms), chunksize=chunksize))


options = make_solver_options(solver_max_step)
for i in range(len(dis_n)):
    Omaga=J/(dis_n[i])
    T=np.pi/(2*Omaga)
    times=np.linspace(0.0, T, N_OUTPUT_POINTS)
    S=0
    if number_drive==1:
        S=qt.QobjEvo([-omaga0/2*H_E0+J/2*H_J, [Omaga*H_dx, osc_cos1], [Omaga*H_dy, osc_sin1]], tlist=times)
    elif number_drive==2:
        S=qt.QobjEvo([-omaga0/2*H_E0+J/2*H_J, [Omaga*H_dx, osc_cos1], [Omaga*H_dx, osc_cos2], [Omaga*H_dy, osc_sin1], [Omaga*H_dy, osc_sin2]], tlist=times)
    elif number_drive==3:
        S=qt.QobjEvo([-omaga0/2*H_E0+J/2*H_J, [Omaga*H_dx, osc_cos1], [Omaga*H_dx, osc_cos2], [Omaga*H_dx, osc_cos3], [Omaga*H_dy, osc_sin1], [Omaga*H_dy, osc_sin2], [Omaga*H_dy, osc_sin3]], tlist=times)
    if deco=="on":
        S=qt.liouvillian(S, c_ops)
    n_terms = 2**(2*N)
    Fidelity = parallel_fidelity_sum(S, times, options, n_terms, N_WORKERS)

    print(str(((Fidelity+2**(2*N))/(2**(2*N)*(2**N+1))).real))

