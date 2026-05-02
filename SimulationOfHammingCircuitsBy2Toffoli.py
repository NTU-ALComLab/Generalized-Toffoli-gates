import os
# Avoid oversubscribing CPU threads inside each worker process.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import multiprocessing as mp
import numpy as np
import qutip as qt
#from tqdm import tqdm
N_qi=5 #number of information qubits
N=N_qi+2  #num of total qubits

n=0
w=np.array([1, 1, 1, 1])  # small indx first (alternating one first)
plateform="cir"
deco="off"

n_range=33
n_min=4
dis_n=np.arange(n_min, n_range, 4)

# Solver/output settings.
# We only use result.final_state, so the output grid contains only the
# initial and final times. Numerical accuracy is controlled by the
# adaptive ODE solver options below, not by a dense output tlist.
N_OUTPUT_POINTS = 2
MAX_INTERNAL_STEPS = 200000
SOLVER_ATOL = 1e-8
SOLVER_RTOL = 1e-6
USE_MAX_STEP = True
POINTS_PER_FAST_OSCILLATION = 20

# Multiprocessing settings.
# N_WORKERS = None uses all available CPU cores. Set N_WORKERS = 1 for serial.
# The parallel implementation uses fork when available so large QuTiP objects
# are inherited by workers instead of being pickled repeatedly.
N_WORKERS = None
POOL_CHUNKS_PER_WORKER = 4
MAX_TASKS_PER_CHILD = 200


J=0.0
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

# Independent per-qubit T1/T2 decoherence rates.
# We use collapse operators sqrt(gamma_1) * sigma_- and
# sqrt(gamma_phi/2) * Z for each qubit.  The factor 1/2 is required
# because D[sqrt(kappa) Z] makes the off-diagonal coherence decay at rate 2*kappa.
gamma_amp = 0.0 if np.isinf(T1) else 1.0/T1
gamma_phi = 1.0/T2 - 0.5*gamma_amp
if gamma_phi < -1e-15:
    raise ValueError('Invalid T1/T2 values: pure-dephasing rate 1/T2 - 1/(2*T1) is negative.')
gamma_phi = max(0.0, gamma_phi)

rt_gamma_amp = np.sqrt(gamma_amp)
rt_gamma_phase = np.sqrt(gamma_phi/2.0)

#J=2*np.pi*40*10**6      #2*np.pi*2*10**3        2*np.pi*40*10**6
J=J/np.max(np.abs(w))
omaga0=0#*np.e/2
#omaga=(-2*n+(N-1))*J-omaga0
omaga=-2*J-omaga0#(-2*n+np.sum(w))*J-omaga0
# Coupling strength and external magnetic field   #(0000 0001 0010 0011 0100 0101 0110 0111 1000 1001 1010 1011 1100 1101 1110 1111)

#T1=30*10**(-6)   #1  30*10**(-6)
#T2=30*10**(-6)      # 50  30*10**(-6)
#rt_gamma_amp=(1/T1)**0.5                  #  (2*np.pi*100*10**3)**0.5
#rt_gamma_phase=(abs(0.5*(1/T2-1/(2*T1))))**0.5                    #(1/T2)**0.5  #(2*np.pi*500*10**3)**0.5


# Construct the Pauli matrices


def Build_H_J(c1, c2, target):
    H_c1_target=qt.qeye(2)
    H_c2_target=qt.qeye(2)

    if c1==0:
        H_c1_target=qt.sigmaz()
    elif target==0:
        H_c1_target=qt.sigmaz()
    if c2==0:
        H_c2_target=qt.sigmaz()
    elif target==0:
        H_c2_target=qt.sigmaz()

    for i in range(1, N):
        if c1==i or target==i:
            H_c1_target=qt.tensor(qt.sigmaz(), H_c1_target)
        else:
            H_c1_target=qt.tensor(qt.qeye(2), H_c1_target)
        if c2==i or target==i:
            H_c2_target=qt.tensor(qt.sigmaz(), H_c2_target)
        else:
            H_c2_target=qt.tensor(qt.qeye(2), H_c2_target)
    return H_c1_target+H_c2_target

def Build_H_x(target):
    H_x=qt.qeye(2)
    if target==0:
        H_x=qt.sigmax()
    for i in range(1, N):
        if target==i:
            H_x=qt.tensor(qt.sigmax(), H_x)
        else:
            H_x=qt.tensor(qt.qeye(2), H_x)
    return H_x

def Build_H_y(target):
    H_y=qt.qeye(2)
    if target==0:
        H_y=qt.sigmay()
    for i in range(1, N):
        if target==i:
            H_y=qt.tensor(qt.sigmay(), H_y)
        else:
            H_y=qt.tensor(qt.qeye(2), H_y)
    return H_y



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
    
  


def osc_cos(t, args):
    return np.cos(omaga*t)
def osc_sin(t, args):
    return np.sin(omaga*t)

def osc_cos_pi(t, args):
    return np.cos(omaga*t+np.pi)
def osc_sin_pi(t, args):
    return np.sin(omaga*t+np.pi)


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


# Independent per-qubit collapse operators.
# Do NOT sum these operators before passing them to qt.liouvillian; summing them
# would implement collective damping/dephasing instead of independent channels.
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

Ubases=np.empty((2**(2*N_qi)), dtype=object)
for i in range(2**N_qi): #|0><i|
    for j in range(2**N_qi): #prefactor
        if N_qi==3:
            U=basis_state(N_qi, 0)*basis_state(N_qi, i).dag()*prefactor_basis_3[j, 0]
        elif N_qi==4:
            U=basis_state(N_qi, 0)*basis_state(N_qi, i).dag()*prefactor_basis_4[j, 0]
        elif N_qi==5:
            U=basis_state(N_qi, 0)*basis_state(N_qi, i).dag()*prefactor_basis_5[j, 0]

        for k in range(1, 2**N_qi): # make each element
            if N_qi==3:
                U=U+prefactor_basis_3[j][k]*basis_state(N_qi, k)*basis_state(N_qi, (k+i)%(2**N_qi)).dag()
            elif N_qi==4:
                U=U+prefactor_basis_4[j][k]*basis_state(N_qi, k)*basis_state(N_qi, (k+i)%(2**N_qi)).dag()   
            elif N_qi==5:
                U=U+prefactor_basis_5[j][k]*basis_state(N_qi, k)*basis_state(N_qi, (k+i)%(2**N_qi)).dag()      
        Ubases[2**N_qi*i+j]=U#/((U.dag()*U).tr())**0.5

#######
#Toffoli
Toffoli=0
for i in range(2**(N_qi-1)):
    total_w=0
    temp_i=i
    for j in range(N_qi-1):
        if temp_i%2!=0:
            total_w=total_w+w[j]
        else:
            total_w=total_w-w[j]
        temp_i=temp_i//2
    if total_w==n:# 
        print(i)
        Toffoli=Toffoli+qt.tensor(basis_state(N_qi-1, i)*basis_state(N_qi-1, i).dag(), -1j*qt.sigmax())
    else:
        #print(i)
        #print(basis_state(N-1, i))
        #print(basis_state(N-1, i)*basis_state(N-1, i).dag())
        Toffoli=Toffoli+qt.tensor(basis_state(N_qi-1, i)*basis_state(N_qi-1, i).dag(), qt.qeye(2))



def X_gate(target):
    X=qt.qeye(2)
    if target==0:
        X=qt.sigmax()
    for i in range(1, N):
        if target==i:
            X=qt.tensor(qt.sigmax(), X)
        else:
            X=qt.tensor(qt.qeye(2), X)
    return X

def Toffoli_S(c1, c2, target, pi, Omaga, times):
    S=0
    if pi==0:
        S=qt.QobjEvo([-omaga0/2*H_E0+J/2*Build_H_J(c1, c2, target), [Omaga*Build_H_x(target), osc_cos], [Omaga*Build_H_y(target), osc_sin]], tlist=times)
    else:
        S=qt.QobjEvo([-omaga0/2*H_E0+J/2*Build_H_J(c1, c2, target), [Omaga*Build_H_x(target), osc_cos_pi], [Omaga*Build_H_y(target), osc_sin_pi]], tlist=times)
    if deco=="on":
        S=qt.liouvillian(S, c_ops)
    return S

def Two_Toffoli_S(c1_1, c1_2, target_1, c2_1, c2_2, target_2, pi, Omaga, times):
    S=0
    if pi==0:
        S=qt.QobjEvo([-omaga0/2*H_E0+J/2*Build_H_J(c1_1, c1_2, target_1)+J/2*Build_H_J(c2_1, c2_2, target_2), [Omaga*Build_H_x(target_1), osc_cos], [Omaga*Build_H_y(target_1), osc_sin], [Omaga*Build_H_x(target_2), osc_cos], [Omaga*Build_H_y(target_2), osc_sin]], tlist=times)
    else:
        S=qt.QobjEvo([-omaga0/2*H_E0+J/2*Build_H_J(c1_1, c1_2, target_1)+J/2*Build_H_J(c2_1, c2_2, target_2), [Omaga*Build_H_x(target_1), osc_cos_pi], [Omaga*Build_H_y(target_1), osc_sin_pi], [Omaga*Build_H_x(target_2), osc_cos_pi], [Omaga*Build_H_y(target_2), osc_sin_pi]], tlist=times)
    if deco=="on":
        S=qt.liouvillian(S, c_ops)
    return S
    

ancilla=qt.tensor(qt.basis(2, 0)*qt.basis(2, 0).dag(), qt.basis(2, 0)*qt.basis(2, 0).dag())
Ubases_ancilla=np.empty((2**(2*N_qi)), dtype=object)
for i in range(2**(2*N_qi)): 
    Ubases_ancilla[i]=qt.tensor(ancilla, Ubases[i])

Toffoli_ancilla=qt.tensor(qt.tensor(qt.qeye(2), qt.qeye(2)),  Toffoli)

Ubases_ref_dag=np.empty((2**(2*N_qi)), dtype=object)
for i in range(2**(2*N_qi)): 
    Ubases_ref_dag[i]=Toffoli*Ubases[i].dag()*Toffoli.dag()

X_Gates=np.empty(N, dtype=object)
for i in range(N):
    X_Gates[i]=X_gate(i)

X_Gates_dag=np.empty(N, dtype=object)
for i in range(N):
    X_Gates_dag[i]=X_Gates[i].dag()

X_Pos=np.array([[3, 4], [2, 4], [2, 3], [1, 3], [1, 2], [1, 4]])
Toffoli_Gates=np.empty(7, dtype=object)
Toffoli_pi_Gates=np.empty(6, dtype=object)

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


# Globals used by worker processes. With the fork start method, workers inherit
# these objects from the parent process when the Pool is created.
_gTimes = None
_gOptions = None


def _apply_circuit_to_basis_index(j):
    """Run the circuit on one operator-basis element and return one fidelity term."""
    state = Ubases_ancilla[j]
    for k in range(6):
        # Apply ideal X gates to convert open controls to closed controls.
        state = (
            X_Gates[X_Pos[k][0]] * X_Gates[X_Pos[k][1]]
            * state
            * X_Gates_dag[X_Pos[k][1]] * X_Gates_dag[X_Pos[k][0]]
        )

        # Compute the two ancilla predicates, flip the target, and uncompute.
        result = qt.mesolve(Toffoli_Gates[k], state, _gTimes, options=_gOptions)
        result = qt.mesolve(Toffoli_Gates[6], result.final_state, _gTimes, options=_gOptions)
        result = qt.mesolve(Toffoli_pi_Gates[k], result.final_state, _gTimes, options=_gOptions)

        # Undo the ideal X gates.
        state = (
            X_Gates[X_Pos[k][0]] * X_Gates[X_Pos[k][1]]
            * result.final_state
            * X_Gates_dag[X_Pos[k][1]] * X_Gates_dag[X_Pos[k][0]]
        )

    reduced_state = state.ptrace([2, 3, 4, 5, 6])
    return (Ubases_ref_dag[j] * reduced_state).tr()


def _serial_fidelity_sum(n_terms, times, options):
    global _gTimes, _gOptions
    _gTimes = times
    _gOptions = options
    total = 0
    for j in range(n_terms):
        total += _apply_circuit_to_basis_index(j)
    return total


def parallel_fidelity_sum(n_terms, times, options, n_workers=None):
    """Parallelize the average-fidelity summation over operator-basis indices."""
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = int(n_workers)

    if n_workers <= 1 or n_terms <= 1:
        return _serial_fidelity_sum(n_terms, times, options)

    if "fork" not in mp.get_all_start_methods():
        print("Warning: multiprocessing start method 'fork' is unavailable; using serial loop.")
        return _serial_fidelity_sum(n_terms, times, options)

    n_workers = min(n_workers, n_terms)

    global _gTimes, _gOptions
    _gTimes = times
    _gOptions = options

    chunksize = max(1, n_terms // (n_workers * POOL_CHUNKS_PER_WORKER))
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=n_workers, maxtasksperchild=MAX_TASKS_PER_CHILD) as pool:
        return sum(pool.imap_unordered(_apply_circuit_to_basis_index, range(n_terms), chunksize=chunksize))


# The fastest drive frequency is |omaga| for this circuit.  max_step is a numerical
# solver safeguard; it does not change the physical circuit.
solver_max_step = None
if USE_MAX_STEP and abs(omaga) > 0:
    solver_max_step = (2*np.pi/abs(omaga)) / POINTS_PER_FAST_OSCILLATION

options = make_solver_options(solver_max_step)

with open("./fidelity_cir_20.txt", 'w') as file:
    for i in range(len(dis_n)):
        Omaga=J/(dis_n[i]) #8 is general
        T=np.pi/(2*Omaga)
        times=np.linspace(0.0, T, N_OUTPUT_POINTS)
        Toffoli_Gates[0]=Two_Toffoli_S(1, 2, 5, 3, 4, 6, 0, Omaga, times)
        Toffoli_Gates[1]=Two_Toffoli_S(1, 3, 5, 2, 4, 6, 0, Omaga, times)
        Toffoli_Gates[2]=Two_Toffoli_S(1, 4, 5, 2, 3, 6, 0, Omaga, times)
        Toffoli_Gates[3]=Two_Toffoli_S(2, 4, 5, 1, 3, 6, 0, Omaga, times)
        Toffoli_Gates[4]=Two_Toffoli_S(3, 4, 5, 1, 2, 6, 0, Omaga, times)
        Toffoli_Gates[5]=Two_Toffoli_S(2, 3, 5, 1, 4, 6, 0, Omaga, times)

        Toffoli_Gates[6]=Toffoli_S(5, 6, 0, 0, Omaga, times)

        Toffoli_pi_Gates[0]=Two_Toffoli_S(1, 2, 5, 3, 4, 6, 1, Omaga, times)
        Toffoli_pi_Gates[1]=Two_Toffoli_S(1, 3, 5, 2, 4, 6, 1, Omaga, times)
        Toffoli_pi_Gates[2]=Two_Toffoli_S(1, 4, 5, 2, 3, 6, 1, Omaga, times)
        Toffoli_pi_Gates[3]=Two_Toffoli_S(2, 4, 5, 1, 3, 6, 1, Omaga, times)
        Toffoli_pi_Gates[4]=Two_Toffoli_S(3, 4, 5, 1, 2, 6, 1, Omaga, times)
        Toffoli_pi_Gates[5]=Two_Toffoli_S(2, 3, 5, 1, 4, 6, 1, Omaga, times)
        print("start")
        n_terms = 2**(2*N_qi)
        Fidelity = parallel_fidelity_sum(n_terms, times, options, N_WORKERS)
        print(str(J/Omaga)+" "+str((Fidelity+2**(2*N_qi))/(2**(2*N_qi)*(2**N_qi+1))))
        print(str(J/Omaga)+" "+str((Fidelity+2**(2*N_qi))/(2**(2*N_qi)*(2**N_qi+1))), file=file)
        file.flush()
