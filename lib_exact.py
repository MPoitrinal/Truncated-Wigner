import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import scipy.constants as cst
from functools import lru_cache
import time


# Create basic operators
si = qt.qeye(2)
sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()
sp = qt.sigmap()
sm = qt.sigmam()



def compute_green_tensor(r, omega, mu0=4*np.pi*1e-7, c=299792458):
    """
    Compute the Green tensor G0(r,ω) for electromagnetic interactions.
    
    Parameters:
    - r: position vector (numpy array of shape (3,) or distance scalar)
    - omega: angular frequency
    - mu0: magnetic permeability of vacuum (default: 4π×10^-7 H/m)
    - c: speed of light in vacuum (default: 299792458 m/s)
    

    Returns:
    - G: Green tensor (numpy array of shape (3,3))
    """
    # Calculate the wave number k = ω/c
    k = omega / c
    
    # Handle both vector and scalar r inputs
    if np.isscalar(r):
        r_mag = r
        # For scalar r, we need a unit vector. Use z-direction by default
        r_vec = np.array([0, 0, r_mag])
    else:
        r_vec = np.array(r)
        r_mag = np.linalg.norm(r_vec)
    
    
    # Calculate the exponential term
    exp_ikr = np.exp(1j * k * r_mag)
    
    # Calculate the prefactor
    prefactor = (mu0 * exp_ikr) / (4 * np.pi * k**2 * r_mag**3)
    
    # Calculate the first term coefficient
    first_term_coef = k**2 * r_mag**2 + 1j * k * r_mag - 1
    
    # Calculate the second term coefficient
    second_term_coef = 3 - 3 * 1j * k * r_mag - k**2 * r_mag**2
    
    # Calculate the identity matrix term
    identity_term = first_term_coef * np.eye(3)
    
    # Calculate the dyadic product term (r⊗r)/r^2
    if np.isscalar(r):
        # For scalar r with default z-direction
        dyadic_term = np.zeros((3, 3))
        dyadic_term[2, 2] = 1  # Only z⊗z component is non-zero
    else:
        # Normalized outer product
        r_normalized = r_vec / r_mag
        dyadic_term = np.outer(r_normalized, r_normalized)
    
    # Combine the terms
    G = prefactor * (identity_term + second_term_coef * dyadic_term)
    
    return G


def compute_gamma_matrix(positions, omega, dipole, Gamma_0):
    """
    Compute the damping matrix Γ for a system of dipoles.
    
    Parameters:
    - positions: List of position vectors for each dipole
    - omega: Frequency
    - dipole: Dipole moment vector (assumed same for all dipoles)
    - c: Speed of light (default: 3e8 m/s)
    
    Returns:
    - Gamma: Damping matrix (numpy array of shape (N, N))
    """
    N = len(positions)
    Gamma = np.zeros((N, N), dtype=complex)
    
    p = dipole
    
    for i in range(N):
        for j in range(N):
            r_ij = np.array(positions[i]) - np.array(positions[j])
            if i == j:
                Gamma[i, j] = Gamma_0
            else :
                G_ij = compute_green_tensor(r_ij, omega)
                Gamma[i, j] = 2 * omega**2 * np.dot(np.conjugate(p), np.dot(np.imag(G_ij), p))/cst.hbar
            
            
            # Calculate Γ_ij = 2ω^2 * p̄ · Im[G(r_i - r_j, ω)] · p
            #print("green tensor times dipole",np.dot(np.imag(G_ij), p))
    
    return Gamma


def compute_J_matrix(positions, omega, dipole, c=3e8):
    """
    Compute the coupling matrix J for a system of dipoles.
    
    Parameters:
    - positions: List of position vectors for each dipole
    - omega: Frequency of the transition
    - dipole: Dipole moment vector (assumed same for all dipoles)
    - c: Speed of light (default: 3e8 m/s)
    
    Returns:
    - J: Coupling matrix (numpy array of shape (N, N))
    """
    N = len(positions)
    J = np.zeros((N, N), dtype=complex)
    
    # Normalize dipole
    p = dipole
    
    for i in range(N):
        for j in range(N):
            # Calculate the Green tensor for the separation vector
            r_ij = np.array(positions[i]) - np.array(positions[j])
            if i == j:
                J[i, j] = 0
            else :
                G_ij = compute_green_tensor(r_ij, omega)
                J[i, j] = -omega**2 * np.dot(np.conjugate(p), np.dot(np.real(G_ij), p))/cst.hbar
                
    return J


# Define basis states for a two-level system
up = qt.basis(2,0)    # Ground state |0⟩
down = qt.basis(2,1)  # Excited state |1⟩

def PowerOut(N,Gamma):
    # Creates collective lowering operator S- = Σ Gamma_ij σ-_i σ+_j
    return np.sum([Gamma[i,j]*sigp(i,N)*sigm(j,N) for i in range(N) for j in range(N)])

# Helper functions to create tensor product operators for N qubits
def sigm(n,N):
    # Creates lowering operator σ- for nth qubit in N-qubit system
    # Returns tensor product: I⊗I⊗...⊗σ-⊗...⊗I
    return qt.tensor([qt.sigmam() if i==n else qt.qeye(2) for i in range(N)])

def sigp(n,N):
    # Creates raising operator σ+ for nth qubit in N-qubit system
    # Returns tensor product: I⊗I⊗...⊗σ+⊗...⊗I
    return qt.tensor([qt.sigmap() if i==n else qt.qeye(2) for i in range(N)])

def sigz(n,N):
    # Creates Pauli-Z operator σz for nth qubit in N-qubit system
    # Returns tensor product: I⊗I⊗...⊗σz⊗...⊗I
    return qt.tensor([qt.sigmaz() if i==n else qt.qeye(2) for i in range(N)])

def sigx(n,N):
    # Creates Pauli-X operator σx for nth qubit in N-qubit system
    # Returns tensor product: I⊗I⊗...⊗σx⊗...⊗I
    return qt.tensor([qt.sigmax() if i==n else qt.qeye(2) for i in range(N)])

def single_ops(N):
    # Returns list of individual lowering operators for each qubit
    return [sigm(n,N) for n in range(N)]

def drive_hamiltonian(N,Ω):
    # Constructs system Hamiltonian for N qubits with Rabi frequency Ω
    # Includes random phases for each qubit's driving term
    H=0
    for n in range(N):
        phase = 0
        # H = Σ (Ω/2 * (e^(iφ)σ- + e^(-iφ)σ+) + δσ+σ-)
        H += Ω/2*(np.exp(1j*phase)*sigm(n,N)+np.exp(-1j*phase)*sigp(n,N))
    return H

def s_minus(N):
    # Creates collective lowering operator S- = Σ σ-_i
    return np.sum([sigm(n,N) for n in range(N)])


def compute_spin_dynamics_exact(N, omega_z, Omega_R, positions, p_vector, omega, tlist,Gamma_0, psi0):
    """
    Simulate a system of N correlated atoms with dipolar interactions and collective decay.
    
    Parameters:
    -----------
    N : int
        Number of atoms
    omega_z : float
        Effective detuning (includes Lamb shift)
    Omega : float
        Rabi frequency
    positions : array-like
        Positions of atoms in 3D space, shape (N, 3)
    p_vector : array-like
        Dipole moment vector, shape (3,)
    omega : float
        Transition frequency
    tlist : array-like
        Time points for simulation
    rho0 : Qobj, optional
        Initial density matrix. If None, all atoms in ground state.
        
    Returns:
    --------
    result : qutip.Result
        Simulation results
    """
    # Convert positions to numpy array
    print('N',N)
    print('positions',positions)
    positions = np.array(positions, dtype=np.float64)
    
    # Normalize dipole vector
    
    # Create basic operators
    si = qt.qeye(2)
    sx = qt.sigmax()
    sy = qt.sigmay()
    sz = qt.sigmaz()
    sp = qt.sigmap()
    sm = qt.sigmam()
    

    # Construct the Hamiltonian efficiently
    # First term: ωz ∑ σz_i
    H_z = sum([omega_z * sigz(i, N) for i in range(N)])
    
    # Second term: Ω ∑ σx_i
    print('Omega_R',Omega_R)
    H_drive = drive_hamiltonian(N,Omega_R)
    
    # Calculate dipolar interaction matrix J_ij
    start_time = time.time()

    J = compute_J_matrix(positions, omega, p_vector)

    print('J',J)
    print(f"J matrix computation time: {time.time() - start_time:.4f} seconds")
    
    # Third term: ∑ J_ij σ+_i σ-_j

    H_dipole_terms = []
    for i in range(N):
        for j in range(N):
            if i != j and abs(J[i, j]) > 1e-10:  # Skip negligible couplings
                H_dipole_terms.append(J[i, j] * sigm(i, N) * sigp(j, N))
    
    H_dipole = sum(H_dipole_terms)
    
    # Total Hamiltonian
    H = H_z + H_drive + H_dipole
    
    # Calculate collective decay matrix Γ_ij
    start_time = time.time()
    
    Gamma = compute_gamma_matrix(positions, omega, p_vector,Gamma_0)
    # debugging


    print('Gamma matrix',Gamma)
    print('J matrix',J)

    print(f"Gamma matrix computation time: {time.time() - start_time:.4f} seconds")
    
    # Construct collapse operators for collective decay
    c_ops = []
    
    # Pre-compute operator lists for tensor products for collective decay operators
    # This creates a 3D array of operator lists where:
    # - sm_sp_list[i][j] contains operators for the term σ-_i σ+_j
    # - For each position k: 
    #   * If k=i, we place σ- (lowering operator)
    #   * If k=j, we place σ+ (raising operator)
    #   * Otherwise, we place identity operator
    # This structure allows efficient construction of collective decay operators
    eigvals, eigvecs = np.linalg.eigh(Gamma)
    

    # Create collapse operators
    c_ops = []

    for k in range(N):
        L_k = sum(eigvecs[j, k] * sigm(j,N) for j in range(N))
        print('L_k',L_k)
        c_ops.append(np.sqrt(np.abs(eigvals[k])) * L_k)
    
    # c_ops.append(np.sqrt(Gamma_0) * lowering_op(N, 0))  # Test

    # Define observables to track
    e_ops = []
    # Pre-compute operator lists for tensor products
    
    # Single-atom observables
    # for i in range(N):
    #     # Population of excited state for each atom
    #     e_ops.append(sigz(i,N))

    e_ops.append(PowerOut(N,Gamma)/(N*Gamma_0))

    # Run the simulation
    options = {"progress_bar": "enhanced", "store_states": True, "method": "adams", "nsteps": 1000}

    # for i, c in enumerate(c_ops):
    #     print(f"Collapse operator {i} norm:", c.norm())


    print(tlist*Gamma_0)

    result = qt.mesolve(H, psi0, tlist, c_ops=c_ops, e_ops=e_ops, options=options)

    print('result',result)

    return result

def plot_results(result, tlist, N, power_t = False):
    """Plot the results of the simulation."""
    plt.figure(figsize=(10, 6))
    
    # Plot population of excited state for each atom
    if not (power_t):
        for i in range(N):
            plt.plot(tlist, result.expect[i], label=f'Atom {i+1}')
    else:
        plt.plot(tlist, result.expect[-1], label=f'Power out')
    
    plt.xlabel('Time')
    plt.xscale('log')
    plt.ylabel('Excited state population')
    plt.title('Dynamics of correlated atoms')
    plt.legend()
    plt.grid(True)
    plt.show()
