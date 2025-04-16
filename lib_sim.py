import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import random
from tqdm import tqdm  # For progress bars
import scipy.constants as cst 


# Libary that compute the evolution of a dissipative system using the DTWA approach 
# as described in arXiv:2503.17443

# Function Hierarchy and Workflow in the DTWA Simulation

# The simulation workflow follows this general pattern:
# 1. compute_spin_dynamics() - Main entry point that sets up and runs simulations
#    ├── compute_green_tensor() - Calculates the Green tensor for dipole interactions
#    ├── compute_gamma_matrix() - Calculates the decay rate matrix
#    ├── compute_coupling_matrix() - Calculates the coupling matrix (J)
#    └── For each of the num_simulations:
#        ├── generate_noises() - Creates noise terms for the stochastic simulation
#        └── SpinDerivative() - Calculates derivatives for the spin equations of motion
#            ├── d_dt_sx() - Calculates derivative for x component of spin
#            ├── d_dt_sy() - Calculates derivative for y component of spin
#            └── d_dt_sz() - Calculates derivative for z component of spin


def d_dt_sx(i, omega_z, J, Gamma, xi_x, s_x, s_y, s_z):
    """
    Compute the time derivative of s_x for all spins.
    
    Parameters:
    - i: index of the spin (not used in vectorized version)
    - omega_z: frequency parameter
    - J: coupling matrix
    - Gamma: damping matrix
    - xi_x: noise term array for x component for all spins
    - s_x, s_y, s_z: spin component arrays
    
    Returns:
    - Array of time derivatives for s_x for all spins
    """
    # d/dt s_x[i] = -2*omega_z*s_y[i] + s_z[i]*sum_j J[i,j]*s_y[j] - (1/2)*s_z[i]*sum_j Gamma[i,j]*s_y[j] + xi_x[i]*s_z[i]
    
    # Initialize the derivatives array
    n = len(s_x)
    derivatives = np.zeros(n, dtype=complex)
    
    # First term: -2*omega_z*s_y
    derivatives = -2 * omega_z * s_y
    
    # Second term: s_z[i]*sum_j J[i,j]*s_y[j] for each i
    # Compute J*s_y for all i (matrix-vector product)
    j_sum = J @ s_y
    j_sum = np.real(j_sum)  # Take the norm
    
    # Third term: (1/2)*s_z[i]*sum_j Gamma[i,j]*s_x[j] for each i
    # Compute Gamma*s_x for all i (matrix-vector product)
    gamma_sum = Gamma @ s_x
    gamma_sum = np.real(gamma_sum)  # Take the norm
    
    # Add the second and third terms
    derivatives += s_z * j_sum
    derivatives += 0.5 * s_z * gamma_sum
    
    # Add the noise term
    derivatives += xi_x * s_z
    
    return derivatives

def d_dt_sy(i, omega_z, Omega, J, Gamma, xi_y, s_x, s_y, s_z):
    """
    Compute the time derivative of s_y for all spins.
    
    Parameters:
    - i: index of the spin (not used in vectorized version)
    - omega_z: frequency parameter
    - Omega: frequency parameter
    - J: coupling matrix
    - Gamma: damping matrix
    - xi_y: noise term array for y component for all spins
    - s_x, s_y, s_z: spin component arrays
    
    Returns:
    - Array of time derivatives for s_y for all spins
    """
    # d/dt s_y[i] = +2*omega_z*s_x[i] - 2*Omega*s_z[i] - s_z[i]*sum_j (1/2)*J[i,j]*s_x[j] + (1/2)*s_z[i]*sum_j Gamma[i,j]*s_x[j] + xi_y[i]*s_z[i]
    
    # Initialize the derivatives array
    n = len(s_x)
    derivatives = np.zeros(n, dtype=complex)
    
    # First term: +2*omega_z*s_x
    derivatives = 2 * omega_z * s_x
    
    # Second term: -2*Omega*s_z
    derivatives -= 2 * Omega * s_z
    
    # Third term: -s_z[i]*sum_j J[i,j]*s_x[j] for each i
    # Compute J*s_x for all i (matrix-vector product)
    j_sum = J @ s_x
    j_sum = np.real(j_sum)  # Take the norm
    
    # Fourth term: (1/2)*s_z[i]*sum_j Gamma[i,j]*s_y[j] for each i
    # Compute Gamma*s_y for all i (matrix-vector product)
    gamma_sum = Gamma @ s_y
    gamma_sum = np.real(gamma_sum)  # Take the norm
    
    # Add the third and fourth terms
    derivatives -= s_z * j_sum
    derivatives += 0.5 * s_z * gamma_sum
    
    # Add the noise term
    derivatives += xi_y * s_z
    
    return derivatives

def d_dt_sz(i, Omega, J, Gamma, xi_x, xi_y, s_x, s_y, s_z):
    """
    Compute the time derivative of s_z for all spins.
    
    Parameters:
    - i: index of the spin (not used in vectorized version)
    - Omega: frequency parameter
    - J: coupling matrix
    - Gamma: damping matrix
    - xi_x, xi_y: noise term arrays for x and y components for all spins
    - s_x, s_y, s_z: spin component arrays
    
    Returns:
    - Array of time derivatives for s_z for all spins
    """
    # d/dt s_z[i] = +2*Omega*s_y[i] + sum_j J[i,j]*(s_y[i]*s_x[j] - s_x[i]*s_y[j]) - sum_j Gamma[i,j]*(s_x[i]*s_x[j] + s_y[i]*s_y[j]) - xi_x[i]*s_x[i] - xi_y[i]*s_y[i]
    
    # Initialize the derivatives array
    n = len(s_x)

    derivatives = np.zeros(n, dtype=complex)
    
    # First term: +2*Omega*s_y
    derivatives = 2 * Omega * s_y
    
    # Second term: J term - vectorized computation
    # Create outer products for the calculations
    # s_y_outer is a column vector of shape (n, 1) containing all s_y values
    s_y_outer = s_y[:, np.newaxis]  # Shape (n, 1)
    # s_x_outer is a row vector of shape (1, n) containing all s_x values
    s_x_outer = s_x[np.newaxis, :]  # Shape (1, n)
    
    # Calculate s_y[i] * s_x[j] for all i,j pairs
    # This creates an n×n matrix where each element (i,j) contains the product s_y[i] * s_x[j]
    # Used for computing the coupling term in the spin equations of motion
    sy_sx_product = s_y_outer * s_x_outer  # Shape (n, n)
    
    # Calculate s_x[i] * s_y[j] for all i,j pairs
    # This creates an n×n matrix where each element (i,j) contains the product s_x[i] * s_y[j]
    # The s_x[:, np.newaxis] creates a column vector (n,1) of all s_x values
    # The s_y[np.newaxis, :] creates a row vector (1,n) of all s_y values
    # Their product is an outer product that gives all possible combinations of s_x[i] * s_y[j]
    # Used in the coupling term calculation with opposite sign from sy_sx_product
    sx_sy_product = s_x[:, np.newaxis] * s_y[np.newaxis, :]  # Shape (n, n)


    # Calculate the J term for all spins at once
    # This computes the coupling term in the equation of motion for s_z
    # For each spin i, it calculates the sum over all spins j of:
    # J[i,j] * (s_y[i]*s_x[j] - s_x[i]*s_y[j])
    # The matrix multiplication with J and element-wise difference of the outer products
    # is summed along axis=1 to get the total coupling effect on each spin
    j_term = np.sum(J * (sy_sx_product - sx_sy_product), axis=1)

    j_term = np.real(j_term)  # Take the norm
    
    # Third term: Gamma term - vectorized computation
    # Calculate s_x[i] * s_x[j] + s_y[i] * s_y[j] for all i,j pairs
    sx_sx_product = s_x[:, np.newaxis] * s_x[np.newaxis, :]  # Shape (n, n)
    sy_sy_product = s_y[:, np.newaxis] * s_y[np.newaxis, :]  # Shape (n, n)
    
    intermed = sx_sx_product+sy_sy_product

    # Calculate the Gamma term for all spins at once
    gamma_term = 0.5 * np.sum(Gamma * (intermed), axis=1)
    
    gamma_term = np.real(gamma_term)  # Take the norm
    
    
    # Add the J term and subtract the Gamma term
    derivatives += j_term
    derivatives -= gamma_term
    # derivatives -= gamma_diag_term
    
    # Add the noise terms
    derivatives -= xi_x * s_x
    derivatives -= xi_y * s_y
    
    return derivatives

def generate_etai_noise(gamma, dt, seed=None):
    """
    Generate the collective noise variables that satisfy the correlation:
    η^α_i(t) η^β_j(t') = γ_i δ_ij δ_αβ δ(t-t')
    
    Parameters:
    - gamma: array of noise strengths for each spin (length N)
    - dt: time step
    - seed: random seed for reproducibility
    
    Returns:
    - noise_x, noise_y, noise_z: arrays of shape (N,) containing Gaussian noise variables
      for the x, y, and z components for all particles
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_particles = len(gamma)
    
    # Generate standard normal random variables for all three components for all particles
    noise_x = np.random.normal(0, 1, size=num_particles)
    noise_y = np.random.normal(0, 1, size=num_particles)    
    # print('gamma',gamma)
    # print('dt',dt)
    # Scale by √(γ_i/dt) to satisfy the correlation function
    scaling_factors = np.sqrt(np.abs(gamma) / dt)
    
    noise_x *= scaling_factors
    noise_y *= scaling_factors
    
    return noise_x, noise_y

def generate_noises(i,Num_particles, gamma, nu, dt, seed=None):
    """
    Generate the noise tensor in the site basis  by combining the Gaussian white noise variables
    with the coupling matrix eigenvectors nu.
    
    Parameters:
    - N: number of spins
    - gamma: array of noise strengths for each spin (length N)
    - nu: coupling matrix of shape (N, N)
    - dt: time step
    - seed: random seed for reproducibility
    
    Returns:
    - xi_x, xi_y, xi_z: arrays of shape (N,) containing the combined noise variables
      for the x, y, and z components
    """
    # Generate the basic Gaussian white noise for all particles at once
    noise_x, noise_y = generate_etai_noise(gamma, dt, seed)
    
    # Initialize the xi arrays for all particles
    xi_x = np.zeros(Num_particles)
    xi_y = np.zeros(Num_particles)
    
    # Vectorized computation of the noise terms using matrix multiplication
    # For each particle i, compute the sum over all j: nu[i,j] * noise_j
    for i in range(Num_particles):
        xi_x[i] = np.sum(nu[i, :] * noise_x)
        xi_y[i] = np.sum(nu[i, :] * noise_y)
    

    return xi_x, xi_y

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


def compute_gamma_matrix(positions, omega,Gamma_0, dipole, c=3e8):
    """
    Compute the damping matrix Γ for a system of dipoles.
    
    Parameters:
    - positions: List of position vectors for each dipole
    - omega: Frequency
    - Gamma_0: Damping rate
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


def compute_J_matrix(positions, omega,Gamma_0, dipole, c=3e8):
    """
    Compute the coupling matrix J for a system of dipoles.
    
    Parameters:
    - positions: List of position vectors for each dipole
    - omega: Frequency of the transition
    - Gamma_0: Damping rate
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




# Define the derivative function for the spin dynamics
def SpinDerivative(spins_current, positions, omega, Gamma_0,omega_z,Omega_Rabi,Gamma,J_matrix, dipole, c,xi_y, xi_x):
    """
    Calculate the derivatives of the spins according to the TWA model.
    
    Parameters:
    - spins_current: Current spin vectors for all particles
    - positions: Positions of all particles
    - omega: Frequency
    - dipole: Dipole moment vector
    - c: Speed of light
    
    Returns:
    - derivatives: Time derivatives of all spin vectors
    """
    # Compute the Gamma and J matrices
    # Initialize the derivatives array

    derivatives = np.zeros_like(spins_current)
    
    # Generate noise arrays for all particles at once    
    # Calculate derivatives for all spins at once

    derivatives[:, 0] = d_dt_sx(None, omega_z, J_matrix, Gamma, xi_x, spins_current[:, 0], spins_current[:, 1], spins_current[:, 2])
    derivatives[:, 1] = d_dt_sy(None, omega_z, Omega_Rabi, J_matrix, Gamma, xi_y, spins_current[:, 0], spins_current[:, 1], spins_current[:, 2])
    derivatives[:, 2] = d_dt_sz(None, Omega_Rabi, J_matrix, Gamma, xi_x, xi_y, spins_current[:, 0], spins_current[:, 1], spins_current[:, 2])


    return derivatives

def chain_positions(num_particles, spacing_factor,lambda_0):
    positions = []
    spacing = spacing_factor*lambda_0
    for i in range(num_particles):
        positions.append([i*spacing, 0, 0])
    return np.array(positions)

def generate_thermal_distribution_in_cylinder(num_particles,spacing_factor,lambda_0,E_0):

    positions = np.zeros((num_particles, 3))
    
    # For a dipole trap, the potential is typically:
    # U(r,z) = U_0 * (1 - exp(-2r²/w_r²) * exp(-2z²/w_z²))
    # where w_r is the radial waist and w_z is the axial waist
    radius = 0.5 * lambda_0 *spacing_factor # cylinder radius
    length = 20 * lambda_0 *spacing_factor # cylinder length
    # Using the waists as the radius and length/2
    w_r = radius

    w_z = length/2
    
    # Temperature parameter (arbitrary units, can be adjusted)
    kB_T = E_0  # in units of U_0
    
    for i in range(num_particles):
        # Implement rejection sampling for thermal distribution
        while True:
            # Generate uniform random positions within cylinder bounds
            r_candidate = radius * np.random.random()
            z_candidate = length * (np.random.random() - 0.5)
            
            # Calculate potential energy at this position (normalized)
            U = (1 - np.exp(-2 * (r_candidate/w_r)**2) * np.exp(-2 * (z_candidate/w_z)**2))
            
            # Calculate Boltzmann factor: exp(-U/kB_T)
            boltzmann_factor = np.exp(-U/kB_T)
            
            # Accept with probability proportional to Boltzmann factor
            if np.random.random() < boltzmann_factor:
                # Random angle for cylindrical symmetry
                theta = 2 * np.pi * np.random.random()
                
                # Convert to Cartesian coordinates
                x = r_candidate * np.cos(theta)
                y = r_candidate * np.sin(theta)
                z = z_candidate
                
                positions[i] = [x, y, z]
                break
    return positions


def compute_spin_dynamics_TWA(
    num_particles=10,
    Gamma_0=2*np.pi *6.065 *1e6,
    t_max_factor=1,
    num_steps=1000,
    time_factor=1,
    omega_z=0,
    Omega_Rabi=0,
    num_simulations=10000,
    dipole_moment = None ,
    listNumparticles=None,
    positions = None    
):
    """
    Compute the dynamics of a linear chain of spins using the TWA approach
    
    Parameters:
    - num_particles: Number of particles in the chain
    - spacing_factor: Spacing between particles as a fraction of wavelength
    - t_max_factor: Maximum simulation time as a factor of Gamma_0
    - num_steps: Number of time steps
    - omega_z: Detuning (in rad/s)
    - Omega_Rabi: Rabi frequency as a factor of Gamma_0
    - num_simulations: Number of simulations to average
    - dipole_direction: Direction of the dipole moment (normalized)
    - listNumparticles: List of particle numbers to simulate (overrides num_particles if provided)
    
    Returns:
    - magnetization_list: List of magnetization arrays
    """
    import scipy.constants as cst 
    from scipy.integrate import solve_ivp
    
    hbar = cst.hbar

    # Rabi frequency in rad/s (50 MHz)

    Gamma_0 = 2*np.pi *6.065 *1e6

    # The relation between Rabi frequency and intensity is:
    # Ω = γ * sqrt(I / (2 * Isat))
    # We're directly setting Rabi frequency rather than calculating from intensity
    c = 2.99792458e8     
                # Speed of light in m/s (in units where ħ = 1)

    lambda_0 = 780e-9          # laser wavelength in m
                # speed of light m/s (more precise value)
    omega = 2*np.pi*c/lambda_0 # laser frequency in rad/s 

    # Define simulation parameters
    t_max =  t_max_factor/Gamma_0 # Maximum simulation time (in appropriate units)

    dt = t_max/num_steps # Time step

    num_steps = int(t_max / dt)


    # Initialize arrays to store the time evolution
    times = np.linspace(0, t_max, num_steps)

    # compute the gamma matrix and the J matrix

    # Perform time evolution using simple Euler method
    print("Simulating spin dynamics...")

    # Initialize array to store the average of all trajectories
    tot_avg_magnetization = np.zeros((num_steps, 3))

    # Run a lot of simulations and average the results
    magnetization_list = []

    Gammas = []

    J_matrices = []


    Omega_Rabi = 0*Gamma_0


    for idx in range(len(listNumparticles)):

        num_particles = listNumparticles[idx]
        # test with no interaction (expect OBEs)

        Gamma = compute_gamma_matrix(positions, omega,Gamma_0, dipole_moment, c)

        J_matrix = compute_J_matrix(positions, omega,Gamma_0, dipole_moment, c)

        Gammas.append(Gamma)

        J_matrices.append(J_matrix) 
        # Compute eigenvalues and eigenvectors of the Gamma matrix

        gamma, nu = np.linalg.eigh(Gamma) # get the eigenvalues and eigenvectors of the coupling matrix for the current configuration

        print('J matrix and Gamma matrix are computed')

        print('Gamma matrix',Gamma)
        print('J matrix',J_matrix)

        magnetizations = np.zeros((num_simulations,num_steps//time_factor, num_particles, 3))

        for sim in tqdm(range(num_simulations), desc="Running simulations"):
            # Initialize spins pointing up (z-direction) with small uncertainty
            spins = np.zeros((num_particles, 3))

            # # Initialize spins pointing up (z-direction) with small uncertainty
            spins[:, 2] = 1.0  # All spins initially pointing up (z-direction)

            # # Add small random noise to the x and y components

            # # Add real-valued noise to x and y components
            # spins[:, 0] = noise_amplitude * np.random.normal(0, 1, num_particles)  # Random real values
            # spins[:, 1] = noise_amplitude * np.random.normal(0, 1, num_particles)  # Random real values
            # Set the x and y components to be randomly +1 or -1
            spins[:, 0] = np.random.choice([-1, 1], size=num_particles)

            spins[:, 1] = np.random.choice([-1, 1], size=num_particles)
            # # Normalize each spin vector to unit length - vectorized version

            # norms = np.linalg.norm(spins, axis=1, keepdims=True)

            # spins = spins / norms
            # print(np.sum(spins,axis=0))
            
            # Initialize the spin evolution array for this simulation
            spin_evolution = np.zeros((num_steps//time_factor, num_particles, 3))

            spin_evolution[0] = spins
            current_state = spins

            for step in range(1, num_steps):
                
                # Implicit midpoint method:
                xi_y, xi_x= generate_noises(None, num_particles, gamma, nu, dt, seed=None)  # generate noise arrays for all particles
                        
                # Iterate until the norm stabilizes within 0.1% of the initial norm
                midpoint_state = current_state

                midpoint_state = current_state + dt * SpinDerivative(midpoint_state, positions, omega,Gamma_0,omega_z,Omega_Rabi,Gamma,J_matrix, dipole_moment, c,xi_y, xi_x)

                midpoint_derivative = SpinDerivative((midpoint_state+current_state)/2, positions, omega,Gamma_0,omega_z,Omega_Rabi,Gamma,J_matrix, dipole_moment, c,xi_y, xi_x)

                current_state = current_state + dt * midpoint_derivative
                # if step%time_factor==0:
                if step%time_factor==0:
                    spin_evolution[step//time_factor] = current_state

                # Test : Normalize spins to maintain unit vectors - vectorized version

                # norms = np.linalg.norm(spin_evolution[step], axis=1, keepdims=True)

                # spin_evolution[step] = spin_evolution[step] /norms
            # Add to the total average
            magnetizations [sim] = spin_evolution

        magnetization_list+=[magnetizations]


    print("All simulations complete!")

    return magnetization_list, Gammas, J_matrices

