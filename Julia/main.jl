using Pkg

# Install required packages if not already installed
# Uncomment these lines if you need to install packages
# Pkg.add("ProgressMeter")
# Pkg.add("LinearAlgebra")
# Pkg.add("Random")

# Import the necessary modules
include("JuliaLibTruncated.jl")
using Main.JuliaLibTruncated
using LinearAlgebra

# Define parameters for the simulation
num_particles = 10
Gamma_0 = 2π * 6.065e6  # Decay rate in Hz
t_max_factor = 1
num_steps = 1000
time_factor = 10  # Save every 10 steps
omega_z = 0.0
Omega_Rabi = 0.0
num_simulations = 100  # Reduced for faster execution

# Define positions of particles (example: particles in a line with spacing 0.1)
positions = [0.1 * i * [1.0, 0.0, 0.0] for i in 1:num_particles]

# Define dipole moment (example: unit vector in z-direction)
dipole_moment = [0.0, 0.0, 1.0]

# Speed of light in vacuum (m/s)
c = 299792458.0

# Transition frequency (example: optical transition)
omega = 2π * 5.0e14  # Hz

# Run the simulation
println("Starting spin dynamics simulation...")
magnetization_list, Gammas, J_matrices = compute_spin_dynamics_TWA(
    num_particles=num_particles,
    Gamma_0=Gamma_0,
    t_max_factor=t_max_factor,
    num_steps=num_steps,
    time_factor=time_factor,
    omega_z=omega_z,
    Omega_Rabi=Omega_Rabi,
    num_simulations=num_simulations,
    dipole_moment=dipole_moment,
    positions=positions
)

println("Simulation completed!")
println("Number of configurations: ", length(magnetization_list))
println("Shape of first magnetization array: ", size(magnetization_list[1]))
