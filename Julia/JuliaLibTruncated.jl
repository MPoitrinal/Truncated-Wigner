# function dtwa_step(spins::Matrix{Float64}, Jx::Matrix{Float64}, Jy::Matrix{Float64}, Jz::Matrix{Float64}, 
#                    ωz::Float64=0.0, Ω::Float64=0.0, Γ::Matrix{Float64}=zeros(0,0), 
#                    ξx::Vector{Float64}=zeros(0), ξy::Vector{Float64}=zeros(0))
#     # spins: 3 × N matrix, each column is a spin vector [Sx; Sy; Sz] for one particle
#     N = size(spins, 2)
#     dS = zeros(3, N)
#     # println("dtwa step")
#     # Extract spin components
#     Sx = spins[1, :]
#     Sy = spins[2, :]
#     Sz = spins[3, :]
    
#     # Check if optional parameters were provided
#     has_Γ = !isempty(Γ)
#     has_ξ = length(ξx) == N && length(ξy) == N

#     # Create matrices for vectorized operations
#     # For each spin i, we need sum_{j≠i} J[i,j]*S[j]
#     # This can be done with matrix multiplication minus the diagonal terms
    
#     # For dSx/dt
#     dS[1, :] = -2 .* ωz .* Sy
    
#     # Interaction terms (Jy*Sy and Jz*Sz)
#     JySy = Jy * Sy
#     JzSz = Jz * Sz
#     # # Remove self-interactions
#     # println(length(JySy))
#     # println(length(Sy))
#     # println(length(dS[1, :]))
    
#     dS[1, :] .+= Sz .* JySy - Sy .* JzSz
#     # println("first interaction term computed")
#     # Dissipation terms
#     if has_Γ
#         ΓSx = Γ * Sx
#         dS[1, :] .+= 0.5 .* Sz .* ΓSx
#     end
    
#     # Noise terms
#     if has_ξ
#         dS[1, :] .+= ξx .* Sz
#     end
    
#     # For dSy/dt
#     dS[2, :] = 2 .* ωz .* Sx - 2 .* Ω .* Sz
    
#     # Interaction terms (Jx*Sx and Jz*Sz already computed)
#     JxSx = Jx * Sx
   
#     dS[2, :] .+= Sz .* JxSx - Sx .* JzSz
    
#     # Dissipation terms
#     if has_Γ
#         ΓSy = Γ * Sy
#         dS[2, :] .+= 0.5 .* Sz .* ΓSy
#     end
    
#     # Noise terms
#     if has_ξ
#         dS[2, :] .+= ξy .* Sz
#     end
    
#     # For dSz/dt
#     dS[3, :] = 2 .* Ω .* Sy
    
#     # Interaction terms
#     # For each i: sum_{j≠i} Jx[i,j]*(Sy[i]*Sx[j] - Sx[i]*Sy[j])
#     SyJxSx = Sy .* (Jx * Sx)
#     SxJxSy = Sx .* (Jx * Sy)
  

#     dS[3, :] .+= SyJxSx - SxJxSy
    
#     # Dissipation terms
#     if has_Γ
#         SxΓSx = Sx .* (Γ * Sx)
#         SyΓSy = Sy .* (Γ * Sy)
#         dS[3, :] .-= 0.5 .* (SxΓSx + SyΓSy)
#     end
    
#     # Noise terms
#     if has_ξ
#         dS[3, :] .-= ξx .* Sx + ξy .* Sy
#     end

#     return spins + dS
# end
function dtwa_step(spins::Matrix{Float64}, Jx::Matrix{Float64}, Jy::Matrix{Float64}, Jz::Matrix{Float64}, 
                  ωz::Float64=0.0, Ω::Float64=0.0, Γ::Union{Matrix{Float64}, Nothing}=nothing, 
                  ξx::Union{Vector{Float64}, Nothing}=nothing, ξy::Union{Vector{Float64}, Nothing}=nothing)
    N = size(spins, 2)
    dS = zeros(3, N)
    
    # Extract spin components
    Sx = spins[1, :]
    Sy = spins[2, :]
    Sz = spins[3, :]
    
    # Precompute interaction terms
    JySy = Jy * Sy
    JzSz = Jz * Sz
    JxSx = Jx * Sx
    
    # For dSx/dt
    dS[1, :] .= -2 .* ωz .* Sy
    dS[1, :] .+= Sz .* JySy .- Sy .* JzSz
    
    # Dissipation terms
    if !isnothing(Γ)
        ΓSx = Γ * Sx
        dS[1, :] .+= 0.5 .* Sz .* ΓSx
    end
    
    # Noise terms
    if !isnothing(ξx)
        dS[1, :] .+= ξx .* Sz
    end
    
    # For dSy/dt
    dS[2, :] .= 2 .* ωz .* Sx .- 2 .* Ω .* Sz
    dS[2, :] .+= Sz .* JxSx .- Sx .* JzSz
    
    if !isnothing(Γ)
        ΓSy = Γ * Sy
        dS[2, :] .+= 0.5 .* Sz .* ΓSy
    end
    
    if !isnothing(ξy)
        dS[2, :] .+= ξy .* Sz
    end
    
    # For dSz/dt
    dS[3, :] .= 2 .* Ω .* Sy
    SyJxSx = Sy .* JxSx
    SxJxSy = Sx .* (Jx * Sy)
    dS[3, :] .+= SyJxSx .- SxJxSy
    
    if !isnothing(Γ)
        SxΓSx = Sx .* (Γ * Sx)
        SyΓSy = Sy .* (Γ * Sy)
        dS[3, :] .-= 0.5 .* (SxΓSx .+ SyΓSy)
    end
    
    if !isnothing(ξx)
        dS[3, :] .-= ξx .* Sx
    end
    
    if !isnothing(ξy)
        dS[3, :] .-= ξy .* Sy
    end
    
    return spins .+ dS
end


"""
    generate_etai_noise(gamma, dt, seed=nothing)

Generate the collective noise variables that satisfy the correlation:
η^α_i(t) η^β_j(t') = γ_i δ_ij δ_αβ δ(t-t')

# Parameters
- `gamma`: array of noise strengths for each spin (length N)
- `dt`: time step
- `seed`: random seed for reproducibility

# Returns
- `noise_x`, `noise_y`: arrays of shape (N,) containing Gaussian noise variables
  for the x and y components for all particles
"""
function generate_etai_noise(gamma, dt)

    
    num_particles = length(gamma)
    
    # Generate standard normal random variables for both components for all particles
    noise_x = randn(num_particles)
    noise_y = randn(num_particles)
    
    # Scale by √(γ_i/dt) to satisfy the correlation function
    scaling_factors = sqrt.(abs.(gamma) ./ dt)
    
    noise_x .*= scaling_factors
    noise_y .*= scaling_factors
    
    return noise_x, noise_y
end

"""
    generate_noises(i, Num_particles, gamma, nu, dt, seed=nothing)

Generate the noise tensor in the site basis by combining the Gaussian white noise variables
with the coupling matrix eigenvectors nu.

# Parameters
- `i`: current iteration (unused in this implementation but kept for API compatibility)
- `Num_particles`: number of spins
- `gamma`: array of noise strengths for each spin (length N)
- `nu`: coupling matrix of shape (N, N)
- `dt`: time step
- `seed`: random seed for reproducibility

# Returns
- `xi_x`, `xi_y`: arrays of shape (N,) containing the combined noise variables
  for the x and y components
"""
function generate_noises(i, Num_particles, gamma, nu, dt)
    # Generate the basic Gaussian white noise for all particles at once
    noise_x, noise_y = generate_etai_noise(gamma, dt)
    
    # Initialize the xi arrays for all particles
    xi_x = zeros(Num_particles)
    xi_y = zeros(Num_particles)
    
    # Compute the noise terms using matrix multiplication
    # For each particle i, compute the sum over all j: nu[i,j] * noise_j
    for i in 1:Num_particles
        xi_x[i] = sum(nu[i, :] .* noise_x)
        xi_y[i] = sum(nu[i, :] .* noise_y)
    end
    
    return xi_x, xi_y
end

"""
    compute_green_tensor(r, omega; mu0=4π*1e-7, c=299792458.0)

Compute the Green tensor G0(r,ω) for electromagnetic interactions.

# Parameters
- `r`: position vector (Vector of length 3) or distance scalar
- `omega`: angular frequency
- `mu0`: magnetic permeability of vacuum (default: 4π×10^-7 H/m)
- `c`: speed of light in vacuum (default: 299792458 m/s)

# Returns
- `G`: Green tensor (3×3 matrix)
"""
function compute_green_tensor(r, omega; mu0=4π*1e-7, c=299792458.0)
    # Calculate the wave number k = ω/c
    k = omega / c
    
    # Handle both vector and scalar r inputs
    if isa(r, Number)
        r_mag = r
        # For scalar r, we need a unit vector. Use z-direction by default
        r_vec = [0.0, 0.0, r_mag]
    else
        r_vec = convert(Vector{Float64}, r)
        r_mag = norm(r_vec)
    end
    
    # Calculate the exponential term
    exp_ikr = exp(1im * k * r_mag)
    
    # Calculate the prefactor
    prefactor = (mu0 * exp_ikr) / (4 * π * k^2 * r_mag^3)
    
    # Calculate the first term coefficient
    first_term_coef = k^2 * r_mag^2 + 1im * k * r_mag - 1
    
    # Calculate the second term coefficient
    second_term_coef = 3 - 3 * 1im * k * r_mag - k^2 * r_mag^2
    
    # Calculate the identity matrix term
    identity_term = first_term_coef * Matrix{Float64}(I, 3, 3)
    
    # Calculate the dyadic product term (r⊗r)/r^2
    if isa(r, Number)
        # For scalar r with default z-direction
        dyadic_term = zeros(ComplexF64, 3, 3)
        dyadic_term[3, 3] = 1.0  # Only z⊗z component is non-zero
    else
        # Normalized outer product
        r_normalized = r_vec / r_mag
        dyadic_term = r_normalized * r_normalized'
    end
    
    # Combine the terms
    G = prefactor * (identity_term + second_term_coef * dyadic_term)
    
    return G
end

"""
    compute_gamma_matrix(positions, omega, Gamma_0, dipole; c=299792458.0)

Compute the damping matrix Γ for a system of dipoles.

# Arguments
- `positions`: Array of position vectors for each dipole
- `omega`: Frequency
- `Gamma_0`: Damping rate
- `dipole`: Dipole moment vector (assumed same for all dipoles)
- `c`: Speed of light (default: 299792458.0 m/s)

# Returns
- `Gamma`: Damping matrix (N×N)
"""
function compute_gamma_matrix(positions, omega, Gamma_0, dipole; c=299792458.0)
    N = length(positions)
    Gamma = zeros(ComplexF64, N, N)
    
    p = dipole
    
    for i in 1:N
        for j in 1:N
            if i == j
                Gamma[i, j] = Gamma_0
            else
                r_ij = positions[i] - positions[j]
                G_ij = compute_green_tensor(r_ij, omega; c=c)
                Gamma[i, j] = 2 * omega^2 * dot(conj(p), imag(G_ij) * p) / ħ

            end
        end
    end
    
    return Gamma
end

"""
    compute_J_matrix(positions, omega, Gamma_0, dipole; c=299792458.0)

Compute the coupling matrix J for a system of dipoles.

# Arguments
- `positions`: Array of position vectors for each dipole
- `omega`: Frequency of the transition
- `Gamma_0`: Damping rate
- `dipole`: Dipole moment vector (assumed same for all dipoles)
- `c`: Speed of light (default: 299792458.0 m/s)

# Returns
- `J`: Coupling matrix (N×N)
"""
function compute_J_matrix(positions, omega, Gamma_0, dipole; c=299792458.0)
    N = length(positions)
    J = zeros(ComplexF64, N, N)
    
    p = dipole
    
    for i in 1:N
        for j in 1:N
            if i == j
                J[i, j] = 0
            else
                r_ij = positions[i] - positions[j]
                G_ij = compute_green_tensor(r_ij, omega; c=c)
                J[i, j] = -omega^2 * dot(conj(p), real(G_ij) * p) / ħ
            end
        end
    end
    
    return J
end

"""
    spin_derivative(spins, positions, omega, Gamma_0, omega_z, Omega_Rabi, dipole; c=299792458.0, xi_x=nothing, xi_y=nothing)

Calculate the derivatives of the spins according to the TWA model.

# Arguments
- `spins`: 3×N matrix where each column is a spin vector [Sx; Sy; Sz] for one particle
- `positions`: Array of position vectors for each dipole
- `omega`: Frequency of the transition
- `Gamma_0`: Damping rate
- `omega_z`: Zeeman splitting frequency
- `Omega_Rabi`: Rabi frequency
- `dipole`: Dipole moment vector (assumed same for all dipoles)
- `c`: Speed of light (default: 299792458.0 m/s)
- `xi_x`: Optional noise array for x-component (default: nothing)
- `xi_y`: Optional noise array for y-component (default: nothing)

# Returns
- `derivatives`: Time derivatives of all spin vectors (3×N matrix)
"""
function spin_derivative(spins, positions, omega, Gamma_0, omega_z, Omega_Rabi, dipole; c=299792458.0, xi_x=nothing, xi_y=nothing)
    # Compute the Gamma and J matrices
    Gamma = compute_gamma_matrix(positions, omega, Gamma_0, dipole; c=c)
    J_matrix = compute_J_matrix(positions, omega, Gamma_0, dipole; c=c)
    
    # Convert complex matrices to real matrices for the DTWA calculation
    Jx = real(J_matrix)
    Jy = real(J_matrix)
    Jz = real(J_matrix)
    Gamma_real = real(Gamma)
    
    # Prepare noise vectors if provided
    N = size(spins, 2)
    xi_x_vec = isnothing(xi_x) ? zeros(0) : xi_x
    xi_y_vec = isnothing(xi_y) ? zeros(0) : xi_y
    
    # println("computing derivatives")
    # Calculate derivatives using the dtwa_step function
    derivatives = dtwa_step(spins, Jx, Jy, Jz, omega_z, Omega_Rabi, Gamma_real, xi_x_vec, xi_y_vec)
    # println("derivatives computed")
    return derivatives
end

"""
    compute_spin_dynamics_TWA(;
        num_particles=10,
        Gamma_0=2π * 6.065e6,
        t_max_factor=1,
        num_steps=1000,
        time_factor=1,
        omega_z=0,
        Omega_Rabi=0,
        num_simulations=10000,
        dipole_moment=nothing,
        listNumparticles=nothing,
        positions=nothing
    )

Compute the dynamics of a linear chain of spins using the TWA approach.

# Arguments
- `num_particles`: Number of particles in the chain
- `Gamma_0`: Decay rate (default: 2π * 6.065e6)
- `t_max_factor`: Maximum simulation time as a factor of Gamma_0
- `num_steps`: Number of time steps
- `time_factor`: Factor to reduce the number of stored time steps
- `omega_z`: Detuning (in rad/s)
- `Omega_Rabi`: Rabi frequency
- `num_simulations`: Number of simulations to average
- `dipole_moment`: Direction of the dipole moment (normalized)
- `listNumparticles`: List of particle numbers to simulate (overrides num_particles if provided)
- `positions`: Array of position vectors for each dipole

# Returns
- `magnetization_list`: List of magnetization arrays
- `Gammas`: List of Gamma matrices
- `J_matrices`: List of J matrices
"""
function compute_spin_dynamics_TWA(;
    num_particles=10,
    Gamma_0=2π * 6.065e6,
    t_max_factor=1,
    num_steps=1000,
    time_factor=1,
    omega_z=0,
    Omega_Rabi=0,
    num_simulations=10000,
    dipole_moment=nothing,
    listNumparticles=nothing,
    positions=nothing
)
    # Define physical constants
    c = 2.99792458e8  # Speed of light in m/s
    λ_0 = 780e-9      # Laser wavelength in m
    omega = 2π * c / λ_0  # Laser frequency in rad/s
    
    # Define simulation parameters
    t_max = t_max_factor / Gamma_0  # Maximum simulation time
    dt = t_max / num_steps  # Time step
    
    # Initialize time array
    times = range(0, t_max, length=num_steps)
    
    println("Simulating spin dynamics...")
    
    # Initialize arrays to store results
    magnetization_list = []
    Gammas = []
    J_matrices = []
    
    Omega_Rabi = 0 * Gamma_0
    
    # Compute interaction matrices
    
    for idx in 1:length(listNumparticles)
        num_particles = listNumparticles[idx]
        
        Gamma = compute_gamma_matrix(positions, omega, Gamma_0, dipole_moment; c=c)
        J_matrix = compute_J_matrix(positions, omega, Gamma_0, dipole_moment; c=c)
        push!(Gammas, Gamma)
        push!(J_matrices, J_matrix)
        gamma_vals, nu = eigen(Gamma)
        
        
        # Compute eigenvalues and eigenvectors of the Gamma matrix
        
        # println("Gamma eigenvalues: ", gamma_vals)
    
        
        # Initialize array to store magnetizations
        magnetizations = zeros(num_simulations, div(num_steps, time_factor), num_particles, 3)
        
        println("Gamma matrix and J matrix are computed")
        # Progress bar for simulations
        # Pre-allocate magnetizations array
        magnetizations = zeros(num_simulations, div(num_steps, time_factor), num_particles, 3)

        # Create a progress bar
        # p = Progress(num_simulations, desc="Running simulations: ")
        # p.counter = Threads.Atomic(0)  # Create an atomic counter

        # Perform simulations with multithreading
        Threads.@threads for sim in 1:num_simulations
            # Initialize spins
            spins = zeros(num_particles, 3)
            spins[:, 3] .= 1.0  # All spins initially pointing up (z-direction)
            spins[:, 1] = rand([-1.0, 1.0], num_particles)  # Random x components
            spins[:, 2] = rand([-1.0, 1.0], num_particles)  # Random y components
            
            # Initialize the spin evolution array for this simulation
            spin_evolution = zeros(div(num_steps, time_factor), num_particles, 3)
            spin_evolution[1, :, :] = spins
            current_state = copy(spins)

            for step in 2:num_steps
                # Generate noise for this step
                xi_x, xi_y = generate_noises(step, num_particles, gamma_vals, nu, dt)
                
                # Implicit midpoint method
                midpoint_state = current_state + dt * spin_derivative(
                    permutedims(current_state, (2, 1)),
                    positions, omega, Gamma_0, omega_z, Omega_Rabi, dipole_moment; 
                    c=c, xi_x=xi_x, xi_y=xi_y
                )'

                midpoint_derivative = spin_derivative(
                    permutedims((midpoint_state + current_state) / 2, (2, 1)),
                    positions, omega, Gamma_0, omega_z, Omega_Rabi, dipole_moment; 
                    c=c, xi_x=xi_x, xi_y=xi_y
                )'

                current_state = current_state + dt * midpoint_derivative

                # Store evolution at each time factor
                if step % time_factor == 0
                    spin_evolution[div(step, time_factor), :, :] = current_state
                end
            end

            # Thread-local storage for magnetizations
            # Threads.atomic_add!(p.counter, 1)  # Update progress bar atomically
            # ProgressMeter.update!(p)

            # Store the result for this simulation
            magnetizations[sim, :, :, :] = spin_evolution
        end

        # After all threads finish, push the results to the magnetization list
        push!(magnetization_list, magnetizations)
    end
    
    println("All simulations complete!")
    
    return magnetization_list, Gammas, J_matrices, times,num_simulations
end

# Define parameters for the simulation
num_particles = 2
Gamma_0 = 2π * 6.065e6  # Decay rate in Hz
t_max_factor = 1
num_steps = 1000
time_factor = 10  # Save every 10 steps
omega_z = 0.0
Omega_Rabi = 0.0
num_simulations = 10000  # Reduced for faster execution
lambda_0 = 780e-9
ħ = 1.054571817e-34
listNumparticles = [2]

using ProgressMeter

using LinearAlgebra
# Define positions of particles (example: particles in a line with spacing 0.1)
positions = [ i * [0.5, 0.0, 0.0]*lambda_0 for i in 1:num_particles]

# Define dipole moment (example: unit vector in z-direction)
atomic_dipole_moment = 2.533e-29 # C·m

dipole_moment = (1/sqrt(2)) * [1.0, 1.0im, 0.0] * atomic_dipole_moment  # Circular polarization

# Speed of light in vacuum (m/s)
c = 299792458.0

# Transition frequency (example: optical transition)
omega = 2π * c / lambda_0  # Hz

# println("computing gamma matrix")
# Gamma = compute_gamma_matrix(positions, omega, Gamma_0, dipole_moment; c=c)
# println("Gamma matrix: ", Gamma)

# println("computing J matrix")
# J = compute_J_matrix(positions, omega, Gamma_0, dipole_moment; c=c)
# println("J matrix: ", J)
using Base.Threads
using ProgressMeter
# Run the simulation
println("Starting spin dynamics simulation...")
t0 = time()
magnetization_list, Gammas, J_matrices, times, num_simulations = compute_spin_dynamics_TWA(
    num_particles=num_particles,
    Gamma_0=Gamma_0,
    t_max_factor=t_max_factor,
    num_steps=num_steps,
    time_factor=time_factor,
    omega_z=omega_z,
    Omega_Rabi=Omega_Rabi,
    num_simulations=num_simulations,
    dipole_moment=dipole_moment,
    positions=positions,
    listNumparticles=listNumparticles
)
t1 = time()
println("Simulation completed in ", t1 - t0, " seconds")
# Save the results to a JSON file
using JSON
# Convert complex numbers to a format that can be serialized to JSON
function prepare_for_json(data)
    if isa(data, Array) && eltype(data) <: Complex
        return Dict("real" => real.(data), "imag" => imag.(data))
    elseif isa(data, Array)
        return [prepare_for_json(item) for item in data]
    elseif isa(data, Dict)
        return Dict(string(k) => prepare_for_json(v) for (k, v) in data)
    elseif isa(data, Complex)
        return Dict("real" => real(data), "imag" => imag(data))
    else
        return data
    end
end

# Prepare the data for JSON serialization
json_data = Dict(
    "magnetization_list" => prepare_for_json(magnetization_list),
    "Gammas" => prepare_for_json(Gammas),
    "J_matrices" => prepare_for_json(J_matrices),
    "times" => times,
    "num_simulations" => num_simulations
)

# Save to a file
open("simulation_results.json", "w") do f
    JSON.print(f, json_data, 4)  # 4 spaces for indentation
end

println("Results saved to simulation_results.json")


println("Simulation completed!")

using Plots
using JSON

using Pkg
include("Plotting.jl")

times = range(0, t_max_factor/Gamma_0, length=div(num_steps, time_factor))
R_t_list = plot_R_t(magnetization_list, times, Gammas, J_matrices, listNumparticles, Gamma_0, num_simulations; Omega_Rabi=0, spacing=nothing, lambda_0=nothing, plot_max=false)
