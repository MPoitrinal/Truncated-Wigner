using JSON
using Plots
using Statistics


"""
    plot_R_t(magnetization_list, times, Gammas, J_matrices, listNumparticles, Gamma_0, num_simulations; Omega_Rabi=0, spacing=nothing, lambda_0=nothing, plot_max=true)

Process magnetization data to calculate and plot R(t) the ratio of outputted power by the incoherently outputted power for different numbers of particles.

# Parameters
- `magnetization_list`: List of magnetization arrays from simulations
- `times`: Array of time points
- `Gammas`: List of Gamma matrices for each particle configuration
- `J_matrices`: List of J matrices for each particle configuration
- `listNumparticles`: List of particle numbers simulated
- `Gamma_0`: Base decay rate
- `num_simulations`: Number of simulations averaged
- `Omega_Rabi`: Rabi frequency (optional)
- `spacing`: Spacing between particles (optional, for plot title)
- `lambda_0`: Wavelength (optional, for plot title)
- `plot_max`: Whether to plot the maximum R(t) vs number of particles (default: true)

# Returns
- `R_t_list`: List of R(t) values for each particle configuration
"""
function plot_R_t(magnetization_list, times, Gammas, J_matrices, listNumparticles, Gamma_0, num_simulations; 
                 Omega_Rabi=0, spacing=nothing, lambda_0=nothing, plot_max=true)
    
    
    # Process magnetization_list to calculate R(t) for each number of particles
    R_t_list = []  # List to store R(t) values for each particle configuration
    R_max_list = []  # List to store maximum R(t) values for each particle configuration
    labels = []  # List to store labels for the plot legend
    
    # Create a color gradient for different particle numbers in the plot
    colors = cgrad(:viridis, length(magnetization_list), categorical=true)

    # Loop through each particle configuration
    for idx in 1:length(magnetization_list)
        # Extract the corresponding Gamma matrix, J matrix, and number of particles
        Gamma = Gammas[idx]
        J_matrix = J_matrices[idx]
        num_particles_current = listNumparticles[idx]

        # Extract the magnetization data
        magnetizations = magnetization_list[idx]
        
        # Calculate the raising and lowering operators from the x and y components
        # s⁺ = s_x + i*s_y and s⁻ = s_x - i*s_y
        s_plus = 0.5 .* (magnetizations[:,:,:,1] .+ im .* magnetizations[:,:,:,2])  # For all simulations, all time steps, all atoms
        s_minus = 0.5 .* (magnetizations[:,:,:,1] .- im .* magnetizations[:,:,:,2])  # For all simulations, all time steps, all atoms
        
        # Extract individual components for clarity
        s_z = magnetizations[:,:,:,3]  # z component
        s_x = magnetizations[:,:,:,1]  # x component
        s_y = magnetizations[:,:,:,2]  # y component
        
        # Initialize R(t) array for this particle configuration
        R_t = zeros(ComplexF64, length(times))
        
        # Calculate average spin operators over all simulations
        # These have shape (time_steps, num_particles)
        print("about to calculate average spin operators")

        print(size(s_plus))

        avg_s_plus = dropdims(mean(s_plus, dims=1), dims=1)  # (N, 3)
        avg_s_minus = dropdims(mean(s_minus, dims=1), dims=1)

        avg_s_z = dropdims(mean(s_z, dims=1), dims=1)

        # Initialize array for incoherent contribution (not used in current implementation)
        R_inch_t = zeros(ComplexF64, length(times))
        
        # Calculate R(t) for each time step
        for t in 1:length(times)
            # Loop through all pairs of particles
            for i in 1:num_particles_current
                for j in 1:num_particles_current
                    # Off-diagonal terms: coherent contribution from dipole-dipole interactions
                    # Calculate <s⁺_i s⁻_j> averaged over all simulations
                    R_t[t] += Gamma[i,j] * (sum(s_plus[:,t,i] .* s_minus[:,t,j]))/num_simulations
                    if i == j 
                        # Diagonal terms: contribution from individual atoms
                        # Calculate <s_z> averaged over all simulations
                        R_t[t] += Gamma_0 * (sum(s_z[:,t,i]))/(2*num_simulations)
                    end
                end
            end
        end
        
        # Normalize R(t) by N*Γ0 to get the enhancement factor
        R_t ./= (Gamma_0 * num_particles_current)
        
        # Store the result for this particle configuration
        push!(R_t_list, R_t)
        
        # Calculate and store the maximum value of R(t) for this configuration
        R_max = maximum(real.(R_t))
        push!(R_max_list, R_max)
        
        # Create label for the plot legend
        push!(labels, "N = $num_particles_current")
    end

    # Create figure for R(t) vs time plot
    p = plot(size=(800, 600), dpi=300, background_color=:white)

    # Plot R(t) for all particle numbers on the same graph
    for idx in 1:length(R_t_list)
        plot!(p, times .* Gamma_0, real.(R_t_list[idx]), 
              label=labels[idx], 
              color=colors[idx], 
              linewidth=2.5,xscale=:log)
    end

    # Add title with parameters if provided
    title_str = "Time evolution of R(t)"
    if Omega_Rabi != 0
        # Include Rabi frequency in the title if provided
        title_str *= " - Ω/Γ₀=$(Omega_Rabi/Gamma_0)"
    end
    if spacing !== nothing && lambda_0 !== nothing
        # Include spacing information in the title if provided
        title_str *= ", a/λ₀=$(spacing/lambda_0)"
    end
    
    # Set plot title, labels, and formatting
    plot!(p, title=title_str, titlefontsize=16,
          xlabel="Time (Γ₀⁻¹)", ylabel="R(t)",
          xguidefontsize=14, yguidefontsize=14,
          legend=:topright, legendfontsize=12,
          grid=true, gridalpha=0.7, gridstyle=:dash,
          tickfontsize=12,
          xscale=:log)  # Use logarithmic scale for time axis
    
    display(p)
    
    # Return the calculated R(t) values
    return R_t_list
end


# # Load the simulation results from the JSON file

# simulation_data = JSON.parsefile("/Users/martinpoitrinsl/Desktop/PhD/FreeSpaceTheory/Code/TruncatedWignerA/Julia/simulation_results.json")
# magnetization_list = simulation_data["magnetization_list"]
# Gammas = simulation_data["Gammas"]
# J_matrices = simulation_data["J_matrices"]
# times = simulation_data["times"]
# num_simulations = simulation_data["num_simulations"]


# println(magnetization_list)
# println("got data from json file")

# R_t_list = plot_R_t(magnetization_list, times, Gammas, J_matrices, listNumparticles, Gamma_0, num_simulations; Omega_Rabi=0, spacing=nothing, lambda_0=nothing, plot_max=false)
