def plot_R_t(magnetization_list, times, Gammas, J_matrices, listNumparticles, Gamma_0, num_simulations, Omega_Rabi=0, spacing=None, lambda_0=None, plot_max=True):
    """
    Process magnetization data to calculate and plot R(t) the ratio of outputted power by the incoherently outputted power for different numbers of particles.
    
    Parameters:
    - magnetization_list: List of magnetization arrays from simulations
    - times: Array of time points
    - Gammas: List of Gamma matrices for each particle configuration
    - J_matrices: List of J matrices for each particle configuration
    - listNumparticles: List of particle numbers simulated
    - Gamma_0: Base decay rate
    - num_simulations: Number of simulations averaged
    - Omega_Rabi: Rabi frequency (optional)
    - spacing: Spacing between particles (optional, for plot title)
    - lambda_0: Wavelength (optional, for plot title)
    - plot_max: Whether to plot the maximum R(t) vs number of particles (default: True)
    
    Returns:
    - R_t_list: List of R(t) values for each particle configuration
    - R_max_list: List of maximum R(t) values for each particle configuration
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Process magnetization_list to calculate R(t) for each number of particles
    R_t_list = []  # List to store R(t) values for each particle configuration
    R_max_list = []  # List to store maximum R(t) values for each particle configuration
    labels = []  # List to store labels for the plot legend
    # Create a color gradient for different particle numbers in the plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(magnetization_list)))

    # Loop through each particle configuration
    for idx, magnetizations in enumerate(magnetization_list):
        # Extract the corresponding Gamma matrix, J matrix, and number of particles
        Gamma = Gammas[idx]
        J_matrix = J_matrices[idx]
        num_particles_current = listNumparticles[idx]

        # Extract the spin components from the magnetization data
        # The magnetizations array has shape (num_simulations, num_time_steps, num_particles, 3)
        # where the last dimension represents the x, y, z components
        
        # Calculate the raising and lowering operators from the x and y components
        # s⁺ = s_x + i*s_y and s⁻ = s_x - i*s_y
        s_plus = 0.5*(magnetizations[:,:,:,0] + 1j * magnetizations[:,:,:,1])  # For all simulations, all time steps, all atoms
        s_minus = 0.5*(magnetizations[:,:,:,0] - 1j * magnetizations[:,:,:,1])  # For all simulations, all time steps, all atoms
        
        # Extract individual components for clarity
        s_z = magnetizations[:,:,:,2]  # z component
        s_x = magnetizations[:,:,:,0]  # x component
        s_y = magnetizations[:,:,:,1]  # y component
        
        # Initialize R(t) array for this particle configuration
        R_t = np.zeros(len(times), dtype=complex)
        
        # Calculate average spin operators over all simulations
        # These have shape (time_steps, num_particles)
        avg_s_plus = np.mean(s_plus, axis=0)
        avg_s_minus = np.mean(s_minus, axis=0)
        avg_s_z = np.mean(s_z, axis=0)

        # Initialize array for incoherent contribution (not used in current implementation)
        R_inch_t = np.zeros(len(times), dtype=complex)
        
        # Calculate R(t) for each time step
        for t in range(len(times)):
            # Loop through all pairs of particles
            for i in range(num_particles_current):
                for j in range(num_particles_current):
                        # Off-diagonal terms: coherent contribution from dipole-dipole interactions
                        # Calculate <s⁺_i s⁻_j> averaged over all simulations
                    R_t[t] += Gamma[i,j] * (np.dot(s_plus[:,t,i], s_minus[:,t,j]))/num_simulations
                    if i == j: 
                        # Diagonal terms: contribution from individual atoms
                        # Calculate <s_z> averaged over all simulations
                        R_t[t] += Gamma_0 * (np.sum(s_z[:,t,i]))/(2*num_simulations)
                        
        # Normalize R(t) by N*Γ0 to get the enhancement factor
        R_t /= Gamma_0*num_particles_current
        
        # Store the result for this particle configuration
        R_t_list.append(R_t)
        
        # Calculate and store the maximum value of R(t) for this configuration
        R_max = np.max(np.real(R_t))
        R_max_list.append(R_max)
        
        # Create label for the plot legend
        labels.append(f'N = {num_particles_current}')

    # Create figure for R(t) vs time plot
    plt.figure(figsize=(12, 8))

    # Plot R(t) for all particle numbers on the same graph
    for idx, R_t in enumerate(R_t_list):
        plt.plot(times*Gamma_0, np.real(R_t), label=labels[idx], color=colors[idx], linewidth=2.5)

    # Add title with parameters if provided
    title = 'Time evolution of R(t)'
    if Omega_Rabi is not None:
        # Include Rabi frequency in the title if provided
        title += f' - $\Omega/\Gamma_0$={Omega_Rabi/Gamma_0:.2f}'
    if spacing is not None and lambda_0 is not None:
        # Include spacing information in the title if provided
        title += f', $a/\lambda_0$={spacing/lambda_0:.2f}'
    
    # Set plot title, labels, and formatting
    plt.title(title, fontsize=16)
    plt.xlabel('Time ($\Gamma_0^{-1}$)', fontsize=14)
    plt.ylabel('R(t)', fontsize=14)
    plt.legend(fontsize=12, framealpha=0.7)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xscale('log')  # Use logarithmic scale for time axis
    plt.gca().set_facecolor('#f8f8f8')  # Set light gray background
    plt.tight_layout()
    plt.show()

    # Plot maximum R(t) vs number of particles if requested
    if plot_max:
        plt.figure(figsize=(10, 6))
        # Plot maximum R(t) values against number of particles
        plt.plot(listNumparticles, R_max_list, 'o-', linewidth=2, markersize=8, color='blue')
        
        # Add title with parameters if provided
        max_title = 'Maximum R(t) vs Number of Particles'
        if Omega_Rabi is not None:
            max_title += f' - $\Omega/\Gamma_0$={Omega_Rabi/Gamma_0:.2f}'
        if spacing is not None and lambda_0 is not None:
            max_title += f', $a/\lambda_0$={spacing/lambda_0:.2f}'
            
        # Set plot title, labels, and formatting
        plt.title(max_title, fontsize=16)
        plt.xlabel('Number of Particles (N)', fontsize=14)
        plt.ylabel('Maximum R(t)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.gca().set_facecolor('#f8f8f8')  # Set light gray background
        plt.tight_layout()
        plt.show()
        
    # Return the calculated R(t) values and maximum R(t) values
    return R_t_list, R_max_list
