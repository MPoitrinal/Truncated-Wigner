def plot_R_t(magnetization_list, times, Gammas, J_matrices, listNumparticles, Gamma_0, num_simulations, Omega_Rabi=0, spacing=None, lambda_0=None, plot_max=True):
    """
    Process magnetization data to calculate and plot R(t) for different numbers of particles.
    
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
    R_t_list = []
    R_max_list = []  # List to store maximum R(t) values
    labels = []
    colors = plt.cm.viridis(np.linspace(0, 1, len(magnetization_list)))  # Create a color gradient

    for idx, magnetizations in enumerate(magnetization_list):
        Gamma = Gammas[idx]
        J_matrix = J_matrices[idx]
        num_particles_current = listNumparticles[idx]

        # Extract the components we need
        # s⁺ = s_x + i*s_y and s⁻ = s_x - i*s_y
        s_plus = 0.5*(magnetizations[:,:,:,0] + 1j * magnetizations[:,:,:,1])  # For all atoms, all time steps
        s_minus = 0.5*(magnetizations[:,:,:,0] - 1j * magnetizations[:,:,:,1])  # For all atoms, all time steps
        s_z = magnetizations[:,:,:,2]  # z component
        s_x = magnetizations[:,:,:,0]  # x component
        s_y = magnetizations[:,:,:,1]  # y component
        
        # Initialize R(t)
        R_t = np.zeros(len(times), dtype=complex)
        
        # Calculate average s_plus and s_minus over all simulations first
        avg_s_plus = np.mean(s_plus, axis=0)  # shape: (time_steps, num_particles)
        avg_s_minus = np.mean(s_minus, axis=0)  # shape: (time_steps, num_particles)
        avg_s_z = np.mean(s_z, axis=0)  # shape: (time_steps, num_particles)

        # Calculate R(t) for each time step using vectorized operations
        R_inch_t = np.zeros(len(times), dtype=complex)
        for t in range(len(times)):
            # Off-diagonal terms (i != j)
            for i in range(num_particles_current):
                for j in range(num_particles_current):
                    if i != j:
                        R_t[t] += Gamma[i,j] * (np.dot(s_plus[:,t, i],s_minus[:,  t, j]))/num_simulations
                    else: 
                        R_t[t] += Gamma_0 * (np.sum(s_z[:,t,i]))/(num_simulations)
                        
        # Normalize by N*Γ0
        R_t /= Gamma_0*num_particles_current
        
        # Store the result
        R_t_list.append(R_t)
        
        # Calculate and store the maximum value of R(t)
        R_max = np.max(np.real(R_t))
        R_max_list.append(R_max)
        
        labels.append(f'N = {num_particles_current}')

    # Plot R(t) for all particle numbers on the same graph
    plt.figure(figsize=(12, 8))

    for idx, R_t in enumerate(R_t_list):
        plt.plot(times*Gamma_0, np.real(R_t), label=labels[idx], color=colors[idx], linewidth=2.5)

    # Add title with parameters if provided
    title = 'Time evolution of R(t)'
    if Omega_Rabi is not None:
        title += f' - $\Omega/\Gamma_0$={Omega_Rabi/Gamma_0:.2f}'
    if spacing is not None and lambda_0 is not None:
        title += f', $a/\lambda_0$={spacing/lambda_0:.2f}'
    
    plt.title(title, fontsize=16)
    plt.xlabel('Time ($\Gamma_0^{-1}$)', fontsize=14)
    plt.ylabel('R(t)', fontsize=14)
    plt.legend(fontsize=12, framealpha=0.7)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xscale('log')
    plt.gca().set_facecolor('#f8f8f8')
    plt.tight_layout()
    plt.show()

    # Plot maximum R(t) vs number of particles if requested
    if plot_max:
        plt.figure(figsize=(10, 6))
        plt.plot(listNumparticles, R_max_list, 'o-', linewidth=2, markersize=8, color='blue')
        
        # Add title with parameters if provided
        max_title = 'Maximum R(t) vs Number of Particles'
        if Omega_Rabi is not None:
            max_title += f' - $\Omega/\Gamma_0$={Omega_Rabi/Gamma_0:.2f}'
        if spacing is not None and lambda_0 is not None:
            max_title += f', $a/\lambda_0$={spacing/lambda_0:.2f}'
            
        plt.title(max_title, fontsize=16)
        plt.xlabel('Number of Particles (N)', fontsize=14)
        plt.ylabel('Maximum R(t)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.gca().set_facecolor('#f8f8f8')
        plt.tight_layout()
        plt.show()
        
    return R_t_list, R_max_list
