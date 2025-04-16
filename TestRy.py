# Import the ARC package for atomic calculations
import arc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Create a Hydrogen atom object
h = arc.Hydrogen()

# Define the quantum numbers for different states to plot
states = [
    (10, 0, 0.5),  # n=10, l=0, j=0.5 (S state)
    (10, 1, 0.5),  # n=10, l=1, j=0.5 (P state)
    (10, 1, 1.5),  # n=10, l=1, j=1.5 (P state)
    (10, 2, 1.5),  # n=10, l=2, j=1.5 (D state)
]

# Create a figure with subplots
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 2, figure=fig)

# Define the radial distance range (in Bohr radii)
r = np.linspace(0, 250, 1000)

# Plot each wavefunction
for i, (n, l, j) in enumerate(states):
    ax = fig.add_subplot(gs[i//2, i%2])
    
    # Calculate the radial wavefunction
    R = h.radialWavefunction(n, l, j, r)
    
    # Plot the wavefunction
    ax.plot(r, R, linewidth=2)
    
    # Calculate the probability density
    prob_density = R**2
    ax.plot(r, prob_density, 'r--', linewidth=1.5, alpha=0.7, label='Probability density')
    
    # Add labels and title
    ax.set_xlabel('Radial distance (Bohr radii)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title(f'n={n}, l={l}, j={j} ({["S", "P", "D", "F", "G"][l] if l < 5 else "l="+str(l)}$_{{{j}}}$)', fontsize=14)
    
    # Add grid and improve appearance
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=10)
    
    # Add vertical line at the expected radius n²a₀
    expected_radius = n**2
    ax.axvline(x=expected_radius, color='green', linestyle='-.', alpha=0.5, 
               label=f'n²={expected_radius}')
    
    # Set reasonable y-limits
    max_val = max(abs(R.max()), abs(R.min()))
    ax.set_ylim(-max_val*1.1, max_val*1.1)

# Add a title for the entire figure
plt.suptitle('Hydrogen Atom Wavefunctions', fontsize=18)

# Add text explaining the wavefunctions
plt.figtext(0.5, 0.01, 
            "Radial wavefunctions of hydrogen atom for different quantum states.\n"
            "Solid blue lines show the wavefunction amplitude, dashed red lines show probability density.",
            ha="center", fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})

plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust layout to make room for the text
plt.show()
