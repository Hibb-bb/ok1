import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
data_dir = './outputs/dim3'  # Path to your extracted folder
output_path = 'icml_simulation_results.pdf'
font_size = 20
icml_width = 6.75  # Standard ICML double-column text width in inches

plt.rcParams.update({
    'font.size': font_size,
    'axes.titlesize': font_size,
    'axes.labelsize': font_size,
    'xtick.labelsize': font_size - 4,
    'ytick.labelsize': font_size - 4,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
})

all_data = []

# --- 1. Data Loading ---
# Walking through the directory to find all result.csv files
for root, dirs, files in os.walk(data_dir):
    if 'result.csv' in files:
        # Extract Radius (R) from the folder name (e.g., "Radius2" -> 2)
        folder_name = os.path.basename(root)
        try:
            r_val = int(folder_name.replace('Radius', ''))
            
            file_path = os.path.join(root, 'result.csv')
            df = pd.read_csv(file_path)
            df['R'] = r_val
            all_data.append(df)
        except ValueError:
            print(f"Skipping directory {folder_name}: Could not parse Radius.")



if not all_data:
    print("No data found.")
else:
    df_total = pd.concat(all_data, ignore_index=True)
    stats = df_total.groupby(['model', 'R', 'M'])['recall rate'].agg(['mean', 'std']).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(icml_width, 3), sharey=True)

    radii = sorted(stats['R'].unique())
    # (2) Using a rainbow/jet-style colormap for R=1 to 6
    colors = plt.cm.turbo(np.linspace(0.1, 0.9, len(radii))) 
    
    # Define marker styles for different models
    model_markers = {'Karcher-Flow': 'o', 'DAM': 's', 'MHN': '^'}

    # --- Left Plot: Karcher-Flow ---
    kf_data = stats[stats['model'] == 'Karcher-Flow']
    for i, r in enumerate(radii):
        subset = kf_data[kf_data['R'] == r].sort_values('M')
        if not subset.empty:
            ax1.plot(subset['M'], subset['mean'], 
                     color=colors[i], marker=model_markers['Karcher-Flow'], linewidth=2)
            ax1.fill_between(subset['M'], subset['mean'] - subset['std'], 
                             subset['mean'] + subset['std'], color=colors[i], alpha=0.2)

    ax1.set_xscale('log')
    ax1.set_title('Hyperbolic')
    ax1.set_xlabel('M')
    ax1.set_ylabel('Recall Rate')
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    # --- Right Plot: Baselines (DAM & MHN) ---
    for i, r in enumerate(radii):
        # DAM
        dam_sub = stats[(stats['model'] == 'DAM') & (stats['R'] == r)].sort_values('M')
        if not dam_sub.empty:
            ax2.plot(dam_sub['M'], dam_sub['mean'], color=colors[i], 
                     marker=model_markers['DAM'], linestyle='-', alpha=0.8)
        
        # MHN
        mhn_sub = stats[(stats['model'] == 'MHN') & (stats['R'] == r)].sort_values('M')
        if not mhn_sub.empty:
            ax2.plot(mhn_sub['M'], mhn_sub['mean'], color=colors[i], 
                     marker=model_markers['MHN'], linestyle='--', alpha=0.8)

    ax2.set_xscale('log')
    ax2.set_title('Euclidean')
    ax2.set_xlabel('M')
    ax2.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    # plt.show()