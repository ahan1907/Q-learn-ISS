import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def compare_coarse_fine_qvalues(coarse_cell_idx, coarse_grid_size=10, fine_grid_size=25, action_bins=2):
    
    # Load Q-tables
    qtable_fine = pd.read_excel(r'C:\Users\Ahan_FOCASLab\OneDrive - Indian Institute of Science\PhD Projects and Other\Ahan PhD work\Q learning using Incremental Stability\Q-learn-with-Inc.-Stability\Q-learn-ISS\My code\Temp Control\Qtable_25_structured.xlsx')
    qtable_coarse = pd.read_excel(r'C:\Users\Ahan_FOCASLab\OneDrive - Indian Institute of Science\PhD Projects and Other\Ahan PhD work\Q learning using Incremental Stability\Q-learn-with-Inc.-Stability\Q-learn-ISS\My code\Temp Control\Qtable_11_structured.xlsx')

    # Unpack coarse cell index
    i, j, k = coarse_cell_idx
    fine_per_coarse = fine_grid_size // coarse_grid_size

    fine_cells = []
    for dx in range(fine_per_coarse):
        for dy in range(fine_per_coarse):
            for dz in range(fine_per_coarse):
                fine_i = i * fine_per_coarse + dx
                fine_j = j * fine_per_coarse + dy
                fine_k = k * fine_per_coarse + dz
                fine_cells.append((fine_i, fine_j, fine_k))

    # Extract Q-values from fine grid
    records = []
    for fx, fy, fz in fine_cells:
        for ax in range(action_bins):
            for ay in range(action_bins):
                for az in range(action_bins):
                    match = qtable_fine[
                        (qtable_fine['state_idx_x'] == fx) &
                        (qtable_fine['state_idx_y'] == fy) &
                        (qtable_fine['state_idx_z'] == fz) &
                        (qtable_fine['action_idx_x'] == ax) &
                        (qtable_fine['action_idx_y'] == ay) &
                        (qtable_fine['action_idx_z'] == az)
                    ]
                    if not match.empty:
                        for _, row in match.iterrows():
                            records.append({
                                "fine_state_x": fx,
                                "fine_state_y": fy,
                                "fine_state_z": fz,
                                "action_x": ax,
                                "action_y": ay,
                                "action_z": az,
                                "fine_Q": row['Q_value']
                            })

    # Create DataFrame of fine Q-values
    fine_df = pd.DataFrame(records)

    # Extract corresponding coarse Q-values (for all 8 actions)
    coarse_rows = qtable_coarse[
        (qtable_coarse['state_idx_x'] == i) &
        (qtable_coarse['state_idx_y'] == j) &
        (qtable_coarse['state_idx_z'] == k)
    ]

    # Create a dictionary for fast lookup of coarse Q-values by action
    coarse_q_map = {(row['action_idx_x'], row['action_idx_y'], row['action_idx_z']): row['Q_value'] for _, row in coarse_rows.iterrows()}

    # Add coarse Q-values to each fine entry
    fine_df["coarse_Q"] = fine_df.apply(lambda row: coarse_q_map.get((row["action_x"], row["action_y"], row["action_z"]), None),axis=1)

    # Create DataFrame for coarse Q-values (once per action)
    coarse_q_records = []
    for (ax, ay, az), qval in coarse_q_map.items():
        coarse_q_records.append({
            "coarse_state_x": i,
            "coarse_state_y": j,
            "coarse_state_z": k,
            "action_x": ax,
            "action_y": ay,
            "action_z": az,
            "coarse_Q": qval
        })
    coarse_df = pd.DataFrame(coarse_q_records)

    return fine_df, coarse_df

coarse_idx = (9, 4, 6)

fine_df, coarse_df = compare_coarse_fine_qvalues(coarse_idx)

print("Fine Q-values:")
print(fine_df.head())