import matplotlib.pyplot as plt
import numpy as np
import os

# Set font sizes
SMALL_SIZE = 20
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

# Set all font sizes
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Get input filename
input_file = input_file = '/Users/shixuang/course/cs243/workspace/fct/fct_L0_N16_N12_N22_N32_W100000000_S1200Gbps_S2400Gbps_S3800Gbps_F10.0_F21.0_F31.0_C2.txt'

# Create output filename based on input filename
base_filename = os.path.basename(input_file)
output_filename = f'cdf_{base_filename[:-4]}.png'

# Step 1: Read flow completion times
completion_times_ns = []

try:
    with open(input_file, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) >= 7:
                completion_time_ns = int(tokens[6])
                completion_times_ns.append(completion_time_ns)
except FileNotFoundError:
    print(f"Error: The file '{input_file}' was not found.")
    exit(1)

if not completion_times_ns:
    print("No flow completion times found in the input file.")
    exit(1)

# Step 2: Convert times from nanoseconds to milliseconds
completion_times_ms = [t / 1e6 for t in completion_times_ns]

# Step 3: Sort the completion times
completion_times_ms_sorted = sorted(completion_times_ms)

# Step 4: Calculate cumulative probabilities
N = len(completion_times_ms_sorted)
cum_prob = np.arange(1, N+1) / N

# Step 5: Plot the CDF
plt.figure(figsize=(7, 10))
plt.plot(completion_times_ms_sorted, cum_prob, marker='.', linestyle='-', color='blue', linewidth=2)
plt.xlabel('Flow Completion Time (milliseconds)', fontsize=MEDIUM_SIZE)
plt.ylabel('Cumulative Probability', fontsize=MEDIUM_SIZE)
plt.title('CDF of Flow Completion Times', fontsize=BIGGER_SIZE, pad=20)
plt.grid(True)

# Add vertical lines for median and 90th percentile
median_fct = np.percentile(completion_times_ms_sorted, 50)
p90_fct = np.percentile(completion_times_ms_sorted, 90)
plt.axvline(median_fct, color='red', linestyle='--', label='Median', linewidth=2)
plt.axvline(p90_fct, color='green', linestyle='--', label='90th Percentile', linewidth=2)
plt.legend(fontsize=MEDIUM_SIZE)

# Make tick labels larger
plt.xticks(fontsize=SMALL_SIZE)
plt.yticks(fontsize=SMALL_SIZE)

# Add some padding to the layout
plt.tight_layout()

# Step 6: Save the figure with the new filename
plt.savefig(output_filename, dpi=300, bbox_inches='tight')

# Print statistical information
mean_fct = np.mean(completion_times_ms_sorted)
print(f"Mean FCT: {mean_fct:.6f} milliseconds")
print(f"Median FCT: {median_fct:.6f} milliseconds")
print(f"90th Percentile FCT: {p90_fct:.6f} milliseconds")
print(f"Figure saved as: {output_filename}")