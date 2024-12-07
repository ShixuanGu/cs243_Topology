import os
import numpy as np

def parse_filename(filename):
    # Remove 'fct_' prefix and '.txt' suffix and split
    parts = filename[4:-4].split('_')
    
    # Parse in strict order following the header
    # L N N1 N2 N3 W S1 S2 S3 F1 F2 F3 C
    values = {
        'L': int(parts[0][1:]),          # L0 -> 0
        'N': int(parts[1][1:]),          # N8 -> 8
        'N1': int(parts[2][2:]),         # N14 -> 4
        'N2': int(parts[3][2:]),         # N22 -> 2
        'N3': int(parts[4][2:]),         # N32 -> 2
        'W': int(parts[5][1:]),          # W50000000 -> 50000000
        'S1': int(parts[6][2:-4]),       # S1200Gbps -> 200
        'S2': int(parts[7][2:-4]),       # S2400Gbps -> 400
        'S3': int(parts[8][2:-4]),       # S3800Gbps -> 800
        'F1': float(parts[9][2:]),       # F10.0 -> 0.0
        'F2': float(parts[10][2:]),      # F20.0 -> 0.0
        'F3': float(parts[11][2:]),      # F30.0 -> 0.0
        'C': int(parts[12][1:])          # C0 -> 0
    }
    return values

def process_fct_file(filepath):
    completion_times = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            completion_times.append(float(parts[-2]) / 1000000)
    return completion_times

def main():
    current_dir = './fct'
    results = []
    
    for filename in os.listdir(current_dir):
        if filename.startswith('fct_') and filename.endswith('.txt'):
            filepath = os.path.join(current_dir, filename)
            values = parse_filename(filename)
            completion_times = process_fct_file(filepath)
            
            if completion_times:
                completion_times = sorted(completion_times)
                median = np.median(completion_times)
                p10 = np.percentile(completion_times, 10)
                p25 = np.percentile(completion_times, 25)
                p50 = np.percentile(completion_times, 50)
                p75 = np.percentile(completion_times, 75)
                p90 = np.percentile(completion_times, 90)
                
                results.append({**values, 'T': median, 'T10': p10, 'T25': p25, 
                              'T50': p50, 'T75': p75, 'T90': p90})
    
    with open('process.txt', 'w') as f:
        f.write('L N N1 N2 N3 W S1 S2 S3 F1 F2 F3 C T T10 T25 T50 T75 T90\n')
        for r in results:
            f.write(f"{r['L']} {r['N']} {r['N1']} {r['N2']} {r['N3']} {r['W']} "
                   f"{r['S1']} {r['S2']} {r['S3']} {r['F1']} {r['F2']} {r['F3']} "
                   f"{r['C']} {r['T']:.6f} {r['T10']:.6f} {r['T25']:.6f} "
                   f"{r['T50']:.6f} {r['T75']:.6f} {r['T90']:.6f}\n")

if __name__ == '__main__':
    main()