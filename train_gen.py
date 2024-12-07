#!/usr/bin/env python2.7
import itertools

def generate_parameters():
    L = [0]
    
    node_configs = [
        # (4, 2, 2, 0),
        # (4, 2, 2, 2),
        # (8, 2, 2, 0),
        # (8, 2, 2, 2),
        # (8, 4, 2, 0),
        # (8, 4, 2, 2),
        # (16, 2, 2, 0),
        # (16, 2, 2, 2),
        (16, 4, 2, 0),
        (16, 4, 2, 2),
        (16, 8, 2, 0),
        (16, 8, 2, 2),
        (16, 8, 4, 0),
        (16, 8, 4, 2),
    ]
    
    # W = [10000000, 25000000, 50000000, 100000000]
    W = [100000000]
    
    speed_triples = [
        (400, 800, 800),
        # (200, 400, 800),
        # (200, 400, 400),
    ]
    
    failure_triples = [
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 1.0),
        (0.0, 0.0, 1.0),
    ]
    
    # C = [0, 1, 2]
    C = [2]

    seen_rows = set()
    with open('train.txt', 'w') as f:
        f.write("L N N1 N2 N3 W S1 S2 S3 F1 F2 F3 C\n")
        
        for w, speeds, failures, c in itertools.product(W, speed_triples, failure_triples, C):
            s1, s2, s3 = speeds
            f1, f2, f3 = failures
            
            for n, n1, n2, n3 in node_configs:
                actual_f3 = 0.0 if n3 == 0 else f3
                
                row = (L[0], n, n1, n2, n3, w, s1, s2, s3, f1, f2, actual_f3, c)
                row_str = " ".join(map(str, row))
                
                if row_str not in seen_rows:
                    seen_rows.add(row_str)
                    f.write("{}\n".format(row_str))

if __name__ == "__main__":
    generate_parameters()