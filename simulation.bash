#!/bin/bash

INPUT_FILE="/workspace/train.txt"
OUTPUT_FILE="/workspace/train_results.txt"

echo "L N N1 N2 N3 W S1 S2 S3 F1 F2 F3 C T T10 T25 T50 T75" > "$OUTPUT_FILE"

mkdir -p /workspace/fct

tail -n +2 "$INPUT_FILE" | while IFS=' ' read -r L N N1 N2 N3 W S1 S2 S3 F1 F2 F3 C _; do
    python /workspace/sp_all.py -L "$L" -N "$N" -N1 "$N1" -N2 "$N2" -N3 "$N3" -W "$W" \
        -S1 "${S1}Gbps" -S2 "${S2}Gbps" -S3 "${S3}Gbps" \
        -F1 "$F1" -F2 "$F2" -F3 "$F3" -C "$C"
    
    cd /workspace/hpcc/simulation || exit
    T=$(./waf --run 'scratch/third mix/config_sp.txt' | tail -n 1)
    
    # Sort completion times and get representative samples (converting to seconds)
    completion_times=$(awk '{print $7/1000000000}' /workspace/hpcc/simulation/mix/fct.txt | sort -n)
    total_lines=$(echo "$completion_times" | wc -l)
    
    # Adding 1 to ensure valid line numbers
    T10_line=$(( (total_lines * 10 / 100) + 1 ))
    T25_line=$(( (total_lines * 25 / 100) + 1 ))
    T50_line=$(( (total_lines * 50 / 100) + 1 ))
    T75_line=$(( (total_lines * 75 / 100) + 1 ))
    
    # Get the percentile values
    [ $T10_line -le $total_lines ] && T10=$(echo "$completion_times" | sed -n "${T10_line}p")
    [ $T25_line -le $total_lines ] && T25=$(echo "$completion_times" | sed -n "${T25_line}p")
    [ $T50_line -le $total_lines ] && T50=$(echo "$completion_times" | sed -n "${T50_line}p")
    [ $T75_line -le $total_lines ] && T75=$(echo "$completion_times" | sed -n "${T75_line}p")
    
    filename="fct_L${L}_N${N}_N1${N1}_N2${N2}_N3${N3}_W${W}_S1${S1}Gbps_S2${S2}Gbps_S3${S3}Gbps_F1${F1}_F2${F2}_F3${F3}_C${C}.txt"
    cp /workspace/hpcc/simulation/mix/fct.txt "/workspace/fct/$filename"
    
    cd /workspace || exit
    
    echo "$L $N $N1 $N2 $N3 $W $S1 $S2 $S3 $F1 $F2 $F3 $C $T $T10 $T25 $T50 $T75" >> "$OUTPUT_FILE"
done

echo "All simulations completed. Results written to $OUTPUT_FILE"