# CS243 Topology Project

## Overview
This project extends the congestion control mechanism based on the [HPCC (High-Precision Congestion Control)](https://github.com/alibaba-edu/High-Precision-Congestion-Control) implementation.

## Environment Requirements
- NS-3.16
- GCC-5
- Ubuntu 16.04
- Python 2.7 (for main simulation)
- Python 3.x (for CDF-CCT figure plotting)

## Docker Setup
Due to specific version requirements, we provide a Docker image with the pre-configured environment:
- Ubuntu 16.04
- Python 2.7
- All necessary dependencies

### Accessing Docker Image
Please request access to our Docker image:
[Docker Image Link](https://drive.google.com/file/d/1JN7oSTRCwtGRH16DNRgLQN-ZOFebW3iN/view?usp=drive_link)

## Usage Notes
1. For running simulations:
   - Use the Docker environment with Python 2.7

2. For plotting CDF-CCT figures:
   - Switch to Python 3.x environment

## Predictive Model Training
The following command will read in saved simulation results (by default `results/process_result.txt`) and save the results plots to your designated folder (by default `modeling_results`)

```bash
python modeling.py
```

## Acknowledgments
This project builds upon the HPCC project by Alibaba. Original repository: [HPCC](https://github.com/alibaba-edu/High-Precision-Congestion-Control)
