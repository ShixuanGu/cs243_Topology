#!/usr/bin/env python2.7
import argparse
import os
import random

def get_comm_pattern(C):
    patterns = {0: "all_to_all", 1: "all_reduce", 2: "all_gather"}
    return patterns.get(C, "all_to_all")

def generate_mini_superpod(L=0, N=4, N3=2, N2=4, N1=8, W=100000000, 
                          S3="800Gbps", S2="800Gbps", S1="400Gbps", 
                          F3=0, F2=0, F1=0, C=0):
    random.seed(42)
    
    # Adjust network size based on failure flags
    if F3 > 0:
        # N3 = max(0, N3 - 1)
        N3 //= 2 
    if F2 > 0:
        # N2 = max(0, N2 - 1)
        N2 //= 2
    
    config = {
        'num_super_spine': N3,
        'num_spine': N2,
        'num_leaf': N1,
        'num_compute_nodes': N,
        'model_size': W,
        'super_spine_speed': S3,
        'spine_leaf_speed': S2,
        'leaf_compute_speed': S1,
        'super_spine_failure': F3,
        'spine_leaf_failure': F2,
        'leaf_compute_failure': F1,
        'topology_constant': L,
        'comm_type': C
    }
    
    total_leaf_switches = config['num_leaf']
    total_spine_switches = config['num_spine']
    total_super_spine_switches = config['num_super_spine'] if config['num_super_spine'] > 0 else 0
    
    total_switches = total_super_spine_switches + total_spine_switches + total_leaf_switches
    total_compute_nodes = config['num_compute_nodes']
    total_nodes = total_switches + total_compute_nodes
    
    if config['num_super_spine'] > 0:
        super_spine_to_spine_links = config['num_super_spine'] * total_spine_switches
        total_super_spine_links = super_spine_to_spine_links * 2
    else:
        total_super_spine_links = 0
    
    spine_to_leaf_links = config['num_spine'] * total_leaf_switches
    total_spine_leaf_links = spine_to_leaf_links * 2
    
    compute_node_connections_per_node = 1
    total_compute_leaf_links = total_compute_nodes * compute_node_connections_per_node * 2
    
    total_links = total_super_spine_links + total_spine_leaf_links + total_compute_leaf_links
    
    content = ["{0} {1} {2}".format(total_nodes, total_switches, total_links)]
    switch_ids = range(total_switches)
    content.append(" ".join(map(str, switch_ids)))
    
    links = []
    
    if config['num_super_spine'] > 0:
        spine_start = config['num_super_spine']
        for super_spine_id in xrange(config['num_super_spine']):
            for spine_id in xrange(spine_start, spine_start + config['num_spine']):
                links.append("{0} {1} {2} 0.001ms 0".format(
                    super_spine_id, spine_id, config['super_spine_speed']))
                links.append("{0} {1} {2} 0.001ms 0".format(
                    spine_id, super_spine_id, config['super_spine_speed']))
    else:
        spine_start = 0
    
    leaf_start = spine_start + config['num_spine']
    for spine_id in xrange(spine_start, spine_start + config['num_spine']):
        for leaf_id in xrange(leaf_start, leaf_start + total_leaf_switches):
            links.append("{0} {1} {2} 0.001ms 0".format(
                spine_id, leaf_id, config['spine_leaf_speed']))
            links.append("{0} {1} {2} 0.001ms 0".format(
                leaf_id, spine_id, config['spine_leaf_speed']))
            
    compute_node_start = total_switches
    compute_node_ids = range(compute_node_start, total_nodes)
    leaf_ids = range(leaf_start, leaf_start + total_leaf_switches)
    
    for idx, compute_node_id in enumerate(compute_node_ids):
        leaf = leaf_ids[idx % total_leaf_switches]
        links.append("{0} {1} {2} 0.001ms 0".format(
            leaf, compute_node_id, config['leaf_compute_speed']))
        links.append("{0} {1} {2} 0.001ms 0".format(
            compute_node_id, leaf, config['leaf_compute_speed']))
    
    content.extend(links)
    topology_content = "\n".join(content)
    
    gpus_per_server = config['num_compute_nodes'] // config['num_leaf']
    servers = {}
    for idx, compute_node_id in enumerate(compute_node_ids):
        server_id = idx // gpus_per_server
        if server_id not in servers:
            servers[server_id] = []
        servers[server_id].append(compute_node_id)
    
    comm_pattern = get_comm_pattern(C)
    compute_nodes = compute_node_ids
    num_compute_nodes = len(compute_nodes)
    flows = []
    
    if comm_pattern == "all_to_all":
        flows_list = []
        start_time = 2.0
        message_size = config['model_size']
        
        for src in compute_nodes:
            src_server = next(server_id for server_id, nodes in servers.items() if src in nodes)
            for dst in compute_nodes:
                dst_server = next(server_id for server_id, nodes in servers.items() if dst in nodes)
                if src != dst and src_server != dst_server:
                    flows_list.append("{0} {1} 3 100 {2} {3}".format(
                        src, dst, message_size, start_time))
        
        num_flows = len(flows_list)
        flows.append(str(num_flows))
        flows.extend(flows_list)
        
    elif comm_pattern == "all_gather":
        flows_list = []
        start_time = 2.0
        portion_size = config['model_size'] // num_compute_nodes
        
        for src in compute_nodes:
            src_server = next(server_id for server_id, nodes in servers.items() if src in nodes)
            for dst in compute_nodes:
                dst_server = next(server_id for server_id, nodes in servers.items() if dst in nodes)
                if src != dst and src_server != dst_server:
                    flows_list.append("{0} {1} 3 100 {2} {3}".format(
                        src, dst, portion_size, start_time))
        
        num_flows = len(flows_list)
        flows.append(str(num_flows))
        flows.extend(flows_list)
        
    elif comm_pattern == "all_reduce":
        flows_list = []
        gradient_size = config['model_size']
        
        start_time = 2.0
        reducer = compute_nodes[0]
        reducer_server = next(server_id for server_id, nodes in servers.items() if reducer in nodes)
        for src in compute_nodes:
            src_server = next(server_id for server_id, nodes in servers.items() if src in nodes)
            if src != reducer and src_server != reducer_server:
                flows_list.append("{0} {1} 3 100 {2} {3}".format(
                    src, reducer, gradient_size, start_time))
        
        start_time = 3.0
        for dst in compute_nodes:
            dst_server = next(server_id for server_id, nodes in servers.items() if dst in nodes)
            if dst != reducer and dst_server != reducer_server:
                flows_list.append("{0} {1} 3 100 {2} {3}".format(
                    reducer, dst, gradient_size, start_time))
        
        num_flows = len(flows_list)
        flows.append(str(num_flows))
        flows.extend(flows_list)
    
    flow_content = "\n".join(flows)
    
    trace_content = str(total_nodes) + "\n"
    trace_content += " ".join(map(str, range(total_nodes)))
    
    return topology_content, flow_content, trace_content, config, total_switches, compute_node_start, total_nodes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate network topology and communication patterns')
    
    parser.add_argument('-L', type=int, default=0, help='Topology constant (default: 0)')
    parser.add_argument('-N', type=int, default=4, help='Number of compute nodes (default: 4)')
    parser.add_argument('-N3', type=int, default=2, help='Number of super spine switches (default: 2, set 0 for 2-tier topology)')
    parser.add_argument('-N2', type=int, default=4, help='Number of spine switches (default: 4)')
    parser.add_argument('-N1', type=int, default=8, help='Number of leaf switches (default: 8)')
    parser.add_argument('-W', type=int, default=100000000, help='Model size in bytes (default: 100000000)')
    parser.add_argument('-S3', type=str, default="800Gbps", help='Super-Spine to Spine link speed (default: 800Gbps)')
    parser.add_argument('-S2', type=str, default="800Gbps", help='Spine-Leaf link speed (default: 800Gbps)')
    parser.add_argument('-S1', type=str, default="400Gbps", help='Leaf-Compute link speed (default: 400Gbps)')
    parser.add_argument('-F3', type=float, default=0, help='Super-Spine failure flag (default: 0)')
    parser.add_argument('-F2', type=float, default=0, help='Spine failure flag (default: 0)')
    parser.add_argument('-F1', type=float, default=0, help='Compute node failure flag (default: 0)')
    parser.add_argument('-C', type=int, default=0, help='Communication type (0:all-to-all, 1:all-reduce, 2:all-gather) (default: 0)')
    
    args = parser.parse_args()
    
    topology_content, flow_content, trace_content, config, total_switches, compute_node_start, total_nodes = \
        generate_mini_superpod(
            L=args.L, N=args.N, N3=args.N3, N2=args.N2, N1=args.N1, W=args.W,
            S3=args.S3, S2=args.S2, S1=args.S1, F3=args.F3, F2=args.F2, F1=args.F1, C=args.C
        )
    
    out_dir = "/workspace/hpcc/simulation/mix"
    with open(os.path.join(out_dir, "mini_superpod_topology.txt"), "w") as f:
        f.write(topology_content)
    with open(os.path.join(out_dir, "mini_superpod_flows.txt"), "w") as f:
        f.write(flow_content)
    with open(os.path.join(out_dir, "mini_superpod_trace.txt"), "w") as f:
        f.write(trace_content)
    
    print "\nGenerated files in", out_dir
    print "Configuration:"
    print "- Topology type:", "3-tier" if config['num_super_spine'] > 0 else "2-tier"
    print "- Topology constant:", config['topology_constant']
    print "- Number of compute nodes:", config['num_compute_nodes']
    if config['num_super_spine'] > 0:
        print "- Super spine switches: 0 to", config['num_super_spine'] - 1
        print "- Spine switches:", config['num_super_spine'], "to", config['num_super_spine'] + config['num_spine'] - 1
        print "- Leaf switches:", config['num_super_spine'] + config['num_spine'], "to", total_switches - 1
    else:
        print "- Spine switches: 0 to", config['num_spine'] - 1
        print "- Leaf switches:", config['num_spine'], "to", total_switches - 1
    print "- Compute nodes:", compute_node_start, "to", total_nodes - 1
    print "- Model size:", config['model_size'], "bytes"
    print "- Communication type:", config['comm_type'], "(", get_comm_pattern(config['comm_type']), ")"