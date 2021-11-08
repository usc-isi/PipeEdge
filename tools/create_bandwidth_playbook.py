import numpy as np
import argparse

host_addr_map = {"m0":"172.30.0.63", "m10": "172.30.0.65", "m20": "172.30.0.30", "m30": "172.30.0.68", "m40": "172.30.0.42", "n0":"172.30.0.23",\
"n10": "172.30.0.10", "n20": "172.30.0.17"}

def create_python_command(bandwidth, action):
    command =f"sudo tc qdisc {action} dev eth0 root tbf rate {bandwidth}mbit burst 32kbit latency 20ms" 
    print(command)
    return command

def create_command_shell(script, node_name, command):
    script.write(f"- hosts: {node_name}\n")
    script.write(f"  tasks:\n")
    script.write(f"    - name: add bandwidth limitation\n")
    script.write(f"      shell: {command}\n")
    script.write("\n")

def create_script(bandwidth_list):
    for i in range(len(bandwidth_list)):
        bw = bandwidth_list[i]
        add_m = f"bw_{bw}mbps_20ms_add_m.yml"
        add_n = f"bw_{bw}mbps_20ms_add_n.yml"
        del_m = f"bw_{bw}mbps_20ms_delete_m.yml"
        del_n = f"bw_{bw}mbps_20ms_delete_n.yml"
        s_add_m = open(add_m, "w")
        s_add_n = open(add_n, "w")
        s_del_m = open(del_m, "w")
        s_del_n = open(del_n, "w")
        create_command_shell(s_add_m, 'm', create_python_command(bw, "add"))
        create_command_shell(s_add_n, 'n', create_python_command(bw, "add"))
        create_command_shell(s_del_m, 'm', create_python_command(bw, "delete"))
        create_command_shell(s_del_n, 'n', create_python_command(bw, "delete"))
        s_add_m.close() 
        s_add_n.close() 
        s_del_m.close() 
        s_del_n.close() 



if __name__=="__main__":
    #########################################################
    #                 Check Enviroment Settings             #
    #########################################################
    parser = argparse.ArgumentParser(description="Create ansible playbook yml script")
    parser.add_argument("-bd", "--bandwidth", type=str, help="bandwidth")
    args = parser.parse_args()

    #########################################################
    #                 Configuration for Network             #
    #########################################################
    # *****  Define the World Size and partition Method ******#

    bandwidth= [i for i in args.bandwidth.split(',')]
    create_script(bandwidth)
    # ***********************  End  **************************#
    