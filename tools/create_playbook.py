import numpy as np
import argparse

host_addr_map = {"m0":"172.30.0.63", "m10": "172.30.0.65", "m20": "172.30.0.30", "m30": "172.30.0.68", "m40": "172.30.0.42", "n0":"172.30.0.23",\
"n10": "172.30.0.10", "n20": "172.30.0.17"}

def create_python_command(file_name, rank, world_size, partition, host_addr, socket_ifname, model_name, batch_size, num_batch, work_threads, splits):
    command = f"python3 {file_name} {rank} {world_size} -m {model_name} -pt {partition} --addr {host_addr} -s {socket_ifname} -b {batch_size} -n {num_batch} -w {work_threads} -sp {splits}"
    print(command)
    return command

def create_shell_command(script, node_name, command, write_async=True, task_name="runtime"):
    script.write(f"- hosts: {node_name}\n")
    script.write(f"  tasks:\n")
    script.write(f"    - name: {task_name}\n")
    script.write(f"      shell: {command}\n")
    if write_async:
        script.write("      async: 10000\n")
        script.write("      poll: 0\n")
    script.write("\n")

def create_script(script_name, node_list, file_name, world_size, partition, host, socket_ifname, model_name, batch_size, num_batch, work_threads, splits):
    script = open(script_name, "w")
    host_addr = host_addr_map[host]
    write_async = True
    rank = 0
    command = create_python_command(file_name, rank, world_size, partition, host_addr, socket_ifname, model_name, batch_size, num_batch, work_threads, splits)
    script.write(f"# {command} \n")
    for node in node_list:
        rank += 1
        command =  create_python_command(file_name, rank, world_size, partition, host_addr, socket_ifname, model_name, batch_size, num_batch, work_threads, splits)
        if node == node_list[-2]:
            write_async = False
        if node == node_list[-1]:
            script.write(f"# {command} \n")
            break

        create_shell_command(script, node, command, write_async)
    script.close()



if __name__=="__main__":
    #########################################################
    #                 Check Enviroment Settings             #
    #########################################################
    parser = argparse.ArgumentParser(description="Create ansible playbook yml script")
    parser.add_argument("-wz", "--world-size", type=int, help="the world size (the number of nodes)")
    parser.add_argument("-f", "--file-name", type=str, default="runtime.py", help="the python runtime file name")
    parser.add_argument("-m", "--model-name", type=str, default="google/vit-base-patch16-224", help="the neural network model for loading")
    parser.add_argument("-pt", "--partition", type=str, default="1,48", help="the partition method")
    parser.add_argument("-ht", "--host", type=str, default="n0", help="the rank 0 node name")
    parser.add_argument("-s", "--socket-ifname", type=str, default="eth0", help="socket iframe name, use [ifconfig | ipaddress] to check")
    parser.add_argument("-n", "--num-batches", default=1, type=int, help="total number of batches")
    parser.add_argument("-b", "--batch-size", default=64, type=int, help="batch size")
    parser.add_argument("-w", "--worker-threads", default=18, type=int, help="the number of worker threads for the communication backend")
    parser.add_argument("-sp", "--splits", default="8", help="the list of microbatch size")
    parser.add_argument("-nz","--nodes", type=str,help="selected nodes")
    parser.add_argument("-wt", "--with-out-first-last",  action="store_true", help="without the first and last node in the script")
    parser.add_argument("-sn", "--script-name", default="playbook.yml", type=str, help="script name")
    args = parser.parse_args()

    #########################################################
    #                 Configuration for Network             #
    #########################################################
    # *****  Define the World Size and partition Method ******#
    world_size = args.world_size
    file_name = args.file_name
    partition =  args.partition
    host_name = args.host
    socket_ifname = args.socket_ifname
    num_batches = args.num_batches
    batch_size = args.batch_size
    num_worker_threads = args.worker_threads
    model_name= args.model_name
    num_split = args.splits
    script_name = args.script_name
    nodes = [i for i in args.nodes.split(',')]
    without_first_last = args.with_out_first_last
    print(f"world size : {world_size}, file name : {file_name}, model name : {model_name}, partition: {partition}, host name : {host_name} ")
    print(f"socket_ifname : {socket_ifname}, name batches : {num_batches}, batch size : {batch_size}, work threads : {num_worker_threads}")
    print(f"nodes : {nodes}")
    print(f"without_first_last_node {without_first_last}")

    create_script(script_name, nodes, file_name, world_size, partition, host_name, socket_ifname, model_name, batch_size, num_batches, num_worker_threads, num_split)
    # ***********************  End  **************************#
    