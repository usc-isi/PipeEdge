from collections import defaultdict
import sys
class Graph:
 
    # Constructor
    def __init__(self, initial_dict = None):
        self.graph = defaultdict(list)
        if initial_dict:
            self.graph = initial_dict

    # function to add an edge to graph
    def addEdge(self,u,v):
        self.graph[u].append(v)

    def print_graph(self):
        print(self.graph)
 
    # Function to print a BFS of graph
    def statistic_microbatch_size(self, s): 
        visited = defaultdict(list)
        # Mark all the vertices as not visited
        for _, key in enumerate(self.graph):
            visited[key] = False

        # Create a queue for BFS
        queue = []
        node_microbatch_size = defaultdict(list)
        queue.append(s)
        visited[s] = True
        stage = 0
 
        while queue:
            # print(f"s is {s}")
            count = len(queue)
            while count > 0:
                s = queue.pop(0)
                for i in self.graph[s]:
                    node = i[0]
                    if visited[node] == False:
                        queue.append(node)
                        visited[node] = True
                    if node in node_microbatch_size:
                        node_microbatch_size[node] += i[1]
                    else:
                        node_microbatch_size[node] = i[1]
                count -= 1
            stage += 1
        return node_microbatch_size
                

