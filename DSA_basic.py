class Graph:
  def __init__(self,num_vertices):
    self.v = num_vertices
    self.adj_matrix = [[0 for _ in range(self.v)] for _ in range(self.v)]
    # self.adj_matrix = np.zeros((self.v,self.v), dtype=int)
  def add_edge(self, u , v,w):
    self.adj_matrix[u][v]=w
    self.adj_matrix[v][u]=w

  def remove_edge(self,u,v):
    self.adj_matrix[u][v]=0
    self.adj_matrix[v][u]=0

  def __len__(self):
    return self.v

  def print_matrix(self):
    for row in self.adj_matrix:
      print(row)

g = Graph(5)
g.add_edge(0,1,34)
g.add_edge(0,4,22)
g.add_edge(1,2,9)

g.print_matrix() 



#adj_list representation of graph 
class Graph:
  def __init__(self):
    self.adj_list={}

  def add_vertex(self,v):
    if v not in self.adj_list:
      self.adj_list[v]=[]

  def add_edge(self,u,v):
    self.add_vertex(u)
    self.add_vertex(v)
    self.adj_list[u].append(v)
    self.adj_list[v].append(u)

  def remove_edge(self,u,v):
    if v in self.adj_list[u]:
      self.adj_list[u].remove(v)
    if u in self.adj_list[v]:
      self.adj_list[v].remove(u)

  def print_graph(self):
    for vertex in self.adj_list:
      print(vertex,":",self.adj_list[vertex])

g = Graph()
g.add_edge(0,1)
g.add_edge(0,4)
g.add_edge(1,2)

g.print_graph()






#dfs - iterative
def reverse_list(arr):
  reversed_arr = []
  for i in range(len(arr)-1,-1,-1):
    reversed_arr.append(arr[i])
  return reversed_arr
def dfs(graph,start):
  visited =set()
  stack = [start]

  while stack:
    node = stack.pop()
    if node not in visited:
      print(node,end=" ")
      visited.add(node)
      for neighbor in reverse_list(graph[node]):
        if neighbor not in visited:
          stack.append(neighbor)

graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

dfs(graph, 'A')




#DFS - recursive

def dfs(graph,start,visited=None):
  if visited is None:
    visited = set()

  if start not in visited:
    print(start,end=" ")
    visited.add(start)
    for neighbor in graph[start]:
      dfs(graph , neighbor ,visited)


graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

dfs(graph, 'A')




#BFS
from collections import deque
def bfs(graph,start):
  visited = set()
  queue = deque([start])

  while queue:
    node = queue.popleft()
    if node not in visited:
      print(node,end=" ")
      visited.add(node)
      for neighbor in graph[node]:
        if neighbor not in visited:
          queue.append(neighbor)
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

bfs(graph,"A")