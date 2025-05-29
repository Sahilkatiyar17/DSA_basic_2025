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





def detect_cycle(node,parent,adj_list,visited):
  visited[node]=1
  for neighbor in adj_list[node]:
    if parent == neighbor:
      continue
    if visited[neighbor]==1:
      return True
    if delect_Cycle(neighbor,node,adj_list,visited):
      return True
  return False


graph = {
    0: [1],
    1: [0,2,4],
    2: [1,3],
    3: [2,4],
    4: [1,3]

}

visited = [0]*len(graph)

if delect_Cycle(0,-1,graph,visited) == True:
  print("Cycle is present")
else:
  print("Cycle is not present")







from collections import deque

def delect_Cycle(start, parent, adj_list, visited):
    visited[start] = 1
    q = deque()
    q.append((start, parent))

    while q:
        node, parent = q.popleft()
        for neighbor in adj_list[node]:
            if neighbor == parent:
                continue
            if visited[neighbor] == 1:
                return True  # Cycle detected
            visited[neighbor] = 1
            q.append((neighbor, node))

    return False

graph = {
    0: [1],
    1: [0, 2, 4],
    2: [1, 3],
    3: [2, 4],
    4: [1, 3]
}

visited = [0] * len(graph)

if delect_Cycle(0, -1, graph, visited):
    print("Cycle is present")
else:
    print("Cycle is not present")











# topological sort using DFS
def dfs(node,graph,visited,stack):
  visited[node]=1
  for neighbor in graph[node]:
    if visited[neighbor]==0:
      dfs(neighbor,graph,visited,stack)
  stack.append(node)

def main(graph):
  visited = {}
  for node in graph:                               # DICTIONARY
    visited[node]=0
  stack = []
  for i in range(len(visited)):
    if visited[i]==0:
      dfs(i,graph,visited,stack)
  return stack[::-1]

graph = {
    0: [1],
    1: [2],
    2: [3],
    3: [],
    4: [5],
    5: [0]
}

print(main(graph))





#topological  sort using BFS
from collections import deque
def topological_bfs(graph):
  ans = []

  # Indeg mein store kiya values
  """Indeg = [0]*len(graph)
  for node in graph:                        # ARRAY
    for neighbor in graph[node]:
      Indeg[neighbor]+=1"""
  Indeg = {node:0 for node in graph}
  for node in graph:                        # DICTIONARY
    for neighbor in graph[node]:
      Indeg[neighbor]+=1

  #bfs 
  queue = deque()
  for i in range(len(Indeg)):
    if Indeg[i]==0:
      queue.append(i)

  while queue:
    node = queue.popleft()
    ans.append(node)

    for neighbor in graph[node]:
      Indeg[neighbor]-=1
      if Indeg[neighbor]==0:
        queue.append(neighbor)
  return ans

g = {
    0:[1,2],
    1:[3,4],
    2:[]
    ,3:[4]
    ,4:[]
    ,5:[6,4]
    ,6:[3]
}


print(topological_bfs(g))








# date - 28/5       #dfs
def detect_c_direct(node,graph,path):
  path[node]=1
  for n in graph[node]:
    if path[n]==1:
      return True
    if detect_c_direct(n,graph,path):
      return True
  path[node]=0
  return False


g = {
    0:[1],
    1:[2],
    2:[3,5,7],
    3:[4],
    4:[6],
    5:[4],
    6:[],
    7:[8],
    8:[]
}

gr = {
    0:[1],
    1:[2],
    2:[3,7],
    3:[4],
    4:[5,6],
    5:[2],
    6:[],
    7:[8],
    8:[]
}
path=[0]*len(g)
a=detect_c_direct(0,gr,path)
print(a)



# date - 28/5    #bfs 
def detect_c_directed(graph):
  indeg=[0]*len(graph)

  for i in range(len(graph)):
    for j in graph[i]:
      indeg[j]+=1

  queue=[]
  for i in range(len(graph)):
    if indeg[i]==0:
      queue.append(i)
  
  cnt=0

  while queue:
    node = queue.pop(0)
    for n in graph[node]:
      indeg[n]-=1 
      if indeg[n]==0:
        queue.append(n)
    cnt+=1
  if cnt==len(graph):
    return False
  else:
    return True
gr = {
    0:[1],
    1:[2],
    2:[3,7],
    3:[4],
    4:[5,6],
    5:[2],
    6:[],
    7:[8],
    8:[]
}

g = {
    0:[1],
    1:[2],
    2:[3,5,7],
    3:[4],
    4:[6],
    5:[4],
    6:[],
    7:[8],
    8:[]
}
print(detect_c_directed(g))









from collections import deque
def bfs(start,graph):
    queue = deque([start])

    color = [-1]*len(graph)
    color[start]=0

    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if color[neighbor]==-1:
                if  color[node]==0:
                    color[neighbor]=1
                else:
                    color[neighbor]=0
                queue.append(neighbor)
            else:
                if color[node]==color[neighbor]:
                    return False
    return True


g = {
    0:[1],
    1:[0,2,3],
    2:[1],
    3:[1,4,5,8],
    4:[3],
    5:[3,6],
    6:[5,7],
    7:[6,8],
    8:[7,3],
}


gr = {
    0: [3, 4],
    1: [3, 5],
    2: [4, 5],
    3: [0, 1],
    4: [0, 2],
    5: [1, 2]
}
print(bfs(0,gr))






def dfs(node,graph,color):
  for i in graph[node]:
    if color[i]==-1:
      color[i]=(color[node]+1)%2
      if not dfs(i,graph,color):
        return False
    else:
      if color[i]==color[node]:
        return False
  return True

def main(graph):
  color = [-1]*len(graph)
  for i in range(len(graph)):
    if color[i]==-1:
      color[i]=0
      if not dfs(i,graph,color):
        return False
  return True


  g = {
    0:[1],
    1:[0,2,3],
    2:[1],
    3:[1,4,5,8],
    4:[3],
    5:[3,6],
    6:[5,7],
    7:[6,8],
    8:[7,3],
}
gr = {
    0: [3, 4],
    1: [3, 5],
    2: [4, 5],
    3: [0, 1],
    4: [0, 2],
    5: [1, 2]
}
print(main(g))
print(main(gr))