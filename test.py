
from PIL import Image
from math import exp
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


sigma = 100

im = Image.open('tower.png').convert('L') #convert to grayscale
plt.imshow(im.convert('RGB'))
plt.show()

h = im.height
w = im.width

#grayscale values; list of size h*w , where each values is integer in range [0,255]
intensity = list(im.getdata()) 

def idx(i,j):
    '''unique index such that of pixel at position (i,j), i =1,...,h; j = 1,...,w
       idx(1,1) = 1; idx(1,w) = w; idx(h,w) = h*w'''
    return w*(i-1) + j

#percentiles
prctl25 = np.percentile(intensity,25)
prctl75 = np.percentile(intensity,75)


#list of coordinates
coords = [(i,j) for i in range(1,h+1) for j in range(1,w+1)] 

#list of nodes
nodes = [idx(i,j) for i,j in coords]
#nodes_min_st = nodes
s = 0 # source node
t = h*w+1 # destination node with index h*w+1
nodes.append(s)
nodes.append(t)

def get_weight(n1,n2): #weight of edge connecting n1 and n2 
    if n1 == s:
        diff = intensity[n2-1] - prctl75
    elif n2 == t:
        diff = intensity[n1-1] - prctl25
    else:
        diff = intensity[n1-1] - intensity[n2-1] 
        
    return exp(-diff**2/sigma)


# neighboring arcs dictionary 
ngbrs = {idx(i,j):[idx(i2,j2) for (i2,j2) in [(i,j-1),(i,j+1),(i-1,j),(i+1,j)]
                    if 1 <= i2 <= h and 1 <= j2 <= w]
          for (i,j) in coords}

for (i,j) in coords: #s and t are neighbors of all other nodes
    ngbrs[idx(i,j)].append(t)
    
ngbrs[s] = [idx(i,j) for (i,j) in coords] # all nodes are neighbors of s

# an edge from node to every neighbor of that node
edges = [(n1,n2) for n1 in nodes[:-1] for n2 in ngbrs[n1]] 


# Create the Graph
G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

#Extract the incidence matrix
A = nx.incidence_matrix(G, oriented=True )

#PRIMAL PROBLEM
m = gp.Model("lp")
m.Params.LogToConsole = 0
m.Params.Method = 0
f = m.addMVar(shape=G.number_of_edges(), vtype=GRB.CONTINUOUS, lb=0, name="")
b = np.zeros(G.number_of_nodes()-2)
m.addConstr(A[:t-1,:]@f==b)

cap = np.zeros(G.number_of_edges())
r = np.zeros(G.number_of_edges())
for i in range(G.number_of_edges()):
    cap[i] = get_weight(edges[i][0],edges[i][1])
    if (edges[i][0] == 0):
        r[i] = 1
    
m.addConstr(f<=cap)
obj = r@f
m.setObjective(obj, GRB.MAXIMIZE)
m.optimize()
flows = m.getAttr("X", m.getVars())
print("Primal objective: ", m.getObjective().getValue())

print(flows)

#DUAL PROBLEM
mm = gp.Model("lp")
mm.Params.LogToConsole = 0
mm.Params.Method = 0
z = mm.addMVar(shape=G.number_of_edges(), vtype=GRB.CONTINUOUS, lb=0, name="z")
y = mm.addMVar(shape=G.number_of_nodes()-2, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="y")
obj = cap@z
mm.addConstr(r<=z+A[:t-1,:].transpose()@y)
mm.setObjective(obj, GRB.MINIMIZE)
mm.optimize()
print("Dual objective: ", mm.getObjective().getValue())
dual_vars = mm.getAttr("X", mm.getVars())

#Check complementary slackness conditions
cuts = dual_vars[:G.number_of_edges()]
y = dual_vars[G.number_of_edges():]
cminflows = cap-flows
print('Complementary slackness: ', cminflows@cuts)
Aty_plus_z_min_r = A[:t-1,:].transpose()@y + cuts - r
print('Complementary slackness: ', Aty_plus_z_min_r@flows)


#Let's draw the foreground image!
for i in range(G.number_of_edges()):
    intensity[edges[i][0]-1] = im.getdata()[edges[i][0]-1]
    if (cuts[i] <= 0.01):
        intensity[edges[i][0]-1] = 255

arr = np.reshape(intensity,(h,w)).astype('uint8')
imout =  Image.fromarray(arr).convert("RGB")
imout.save('new.png')



