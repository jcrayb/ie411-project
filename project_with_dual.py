from PIL import Image
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import networkx as nx
import scipy as sp
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


im = Image.open('tower.png').convert('L') #convert to grayscale

im.save('new_img.png')
tic = time.time()


s = 0 

real_source = 50
#grayscale values; list of size h*w , where each values is integer in range [0,255]

iterations = 25


def idx(i,j):
    return w*(i-1) + j

for z in range(iterations):
    def get_weight(n1,n2): #weight of edge connecting n1 and n2 
        if n1== s :
            diff = 0
        elif n2 == t:
            diff = 0
        else:
            diff = np.abs(intensity[n1-1] - intensity[n2-1]) 
        return diff

    im = Image.open('new_img.png').convert('L')
    h = im.height
    w = im.width
    t = h*w+1
    intensity = list(im.getdata())
    coords = [(i,j) for i in range(1,h+1) for j in range(1,w+1)]

    nodes = [0]
    nodes += [idx(i, j) for i, j in coords]

    nodes += [h*w+1]
    #print(nodes)





    ngbrs = {idx(i,j) :[idx(i2,j2) for (i2,j2) in [(i+1,j-1),(i+1,j),(i+1,j+1)]
            if 1 <= i2 <= h and 1 <= j2 <= w]
            for (i,j) in coords}



    ngbrs[0] = [real_source]

    for j in range(1, w+1):
        ngbrs[idx(h, j)].append(h*w+1)


    edges = [(n1,n2) for n1 in nodes[:-1] for n2 in ngbrs[n1]]
    #print(nodes)
    #print(edges)

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    A = nx.incidence_matrix(G, oriented=True)
    #print(A)
    b = np.zeros(G.number_of_nodes())

    b[0] = -1
    b[-1] = 1

    #print(b)

    r = np.zeros(G.number_of_edges())

    for i in range(G.number_of_edges()):
        
        r[i] = get_weight(edges[i][0],edges[i][1])
        #print(edges[i], r[i])

    #print(r)

    m = gp.Model("lp")
    m.Params.LogToConsole = 0
    m.Params.Method = 0
    f = m.addMVar(shape=G.number_of_edges(), vtype=GRB.CONTINUOUS, lb=0, name="")

    m.addConstr(A@f==b)
    obj = r@f
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()
    flows = m.getAttr("X", m.getVars())
    print("Primal objective: ", m.getObjective().getValue())
    #print(flows)
    
    #DUAL
    mm = gp.Model("lp")
    mm.Params.LogToConsole = 0
    mm.Params.Method = 0
    y = mm.addMVar(shape=G.number_of_nodes(), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="y")
    obj = b@y
    mm.addConstr(r>=A.transpose()@y)
    mm.setObjective(obj, GRB.MAXIMIZE)
    mm.optimize()
    print("Dual objective: ", mm.getObjective().getValue())
    dual_vars = mm.getAttr("X", mm.getVars())
    
    #COMPLEMENTARY SLACKNESS
    
    img = np.array(im)

    flowing = []
    new_img = []
    flows = flows[1:]
    for i in range(len(flows)):
        if flows[i]==1:
            endpoint = edges[i][0]
            coord = (endpoint//w, endpoint%w)
            flowing += [coord]
            #print(coord)
            new_img += [np.delete(img[coord[0]], coord[1]-1)]

    img = Image.fromarray(np.array(new_img))
    img.save('new_img.png')
    img.show()
toc = time.time()
print ('elapsed ', toc - tic)
 