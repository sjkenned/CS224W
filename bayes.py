import networkx as nx
import pandas
import sys
import numpy
from scipy.special import loggamma
import copy
import random
import matplotlib.pyplot as plt

def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))

def compute(infile, outfile, record = True):
    idx2names = {}
    var_list = process_infile(infile, idx2names)
    G = K2_search(var_list)
    nx.draw(G, with_labels = True)
    plt.show()
    if(record):
	write_gph(G, idx2names, outfile)
    return G

def infer_bayesian_structure(var_list):
    G = K2_search(var_list)
    return G

def process_infile(infile, idx2names):
    values = []
    with open(infile) as f:
        l1 = f.readline().strip().split(",")
        print(len(l1))
        for i in range(len(l1)):
            idx2names[i] = l1[i]
        for line in f:
            values.append(line.strip().split(","))
    return values

def process_infile_simple(infile):
    values = []
    with open(infile) as f:
        for line in f:
            values.append(line.strip().split(","))
    return values

def K2_search(var_list):
    G = nx.DiGraph()
    for i in range(len(var_list[0])):
        G.add_node(i)
    occurences = init_occurences(var_list)
    for i in range(len(var_list[0])):
        update_occurences(G, var_list, i, occurences)
    nodes = range(len(var_list[0]))
    for i in nodes:
        #print(".")
        while(True):
            s = get_score(G, occurences)
            neighbors = get_K2_neighbors(G, i)
            max_neighbor_score = float('-inf')
            max_neighbor = nx.DiGraph()
            max_occurences_new = occurences[:]
            if(len(neighbors)==0):
                break
            for neighbor in neighbors:
                occurences_new = update_occurences(neighbor, var_list, i, occurences[:])
                nscore = get_score(neighbor, occurences_new)
                if(nscore > max_neighbor_score):
                    max_neighbor = neighbor
                    max_neighbor_score = nscore
                    max_occurences_new = occurences_new[:]
            if(max_neighbor_score < s):
                break
            else:
                G = max_neighbor
                occurences = max_occurences_new
    return G     

def get_score(G, occurences): 
    if(len(G.edges())==0):
        return float('-inf')
    score = 0.0
    for i in range(len(occurences)):
        d = occurences[i]
        for key in d.keys():
            m_ij0 = 0.0
            a_ij0 = 0.0
            d2 = occurences[i][key]
            if(len(d2.items())>0):
                for k,v in d2.items():
                    score += loggamma(1 + v) 
                    m_ij0 += v
                    a_ij0 += 1
                score += (loggamma(a_ij0) - loggamma(a_ij0 + m_ij0))
    return score

def get_K2_neighbors(G, i): 
    legal_neighbors = []
    for j in range(len(G.nodes())):
        if j!= i and (j, i) not in G.edges():
            A = copy.deepcopy(G)
            A.add_edge(j, i)
            if(isAcyclic(A)):
                legal_neighbors.append(A)
    return legal_neighbors


def isAcyclic(G):
    try:
        nx.find_cycle(G)
        return False
    except:
        return True

def init_occurences(var_list): 
    occurences = []
    for i in range(len(var_list[1])):
        d = {}
        d[tuple()] = {}
        occurences.append(d)
    return occurences

def update_occurences(G, var_list, i, occurences): 
    new_i = {}
    parents = [] 
    for a in G.edges():
        if(a[1] == i):
            parents.append(a[0])
    for row in var_list:
        parent_values = [row[a] for a in parents]
        if(tuple(parent_values) in new_i):
            if(row[i] in new_i[tuple(parent_values)]):
                new_i[tuple(parent_values)][row[i]] = new_i[tuple(parent_values)][row[i]] + 1
            else:
                new_i[tuple(parent_values)][row[i]] = 1
        else:
            new_i[tuple(parent_values)] = {}
            new_i[tuple(parent_values)][row[i]] = 1 
    occurences[i] = new_i
    return occurences

