
# coding: utf-8

# In[2]:


import bayes
import numpy
import networkx as nx
import random
import matplotlib.pyplot as plt
import sympy


# In[3]:


# generates a random directed acyclic graph with n nodes and e edges
def generateRandomBayesNet(n, e):
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i)
    for j in range(e):
        #print "."
        while(True):
            n1 = random.randint(0,n)
            n2 = random.randint(0,n)
            if(n1 == n2 or G.has_edge(n1,n2)): 
                continue
            G.add_edge(n1, n2, weight = random.random())
            if(nx.is_directed_acyclic_graph(G)):
                break
            else:
                G.remove_edge(n1,n2)
    return G


# In[4]:


def add_indirect_edges(G):
    Gp = G.copy()
    for i in range(Gp.number_of_nodes()):
        for j in range(Gp.number_of_nodes()):
            if(i!=j):
                if(nx.has_path(Gp, i, j)):
                    Gp.add_edge(i, j)
    return Gp


# In[5]:


# generates a random directed acyclic graph with n nodes and e edges
def generateRandomBayesNet_withPriors(n, e, npriors):
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i)
    for j in range(e):
        #print "."
        while(True):
            n1 = random.randint(0,n-1)
            n2 = random.randint(npriors,n-1)
            if(n1 == n2 or G.has_edge(n1,n2)): 
                continue
            G.add_edge(n1, n2, weight = random.random())
            if(nx.is_directed_acyclic_graph(G)):
                break
            else:
                G.remove_edge(n1,n2)
    return G


# In[6]:


# generates nsamples samples from bayesian net graph g
def generateSamples_reverse(G, nsamples): # no parents case
    values_large = []
    for i in range(nsamples):
        values = [0 for i in range(G.number_of_nodes())]
        for node in nx.topological_sort(G):
            influence = 0.0
            total = 0.001
            parents = G.in_edges(node)
            if(len(parents) == 0):
                if(random.random() > .5):
                    values[node] = 1
            else:
                for edge in parents:
                    w = G.get_edge_data(edge[0], edge[1])['weight']
                    total += w
                    if(values[edge[0]] == 1):
                        influence += w
                on = 1
                off = 0
                if(node%2 == 1):
                    on = 0
                    off = 1
                if(random.random() < influence/total):
                    values[node] = on
                else:
                    values[node] = off
        values_large.append(values)
        print values_large[:100]
    return values_large


# In[7]:


def generateSamples_modified(G, nsamples): # no parents case
    values_large = []
    for i in range(nsamples):
        values = [0 for i in range(G.number_of_nodes())]
        for node in nx.topological_sort(G):
            influence = 0.0
            total = 0.001
            parents = G.in_edges(node)
            if(len(parents) == 0):
                if(random.random() > .5):
                    values[node] = 1
            else:
                for edge in parents:
                    total += 1
                    if(values[edge[0]] == 1):
                        influence += 1
                on = 1
                off = 0
                if(node%2 == 1):
                    on = 0
                    off = 1
                if(random.random() < influence/total):
                    values[node] = on
                else:
                    values[node] = off
        values_large.append(values)
        print values_large[:100]
    return values_large


# In[8]:


# generates nsamples samples from bayesian net graph g
def generateSamples_alt(G, nsamples): # no parents case
    values_large = []
    for i in range(nsamples):
        values = [0 for i in range(G.number_of_nodes())]
        for node in nx.topological_sort(G):
            influence = 0.0
            total = 0.001
            parents = G.in_edges(node)
            if(len(parents) == 0):
                if(random.random() < .5):
                    values[node] = 1
            else:
                for edge in parents:
                    w = G.get_edge_data(edge[0], edge[1])['weight']
                    total += w
                    if(values[edge[0]] == 1):
                        influence += w
                if(random.random() < influence/total):
                    values[node] = 1
            values_large.append(values)
    print values_large[:100]
    return values_large


# In[9]:


def generateSamples_weightless(G, nsamples): # no parents case
    values_large = []
    for i in range(nsamples):
        values = [0 for i in range(G.number_of_nodes())]
        for node in nx.topological_sort(G):
            influence = 0.0
            total = 0.001
            parents = G.in_edges(node)
            if(len(parents) == 0):
                if(random.random() < .5):
                    values[node] = 1
            else:
                for edge in parents:
                    w = 1
                    total += w
                    if(values[edge[0]] == 1):
                        influence += w
                if(random.random() < influence/total):
                    values[node] = 1
            values_large.append(values)
    print values_large[:100]
    return values_large


# In[10]:


def generateSamples3(G, nsamples): # no parents case
    values_large = []
    for i in range(nsamples):
        values = [0 for i in range(G.number_of_nodes())]
        for node in nx.topological_sort(G):
            influence0 = 0.0
            influence1 = 0.0
            influence2 = 0.0
            total = 0.000
            parents = G.in_edges(node)
            if(len(parents) == 0):
                values[node] = numpy.random.randint(0,3)
            else:
                for edge in parents:
                    total += 1
                    if(values[edge[0]] == 0):
                        influence0 += 1
                    elif(values[edge[0]] == 1):
                        influence1 += 1
                    elif(values[edge[0]] == 2):
                         influence2 += 1
                values[node] = numpy.random.choice((0,1,2), p=[influence0/total, influence1/total,influence2/total])
        values_large.append(values)
    print values_large[:100]
    return values_large


# In[11]:


def evaluate_estimate(estimate, orig):
    false_positives = 0.0
    true_positives = 0.0
    for edge in estimate.edges():
        if orig.has_edge(edge[0], edge[1]):
            true_positives += 1
        else:
            false_positives +=1
    precision = true_positives/(true_positives + false_positives)
    recall = true_positives/(orig.number_of_edges())
    F1 = 2*precision*recall/(precision + recall)
    return precision, recall, F1


# In[12]:


def run_experiment(n_nodes, n_edges, n_samples, verbose = False):
    G = generateRandomBayesNet_withPriors(n_nodes, n_edges, 3)
    data = generateSamples3(G, n_samples)
    G2 = bayes.infer_bayesian_structure(data)
    precision, recall, F1 = evaluate_estimate(G2, G)
    if(verbose == True):
        print 'precision', precision
        print 'recall', recall
        print 'F1', F1
    return G, G2,precision, recall, F1, data


# In[13]:


def k_score_based_pruning(G, data, n):
    score = test_bayesian_fit(G, data)
    weights = []
    G_c = G.copy()
    for edge in G.edges():
        G_minus = G.copy()
        G_minus.remove_edge(edge[0], edge[1])
        fit_diff = score - test_bayesian_fit(G_minus, data)
        weights.append((edge, fit_diff))
        weights = sorted(weights, key = lambda x: x[1])
    for i in range(n):
        G_c.remove_edge(weights[i][0][0],weights[i][0][1])
    return G_c


# In[14]:


#smaller graphs
def deconvolve(G): 
    A = nx.adjacency_matrix(G)
    i = numpy.identity(G.number_of_nodes())
    L = numpy.linalg.matrix_power(i - A.A, -1)
    A2 = numpy.matmul(A.A, L)
    Gc = nx.convert_matrix.from_numpy_matrix(A2, create_using = nx.DiGraph)
    return Gc


# In[15]:


#much larger graphs
def eigendecomp_deconvolve(G):
    n = G.number_of_nodes()
    A = nx.adjacency_matrix(G)
    w, v = numpy.linalg.eig(A.A)
    dirs = numpy.zeros(n, dtype = complex)
    for i in range(len(w)):
        inv = 1/(w+1)
        d = w[i]*inv
        dirs[i] = sum(d)
    diag = numpy.zeros((n,n), dtype= complex)
    for i in range(len(dirs)):
        diag[i][i] = dirs[i]
    A_prime = numpy.zeros((n,n), dtype = complex)
    numpy.matmul(v, diag, A_prime)
    A_final = numpy.zeros((n,n), dtype = complex)
    numpy.matmul(A_prime, numpy.linalg.matrix_power(v, -1), A_final)
    return nx.convert_matrix.from_numpy_matrix(A_final, create_using = nx.DiGraph)


# In[43]:


def add_edge_weights(G, data):
    score = test_bayesian_fit(G, data)
    weights = {}
    for edge in G.edges():
        G_minus = G.copy()
        G_minus.remove_edge(edge[0], edge[1])
        fit_diff = score - test_bayesian_fit(G_minus, data)
        weights[edge] = fit_diff
    for edge in G.edges():
        G.add_edge(edge[0], edge[1], weight = weights[edge])
        


# In[48]:


def add_edge_weights_specified(G,weights):
    score = test_bayesian_fit(G, data)
    e = []
    for edge in G.edges():
        e.append(edge)
    for edge in e:
        G.add_edge(edge[0], edge[1], weight = weights[edge])


# In[16]:


def test_bayesian_fit(G, data):
    occurences = bayes.init_occurences(data)
    for node in G.nodes():
        occurences = bayes.update_occurences(G, data, node, occurences)
    return bayes.get_score(G, occurences)
