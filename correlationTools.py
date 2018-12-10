
# coding: utf-8

# In[1]:


import bayes
import networkx as nx
import numpy


# In[2]:


import StructureLearn


# In[3]:


def calculate_correlation_matrix(data):
    n = len(data[0])
    M = numpy.zeros((n,n))
    for k in range(len(data)):
        if(k/100 == 1):
            print "."
        for i in range(n):
            for j in range(n):
                if i!= j:
                    if(data[k][i] == data[k][j]):
                        M[i][j] += 1
                    else:
                        M[i][j] -= 1
    M *= 1.0/len(data)
    return M


# In[4]:


G = StructureLearn.generateRandomBayesNet_withPriors(25, 80, 5)


# In[5]:


def genGraph(M,threshold):
    G = nx.Graph()
    for i in range(len(M[0])):
        G.add_node(i)
    for i in range(len(M[0])):
        for j in range(len(M[0])):
            if(M[i][j] > threshold):
                G.add_edge(i,j, weight = M[i][j])
    return G


# In[6]:


data = StructureLearn.generateSamples_alt(G, 5000)


# In[7]:


M2 = calculate_correlation_matrix(data)


# In[33]:


M


# In[8]:


M2


# In[9]:


import matplotlib.pyplot as plt


# In[10]:


plt.plot(M2)


# In[14]:


precs = []
recalls = []
F1s = []
for i in range(10):
    G2 = genGraph(M2, .1 + i*.1)
    precision, recall, F1 = StructureLearn.evaluate_estimate(G2, G, verbose = True)
    precs.append(precision)
    recalls.append(recall)
    F1s.append(F1)


# In[15]:


plt.plot(precs)


# In[16]:


plt.show()


# In[17]:


plt.plot(recalls)


# In[18]:


plt.show()


# In[19]:


plt.plot(F1s)


# In[20]:


plt.show()


# In[24]:


def Mat_deconvolve(M):
    i = numpy.identity(len(M[0]))
    L = numpy.linalg.matrix_power(i - M, -1)
    A2 = numpy.matmul(M, L)
    return A2


# In[30]:


M2


# In[28]:


Mat_deconvolve(M2)


# In[31]:


Dprecs = []
Drecalls = []
DF1s = []
M3 = Mat_deconvolve(M2)
for i in range(10):
    G2 = genGraph(M3, .1 + i*.1)
    precision, recall, F1 = StructureLearn.evaluate_estimate(G2, G, verbose = True)
    Dprecs.append(precision)
    Drecalls.append(recall)
    DF1s.append(F1)


# In[32]:


plt.plot(Dprecs)


# In[33]:


plt.show()


# In[34]:


plt.plot(Drecalls)


# In[35]:


plt.show()


# In[36]:


plt.plot(F1s)


# In[37]:


plt.show()


# In[ ]:




