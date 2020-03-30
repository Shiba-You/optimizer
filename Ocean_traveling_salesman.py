#!/usr/bin/env python
# coding: utf-8

# ## 巡回セールスマン問題

# In[ ]:


import networkx as nx
import matplotlib.pyplot as plt
import blueqat.opt as wq
import numpy as np
from dimod import *
from dwave.cloud import Client
from dwave_qbsolv import QBSolv
client = Client.from_config(token='DEV-92b24dbe3f6a16cec0d1069a8e52445c5967491a')
from dwave.system.samplers import *
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import neal
import itertools
import time
from sklearn import preprocessing
import hybrid
client.get_solvers()


# In[ ]:


#入力セル
print("Number of cities:")
pu_cities_size = int(input())
print("The cost of constraint:")
pu_B = float(input()) #0.2

originList = [i for i in range(pu_cities_size)]
ObjectList = list(itertools.combinations(originList, 2))
cost_matrix = list(itertools.combinations(originList, 2))
for i in range(len(cost_matrix)):
    if abs(cost_matrix[i][0] - cost_matrix[i][1]):
        cost_matrix[i] += (cost_matrix[i][0] + cost_matrix[i][1], )


# In[ ]:


#グラフ描写
options = {'node_color': '#efefef','node_size': 1200,'with_labels':'True'}
G = nx.Graph()
G.add_nodes_from(nx.path_graph(pu_cities_size))
G.add_edges_from(ObjectList)
nx.draw(G, **options)


# In[ ]:


#コスト関数：第１項
def get_traveling_qubo(cities_size):
    qubo_size = cities_size*cities_size
    traveling_qubo = np.zeros((qubo_size, qubo_size))
    indices = [(u, v, i, j) for u in range(cities_size) for v in range(cities_size) for i in range(cities_size) for j in range(cities_size)]
    for u, v, i, j in indices:
        ui = u * cities_size + i
        vj = v * cities_size + j
        if ui > vj:
            continue
        if ui == vj:
            traveling_qubo[ui][vj] -= 2
        if u == v and i != j:
            traveling_qubo[ui][vj] += 2
        if u < v and i == j:
            traveling_qubo[ui][vj] += 2
    return traveling_qubo, qubo_size, cities_size
# traveling_qubo, qubo_size, cities_size = get_traveling_qubo(pu_cities_size)
# print(traveling_qubo)


# In[ ]:


#コスト関数：第２項
def get_traveling_cost_qubo(qubo_size, cities_size, cost_matrix):
    traveling_cost_qubo = np.zeros((qubo_size, qubo_size))
    indices = [(u, v, i, j) for u in range(cities_size) for v in range(cities_size) for i in range(cities_size) for j in range(cities_size)]
    for u, v, i, j in indices:
        ui = u * cities_size + i
        vj = v * cities_size + j
        k = abs(i - j)
        if ui > vj:
            continue
        if (k ==1 or k == cities_size-1) and u < v:
            for r in range(len(cost_matrix)):
                if cost_matrix[r][0] == u and cost_matrix[r][1] == v:
                    traveling_cost_qubo[ui][vj] += cost_matrix[r][2]
    return traveling_cost_qubo
# traveling_cost_qubo = get_traveling_cost_qubo(qubo_size, cities_size, cost_matrix)
# print(traveling_cost_qubo)

samplerは低エネルギーのものからサンプリングするプロセス。
「SimulatedAnnealingSampler」で呼ぶことができ、「sample_qubo」と「sample_ising」に対応している。
# In[ ]:


def get_ocean_qubo_re(traveling_cost_qubo, traveling_qubo, B, cities_size):
    with Client.from_config() as client:
        solver = client.get_solver()
        p_qubo = traveling_qubo + traveling_cost_qubo * B
        keys = []
        for i in range(len(p_qubo[0])):
            for j in range(len(p_qubo[1])):
                keys.append((i, j))
        Q = {key: val for key, val in zip(keys, p_qubo.reshape(-1))}
#         sampler = neal.SimulatedAnnealingSampler()
#         print(Q)
        computation = solver.sample_qubo(Q,num_reads=10)
        for sample in computation.samples:
            result = sample
            print(result)
    return result


# In[ ]:


#観測
def get_travelingsalesman_qubo(traveling_cost_qubo, traveling_qubo, B, cities_size):
    a = wq.opt()
    a.qubo = traveling_qubo + traveling_cost_qubo * B
#     idx = np.unravel_index(np.argmax(a.qubo), a.qubo.shape)
#     a.qubo /= a.qubo[idx]
    print(a.qubo)
    answer = a.sa()
    result = np.zeros([cities_size,cities_size])
    k = 0
    for i in range(int(len(answer)/cities_size)):
        result[i] += answer[k : k+cities_size]
        k += cities_size
    print(answer)
    print(result)
#     a.plot()
    return answer, a.qubo


# In[ ]:


#ハミルトニアンの検証
def calculate_H_f(q, cities_size):
    H_f = 0
    for v in range(cities_size):
        sum_x = 0
        for i in range(cities_size):
            index = v * cities_size + i
            sum_x += q[index]
        H_f += (1 - sum_x) ** 2
    print(H_f)
    return H_f

def calculate_H_s(q, cities_size):
    H_s = 0
    for i in range(cities_size):
        sum_x = 0
        for v in range(cities_size):
            index = v * cities_size + i
            sum_x += q[index]
        H_s += (1 - sum_x) ** 2
    print(H_s)
    return H_s

def calculate_H_t(q, cities_size, cost_matrix):
    H_t = 0
    indices = [(u, v, i, j) for u in range(cities_size) for v in range(cities_size) for i in range(cities_size) for j in range(cities_size)]
    for u, v, i, j in indices:
        ui = u * cities_size + i
        vj = v * cities_size + j
        k = abs(i - j)
        if ui >= vj:
            continue
        if k == 1:
            if q[ui] == 1 and q[vj] == 1:
                for k in range(len(cost_matrix)):
                    if cost_matrix[k][0] == u and cost_matrix[k][1] == v:
                        H_t += cost_matrix[k][2]
    print(H_t)
    return H_t

def calculate_H(q, cities_size, cost_matrix, B):
    print("hamiltonian_f =")
    H_f = calculate_H_f(q, cities_size)
    print("hamiltonian_s =")
    H_s = calculate_H_s(q, cities_size)
    print("hamiltonian_t =")
    H_t = calculate_H_t(q, cities_size, cost_matrix)
    H =  H_f + H_s + H_t * B
    print("hamiltonian =")
    print(H)
    return H
# calculate_H(q, cities_size, cost_matrix, pu_B)


# In[ ]:


# 入力セル
print("Number of cities:")
pu_cities_size = int(input())
print("The cost of constraint:")
pu_B = float(input()) #0.2 # 10: 0.0001

originList = [i for i in range(pu_cities_size)]
ObjectList = list(itertools.combinations(originList, 2))
cost_matrix = list(itertools.combinations(originList, 2))
cost_len = len(cost_matrix)

# for i in range(len(cost_matrix)):
#     cost_matrix[i] += (cost_matrix[i][0] + cost_matrix[i][1], )
# print(cost_matrix)


for i in range(cost_len):
    if abs(cost_matrix[i][0] - cost_matrix[i][1])==3:
        cost_matrix[i] += (2, )
    elif abs(cost_matrix[i][0] - cost_matrix[i][1])==7:
        cost_matrix[i] += (2, )
    else:
        cost_matrix[i] += (10000, )

# for i in range(len(origin_cost_matrix)):
#     if abs(origin_cost_matrix[i][0] - origin_cost_matrix[i][1])==3:

# cost_matrix = []
# for i in range(len(origin_cost_matrix)):
#     if origin_cost_matrix[i][2]!=0:
#         cost_matrix.append(origin_cost_matrix[i])


# In[ ]:


#出力セル
traveling_qubo, qubo_size, cities_size = get_traveling_qubo(pu_cities_size)
traveling_cost_qubo = get_traveling_cost_qubo(qubo_size, cities_size, cost_matrix)


# In[ ]:


for i in range(1):
    q, aqubo = get_travelingsalesman_qubo(traveling_cost_qubo, traveling_qubo, pu_B, pu_cities_size)
    calculate_H(q, cities_size, cost_matrix, pu_B)


# In[ ]:


def get_ocean_qubo_re(traveling_cost_qubo, traveling_qubo, B, cities_size):
#     solver = client.get_solver()
    p_qubo = traveling_qubo + traveling_cost_qubo * B
    keys = []
    for i in range(len(p_qubo[0])):
        for j in range(len(p_qubo[1])):
            keys.append((i, j))
    Q = {key: val for key, val in zip(keys, p_qubo.reshape(-1))}
#     sampler = neal.SimulatedAnnealingSampler()
    sampler = EmbeddingComposite(DWaveSampler())
#     print(Q)
    result = QBSolv(qpu_reads = 1000).sample_qubo(Q, num_reads=10, qpu_sampler=sampler)
    result_dw = result.first.sample
#     result = sampler.sample_qubo(Q,num_reads=10)
#         for sample in computation.samples:
#             result = sample
    print(result_dw)
    return result_dw


# In[ ]:


#Oceanによる出力
timelist = []
ocean_ans = []
for i in range(3):
    start = time.time()
#     ocean_ans = get_ocean_qubo_re(traveling_cost_qubo, traveling_qubo, pu_B, cities_size)
    ocean_ans.append(get_ocean_qubo_re(traveling_cost_qubo, traveling_qubo, pu_B, cities_size))
    timelist.append(time.time() - start)
print(timelist)


# In[ ]:


#最小を取る
ans = np.zeros([len(ocean_ans[:]), pu_cities_size, pu_cities_size])
for j in range(len(ocean_ans[:])):
    k = 0
    l = list(ocean_ans[j].values())
    for i in range(int(pu_cities_size)):
        ans[j, i] += l[k: k+pu_cities_size]
        k += pu_cities_size
        print(len(l))
        print(ans[j,i])
        print(ans.shape)
print(ans)


# In[ ]:


#最小以外も取る
ans = np.zeros([10,pu_cities_size,pu_cities_size])
for j in range(len(ocean_ans.record.sample[:])):
    k = 0
    for i in range(int(len(ocean_ans.record.sample[0])/pu_cities_size)):
        ans[j, i] += ocean_ans.record.sample[j, k: k+pu_cities_size]
        k += pu_cities_size
print(ans)


# In[ ]:


index = np.zeros([10,pu_cities_size])
for j in range(len(ans[:])):
    k = 0
    for i in range(int(len(ans[0]))):
        if (np.count_nonzero(ans[j, i] == 1)) == 1:
            index[j, i] += np.where(ans[j, i] == 1)[0]
        elif (np.count_nonzero(ans[j, i] == 1)) > 1:
            index[j, i] += np.count_nonzero(ans[j, i] == 1)
        else:
            index[j, i] += 0
print(index)

