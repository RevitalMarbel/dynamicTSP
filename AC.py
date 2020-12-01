import datetime

import math

import draw_graphs
import visibilityGraph
import os
import sys
import time
from scipy import fft
import re
import numpy as np
import networkx as nx
import pylab as plt
import visibilityGraph_for_AC
import distance_functions
from networkx.utils import open_file

import pickle


def AB_DCST(G, s=1, nue=0.5):
    res = []
    cost_b=math.inf
    steps_with_no_improvement=0
    B=nx.Graph()
    for i in range(1):
        for step in range(s):
            if step== (s/3) or step== (2*s/3):
                G.update_phermones(nue)
            print("move all")
            G.move_all()
        print("update phermons")
        G.update_phermones()
        print("construct tree")
        T = G.treeConstruct(buttomPhermonesnum=7000)
        t_cost = visibilityGraph_for_AC.cost(T)
        print("cost ", cost_b, t_cost)
        if cost_b > t_cost:
            cost_b = t_cost
            B=T

            steps_with_no_improvement=0
        else:
            steps_with_no_improvement=+1
        B = G.phermon_enjancement(T)
        nue=- 0.01
        if steps_with_no_improvement>10:
            B = G.phermon_enjancement(T,enahncmentFactor=0.2)
            print("evaporate")
        res.append([i, T, cost_b])
    return B
