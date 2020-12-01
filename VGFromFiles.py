import random
from networkx.utils import open_file

import pickle
import networkx as nx
import matplotlib.pyplot as plt
import math
#class that represent the visibility graph for a given data in a specific time period
#parameter: nodes: a list of nodes
#nodes=[id, lat, lon, alt]
import collections
#from matplotlib import pyplot as plt

#this dictionary holds the edges list (with its weight) for each time
from distance_functions import get_area_from_distances_file

edges_by_time={}
object_by_time={}


class VG:
    def __init__(self, nodes,names,dist_file_name,max_neighbors=3, time=0, minDist=1500 , initPher=1, ratio=1):
        self.nodes=nodes
        self.time=int(time)
        self.names=names
        a={}
        self.ants=a
        self.ants[1]=1

        #max edge cost
        self.max_cost=0
        #min edge cost
        self.min_cost=0
        self.initPher=initPher
        self.max_neighbors=max_neighbors
        self.maxP=0
        self.minP=0


        G= nx.Graph()
        pos={}
        alt={}
        name={}
        visited={}
        #add nodes by id
        for i in nodes:
            #G.add_node(i[0])
            G.add_node(names[i[0]])
        #add location as pos:
        for i in nodes:
            pos[names[i[0]]]=(i[1],i[2])
        #add altitude as alt
        for i in nodes:
            alt[names[i[0]]] = (i[3])
        #add names to to self dict
        for i in nodes:
            name[names[i[0]]] = (names[i[0]])
        for i in nodes:
            visited[names[i[0]]] = 'no'

        nx.set_node_attributes(G, name, 'name')
        nx.set_node_attributes(G, pos, 'pos')
        nx.set_node_attributes(G, alt, 'alt')
        nx.set_node_attributes(G, visited, 'visited')
        self.vg=G
        self.pos=pos

        self.dists=get_area_from_distances_file(dist_file_name, tresh=minDist)
        #basicly - add the edges according the vg mindist factor
        #self.compute_edges_robust()


        self.compute_vg_for_robust(50000, ratio)

        self.init_ants()
        edges_by_time[time]=self.vg
        object_by_time[time] = self
        # assign each ant to a node for AC algorithm


        #assign phermodne level to each edge
        #self.init_phermones(initPher)


    def draw_graph(self, ant=False, edgeList=[], draw=False, name=""):
        plt.figure(figsize=(16, 8))
        #fig, ax = plt.subplots()
        nx.draw_networkx_labels(self.vg, self.pos,font_size=5,font_color='r')
        #nx.draw_networkx_edges(self.vg, self.pos, nodelist=self.nodes[0], alpha=0.4)
        nx.draw_networkx_nodes(self.vg, self.pos, nodelist=list(self.pos.keys()),node_size=60)
        nx.draw_networkx_edges(self.vg, self.pos, edgelist=edgeList, edge_color='b', width=2)  # highlight elist
        #if ant:
            #nx.draw_networkx_nodes(self.vg, self.pos, nodelist=[self.ants[1],self.ants[197], self.ants[41]],node_size=50, node_color='red')
                               #,cmap=plt.cm.Reds_r, with_labels = True)
        #plt.draw()
        laat_min= min(self.pos.items(), key=lambda x: x[1])
        lat_max = max(self.pos.items(), key=lambda x: x[1])
#        lon_min = min(self.pos.items(), key=lambda x: x[2])
#        lon_max = max(self.pos.items(), key=lambda x: x[2])
#        plt.set_xlim(laat_min, lat_max)
        #plt.grid(axis='y', linestyle='-')
        #plt.grid(True, which='both',axis='both')
        if draw==True:
            plt.show()
        else:
            plt.savefig(name+'.png')

    def draw_Tree(self, Tree, edgeList=[], draw=False, name=""):
            plt.figure(figsize=(16, 8))
            # fig, ax = plt.subplots()
            nx.draw_networkx_labels(self.vg, self.pos, font_size=5, font_color='r')
            nx.draw_networkx_edges(Tree, self.pos, nodelist=self.nodes[0], alpha=0.4)
            nx.draw_networkx_nodes(self.vg, self.pos, nodelist=list(self.pos.keys()), node_size=60)
            nx.draw_networkx_edges(Tree, self.pos, edgelist=edgeList, edge_color='r', width=2)  # highlight elist
            # if ant:
            # nx.draw_networkx_nodes(self.vg, self.pos, nodelist=[self.ants[1],self.ants[197], self.ants[41]],node_size=50, node_color='red')
            # ,cmap=plt.cm.Reds_r, with_labels = True)
            # plt.draw()
            laat_min = min(self.pos.items(), key=lambda x: x[1])
            lat_max = max(self.pos.items(), key=lambda x: x[1])
            #        lon_min = min(self.pos.items(), key=lambda x: x[2])
            #        lon_max = max(self.pos.items(), key=lambda x: x[2])
            #        plt.set_xlim(laat_min, lat_max)
            # plt.grid(axis='y', linestyle='-')
            # plt.grid(True, which='both',axis='both')
            if draw == True:
                plt.show()
            else:
                plt.savefig(name + '.png')

    def distance4(self,lat1, lat2, lon1, lon2, ratio):

        R = 6373.0+550

        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)
        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = (R * c)/ratio
        return distance

    def ang_distance4(self, lat1, lat2, lon1, lon2, ratio):
        lat1 = math.radians(lat1)
        lat2 = math.radians(lat2)
        lon1 = math.radians(lon1)
        lon2 = math.radians(lon2)

        d=math.atan2(math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1),math.sin(lon2 - lon1) * math.cos(lat2))
        d=math.degrees(d)
        d=math.fabs(d)
        return d/ratio


    def ang_distance(self, node1, node2, ratio=1):
        p1 = self.vg.nodes[node1]['pos']
        p2 = self.vg.nodes[node2]['pos']
        return self.ang_distance4(p1[0], p2[0], p1[1], p2[1], ratio)



    def distance(self, node1, node2, ratio=1):
        p1=self.vg.nodes[node1]['pos']
        p2 = self.vg.nodes[node2]['pos']

        return self.distance4(p1[0], p2[0],p1[1],p2[1],ratio)

    def isLOS(dist, tresh):
        if (dist < tresh):
            return True
        return False

    #i is always smaller than j
    def compute_edges_robust(self):
        for u, v, a in self.vg.edges(data=True):
            if(u<v):
                self.vg[u][v]["robust"]=self.compute_edge_robust(u,v)
            else:
                self.vg[u][v]["robust"] = self.compute_edge_robust(v, u)

    def compute_edge_robust(self, i,j):
        if str(i)+"_"+str(j) in  self.dists:
            robust=self.dists[str(i)+"_"+str(j)]
        else:
            robust=-1
        return robust

    def compute_vg_for_robust(self, upperBound,ratio=1):
        min = 0
        max = 1
        for i in self.vg.nodes:
            for j in self.vg.nodes:
                if (i < j):
                        r = self.compute_edge_robust(i,j)
                        d=self.distance(i,j, ratio)
                        if r==-1:
                            r=d
                        if (len(edges_by_time) > 0):
                            diff_c = 0
                            if (r <= upperBound / ratio):
                                self.vg.add_edge(i, j, weight=d ,robust=r, ph=self.initPher, update=0, diff=diff_c)
                                max = self.vg[i][j]['weight']
                                min = self.vg[i][j]['weight']
                        else:  # first time ....

                            if (r <= upperBound / ratio):
                                self.vg.add_edge(i, j, weight=d,robust=r, ph=self.initPher, update=0, diff=d)
                                max = self.vg[i][j]['weight']
                                min = self.vg[i][j]['weight']
        for w in self.vg.edges.data('weight'):
            w=w[2]
            if max<w:
                max=w
            if min>w:
                min=w
        self.max_cost=max
        self.min_cost=min
        #iso=list(nx.isolates(self.vg))
        #print(iso)
        #self.ants=[x for x in self.ants if x not in iso]

        #update the initial phermones level for each edge
        for w in self.vg.edges.data():
            w=w[2]
            w['ph']=(self.max_cost-w['weight'])+(self.max_cost-self.min_cost)/3

        # self.vg.remove_nodes_from(iso)


    def compute_vg(self, minDist, ratio=1, func=0):  # 0 for regular distance and 1 for ang_distance
        min=0
        max=1
        for i in self.vg.nodes:
            for j in self.vg.nodes:
                if(i!=j):
                    if(func==0):
                        d= self.distance(i,j, ratio)
                        if (len(edges_by_time)>0):
                          #  diff_c = d - edges_by_time[self.time - 1][i][j]['weight']
                            diff_c=0
                            #if(d>minDist/ratio):
                                #self.vg.add_edge(i, j, weight=math.inf, ph=self.initPher, update=0, diff=math.inf)
                                #min = self.vg[i][j]['weight']
                            if(d<=minDist/ratio):
                                self.vg.add_edge(i,j,weight=d, ph=self.initPher, update=0, diff=diff_c)
                                max = self.vg[i][j]['weight']
                                min = self.vg[i][j]['weight']
                        else: #first time ....
                           # if (d > minDist / ratio):
                               # self.vg.add_edge(i, j, weight=math.inf, ph=self.initPher, update=0, diff=math.inf)
                               #min = self.vg[i][j]['weight']
                            if (d <= minDist / ratio):
                                self.vg.add_edge(i, j, weight=d, ph=self.initPher, update=0, diff=d)
                                max = self.vg[i][j]['weight']
                                min = self.vg[i][j]['weight']
                    else:
                        d = self.ang_distance(i, j, ratio)
                        #print(d, minDist, ratio)
                        if (len(edges_by_time) > 0):
                            #  diff_c = d - edges_by_time[self.time - 1][i][j]['weight']
                            diff_c = 0
                            # if (d > minDist / ratio):
                            #     self.vg.add_edge(i, j, weight=math.inf, ph=self.initPher, update=0, diff=math.inf)
                            #     min = self.vg[i][j]['weight']
                            if (d <= minDist / ratio):
                                self.vg.add_edge(i, j, weight=d, ph=self.initPher, update=0, diff=diff_c)
                                max = self.vg[i][j]['weight']
                        else:  # first time ....
                            # if (d > minDist / ratio):
                            #     self.vg.add_edge(i, j, weight=math.inf, ph=self.initPher, update=0, diff=math.inf)
                            #     min = self.vg[i][j]['weight']
                            if (d <= minDist / ratio):
                                self.vg.add_edge(i, j, weight=d, ph=self.initPher, update=0, diff=d)
                                max = self.vg[i][j]['weight']
            #remove isolated nodes- the graph must be connected



        for w in self.vg.edges.data('weight'):
            w=w[2]
            if max<w:
                max=w
            if min>w:
                min=w
        self.max_cost=max
        self.min_cost=min
        #iso=list(nx.isolates(self.vg))
        #print(iso)
        #self.ants=[x for x in self.ants if x not in iso]

        #update the initial phermones level for each edge
        for w in self.vg.edges.data():
            w=w[2]
            w['ph']=(self.max_cost-w['weight'])+(self.max_cost-self.min_cost)/3

        # self.vg.remove_nodes_from(iso)
    #
    # def init_phermones(self, initLevel):
    #     for e in self.vg.edges:
    #         e['ph']=initLevel
    #         e['update']='no'

    def init_ants(self):
        self.ants={}
        for i in self.vg.nodes:
           self.ants[i]=1



    def move(self, ant):
        #print("ant", ant)
        n=self.vg.neighbors(ant)
       # edges = self.vg.edges(ant)
        edgesCose={}

        for i in n:
            edgesCose[i]=(self.vg[ant][i]['weight'])
        #print(edgesCose)
        edgesCose = {k: v for k, v in sorted(edgesCose.items(), key=lambda item: item[1], reverse=True)}
        #edgesCose=collections.OrderedDict(sorted(edgesCose.items()))

        for i in range(5):
            #choose the next neighbor
            # if(len(edgesCose)<1):
            #     print(edgesCose)
            k=self.wheel(edgesCose)
            #print(k)
            if self.vg.nodes[k]['visited']=='no':
                self.vg.nodes[k]['visited']='yes'
                self.vg[ant][k]['update']=self.vg[ant][k]['update']+1
                #move the ant to the new node
                #print(k)
                self.ants[ant]=k
                #print('ant', ant, 'moved to',k)
                break

    #assume the costs is sorted dict
    #this fuction returns the key (neighbor) tht was choosen randomly with respect to its weight
    def wheel(self, costs):
        #this dict is the prob array
        costs=self.normalize(costs)
        accumulative={}
        sum=0
        for k,v in costs.items():
            accumulative[k]=sum+v
            sum+=v
        r= random.random()
        for k,v in costs.items():
            if r<v:
                return k
        return k


    #data is a dictionary, the items needs to be normalized
    def normalize(self, data):
        minn= min(data.items())[1]
        maxx=max(data.items())[1]
        res={}
        if minn==maxx:
            for k, v in data.items():
                res[k] = 0.5
        else:
            for k, v in data.items():
                res[k]= (v-minn)/(maxx-minn)
        return res

    def move_all(self):
        for a in self.ants:
            self.move(a)
        for a in self.ants:
            self.vg.node[a]['visited']='no'

    def update_edge_phermone(self,P, number_of_updates, IP,nue):
        new_p=(1-nue)*(P)+number_of_updates*IP
        return new_p

    def update_phermones(self, nue=0.5):
        self.maxP=1000*((self.max_cost-self.min_cost)+(self.max_cost-self.min_cost)/3)
        self.minP=(self.max_cost-self.min_cost)/3
        for e in self.vg.edges.data():
            #print(e[2])
            e=e[2]
            if e['update']>0:
                IP=((self.max_cost-e['weight'])+(self.max_cost-self.min_cost))/3
                u=self.update_edge_phermone(e['ph'], e['update'], IP, nue)
                #print(e['update'])
                if u>self.maxP:
                    e['ph']=self.maxP-u
                else:
                    if u<self.minP:
                        e['ph']=self.minP+u

    def print_phermones_level(self):
        for e in self.vg.edges.data('ph'):
            print("ph",e)


    def treeConstruct(self, buttomPhermonesnum):

        #create a new edge dict for construction
        topPhermones = self.vg.edges.data()
        #print(topPhermones)
        #sort by phermones level decending order
        phermones = sorted([i[2] for i in self.vg.edges.data('ph')], reverse=True)
        #print("phermoes" ,phermones )
        phermones = phermones[:len(phermones) - buttomPhermonesnum]
        #print("ph",phermones)
        costs={}

        # remove buttom of list
        for i in list(topPhermones):
             if i[2]['ph']  in phermones:
            #if i in phermones:
               costs[(i[0],i[1])]=i[2]['weight']
               #costs[i[2]['weight']] = (i[0],i[1])

        costs={k: v for k, v in sorted(costs.items(), key=lambda item: item[1])}
        #costs = collections.OrderedDict(sorted(costs.items()))
        #print(costs.items())
        T = nx.Graph()
        for k,v in costs.items():
            #print(k[0])
            if not T.has_node(k[0]):
                T.add_node(k[0], pos=self.vg.nodes[k[0]]['pos'])
            if not T.has_node(k[1]):
                T.add_node(k[1], pos=self.vg.nodes[k[1]]['pos'])
            #if(k[1] not in nx.algorithms.components.node_connected_component(T, k[0])):
            if(len(list(T.neighbors(k[0]))) <self.max_neighbors   and len(list(T.neighbors(k[1]))) <self.max_neighbors):
                    T.add_edge(k[0],k[1], weight= v, ph=self.vg[k[0]][k[1]]['ph'])
        if(nx.is_connected(T)):
            print("connected")
        return T

    def phermon_enjancement(self, G, enahncmentFactor=1.5):
        for i in G.edges.data('ph'):
            t=enahncmentFactor*G[i[0]][i[1]]['ph']
            if t>self.maxP:
                G[i[0]][i[1]]['ph'] =self.maxP-t
            else:
                if t<self.minP:
                    G[i[0]][i[1]]['ph'] = self.minP + t

        return G


def cost(G):
    # cosst=0
    # for i in G.edges.data('weight'):
    #     cosst+=i[2]
    # return cosst
    sum=0
    print("check cost")
    for n, (dist, path) in nx.all_pairs_dijkstra(G,weight='weight'):
        for k in dist:
            sum+=dist[k]
    print("sum",sum)
    return sum

def draw_G(G,pos , edgeList=[], draw=False, name=""):
    plt.figure(figsize=(16, 8))
    # fig, ax = plt.subplots()
    nx.draw_networkx_labels(G, pos, font_size=5, font_color='r')
    nx.draw_networkx_edges(G, pos, nodelist=G.nodes, alpha=0.4)
    nx.draw_networkx_nodes(G, pos, nodelist=list(pos.keys()), node_size=60)
    nx.draw_networkx_edges(G, pos, edgelist=edgeList, edge_color='r', width=2)  # highlight elist
        # ,cmap=plt.cm.Reds_r, with_labels = True)
    # plt.draw()
    laat_min = min(pos.items(), key=lambda x: x[1])
    lat_max = max(pos.items(), key=lambda x: x[1])
    #        lon_min = min(self.pos.items(), key=lambda x: x[2])
    #        lon_max = max(self.pos.items(), key=lambda x: x[2])
    #        plt.set_xlim(laat_min, lat_max)
    # plt.grid(axis='y', linestyle='-')
    # plt.grid(True, which='both',axis='both')
    if draw == True:
        plt.show()
    else:
        plt.savefig(name + '.png')



        # C={topPhermones[]}
        #
        # edgesCose = collections.OrderedDict(sorted(edgesCose.values()))
        #
        # #sort by phermones:
        # phermones =sorted(self.vg.edges.data('ph'))
        # #remove last elements
        # phermones = phermones[:len(phermones) - topPhermonesnum]
        # C = collections.OrderedDict(sorted(phermones.items()))
        #
        # phermones = self.vg.edges.data('weight')
        # #C = collections.OrderedDict(sorted(C.items()))