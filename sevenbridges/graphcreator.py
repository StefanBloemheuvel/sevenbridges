def joke_new():
    return (u'Wenn ist das Nunst\u00fcck git und Slotermeyer? Ja! ... '
            u'Beiherhund das Oder die Flipperwaldt gersput.')
    
#%%
import os
import sys
import pandas as pd
# from geopy.distance import geodesic
import networkx as nx
import numpy as np
np.set_printoptions(suppress=True)
import warnings
from libpysal import weights
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker
from sklearn.cluster import KMeans
from mpl_toolkits.basemap import Basemap as Basemap
from math import radians, sin, cos, sqrt, asin
from scipy.spatial.distance import  pdist, squareform
from sklearn.cluster import DBSCAN
from scipy.cluster.vq import kmeans, vq
from sklearn.cluster import OPTICS
import scipy as sp
from scipy.spatial import Delaunay
from scipy.sparse import lil_matrix
from sklearn.preprocessing import minmax_scale
from dtaidistance import dtw
import time
import random
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.stats import entropy
from minepy import pstats, cstats
from numpy import arctan2, cos, sin, sqrt, pi, power, append, diff, deg2rad
# other dependencies
# geopandas
from sklearn.neighbors import DistanceMetric
dist = DistanceMetric.get_metric('haversine')
from math import sin, cos, sqrt, atan2, radians

R = 6373.0

class graph_generator:
    def load_location_data(self, path):
        self.path = path
        
        if self.path.split('.')[-1] == 'csv':
            self.data = pd.read_csv(path, dtype={'station': 'str', 'lat': 'float', 'lon':'float'})
        else:
            print('incorrect path!')
        
    def minmax(self, path, cutoff=0.3):
        print(f'Package Print: went for minmax with cutoff = {cutoff}')
        self.created_with = 'minmax'
         
        # Load the data
        self.load_location_data(path)
        
        graph = nx.Graph()

        for k in self.data[['station','lat','lon']].iterrows():
            graph.add_node(k[1][0], pos=(k[1][1],k[1][2]))
            
        for idx1, itm1 in self.data[['station','lat','lon']].iterrows():
            for idx2, itm2 in self.data[['station','lat','lon']].iterrows():
                        pos1 = (itm1[1],itm1[2])
                        pos2 = (itm2[1],itm2[2])
                        X = [[radians(itm1[1]), radians(itm1[2])], [radians(itm2[1]), radians(itm2[2])]]
                        distance = R * dist.pairwise(X)
                        distance = np.array(distance).item(1)
                        if distance != 0: # this filters out self-loops and also the edges between the artificial nodes
                            graph.add_edge(itm1[0], itm2[0], weight=distance)


        min_weight, max_weight = min(nx.get_edge_attributes(graph, "weight").values()), max(nx.get_edge_attributes(graph, "weight").values())

        for i,j in enumerate(graph.edges(data=True)):
            graph[j[0]][j[1]]['weight'] = 0.98 - (graph[j[0]][j[1]]['weight'] - min_weight) / (max_weight - min_weight)

        graph.remove_edges_from((e for e, w in nx.get_edge_attributes(graph,'weight').items() if w < cutoff))

        
        self.networkx_graph = graph
        
    def create_adjacency_matrix(self, fill_diagonal = False):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            epsilon = 5e-8
            adj = np.asarray(nx.adjacency_matrix(self.networkx_graph, nodelist=sorted(self.networkx_graph.nodes())).todense())
            
            if fill_diagonal == True:
                print(f'filled diagonal with ones')
                np.fill_diagonal(adj, 1)
                
            self.adjacency_matrix = adj
            
    def summary_statistics(self):
        self.number_of_nodes = nx.number_of_nodes(self.networkx_graph)
        self.number_of_edges = nx.number_of_edges(self.networkx_graph)
        print('\n','####### Summary Statistics #######')
        print(f'Graph created with = {self.path.split("/")[-1]} and {self.created_with}')
        print(f'Number of nodes = {self.number_of_nodes}')
        print(f'Number of edges = {self.number_of_edges}')
        degree_centrality_scores = list(sorted(nx.degree_centrality(self.networkx_graph).items(), key=lambda x : x[1], reverse=True)[:1])
        print(f'The most important node is {degree_centrality_scores[0][0]}({degree_centrality_scores[0][1]:2f})')
        print(f'Number of connected components = {nx.number_connected_components(self.networkx_graph)}')
        print(f"Density: {nx.density(self.networkx_graph):.2f}")
        print('#######         END         #######','\n')