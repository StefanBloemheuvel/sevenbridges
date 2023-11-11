#%%
import os
import sys
import pandas as pd
import networkx as nx
import numpy as np
import warnings
from math import radians, sin, cos, sqrt, asin
import scipy as sp
from scipy.spatial.distance import cdist
import time
import random
from numpy import arctan2, cos, sin, sqrt, pi, power, append, diff, deg2rad
from sklearn.metrics import DistanceMetric
dist = DistanceMetric.get_metric('haversine')
from math import sin, cos, sqrt, atan2, radians

R = 6373.0

class graph_generator:
    def load_location_data(self, input_data):
        """
        Convert the input data to a Pandas DataFrame.

        Parameters:
        data : Pandas DataFrame, NumPy array, or path str
            The input data to be converted. It can be a Pandas DataFrame,
            a NumPy array, or a file path to a CSV file.

        Returns:
        pandas.DataFrame
            A Pandas DataFrame containing the data from the input.

        Raises:
        ValueError: If the input data is of an unsupported type or the file is not found.

        Example usage:

        Now you have a Pandas DataFrame, regardless of the input type.
        """
        
        if isinstance(input_data, pd.DataFrame):
            if all(col in input_data.columns for col in ['node_name', 'lat', 'lon']):
                return input_data
            else:
                raise ValueError("Input DataFrame must have columns 'node_name', 'lat', and 'lon'.")
        elif isinstance(input_data, np.ndarray):
            if input_data.shape[1] == 2:
                self.data = pd.DataFrame(input_data, columns=['lat', 'lon'])
                self.data.insert(0, 'node_name', self.data.index.astype(str))
            elif input_data.shape[1] == 3:
                return pd.DataFrame(input_data, columns=['node_name', 'lat', 'lon'])
            else:
                raise ValueError("Input Numpy Array must have 3 columns.")
        elif isinstance(input_data, str):
            try:
                self.data = pd.read_csv(input_data)
                if not all(col in self.data.columns for col in ['node_name', 'lat', 'lon']):
                    raise ValueError("CSV file must have columns 'node_name', 'lat', and 'lon'.")
                return self.data
            except FileNotFoundError:
                raise ValueError(f"File not found: {input_data}")
    
        else:
            raise ValueError("Unsupported data type. Input must be a Pandas DataFrame, Numpy array or CSV file path.")
        
    def kmeans(self, path, n_clusters, max_iters=200):
        print(f'Package Print: went for kmeans with {n_clusters} clusters')
        self.created_with = 'kmeans'
         
        # Load the data
        self.load_location_data(path)
    
        # create the graph
        graph = nx.Graph()
    
        centroids = self.data[['lat','lon']].values[np.random.choice(self.data.shape[0], n_clusters, replace=False)]
        
        for i in range(max_iters):
            # Calculate distances between data points and centroids
            distances = cdist(self.data[['lat','lon']].values, centroids, 'euclidean')
            
            # Assign each data point to the closest centroid
            labels = np.argmin(distances, axis=1)
            
            # Update centroids by taking the mean of all points assigned to each centroid
            new_centroids = np.array([self.data[['lat','lon']][labels == j].mean(axis=0) for j in range(n_clusters)])
            
            # Check for convergence
            if np.all(centroids == new_centroids):
                break
            
            centroids = new_centroids
            
        self.data['cluster'] = labels
        for k in self.data[['node_name','lat','lon','cluster']].itertuples():
            graph.add_node(k[1], pos=(k[2],k[3]), cluster=k[4])    
        
        for node_r in graph.nodes(data=True):
            for node in graph.nodes(data=True):
                if node != node_r and node[1]['cluster'] == node_r[1]['cluster'] and node_r[1]['cluster'] != -1 and node[1]['cluster'] != -1:
                    graph.add_edge(node[0], node_r[0], weight=1)
                
        self.networkx_graph = graph
        
    def minmax(self, path, cutoff=0.3):
        print(f'Package Print: went for minmax with cutoff = {cutoff}')
        self.created_with = 'minmax'
         
        # Load the data
        self.load_location_data(path)
        
        graph = nx.Graph()

        for k in self.data[['node_name','lat','lon']].iterrows():
            graph.add_node(k[1][0], pos=(k[1][1],k[1][2]))
            
        for idx1, itm1 in self.data[['node_name','lat','lon']].iterrows():
            for idx2, itm2 in self.data[['node_name','lat','lon']].iterrows():
                X = [[radians(itm1[1]), radians(itm1[2])], [radians(itm2[1]), radians(itm2[2])]]
                distance = R * dist.pairwise(X)
                distance = np.array(distance).item(1)
                if distance != 0: # this filters out self-loops and also the edges between the artificial nodes
                    graph.add_edge(itm1[0], itm2[0], weight=distance)

        min_weight, max_weight = min(nx.get_edge_attributes(graph, "weight").values()), max(nx.get_edge_attributes(graph, "weight").values())

        for i,j in enumerate(graph.edges(data=True)):
            graph[j[0]][j[1]]['weight'] = 1 - (graph[j[0]][j[1]]['weight'] - min_weight) / (max_weight - min_weight)

        graph.remove_edges_from((e for e, w in nx.get_edge_attributes(graph,'weight').items() if w < cutoff))
        
        self.networkx_graph = graph
        
    def create_adjacency_matrix(self, fill_diagonal = False):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            adj = np.asarray(nx.adjacency_matrix(self.networkx_graph, nodelist=sorted(self.networkx_graph.nodes())).todense())
            
            if fill_diagonal == True:
                print(f'filled diagonal with ones')
                np.fill_diagonal(adj, 1)
                
            self.adjacency_matrix = adj
            
    def summary_statistics(self):
        self.number_of_nodes = nx.number_of_nodes(self.networkx_graph)
        self.number_of_edges = nx.number_of_edges(self.networkx_graph)
        print('\n','####### Summary Statistics #######')
        print(f'Graph created with {self.created_with}')
        print(f'Number of nodes = {self.number_of_nodes}')
        print(f'Number of edges = {self.number_of_edges}')
        degree_centrality_scores = list(sorted(nx.degree_centrality(self.networkx_graph).items(), key=lambda x : x[1], reverse=True)[:1])
        print(f'The most important node is {degree_centrality_scores[0][0]}({degree_centrality_scores[0][1]:2f})')
        print(f'Number of connected components = {nx.number_connected_components(self.networkx_graph)}')
        print(f"Density: {nx.density(self.networkx_graph):.2f}")
        print('#######         END         #######','\n')
        
        
