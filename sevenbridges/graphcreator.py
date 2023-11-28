#%%
import os
import sys
import pandas as pd
import networkx as nx
import numpy as np
import warnings
from math import radians, sin, cos, sqrt, asin
import scipy as sp
from scipy.spatial.distance import cdist, pdist, squareform
import time
import random
from numpy import arctan2, cos, sin, sqrt, pi, power, append, diff, deg2rad
from sklearn.metrics import DistanceMetric
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt
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
                self.data = input_data
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
        
    def kmeans(self, path, n_clusters=5, max_iters=6000):
        """
        Use K-Means algorithm to create graph.

        Parameters:
        data : Pandas DataFrame, NumPy array, or path str
            The input data to be converted. It can be a Pandas DataFrame,
            a NumPy array, or a file path to a CSV file.

        Returns:
        networkx.Graph
            A Networkx Graph object.

        Raises:
        ValueError: If the input data is of an unsupported type or the file is not found.
        """
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
                
        self.G = graph
        
    def minmax(self, path, cutoff=0.3):
        """
        Use MinMax algorithm to create graph.

        Parameters:
        path : Pandas DataFrame, NumPy array, or path str
            The input data to be converted. It can be a Pandas DataFrame,
            a NumPy array, or a file path to a CSV file.
        cutoff : Relative cutoff point of distances scaled between 0-1.

        Returns:
        networkx.Graph
            A Networkx Graph object.

        Raises:
        ValueError: If the input data is of an unsupported type or the file is not found.
        """
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
        
        self.G = graph
        
    def relative_neighborhood(self, path):
        self.created_with = 'relative_neighborhood'
        self.load_location_data(path)
        
        self.G = nx.Graph()
        for k in self.data[['node_name','lat','lon']].itertuples():
            self.G.add_node(k[1], pos=(k[2],k[3])) 
        
        coords = self.data[['lat', 'lon']].to_numpy()
        node_names = self.data['node_name'].tolist()
        
        n = len(coords)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distances[i, j] = self.haversine(coords[i], coords[j])

        # Create a Relative Neighborhood Graph
        def is_rng_edge(i, j, distances):
            d_ij = distances[i, j]
            for k in range(len(distances)):
                if k != i and k != j:
                    if max(distances[i, k], distances[j, k]) < d_ij:
                        return False
            return True

        edges = [(node_names[i], node_names[j]) for i in range(n) for j in range(i + 1, n) if is_rng_edge(i, j, distances)]
        # Build Graph
        self.G.add_edges_from(edges)
        
    def plot(self):
        fig, ax = plt.subplots()
        nx.draw(self.G,nx.get_node_attributes(self.G, 'pos'),ax=ax) #notice we call draw, and not draw_networkx_nodes
        plt.axis('on') # turns on axis
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.show()
        
    def knn(self, path, k=4, weighted=False):   
        """
        Use K-NN algorithm to create graph.

        Parameters:
        data : Pandas DataFrame, NumPy array, or path str
            The input data to be converted. It can be a Pandas DataFrame,
            a NumPy array, or a file path to a CSV file.
        k : Number of neighbors to consider.
        
        Returns:
        networkx.Graph
            A Networkx Graph object.

        Raises:
        ValueError: If the input data is of an unsupported type or the file is not found.
        """
        print(f'Package Print: went for knn_unweighted with k={k} and weighted={weighted}')
        self.created_with = 'knn_unweighted'
         
        # Load the data
        self.load_location_data(path)
        
        graph = nx.Graph()
        for i in self.data.iterrows():
            graph.add_node(i[0], pos=(i[1][1],i[1][2]))
            
        self.data['lat_rad'] = np.deg2rad(self.data['lat'])
        self.data['lon_rad'] = np.deg2rad(self.data['lon'])
        
        tree = BallTree(self.data[['lat_rad','lon_rad']], metric="haversine")
        distances, indices = tree.query(self.data[['lat_rad','lon_rad']], k=k)
        distances[:,1:] = distances[:,1:] * 6371

        if weighted == False:
            for i in indices:
                [graph.add_edge(i[0],j, weight=1) for j in i[1:]]
            
        if weighted == True:
            edge_list = []


            for i, neighbors in enumerate(indices):
                # Start from 1 to skip the point itself
                for j in range(1, len(neighbors)):
                    edge = (i, neighbors[j], distances[i][j])
                    edge_list.append(edge)
            
            graph.add_weighted_edges_from(edge_list)


        self.G = graph
        self.G = nx.relabel_nodes(self.G, dict(zip(range(0,self.data['node_name'].shape[0]),self.data['node_name'])))
        
        self.tree = tree
        
        
    def haversine(self, coord1, coord2):
        # Convert latitude and longitude from degrees to radians
        lat1, lon1 = np.radians(coord1)
        lat2, lon2 = np.radians(coord2)

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return 6371 * c  # Earth's radius in kilometers

    def spherical_midpoint(self, coord1, coord2):
        # Convert latitude and longitude from degrees to radians
        lat1, lon1 = np.radians(coord1)
        lat2, lon2 = np.radians(coord2)

        # Midpoint calculation
        Bx = np.cos(lat2) * np.cos(lon2 - lon1)
        By = np.cos(lat2) * np.sin(lon2 - lon1)
        lat_mid = np.arctan2(np.sin(lat1) + np.sin(lat2), np.sqrt((np.cos(lat1) + Bx)**2 + By**2))
        lon_mid = lon1 + np.arctan2(By, np.cos(lat1) + Bx)
        return np.degrees(lat_mid), np.degrees(lon_mid)

    def is_edge_in_gabriel_graph(self, points, i, j):
            midpoint = self.spherical_midpoint(points[i], points[j])
            radius = self.haversine(points[i], midpoint)
            for k in range(len(points)):
                if k != i and k != j and self.haversine(points[k], midpoint) < radius:
                    return False
            return True
        
    def gabriel(self, path):
        """
        Use Gabriel requirements to create graph.

        Parameters:
        data : Pandas DataFrame, NumPy array, or path str
            The input data to be converted. It can be a Pandas DataFrame,
            a NumPy array, or a file path to a CSV file.

        Returns:
        networkx.Graph
            A Networkx Graph object.

        Raises:
        ValueError: If the input data is of an unsupported type or the file is not found.
        """
        self.created_with = 'Gabriel'
        
        self.load_location_data(path)
        
        self.G = nx.Graph()
        for i in self.data.iterrows():
            self.G.add_node(i[0], pos=(i[1][1],i[1][2]))
            
        edges = []
        for i in range(self.data.shape[0]):
            for j in range(i + 1, self.data.shape[0]):
                
                if self.is_edge_in_gabriel_graph(self.data[['lat','lon']].values, i, j):
                    edges.append((i, j))
        
        self.G.add_edges_from(edges)
        self.G = nx.relabel_nodes(self.G, dict(zip(range(0,self.data['node_name'].shape[0]),self.data['node_name'])))
        
    
    def thresholded_gaussian_kernel(self, path, kappa):
        """
        Use Thresholded Gaussian Kernel to create graph.

        Parameters:
        dataframe (pd.DataFrame): DataFrame with columns ['node_name', 'latitude', 'longitude'].
        kappa (float): The threshold below which the edge weight is considered zero.

        Returns:
        pd.DataFrame: Adjacency matrix with node names as indices and columns.
        """
        self.created_with = 'Gaussian Kernel'
        self.load_location_data(path)
        
        # Extract latitude and longitude values
        locations = self.data[['lat', 'lon']].values
        
        # Calculate pairwise Euclidean distances between locations
        dists = squareform(pdist(locations))
        
        # Apply Gaussian kernel
        adjacency_matrix = np.exp(- (dists ** 2) / (2 * dists.std() ** 2))
        
        # Apply threshold
        adjacency_matrix[adjacency_matrix < kappa] = 0
        self.G = nx.from_numpy_array(adjacency_matrix)
        

        
    def create_adjacency_matrix(self, fill_diagonal = False):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            adj = np.asarray(nx.adjacency_matrix(self.G, nodelist=sorted(self.G.nodes())).todense())
            
            if fill_diagonal == True:
                print(f'filled diagonal with ones')
                np.fill_diagonal(adj, 1)
                
            self.adjacency_matrix = adj
            
    def summary_statistics(self):
        self.number_of_nodes = nx.number_of_nodes(self.G)
        self.number_of_edges = nx.number_of_edges(self.G)
        print('\n','####### Summary Statistics #######')
        print(f'Graph created with {self.created_with}')
        print(f'Number of nodes = {self.number_of_nodes}')
        print(f'Number of edges = {self.number_of_edges}')
        degree_centrality_scores = list(sorted(nx.degree_centrality(self.G).items(), key=lambda x : x[1], reverse=True)[:1])
        print(f'The most important node is {degree_centrality_scores[0][0]}({degree_centrality_scores[0][1]:2f})')
        print(f'Number of connected components = {nx.number_connected_components(self.G)}')
        print(f"Density: {nx.density(self.G):.2f}")
        print('#######         END         #######','\n')
        