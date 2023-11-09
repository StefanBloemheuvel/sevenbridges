#%%
import os
import sys
import time
import random
import warnings
from math import sin, cos, sqrt, atan2, radians
from math import radians, sin, cos, sqrt, asin

import pandas as pd
import networkx as nx
import numpy as np
from numpy import arctan2, cos, sin, sqrt, pi, power, append, diff, deg2rad
import scipy as sp
from scipy.spatial.distance import  pdist, squareform
from scipy.cluster.vq import kmeans, vq
from scipy.spatial import Delaunay
from scipy.sparse import lil_matrix

from sklearn.cluster import OPTICS, DBSCAN, KMeans
from sklearn.neighbors import DistanceMetric, BallTree
import matplotlib.pyplot as plt 
from dtaidistance import dtw
from libpysal import weights

dist = DistanceMetric.get_metric('haversine')

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
            graph[j[0]][j[1]]['weight'] = 1 - (graph[j[0]][j[1]]['weight'] - min_weight) / (max_weight - min_weight)

        graph.remove_edges_from((e for e, w in nx.get_edge_attributes(graph,'weight').items() if w < cutoff))

        
        self.networkx_graph = graph
    
    def knn_unweighted(self, path, k=4):   
        print('Package Print: went for knn_unweighted')
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
        
        # Method 1
        for i in indices:
            [graph.add_edge(i[0],j, weight=1) for j in i[1:]]
            
        self.networkx_graph = graph
        self.networkx_graph = nx.relabel_nodes(self.networkx_graph, dict(zip(range(0,self.data['station'].shape[0]),self.data['station'])))
        
        self.tree = tree
    
    def NormalizeData(self, input_data):
        return (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))
    
    def knn_weighted(self, path, k=5):
        print('Package Print: went for knn_weighted')
        self.created_with = 'knn_weighted'
        
        # Load the data
        self.load_location_data(path)
        old_names = self.data['station'].copy()
        self.data['station'], tesst_key = pd.factorize(self.data['station'])

        graph = nx.Graph()
        for i in self.data[['station','lat','lon']].itertuples():
            graph.add_node(i[1], pos=(i[2],i[3]))
        
        pos = nx.get_node_attributes(graph,'pos')
        names = [i for i in graph.nodes()]
        indexes = [i for i in range(0,nx.number_of_nodes(graph))]
        a_dictionary = dict(zip(indexes, names))

        self.data['lat_rad'] = np.deg2rad(self.data['lat'])
        self.data['lon_rad'] = np.deg2rad(self.data['lon'])
        
        tree = BallTree(self.data[['lat_rad','lon_rad']], metric="haversine")
        distances, indices = tree.query(self.data[['lat_rad','lon_rad']], k=k)
        distances[:,1:] = distances[:,1:] * 6371
        distances = self.NormalizeData(distances)

        y = indices.copy()
        for item, v in a_dictionary.items():
            indices[y == item] = v
        
        depth = k
        for indices_i,distances_j in zip(indices,distances):
            for k in range(1,depth):
                graph.add_edge(indices_i[0],indices_i[k],weight=distances_j[k])

        graph = nx.relabel_nodes(graph, dict(zip(range(0,len(old_names)),old_names)))

        self.networkx_graph = graph
        self.networkx_graph = nx.relabel_nodes(self.networkx_graph, dict(zip(range(0,self.data['station'].shape[0]),self.data['station'])))
        
        self.tree = tree
        
    def gabriel(self, path):
        print('Package Print: went for gabriel')
        # Load the data
        self.created_with = 'gabriel'

        self.load_location_data(path)
        
        tri = Delaunay(self.data[['lat','lon']].values)
        node_names = self.data['station']
        self.simplices = tri.simplices
        
        edges = []
        for i in tri.simplices:
            for j in range(0,3):
                for k in range(0,3):
                    if j != k:
                        edges.append((i[j],i[k]))
        new_df = pd.DataFrame(edges).drop_duplicates().sort_values([0, 1]).groupby(0)[1].apply(list).to_dict()
    
        lil = lil_matrix((tri.npoints, tri.npoints))
        indices, indptr = tri.vertex_neighbor_vertices
        for k in range(tri.npoints):
            lil.rows[k] = indptr[indices[k]:indices[k+1]]
            lil.data[k] = np.ones_like(lil.rows[k])  # dummy data of same shape as row
        
        coo = sp.sparse.csr_matrix(lil.toarray()).tocoo()
        conns = np.vstack((coo.row, coo.col)).T
        
        delaunay_conns = np.sort(conns, axis=1)
        
        c = tri.points[delaunay_conns]
        m = (c[:, 0, :] + c[:, 1, :])/2
        r = np.sqrt(np.sum((c[:, 0, :] - c[:, 1, :])**2, axis=1))/2
        tree = sp.spatial.cKDTree(self.data[['lat','lon']].values)
        n = tree.query(x=m, k=1)[0]
        g = n >= r*(0.999)  # The factor is to avoid precision errors in the distances
        gabriel_conns = delaunay_conns[g]
        graph = nx.from_edgelist(gabriel_conns)
        
        graph = nx.relabel_nodes(graph, dict(zip(range(0,self.data['station'].shape[0]),self.data['station'])))
        self.networkx_graph = graph
        
    def distance_limiter(self, path, k):
        print('Package Print: went for Distance Limiter')
        self.created_with = 'distance_limiter'
        self.load_location_data(path)
        
        self.distance_limiter_obj = weights.DistanceBand.from_array(self.data[['lat','lon']], threshold=k)
        self.networkx_graph = self.distance_limiter_obj.to_networkx()
        self.networkx_graph = nx.relabel_nodes(self.networkx_graph, dict(zip(range(0,self.data['station'].shape[0]),self.data['station'])))
        
    def relative_neighborhood(self, path):
        print('Package Print: went for relative neighborhood')
        self.created_with = 'relative neighborhood'
        
        # Load the data
        self.load_location_data(path)
        
        graph = weights.Relative_Neighborhood(self.data[['lat','lon']]).to_networkx()
        
        self.networkx_graph = graph
        self.networkx_graph = nx.relabel_nodes(self.networkx_graph, dict(zip(range(0,self.data['station'].shape[0]),self.data['station'])))

    def load_sensor_data(self, sensor_data_path):
        if sensor_data_path.split('.')[-1] == 'npy':
            self.sensor_data = np.load(sensor_data_path)
        else:
            print(f'error, this data is not npy, it is {sensor_data_path.split(".")[-1]} ')
            
    
    def from_signal(self, location_path, data_path, variant, clips, threshold):
        print(f'Package Print: went for signal based graph with variant = {variant}')
        self.created_with = variant
        
        # Load the data
        self.load_location_data(location_path)
        self.load_sensor_data(data_path)
        
        print(f'data shape = {self.sensor_data.shape}')
        graph = nx.Graph()
        for i in self.data[['station','lat','lon']].itertuples():
            graph.add_node(i[1], pos=(i[2],i[3]))
        
        start = time.time()
        if clips == True:
            print(f'went for clips')
            self.sensor_data = self.sensor_data.astype('double')
            length = 24 
            sens = self.sensor_data.shape[1]

            idx_last = -(self.sensor_data.shape[0] % length)
            if idx_last < 0:    
                clips = self.sensor_data[:idx_last].reshape(-1, self.sensor_data.shape[1],length)
            else:
                clips = self.sensor_data[idx_last:].reshape(-1, self.sensor_data.shape[1],length)                
            
            np.random.seed(1)
            random.seed(1)
            clips = clips[np.random.choice(range(clips.shape[0]), 50, replace=False),:,:]
            
            collection_of_graphs = np.empty((clips.shape[0],sens,sens))
            for i,j in enumerate(clips):
                if i % 50 == 0:
                    print(f'iteration {i} of {variant}')
                try:
                    if variant == 'dtw':
                        R1 = dtw.distance_matrix_fast(j)
                    if variant == 'correlation':
                        R1 = np.corrcoef(j)
                        
                    collection_of_graphs[i] = R1
                
                except Exception as e:
                    print(f'error was found = {e}')
                    continue             
            
        else:
            np.random.seed(1)
            random.seed(1)
            self.sensor_data = self.sensor_data[np.random.choice(range(self.sensor_data.shape[0]), 50, replace=False),:,:,:]
            collection_of_graphs = np.empty((self.sensor_data.shape[0],self.sensor_data.shape[1],self.sensor_data.shape[1]))
            for i,j in enumerate(self.sensor_data):
                if i % 50 == 0:
                    print(f'iteration {i}')
                try:
                    if variant == 'dtw':
                        R1 = dtw.distance_matrix_fast(j[:,:1000,0])
                    if variant == 'correlation':
                        R1 = np.corrcoef(j[:,:1000,0])
                                
                    collection_of_graphs[i] = R1
                        
                except Exception as e:
                    print(e)
                    continue 
        
        end = time.time()
        
        collection_of_graphs = np.nan_to_num(collection_of_graphs)
        collection_of_graphs = np.sort(collection_of_graphs, axis= 0) # this is from small to large!
        
        if variant == 'correlation':
            collection_of_graphs = collection_of_graphs[-20:,:,:]
        if variant == 'dtw':
            collection_of_graphs = collection_of_graphs[:,:,:]
        
        collection_of_graphs = collection_of_graphs.mean(axis=0)
        
        if variant == 'dtw':
            collection_of_graphs = 1 - (collection_of_graphs - collection_of_graphs.min()) / (collection_of_graphs.max() - collection_of_graphs.min())
        collection_of_graphs[collection_of_graphs < threshold] = 0
        
        self.networkx_graph = nx.from_numpy_array(collection_of_graphs)
        self.networkx_graph = nx.relabel_nodes(self.networkx_graph, dict(zip(range(0,self.data['station'].shape[0]),self.data['station'])))

    
    def kmeans(self, path, num_clusters):
        print('Package Print: went for kmeans own')
        self.created_with = 'kmeans_own'
        
        # Load the data
        self.load_location_data(path)
        
        print('went for kmeans_own')
        graph = nx.Graph()

        centroids, mean_dist = kmeans(self.data[['lat','lon']], num_clusters, seed=1)
        clusters, dist = vq(self.data[['lat','lon']], centroids)
        self.data['cluster'] = clusters

        for k in self.data[['station','lat','lon','cluster']].itertuples():
            graph.add_node(k[1], pos=(k[2],k[3]), cluster=k[4])    
        
        for node_r in graph.nodes(data=True):
            for node in graph.nodes(data=True):
                if node != node_r and node[1]['cluster'] == node_r[1]['cluster'] and node_r[1]['cluster'] != -1 and node[1]['cluster'] != -1:
                    graph.add_edge(node[0], node_r[0], weight=1)
                
        self.networkx_graph = graph
        
        
    def dbscan(self, path, visualize=False, eps=4, min_samples=2):
        print(f'Package Print: went for dbscan with {eps=} and {min_samples}')
        self.created_with = 'dbscan'
        
        # Load the data
        self.load_location_data(path)
        
        graph = nx.Graph()

        for k in self.data[['station','lat','lon']].iterrows():
            graph.add_node(k[1][0], pos=(k[1][1],k[1][2]))
        number_of_nodes = self.data.shape[0]

        X=self.data.loc[:,['lat','lon']]
        distance_matrix = squareform(pdist(X, (lambda u,v: self.haversine(u,v))))

        db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')  # using "precomputed" as recommended by @Anony-Mousse
        y_db = db.fit_predict(distance_matrix)

        X['cluster'] = y_db
        
        if visualize==True:
            plt.scatter(X['lon'], X['lat'], c=X['cluster'],cmap='viridis')
            plt.show()
        
        adj = np.zeros((number_of_nodes,number_of_nodes))
        for i,j in enumerate(X['cluster']):
            for k,l in enumerate(X['cluster']):
                if j == l and j != -1 and l != -1:
                    graph.add_edge(i,k, weight=1)

        graph.remove_edges_from(nx.selfloop_edges(graph))
        self.networkx_graph = graph
        self.networkx_graph = nx.relabel_nodes(self.networkx_graph, dict(zip(range(0,self.data['station'].shape[0]),self.data['station'])))
        
    def optics(self, path, min_samples=2):
        print('Package Print: went for optics')
        self.created_with = 'optics'

        visualize=False
        self.load_location_data(path)
        
        graph = nx.Graph()

        for k in self.data[['station','lat','lon']].iterrows():
            graph.add_node(k[1][0], pos=(k[1][1],k[1][2]))
        number_of_nodes = self.data.shape[0]

        X=self.data.loc[:,['lat','lon']]
        distance_matrix = squareform(pdist(X, (lambda u,v: self.haversine(u,v))))

        optics_clustering = OPTICS(min_samples=min_samples, metric='precomputed')
        y_db = optics_clustering.fit_predict(distance_matrix)

        self.data['cluster'] = y_db
        
        if visualize==True:
            plt.scatter(self.data['lon'], self.data['lat'], c=self.data['cluster'],cmap='viridis')
            plt.show()
        
        # graph = nx.Graph()
        adj = np.zeros((number_of_nodes,number_of_nodes))
        for i,j in enumerate(self.data['cluster']):
            for k,l in enumerate(self.data['cluster']):
                if j == l and j != -1 and l != -1:
                
                    graph.add_edge(i,k, weight=1)

        graph.remove_edges_from(nx.selfloop_edges(graph))
        pos = nx.get_node_attributes(graph,'pos')
        self.networkx_graph = graph
        self.networkx_graph = nx.relabel_nodes(self.networkx_graph, dict(zip(range(0,self.data['station'].shape[0]),self.data['station'])))



        
    def create_adjacency_matrix(self, fill_diagonal = False):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            epsilon = 5e-8
            adj = np.asarray(nx.adjacency_matrix(self.networkx_graph, nodelist=sorted(self.networkx_graph.nodes())).todense())
            
            if fill_diagonal == True:
                print(f'filled diagonal with ones')
                np.fill_diagonal(adj, 1)
                
            self.adjacency_matrix = adj

    def create_normalized_laplacian_matrix(self):
        print('created normalized laplacian')
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.normalized_laplacian_matrix = nx.normalized_laplacian_matrix(self.networkx_graph, nodelist=sorted(self.networkx_graph.nodes())).todense()
            np.fill_diagonal(self.normalized_laplacian_matrix,1)

    def haversine(self, lonlat1, lonlat2):
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians 
        lat1, lon1 = lonlat1
        lat2, lon2 = lonlat2
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 6371 # Radius of earth in kilometers. Use 3956 for miles
        return c * r
    
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
        # print(f'Average clustering coefficient = {nx.average_clustering(self.networkx_graph):.2f}')
        print(f"Density: {nx.density(self.networkx_graph):.2f}")
        print('#######         END         #######','\n')

    def loglog_degree_histogram(self):
        degree_sequence = sorted((d for n, d in self.networkx_graph.degree()), reverse=True)
        fig = plt.figure(figsize=(3,3))
        plt.loglog(range(1,self.networkx_graph.order()+1),degree_sequence,'k.')
        plt.xlabel('Rank')
        plt.ylabel('Degree')
        fig.tight_layout()
        plt.show()
        
    def loglog_hist(self):
        degree_sequence = sorted((d for n, d in self.networkx_graph.degree()), reverse=True)
        dmax=max(degree_sequence)

        fig = plt.figure(figsize=(3,3))
        plt.plot(degree_sequence, '#36558f', marker=".")
        plt.title(f"Density: {nx.density(self.networkx_graph):.2f}")
        plt.ylabel("Degree", fontsize=10)
        plt.xlabel("Rank", fontsize=10)
        fig.tight_layout()
        plt.locator_params(axis="both", integer=True, tight=True)
        plt.show()