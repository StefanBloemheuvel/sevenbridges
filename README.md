<img src="logo.jpg" alt="The proposed framework" width="400"/>


## Welcome to SevenBridges

SevenBridges is a Python library for adjacency matrix calculation. The main goal of this project is to provide a simple and flexible framework for creating the adjacency matrices necessary for implicit graphs for graph neural networks (GNNs) or Graph Signal Processing. 

You can use SevenBridges for calculating the adjacency matrix based on geographical locations of the nodes, or by examining the actual time series values in a dataset. 

SevenBridges implements the most popular techniques for graph construction.

### Current Implemented Techniques
- MinMax 
- Kmeans
- KNN (Weighted)
- Gabriel
### Future Implemented Techniques
- Optics
- Correlation (based on time series data)
- Dynamic Time Warping
- Mutual Information Coefficient
- Relative Neighborhood 

## Installation
SevenBridges is compatible with python 3.8 and above.
The simplest way to install SevenBridges is from PyPi:

```
pip install sevenbridges
```

## Requirements
* pandas
* networkx
* numpy
* scikit-learn
* scipy

## Usage
SevenBridges work very simple.
It will load in the data in either Pandas, Numpy or .csv file format.
However, keep in mind that the data must have the node names in the first column, and the longitude and latitude in the other two.
If no names are given, we will automatically assign names.

In the following example, we load in some example location data of 8 nodes as a Numpy array.
Then, we load in the graphcreator class and initiate the graphcreator instance.
Then, we try out the kmeans technique to find our graph!
```
latitude_longitude_data = np.array([
    [40.09068,116.17355],
     [40.00395,116.20531],
     [39.91441,116.18424],
     [39.81513,116.17115],
     [39.742767,116.13605],
     [39.987312,116.28745],
     [39.98205,116.3974],
     [39.95405,116.34899],
])

n_clusters = 2

from sevenbridges.graphcreator import graph_generator
generator = graph_generator()
generator.kmeans(latitude_longitude_data, n_clusters)
```

We can then see the information of our graph in the following way:
```
adj = generator.networkx_graph
print(adj)
>>> Graph with 8 nodes and 13 edges
```

To retrieve the numpy adjacency matrix:
```
nx.to_numpy_array(generator.networkx_graph)
>>> array([[0., 1., 0., 0., 0., 1., 1., 1.],
           [1., 0., 0., 0., 0., 1., 1., 1.],
           [0., 0., 0., 1., 1., 0., 0., 0.],
           [0., 0., 1., 0., 1., 0., 0., 0.],
           [0., 0., 1., 1., 0., 0., 0., 0.],
           [1., 1., 0., 0., 0., 0., 1., 1.],
           [1., 1., 0., 0., 0., 1., 0., 1.],
           [1., 1., 0., 0., 0., 1., 1., 0.]])
```


## Contributing
SevenBridges is an open-source project available on Github, and contributions of all types are welcome. Feel free to open a pull request if you have something interesting that you want to add to the framework.