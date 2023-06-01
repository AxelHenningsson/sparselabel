# sparselabel
Label 3D sparse data blocks by finding the connected components of the related directed graph.

Labeling is done by constructing a sparse connected graph relating to the sparse
3D image specified by the input indices row, col, frame. The graph connects the voxels
in the 3d framestack. For a given (nonzero) voxel, at (i,j,k) the algorithm checks which
of the forward neighbours (i+1,j,k), (i,j+1,k) and (i,j,k+1) that are nonzero (if any).
These are then connected to the node number located at (i,j,k) constructing a graph that
allows for forward movement within the local clusters. The graph is represented by a sparse
matrix as described in scipy.sparse.csgraph. The 2D analogy of the graph construction is illustrated below:

![example](https://github.com/AxelHenningsson/sparselabel/assets/31615210/b9ca9116-7c0b-404e-acd4-6fc9c6954074)


# Example
Imagine the that we want to segment the binary dense array

```python
    import numpy as np
    array3d = np.zeros((5,5,5))
    array3d[0:2, 0:2, 1:3] = 1
    array3d[4, 4, 3:] = 1
```

The `array3d` may be represented as a sparse array by specifying the nonzero indices as

```python
    row   = np.array([0, 0, 1, 1, 0, 0, 1, 1, 4, 4], dtype=int)
    col   = np.array([0, 1, 0, 1, 0, 1, 0, 1, 4, 4], dtype=int)
    frame = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 4], dtype=int)
```

This means that `array3d[row, col, frame]!=0`. We can now label the sparse array using `sparselabel.connected3d.label` as

```python
    from sparselabel.connected3d import label
    labels, num_features = label(row, col, frame)
```

```
    >> out: labels=[1 1 1 1 1 1 1 1 2 2], num_features=2
```

The `labels` array specifies the voxel cluster index that each nonzero voxel in `array3d` belongs to. I.e `array3d[row[labels==k], col[labels==k], frame[labels==k]]` are all of the voxels that belong to cluster with index `k`. Cluster belonging is here defined from a 6-neighbourhood, i.e voxels sharing a common face will belong to the same cluster.

# Benchmarks
----------
Benchmarks using test arrays of shape=(2048 x 2048 x N) with Gaussian 3D voxel clusters with approximate diameter 15 voxels. The number of nonzero components was increased by copying and restacking the sparse frames to increase N.

![image](https://github.com/AxelHenningsson/sparselabel/assets/31615210/c33071f6-cfc7-4dd6-8249-7d328e92a952)

A sample 2048 x 2048 sparse frame from the benchmark dataset can be viewed below:
![image](https://github.com/AxelHenningsson/sparselabel/assets/31615210/cf047e68-3c62-4721-88ac-b530675d1953)


# Installation
----------
Clone the repo and go to the root

    git clone https://github.com/AxelHenningsson/sparselabel.git
    cd lauesim

Create a new pip environment and install

    python3 -m venv env
    pip install -e .

