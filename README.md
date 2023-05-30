# sparselabel
Label 3D sparse data blocks by finding the connected components of the related directed graph.

Labeling is done by constructing a sparse connected graph relating to the sparse
3D image specified by the input indices row, col, frame. The graph connects the voxels
in the 3d framestack. For a given (nonzero) voxel, at (i,j,k) the algorithm checks which
of the forward neighbours (i+1,j,k), (i,j+1,k) and (i,j,k+1) that are nonzero (if any).
These are then connected to the node number located at (i,j,k) constructing a graph that
allows for forward movement within the local clusters. The graph is represented by a sparse
matrix as described in scipy.sparse.csgraph. The 2D analogy of the graph construction is illustrated below:


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
