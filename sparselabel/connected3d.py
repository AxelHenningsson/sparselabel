import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as cc
from numba import jit

def moments(row, col, frame, labels, weights):
    """Compute center of gravity of the voxel clusters described by labels.

    Args:
        row (`array like`): first index of 3d nonzero components of the 3D array structure.
        col (`array like`): second index of 3d nonzero components of the 3D array structure.
        frame (`array like`): third index of 3d nonzero components of the 3D array structure.
        labels (`array like`): The voxel cluster labels of the nonzero components.
        weights (`array like`): weights to take into account for CoG computation.

    Returns:
        `tuple`: The row, col and frame center of gravity of the voxel clusters as well
            as the sum of the weights of the clusters (rc,cc,fc,W).
    """
    index = np.argsort(labels)
    srow, scol, sframe, sweights = row[index], col[index], frame[index], weights[index]
    slabels = labels[index]
    breakpoints = np.insert(np.where(slabels[:-1] != slabels[1:])[0]+1, 0, 0, axis=0)
    W,cc,rc,fc = [],[],[],[]
    for b1,b2 in zip(breakpoints[0:],breakpoints[1:]):
        r = scol[b1:b2]
        c = srow[b1:b2]
        f = sframe[b1:b2]
        w = sweights[b1:b2]
        W.append( np.sum(w) )
        rc.append( r.dot(w) / W[-1] )
        cc.append( c.dot(w) / W[-1] )
        fc.append( f.dot(w) / W[-1] )
    r = scol[breakpoints[-1]:]
    c = srow[breakpoints[-1]:]
    f = sframe[breakpoints[-1]:]
    w = sweights[breakpoints[-1]:]
    W.append( np.sum(w) )
    rc.append( r.dot(w) / W[-1] )
    cc.append( c.dot(w) / W[-1] )
    fc.append( f.dot(w) / W[-1] )
    return np.array(rc), np.array(cc), np.array(fc), np.array(W)

def label(row, col, frame):
    """Label a 3D sparse array given the indices of the nonzero components.

    Labeling is done by constructing a sparse connected graph relating to the sparse
    3D image specified by the input indices row, col, frame. The graph connects the voxels
    in the 3d framestack. For a given (nonzero) voxel, at (i,j,k) the algorithm checks which
    of the forward neighbours (i+1,j,k), (i,j+1,k) and (i,j,k+1) that are nonzero (if any).
    These are then connected to the node number located at (i,j,k) constructing a graph that
    allows for forward movement within the local clusters. The graph is represented by a sparse
    matrix as described in scipy.sparse.csgraph.

    Args:
        row (`array like`): first index of 3d nonzero components of the 3D array structure.
        col (`array like`): second index of 3d nonzero components of the 3D array structure.
        frame (`array like`): third index of 3d nonzero components of the 3D array structure.

    Returns:
        tuple: The labels and the integer number of features. Each instance in the output labels
            array specifies the cluster index of the corresponding voxel in the 3D array. I.e if
            labels[i]==k then the voxel at position (row[i],col[i],frame[i]) belongs to the 
            spatially connected voxel cluster with index k.
    """
    graph_node_labels = np.linspace(1, len(row), len(row), dtype=np.int64)
    np.random.shuffle(graph_node_labels)
    graph = _get_graph(graph_node_labels, row, col, frame)
    num_features, labels = cc(csgraph=graph)
    return np.array( labels[graph_node_labels] ), num_features-1


@jit(nopython=True)
def _gritty_loop(frc_rows, 
                 frc_cols, 
                 frc_frames, 
                 frc_data,
                 fcr_cols, 
                 fcr_rows,
                 fcr_frames,
                 fcr_data,
                 rcf_cols,  
                 rcf_rows,  
                 rcf_frames,
                 rcf_data ):
    row, col = [], []
    for c in range(len(frc_rows)-1):

            if frc_rows[c]==frc_rows[c+1] and frc_frames[c]==frc_frames[c+1] and frc_cols[c]==frc_cols[c+1]-1:
                row.append(frc_data[c])
                col.append(frc_data[c+1])

            if fcr_cols[c]==fcr_cols[c+1] and fcr_frames[c]==fcr_frames[c+1] and fcr_rows[c]==fcr_rows[c+1]-1:
                row.append(fcr_data[c])
                col.append(fcr_data[c+1])

            if rcf_cols[c]==rcf_cols[c+1] and rcf_rows[c]==rcf_rows[c+1] and rcf_frames[c]==rcf_frames[c+1]-1 :
                row.append(rcf_data[c])
                col.append(rcf_data[c+1])

    return row, col

def _get_graph(graph_node_labels, rows, cols, frames):
    
    frc_index    = np.lexsort( (cols, rows, frames) )
    frc_cols     = cols[frc_index]
    frc_rows     = rows[frc_index]
    frc_frames   = frames[frc_index]
    frc_data     = graph_node_labels[frc_index]

    fcr_index    = np.lexsort( (frc_rows, frc_cols, frc_frames) )
    fcr_cols     = frc_cols[fcr_index]
    fcr_rows     = frc_rows[fcr_index]
    fcr_frames   = frc_frames[fcr_index]
    fcr_data     = graph_node_labels[fcr_index]

    rcf_index    = np.lexsort( (frc_frames, frc_cols, frc_rows) )
    rcf_cols     = frc_cols[rcf_index]
    rcf_rows     = frc_rows[rcf_index]
    rcf_frames   = frc_frames[rcf_index]
    rcf_data     = graph_node_labels[rcf_index]

    row,col = _gritty_loop( frc_rows, 
                            frc_cols, 
                            frc_frames, 
                            frc_data,
                            fcr_cols, 
                            fcr_rows,
                            fcr_frames,
                            fcr_data,
                            rcf_cols,  
                            rcf_rows,  
                            rcf_frames,
                            rcf_data )

    nid = np.max(graph_node_labels)
    return csr_matrix(([1]*len(row), (row, col)), shape=(nid+1, nid+1), dtype=np.int8)
