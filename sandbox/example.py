# Imagine the that we want to segment the binary dense array
import numpy as np
array3d = np.zeros((5,5,5))
array3d[0:2, 0:2, 1:3] = 1
array3d[4, 4, 3:] = 1

# This may be represented as a sparse array by the nonzero indices as
row   = np.array([0, 0, 1, 1, 0, 0, 1, 1, 4, 4], dtype=int)
col   = np.array([0, 1, 0, 1, 0, 1, 0, 1, 4, 4], dtype=int)
frame = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 4], dtype=int)

# We can label the sparse array using the sparselabel package as
from sparselabel.connected3d import label
labels, N = label(row, col, frame)

# >> labels=[1 1 1 1 1 1 1 1 2 2], N=2

