import unittest
import numpy as np
from scipy.sparse import coo_matrix
from sparselabel.connected3d import label, moments

class TestLabel(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def test_label(self):

        imstack = np.zeros((5,5,5))
        imstack[2,2,2] = 1
        imstack[2,2,3] = 2
        imstack[4,4,4] = 1.436

        row, col, frame, weights = [],[],[],[]
        for f,image in enumerate(imstack):
            I = coo_matrix(imstack[:,:,f])
            row.extend( I.row )
            col.extend( I.col )
            frame.extend( [f]*len(I.col) )
            weights.extend( I.data )

        row       = np.array(row).astype(int)
        col       = np.array(col).astype(int)
        frame     = np.array(frame).astype(int)
        weights   = np.array(weights).astype(float)

        labels, N = label(row, col, frame)
        rc, cc, fc, W = moments(row, col, frame, labels, weights)

        self.assertAlmostEqual(rc[0], 2)
        self.assertAlmostEqual(cc[0], 2)
        self.assertAlmostEqual(fc[0], 2.66666666666667)
        self.assertAlmostEqual(W[0], 3)

        self.assertAlmostEqual(rc[1], 4)
        self.assertAlmostEqual(cc[1], 4)
        self.assertAlmostEqual(fc[1], 4)
        self.assertAlmostEqual(W[1], 1.436)


if __name__ == '__main__':
    unittest.main()
