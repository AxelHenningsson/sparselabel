import unittest
import numpy as np
from scipy.sparse import coo_matrix
from sparselabel.connected3d import label

class TestLabel(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def test_label(self):

        imstack = np.zeros((25,25,25))
        imstack[1:4, 0:4, 0:4] = 1
        imstack[5, 5:6, 5:7] = 2
        imstack[7, 7, 7:9] = 3
        imstack[0:4, 0:4, 5:] = 4

        row, col, frame, val = [],[],[],[]
        for f,image in enumerate(imstack):
            I = coo_matrix(imstack[:,:,f])
            row.extend( I.row )
            col.extend( I.col )
            frame.extend( [f]*len(I.col) )
            val.extend( I.data )

        row   = np.array(row).astype(int)
        col   = np.array(col).astype(int)
        frame = np.array(frame).astype(int)
        val   = np.array(val).astype(float)

        labels, N = label(row, col, frame)
        self.assertEqual(N, np.max(imstack))

        l1 = labels[(row==1)*(col==0)*(frame==0)]
        l2 = labels[(row==5)*(col==5)*(frame==5)]
        l3 = labels[(row==7)*(col==7)*(frame==7)]
        l4 = labels[(row==0)*(col==0)*(frame==5)]

        vv = val.copy()
        vv[val==1]=l1
        vv[val==2]=l2
        vv[val==3]=l3
        vv[val==4]=l4

        for i in range(len(labels)):
            self.assertEqual(vv[i], labels[i])


if __name__ == '__main__':
    unittest.main()
