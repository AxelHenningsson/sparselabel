from sparselabel.connected3d import label
import cProfile
import pstats
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import time
import os

if __name__ == "__main__":


    if 0:
        npeaks   = 320
        peaksize = 15
        shape = (16, 2048, 2048)
        image = np.zeros((np.prod(shape)))
        index = np.random.randint(0, high=len(image), size=npeaks)
        image[index] = np.random.rand(len(index), )
        image = image.reshape(shape)
        image = gaussian_filter(image, sigma=peaksize/5.)
        image[image<1e-4] = 0
        plt.imshow(image[0,:,:])
        plt.show()
        raise

        row, col, frame = [],[],[]
        for f,im in enumerate(image):
            sim = coo_matrix(im)
            row.extend( sim.row )
            col.extend( sim.col )
            frame.extend([f]*len(sim.row))

        np.save(os.path.join(os.path.dirname(__file__), 'row.npy'), row)
        np.save(os.path.join(os.path.dirname(__file__), 'col.npy'), col)
        np.save(os.path.join(os.path.dirname(__file__), 'frame.npy'), frame)

    row   = list(np.load(os.path.join(os.path.dirname(__file__), 'row.npy')))
    col   = list(np.load(os.path.join(os.path.dirname(__file__), 'col.npy')))
    frame = list(np.load(os.path.join(os.path.dirname(__file__), 'frame.npy')))

    c = len(row)//2
    rincs = [ row[0:c], row[c:] ]
    cincs = [ col[0:c], col[c:]  ]
    fincs = [ frame[0:c],  frame[c:] ]

    times = []
    nnz = []

    # warmup: 
    labels, num_features = label(np.array(row),np.array(col),np.array(frame))
    for i in range(10):
        ff = np.max(frame)
        for r,c,f in zip(rincs, cincs, fincs):
            print('nnz', len(row), 'stackshape', np.max(frame) )
            pr = cProfile.Profile()
            pr.enable() 
            t1 = time.perf_counter()
            labels, num_features = label(np.array(row),np.array(col),np.array(frame))
            t2 = time.perf_counter()
            pr.disable()
            pr.dump_stats('tmp_profile_dump')
            ps = pstats.Stats('tmp_profile_dump').strip_dirs().sort_stats('cumtime')
            ps.print_stats(15)
            
            nnz.append(len(row))
            times.append(t2-t1)
            print(times[-1])

            row.extend( r )
            col.extend( c )
            frame.extend( list(np.array(f)+ff) )

    plt.style.use('seaborn-v0_8-dark')
    plt.figure(figsize=(12,6))
    plt.title('Benchmark for Labeling 3d Sparse Arrays', fontsize=16)
    plt.plot(nnz, times, 'ko--')
    plt.xlabel('Number of Nonzero Elements in 3d Sparse Array', fontsize=16)
    plt.ylabel('Time to Label 3d Sparse Array [s]', fontsize=16)
    plt.show()




