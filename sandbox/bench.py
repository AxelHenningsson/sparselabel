from sparselabel.connected3d import label
import cProfile
import pstats
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt

if __name__ == "__main__":

    pr = cProfile.Profile()
    pr.enable()
    npeaks   = 640
    peaksize = 15
    shape = (32, 2024, 2024)
    image = np.zeros((np.prod(shape)))
    #index = np.random.choice(len(image), npeaks, replace=False)
    index = np.random.randint(0, high=len(image), size=npeaks)
    image[index] = np.random.rand(len(index), )
    image = image.reshape(shape)
    image = gaussian_filter(image, sigma=peaksize/5.)
    image[image<1e-4] = 0
    plt.imshow(image[0,:,:])
    plt.show()

    row, col, frame = [],[],[]
    for f,im in enumerate(image):
        sim = coo_matrix(im)
        row.extend( sim.row )
        col.extend( sim.col )
        frame.extend([f]*len(sim.row)) # 32

    row.extend( row )
    col.extend( col )
    frame.extend( frame ) # 64

    row.extend( row )
    col.extend( col )
    frame.extend( frame ) # 128

    pr.disable()
    pr.dump_stats('tmp_profile_dump')
    ps = pstats.Stats('tmp_profile_dump').strip_dirs().sort_stats('cumtime')
    ps.print_stats(15)
    print("")
    pr = cProfile.Profile()
    pr.enable()

    row, col, frame = np.array(row),np.array(col),np.array(frame)
    labels, num_features = label(row, col, frame)

    pr.disable()
    pr.dump_stats('tmp_profile_dump')
    ps = pstats.Stats('tmp_profile_dump').strip_dirs().sort_stats('cumtime')
    ps.print_stats(15)
