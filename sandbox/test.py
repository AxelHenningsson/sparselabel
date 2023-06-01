from sparselabel.connected3d import label
import sparselabel
import cProfile
import pstats
import h5py
import os
import numpy as np

if __name__ == "__main__":
    
    pr = cProfile.Profile()
    pr.enable()

    with h5py.File(os.path.join(os.path.dirname(__file__),'sparse.h5') , "r") as hin:
        row = hin['14.1']['row'][:].astype(int)
        col = hin['14.1']['col'][:].astype(int)
        frame = hin['14.1']['frame'][:].astype(int)
        labels, num_features = label(row, col, frame)
    print(np.max(labels))
    pr.disable()
    pr.dump_stats('tmp_profile_dump')
    ps = pstats.Stats('tmp_profile_dump').strip_dirs().sort_stats('cumtime')
    ps.print_stats(15)

    pr = cProfile.Profile()
    pr.enable()

    import os
    os.environ['SLURM_CPUS_PER_TASK'] = "1"
    labels = sparselabel.connected3d.ImageD11_cp(os.path.join(os.path.dirname(__file__),'sparse.h5'), '14.1')
    print(np.max(labels))

    pr.disable()
    pr.dump_stats('tmp_profile_dump')
    ps = pstats.Stats('tmp_profile_dump').strip_dirs().sort_stats('cumtime')
    ps.print_stats(15)

