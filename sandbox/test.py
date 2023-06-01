from sparselabel.connected3d import label
import cProfile
import pstats
import h5py
import os

if __name__ == "__main__":
    
    pr = cProfile.Profile()
    pr.enable()

    with h5py.File(os.path.join(os.path.dirname(__file__),'sparse.h5') , "r") as hin:
        row = hin['14.1']['row'][:].astype(int)
        col = hin['14.1']['col'][:].astype(int)
        frame = hin['14.1']['frame'][:].astype(int)
        labels, num_features = label(row, col, frame)

    pr.disable()
    pr.dump_stats('tmp_profile_dump')
    ps = pstats.Stats('tmp_profile_dump').strip_dirs().sort_stats('cumtime')
    ps.print_stats(15)
