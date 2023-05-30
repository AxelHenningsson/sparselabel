# def save_as_vtk_particles(filename, row, col, frame, intensity, label):
#     coordinates = np.vstack((row, col, frame)).T

#     cells = [("vertex", np.array([[i] for i in range(coordinates.shape[0])]) )]
#     meshio.Mesh(
#         coordinates,
#         cells,
#         point_data={"label" : label, "intensity": intensity},
#         ).write(filename)

if __name__ == "__main__":



    # imstack = np.random.rand(256,256,256)
    # imstack[imstack < 0.999995] = 0
    # imstack = gaussian_filter(imstack, sigma=1)
    # imstack[imstack<1e-6]=0
    # imstack = imstack*1e4

    # row, col, frame = [],[],[]
    # intensity = []
    # for f,image in enumerate(imstack):
    #     I = coo_matrix(image)
    #     row.extend( I.row )
    #     col.extend( I.col )
    #     frame.extend( [f]*len(I.col) )
    #     intensity.extend( I.data )

    # row = np.array(row).astype(int)
    # col = np.array(col).astype(int)
    # frame = np.array(frame).astype(int)
    # intensity = np.array(intensity).astype(float)

    pr = cProfile.Profile()
    pr.enable()

    with h5py.File('sparse.h5' , "r") as hin:
        row = hin['14.1']['row'][:].astype(int)
        col = hin['14.1']['col'][:].astype(int)
        frame = hin['14.1']['frame'][:].astype(int)
        intensity = hin['14.1']['intensity'][:].astype(int)
    label = segment(row, col, frame)


    # with h5py.File("sparse_labeled.h5", "r") as hin:
    #     dset = hin['14.1'].create_dataset("label", (len(label),), dtype='i')
    #     hin['14.1']['label'][:] = label

    pr.disable()
    pr.dump_stats('tmp_profile_dump')
    ps = pstats.Stats('tmp_profile_dump').strip_dirs().sort_stats('cumtime')
    ps.print_stats(15)

    shape = (2048, 2048)

    save_as_vtk_particles('blobs3d.xdmf', row, col, frame, intensity, label)

    fig,ax = plt.subplots(2,2, sharex=True, sharey=True)
    mask = frame==1
    image = coo_matrix((intensity[mask], (row[mask], col[mask])), shape=shape, dtype=np.int32)
    segi = coo_matrix((label[mask], (row[mask], col[mask])), shape=shape, dtype=np.int32)
    segi  = np.array(segi.todense())
    image = np.array(image.todense())
    ax[0,0].imshow(image)
    ax[0,1].imshow( segi )
    mask = frame==2
    image = coo_matrix((intensity[mask], (row[mask], col[mask])), shape=shape, dtype=np.int32)
    segi = coo_matrix((label[mask], (row[mask], col[mask])), shape=shape, dtype=np.int32)
    segi  = np.array(segi.todense())
    image = np.array(image.todense())
    ax[1,0].imshow(image)
    ax[1,1].imshow( segi )

    plt.show()