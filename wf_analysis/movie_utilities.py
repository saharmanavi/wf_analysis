import os
import h5py
import shutil
import time
import numpy as np
import tables as tb
import pdb






def decimate_JCamF(input_array, output_file, spatial_compression, temporal_compression):

    ia = input_array
    height = width = ia.shape[1]
    number_of_frames = ia.shape[0]
    number_of_frame_chunks = int(ia.shape[0]*1./temporal_compression)


    fd = tb.open_file(output_file, 'w')
    # fd = output_file
    filters = tb.Filters(complevel=1, complib='blosc')
    hdf5data = fd.create_earray(fd.root, 
                    'data', 
                    tb.Float64Atom(), 
                    filters=filters,
                    shape=(0, int(height*1./spatial_compression),int(width*1./spatial_compression)))

    for frm in range(number_of_frame_chunks):
        # data = ia[frm].reshape((temporal_compression,height,width))

        start = frm*temporal_compression
        end = (frm+1)*temporal_compression
        data = ia[start:end, :height, :width]
        data = np.transpose(data, (0, 2, 1))

        D = np.zeros((int(height*1./spatial_compression), int(width*1./spatial_compression)), dtype=np.float64)
        for ii in range(spatial_compression):
            for jj in range(spatial_compression):
                for kk in range(temporal_compression): 
                    D += data[kk, ii::spatial_compression,jj::spatial_compression]

        # Normalize:
        D -= spatial_compression*spatial_compression*temporal_compression     # Corrects for single-bit errors in raw data, with line above
        D /= spatial_compression*spatial_compression*temporal_compression

        D.reshape((np.prod(D.shape),))
        hdf5data.append(D[None])

    fd.close()
    del ia

def concat_all_chunks(output_file, chunks_loc, length):
    ds = h5py.File(output_file, 'w')
    dset = ds.create_dataset("data", (length, 256, 256))

    for f in os.listdir(chunks_loc):
        f_path = os.path.join(chunks_loc, f)
        chunk = h5py.File(f_path, 'r')
        c = chunk['data']

        start = int(f.split("_")[0])
        try:
            end = int(f.split("_")[1])
        except ValueError:
            end = length

        print "starting {}-{} chunk".format(start, end)
        dset[start:end, :256, :256] = c.value
        
        chunk.close()
        del c

    ds.close()
    del dset


def full_method(date, mouse, raw_movie_path, spatial_compression, temporal_compression, suffix=False, create=True, xfer=True):
    if suffix is not None:
        label = "{}_{}_{}".format(date, mouse, suffix)
    else:
        label = "{}_{}".format(date, mouse)

    f = h5py.File(raw_movie_path, 'r+')
    h = f['data']
    movie_len = h.shape[0]
    chunks = np.arange(0,movie_len,10000)

    local_dir = os.path.join(r"C:\Users\saharm\Desktop\movie_folder", "decimated_chunks_{}".format(label))
        

    if create==True:
        if os.path.exists(local_dir)==False:
            os.makedirs(local_dir)

        for n, c in enumerate(chunks):
            print 'making array {}'.format(n)
            try:
                v1 = h[chunks[n]:chunks[n+1], :, :]
                output_name = "{}_{}_{}_cam2_256x256_{}hz.h5".format(chunks[n]/tc, chunks[n+1]/tc, label, hz)
            except IndexError:
                v1 = h[chunks[n]:, :, :]
                output_name = "{}_end_{}_cam2_256x256_{}hz.h5".format(chunks[n]/tc, label, hz)


            print 'decimating {} of {}'.format(n, len(chunks)-1) 
            decimate_JCamF(input_array = v1, 
                            output_file = os.path.join(save_dir, output_name), 
                            spatial_compression = 2, 
                            temporal_compression = tc)

            del v1
        print "=====DONE W/DECIMATION====="


if __name__ == "__main__":
    
    ########################
    create = True
    xfer = True
    date = '181003'
    mouse = 'M395926_hemo'
    ########################


    label = "{}_{}".format(date, mouse)
    if create==True:
        ########################
        path = r"C:\Users\saharm\Desktop\movie_folder\181003JCamF_cam1_100.dcimg_2_2_1.h5"       
        hz = 100
        ########################
        
        save_dir = os.path.join(r"C:\Users\saharm\Desktop\movie_folder", "decimated_chunks_{}hz_{}".format(hz, mouse))
        if os.path.exists(save_dir)==False:
            os.makedirs(save_dir)

        f = h5py.File(path, 'r+')
        h = f['data']
        chunks = np.arange(0,362000,10000)
        tc = int(100./hz)

        for n, c in enumerate(chunks):
            print 'making array {}'.format(n)
            try:
                v1 = h[chunks[n]:chunks[n+1], :, :]
                output_name = "{}_{}_{}_cam2_256x256_{}hz.h5".format(chunks[n]/tc, chunks[n+1]/tc, label, hz)
            except IndexError:
                v1 = h[chunks[n]:, :, :]
                output_name = "{}_end_{}_cam2_256x256_{}hz.h5".format(chunks[n]/tc, label, hz)


            print 'decimating {} of {}'.format(n, len(chunks)-1) 
            decimate_JCamF(input_array = v1, 
                            output_file = os.path.join(save_dir, output_name), 
                            spatial_compression = 2, 
                            temporal_compression = tc)

            del v1
        print "=====DONE W/DECIMATION====="

    if xfer==True:
        ########################
        hzs = [100]
        ########################

        xfer_loc = os.path.join(r"\\allen\programs\braintv\workgroups\nc-ophys\Sahar", label)
        if os.path.exists(xfer_loc)==False:
            os.makedirs(xfer_loc)
        print xfer_loc
        for hz in hzs:
            file_name = "{}_256x256_{}hz.h5".format(label, hz)
            output_file = os.path.join(xfer_loc, file_name)
            chunks_loc = os.path.join(r"C:\Users\saharm\Desktop\movie_folder", "decimated_chunks_{}hz_{}".format(hz, mouse))
            length = int(hz/100.*362000)
            print "starting {}".format(file_name)
            concat_all_chunks(output_file, chunks_loc, length)
            print '====={}hz XFER COMPLETE====='.format(hz)    


    ########################
    create = True
    xfer = True
    date = '181017'
    mouse = 'M395929_hemo'
    ########################


    label = "{}_{}".format(date, mouse)
    if create==True:
        ########################
        path = r"C:\Users\saharm\Desktop\movie_folder\181017JCamF_cam1_100.dcimg_2_2_1.h5"      
        hz = 100
        ########################
        
        save_dir = os.path.join(r"C:\Users\saharm\Desktop\movie_folder", "decimated_chunks_{}hz_{}".format(hz, mouse))
        if os.path.exists(save_dir)==False:
            os.makedirs(save_dir)

        f = h5py.File(path, 'r+')
        h = f['data']
        chunks = np.arange(0,362000,10000)
        tc = int(100./hz)

        for n, c in enumerate(chunks):
            print 'making array {}'.format(n)
            try:
                v1 = h[chunks[n]:chunks[n+1], :, :]
                output_name = "{}_{}_{}_cam2_256x256_{}hz.h5".format(chunks[n]/tc, chunks[n+1]/tc, label, hz)
            except IndexError:
                v1 = h[chunks[n]:, :, :]
                output_name = "{}_end_{}_cam2_256x256_{}hz.h5".format(chunks[n]/tc, label, hz)


            print 'decimating {} of {}'.format(n, len(chunks)-1) 
            decimate_JCamF(input_array = v1, 
                            output_file = os.path.join(save_dir, output_name), 
                            spatial_compression = 2, 
                            temporal_compression = tc)

            del v1
        print "=====DONE W/DECIMATION====="

    if xfer==True:
        ########################
        hzs = [100]
        ########################

        xfer_loc = os.path.join(r"\\allen\programs\braintv\workgroups\nc-ophys\Sahar", label)
        if os.path.exists(xfer_loc)==False:
            os.makedirs(xfer_loc)
        print xfer_loc
        for hz in hzs:
            file_name = "{}_256x256_{}hz.h5".format(label, hz)
            output_file = os.path.join(xfer_loc, file_name)
            chunks_loc = os.path.join(r"C:\Users\saharm\Desktop\movie_folder", "decimated_chunks_{}hz_{}".format(hz, mouse))
            length = int(hz/100.*362000)
            print "starting {}".format(file_name)
            concat_all_chunks(output_file, chunks_loc, length)
            print '====={}hz XFER COMPLETE====='.format(hz)    

