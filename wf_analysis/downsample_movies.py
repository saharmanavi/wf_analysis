import os
import h5py
import shutil
import time
import numpy as np
import tables as tb
import pdb


class DownsampleMovies(object):
    def __init__(self, name_str, spatial_compression, temporal_compression, raw_movie_path, chunk_dir, final_dir=None, create=True, concat=True):
        
        self.label = name_str
        self.sc = spatial_compression
        self.tc = temporal_compression
        
        f = h5py.File(raw_movie_path, 'r+')
        self.movie = f['data']
        
        self.movie_len = self.movie.shape[0]
        self.movie_size = self.movie.shape[1]
        self.cam = os.path.split(raw_movie_path)[-1].split('CamF_')[1].split('_')[0]

        self.chunk_dir = chunk_dir
        if final_dir is not None:
            self.final_dir = final_dir
        else:
            self.final_dir = self.chunk_dir

        self.local_dir = os.path.join(self.chunk_dir, "chunks_{}_tc{}".format(self.label, self.tc))

        if create==True:
            self.chunk_it()
        if concat==True:
            self.concat_it()



    def chunk_it(self):
             
        if os.path.exists(self.local_dir)==False:
            os.makedirs(self.local_dir)

        chunks = np.arange(0,self.movie_len,10000)
        for n, c in enumerate(chunks):
            print 'making array {}'.format(n)
            try:
                v1 = self.movie[chunks[n]:chunks[n+1], :, :]
                output_name = "{}_{}_{}_{}_{}_tc{}.h5".format(chunks[n]/self.tc, chunks[n+1]/self.tc, self.label, self.cam, self.movie_size/self.sc, self.tc)
            except IndexError:
                v1 = self.movie[chunks[n]:, :, :]
                output_name = "{}_end_{}_{}_{}_tc{}.h5".format(chunks[n]/self.tc, self.label, self.cam, self.movie_size/self.sc, self.tc)

            print 'decimating {} of {}'.format(n, len(chunks)-1) 
            self.decimate_JCamF(input_array = v1, 
                            output_file = os.path.join(self.local_dir, output_name), 
                            spatial_compression = self.sc, 
                            temporal_compression = self.tc)

            del v1
        print "=====DONE W/DECIMATION====="


    def concat_it(self):

        xfer_loc = os.path.join(self.final_dir, self.label)
        if os.path.exists(xfer_loc)==False:
            os.makedirs(xfer_loc)
        
        file_name = "{}_{}_{}x{}_tc{}.h5".format(self.label, self.cam, self.movie_size/self.sc, self.movie_size/self.sc, self.tc)
        output_file = os.path.join(xfer_loc, file_name)
        chunks_loc = self.local_dir
        length = int(self.movie_len/self.tc)
        
        print "starting {}".format(file_name)
        self.concat_all_chunks(output_file, chunks_loc, length)
        print '====={} XFER COMPLETE====='.format(self.label)    



    def decimate_JCamF(self, input_array, output_file, spatial_compression, temporal_compression):

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
            if 'cam1' in self.cam:
                data = np.flip(data, 1)

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

    def concat_all_chunks(self, output_file, chunks_loc, length):
        ds = h5py.File(output_file, 'w')
        shape = int(self.movie_size/self.sc)
        dset = ds.create_dataset("data", (length, shape, shape))

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
            dset[start:end, :shape, :shape] = c.value
            
            chunk.close()
            del c

        ds.close()
        del dset




if __name__ == "__main__":
    
    thing = {'181009_M395929_hemo_blank': r"C:\Users\saharm\Desktop\movie_folder\181009JCamF_cam2_200.dcimg_2_2_1.h5",
            # '181017_M395929_hemo': r"C:\Users\saharm\Desktop\movie_folder\181017JCamF_cam1_100.dcimg_2_2_1.h5"
            }

    # for t in thing.keys():
    # for tc in [1, 5]:
    DownsampleMovies(name_str = '181009_M395929_hemo_DoC', 
                    spatial_compression=2, 
                    temporal_compression=1, 
                    raw_movie_path=r"\\ALLEN\programs\braintv\workgroups\nc-ophys\CorticalMapping\IntrinsicImageData\181009-M395929\DoC\181009JCamF_cam1_100.dcimg_2_2_1.h5", 
                    chunk_dir=r"C:\Users\saharm\Desktop\movie_folder", 
                    final_dir=r"\\allen\programs\braintv\workgroups\nc-ophys\Sahar",
                    create=True, concat=True)


