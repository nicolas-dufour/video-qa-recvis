import numpy as np
import h5py
from tqdm.notebook import tqdm

def reformat_frames(input_file,output_file):
    input_fts = h5py.File(input_file,'r')
    keys = list(input_fts.keys())
    keys = [n.encode("ascii", "ignore") for n in list(input_fts.keys())]
    dataset_size = len(keys)
    C,F, D = input_fts[keys[0]].shape
    ids ={}
    with h5py.File(output_file,'w') as fd:
        feat_dset = fd.create_dataset('resnet_features', (dataset_size, C, F, D),dtype=np.float32)
        video_ids_dset = fd.create_dataset('ids', (dataset_size,), h5py.string_dtype("ascii") ,keys)
        for i,key in enumerate(tqdm(keys)):
            feat_dset[i]=input_fts[key]