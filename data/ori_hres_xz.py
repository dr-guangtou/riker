#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reducing the Illustris high-resolution map using X-Y projection."""

import os

import numpy as np

from functools import partial
from multiprocessing import Pool

from riker.galaxy import GalaxyMap
from riker.data import BeneMassAgeZMaps

# Input HDF5 from simulation
ori_dir = '/Users/song/data/massive/simulation/riker/ori/'

# HDF5 files
ori_file = os.path.join(ori_dir, 'galaxies_orig_108_agez_highres.hdf5')

# Label of the survey
ori_label = 'ori100_z0.4_hres'

# Ready the data file
ori_data = BeneMassAgeZMaps(ori_file, label=ori_label)

# Number of galaxies 
n_gal = ori_data.n_gal

# Projection
proj = 'xz'

# Close the file
ori_data.data.flush()
ori_data.data.close()

n_jobs = 1

# Worker function
def reduce(idx, proj='xz'):
    """Reduce a single galaxy."""
    print("# Deal with galaxy: {}".format(idx))
    # Input HDF5 from simulation
    hdf5_dir = '/Users/song/data/massive/simulation/riker/ori/'
    hdf5_file = os.path.join(ori_dir, 'galaxies_orig_108_agez_highres.hdf5')

    # Label of the survey
    sim_label = 'ori100_z0.4_hres'

    # Ready the data file
    # Need to put this inside the worker function to use multiprocessing
    # See: https://github.com/h5py/h5py/issues/1092
    hdf5_data = BeneMassAgeZMaps(hdf5_file, label=sim_label)

    # Deal with each galaxy
    gal = GalaxyMap(hdf5_data, idx, proj=proj, aper_force=None)
    gal.run_all(plot=False, output=False)

    # Close the HDF5 file
    hdf5_data.data.close()


reduce_one = partial(reduce, proj=proj)

if n_jobs == 1:
    for idx in np.arange(n_gal):
        reduce_one(idx)
else:
    with Pool(processes=n_jobs) as p:
        p.map(reduce_one, np.arange(n_gal))

