"""
Author: Amit Jaiswal <amit.jaiswal@megin.fi> <amit.jaiswal@aalto.fi>

"""
# fmt:off
import numpy as np
# from nibabel import load as nib_load
# import mne
# import pyvista as pv
# from .oncology import (get_mask_data, plot_mri_planes_and_surfaces, \
#                        plot_mri_planes, read_tumor_blob_from_3DSlicer)
# from skimage.measure import marching_cubes
# from copy import deepcopy
# import os
# from os.path import join, splitext, dirname, exists
# from .viz import myMlabTriagularMesh, myMlabPoint3d
# from mne.utils import run_subprocess
# from trimesh import Trimesh
# import numpy as np
# from .utils import min_mean_max, run
# import nibabel as nib
# from . import pseudoMRI
# import tqdm
# import pydicom
# from pathlib import Path 

# def tic():
#     # Homemade version of matlab tic and toc functions
#     global startTime_for_tictoc
#     startTime_for_tictoc = time()
# def toc():
#     if 'startTime_for_tictoc' in globals():
#         print("Elapsed time is " + str(time() - startTime_for_tictoc) + " seconds.")
#     else:
#         print("Toc: start time not set") 