#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 09:59:05 2022

@author: Amit Jaiswal(1,2)
1.MEGIN OY, Espoo, Finland 
2.NBE, Aalto University, Espoo, Finland
Contact: amit.jaiswal@megin.fi OR amit.jaiswal@aalto.fi

Using pseudo-MRI engine version 1.0
Jaiswal, A., Nenonen, J., & Parkkonen, L., Pseudo-MRI engine for MRI-free
electromagnetic source imaging. Submitted to Human Brain Mapping in Apr.
2024.
pseudo-MRI engine is a research tool intended for generating pseudo-MRI, 
i.e., the subject-spectific segmented set of a template MRI. 
"""
print(__doc__)
import nimi; del nimi

#%% 1. Set subject and directories
import argparse
parser = argparse.ArgumentParser(description='pseudo-MRI engine')
parser.add_argument('-s',       '--subject',        default=None,  type=str,  help='subject name')
parser.add_argument('-s',       '--template',       default=None,  type=str,  help='Template MRI name')
parser.add_argument('-s',       '--template_dir',   default=None,  type=str,  help='Template MRI parent directory')
parser.add_argument('-dig',     '--isotrakfile',    default=None,  type=str,  help='MEG file with isotrak information (default=None)')
parser.add_argument('-trans',   '--trans_file',     default=None,  type=str,  help='Co-registration (sRAS-to-head) file (default=None)')
parser.add_argument('--dense_hsp',          action='store_true',      help='densify HSP?')
parser.add_argument('--show_good_hsps_idx', action='store_true',      help='show good hsps idices?')
parser.add_argument('-v', '--verbose',      action='store_true',      help='verbose mode or not?')
parser.add_argument('--nmax_Ctrl',          default=200,    type=int, help='Number of maximum control points.')
args = parser.parse_args()

# Load modules
from mne.surface import read_surface
import os
import meginpy
n_jobs = os.cpu_count()
print("\nhostname \t= %s \n#cores \t\t= %d" % (os.uname().nodename, n_jobs))
meginpy.utils.tic()

#%% Generate pseudo MRI
templ_surf_fname  = '%s/%s/bem/watershed/%s_outer_skin_surface'%(args.templates_dir, args.template, args.template)
template_headsurf = read_surface(templ_surf_fname, return_dict=True)[2]
template_headsurf['rr'] /= 1000  # covert to meter    
template_headsurf['rr'] *= 0.975 # shrink inward to reduce mismatch (FIX it by checking the mean distance for points above fids) 

args.use_hpi             = True
args.show_good_hsps_idx  = False
args.rem_good_pts_idx    = []
args.which_mri_to_warp   = ['T1.mgz', 'brain.mgz'][:1] # 'all'

#%% Run PseudoMRIGenerator_ver1
meginpy.pseudoMRI.PseudoMRIGenerator_ver1(args.isotrak_file, args.template, args.templates_dir, args.fiducial_file, args.pseudo_subject, 
                                          args.pseudo_subjects_dir, isotrak_from=args.subject, dense_hsp=False, mirror_hsps=True, 
                                          template_headsurf=template_headsurf, warp_anatomy=True, which_mri=args.which_mri_to_warp, 
                                          write2format=['.mgz'], n_jobs=n_jobs, save_pseudomri_plot=True,  
                                          isotrak_reject_min_max=[2, 10], use_hpi=args.use_hpi, 
                                          show_good_hsps_idx=args.show_good_hsps_idx, rem_good_pts_idx=args.rem_good_pts_idx,
                                          nmax_Ctrl=args.nmax_Ctrl)
meginpy.utils.toc()
