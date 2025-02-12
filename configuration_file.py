#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 09:59:05 2022
@author: Amit Jaiswal(1,2)
1.MEGIN OY, Espoo, Finland 
2.NBE, Aalto University, Espoo, Finland
Contact: amit.jaiswal@megin.fi OR amit.jaiswal@aalto.fi
Mid-level configuration file for pseudo-MRI-engine
"""
import numpy as np
from os import cpu_count, listdir, remove, unlink, symlink
from os.path import exists, join, islink
from glob2 import glob
from mne.surface import read_surface
from shutil import copyfile

def check_set_input_config(args):
    # checks directories
    assert args.pseudo_MRI_name is not None
    assert exists(join(args.template_MRI_dir, args.template_MRI_name))
    args.pseudo_MRI_dir = args.template_MRI_dir if args.pseudo_MRI_dir in [None, ''] else args.pseudo_MRI_dir
    args.pseudo_MRI_path = join(args.pseudo_MRI_dir, args.pseudo_MRI_name)
    assert (not exists(args.pseudo_MRI_path) or not listdir(args.pseudo_MRI_path))
    
    # set and set fiducial file for the template
    if not args.preauri_loc in  ['CrusHelix', 'Targus', 'ITnotch', 'original']:
        args.preauri_loc = '_original'  
    else:
        args.preauri_loc = ('_' + args.preauri_loc)        
    args.def_fiducial_file = join(args.template_MRI_dir, args.template_MRI_name, 
                             'bem', f'{args.template_MRI_name}-fiducials.fif')
    remove(args.def_fiducial_file) if exists(args.def_fiducial_file) else None
    unlink(args.def_fiducial_file) if islink(args.def_fiducial_file) else None
    if args.fiducial_file in [None, '']:
        args.fiducial_file = args.def_fiducial_file.replace('.fif', f'{args.preauri_loc}.fif')
    print(f'\nUsing {args.fiducial_file} as the template MRI fiducial.\n')
    # try:
    #     symlink(args.fiducial_file, args.def_fiducial_file)
    # except OSError:
    copyfile(args.fiducial_file, args.def_fiducial_file)
    print(args.template_MRI_dir, args.template_MRI_name, '\n')

    args.n_jobs            = cpu_count()-1  if not hasattr(args, 'n_jobs')             else int(args.n_jobs)    
    args.dense_hsp         = False          if not hasattr(args, 'dense_hsp')          else args.dense_hsp 
    args.mirror_hsps       = True           if not hasattr(args, 'mirror_hsps')        else args.mirror_hsps
    args.template_headsurf = 'outer_skin'   if not hasattr(args, 'template_headsurf')  else args.template_headsurf
    args.dense_surf        = False          if not hasattr(args, 'dense_surf')         else args.dense_surf
    args.z_thres           = 0.020          if not hasattr(args, 'z_thres')            else args.z_thres 
    args.destHSPsShiftInwrd= 0.0015         if not hasattr(args, 'destHSPsShiftInwrd') else args.destHSPsShiftInwrd
    args.Wreg_est          = 1e-10          if not hasattr(args, 'Wreg_est')           else args.Wreg_est
    args.Wreg_apply        = 1e-10          if not hasattr(args, 'Wreg_apply')         else args.Wreg_apply
    args.wtol              = 1e-06          if not hasattr(args, 'wtol')               else int(args.wtol)
    args.warp_anatomy      = True           if not hasattr(args, 'warp_anatomy')       else args.warp_anatomy
    args.blocksize         = 500000         if not hasattr(args, 'blocksize')          else int(args.blocksize)
    args.write2format      = ['.mgz']       if not hasattr(args, 'write2format')       else args.write2format
    args.use_hpi           = True           if not hasattr(args, 'use_hpi')            else args.use_hpi
    args.rem_good_pts_idx  = []             if not hasattr(args, 'rem_good_pts_idx')   else args.rem_good_pts_idx
    args.dig_reject_min_max= [2, 10]        if not hasattr(args, 'dig_reject_min_max') else args.dig_reject_min_max
    args.nmax_Ctrl         = 200            if not hasattr(args, 'nmax_Ctrl')          else int(args.nmax_Ctrl)
    args.which_mris        = 'all'          if not hasattr(args, 'which_mris')          else args.which_mris
    
    for arg in list(args.__dict__.keys()):
        print('%s: %s'%(arg.ljust(18), vars(args)[arg])) if not 'Browse' in arg else None
        
    # plotting-related
    args.show_good_hsps_idx= False          if not hasattr(args, 'show_good_hsps_idx') else args.show_good_hsps_idx
    args.toplot            = True           if not hasattr(args, 'toplot')             else args.toplot
    args.toooplot          = True           if not hasattr(args, 'toooplot')           else args.toooplot
    args.pyplot_fsize      = 12             if not hasattr(args, 'pyplot_fsize')       else int(args.pyplot_fsize)
    args.plot_zoom_in      = '10%'          if not hasattr(args, 'plot_zoom_in')       else args.plot_zoom_in
    args.plot_nslice       = 16             if not hasattr(args, 'plot_nslice')        else int(args.plot_nslice)
    args.plot_tol          = 3              if not hasattr(args, 'plot_tol')           else int(args.plot_tol)
    args.plot_side_leave   = '25%'          if not hasattr(args, 'plot_side_leave')    else args.plot_side_leave
    args.plot_lw           = 0.5            if not hasattr(args, 'plot_lw')            else args.plot_lw
    args.plot_titlecolor   = (.8,.9,.2)     if not hasattr(args, 'plot_titlecolor')    else args.plot_titlecolor
    args.plot_titlefsize   = 10             if not hasattr(args, 'plot_titlefsize')    else int(args.plot_titlefsize)
    args.figsize           = (15,4)         if not hasattr(args, 'figsize')            else args.figsize
    if not hasattr(args, 'snap_config'):
        args.snap_config = dict(top=.999, bottom=0.0, left=-0.0, right=.999, wspace=-0.2, 
                                         titlepad=-5, tight=True, nsnap=3, figsize=args.figsize)
    return args