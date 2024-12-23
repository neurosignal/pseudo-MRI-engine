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
The software is intended to generate pseudo-MRI, 
i.e., the subject-spectific segmented set of a template MRI.
"""
print(__doc__)

#%% 1. Set subject and directories
import argparse
parser = argparse.ArgumentParser(description='pseudo-MRI engine for MRI-free electromagnetic source imaging.')
parser.add_argument('-p',       '--pseudo_MRI_name',    default=None,  type=str,  help='subject name')
parser.add_argument('-pd',      '--pseudo_MRI_dir',     default=None,  type=str,  help='Parent directory for the pseudo-MRI folder (optional)')
parser.add_argument('-dig',     '--headshape',          default=None,  type=str,  help='File with headshape digitization information')
parser.add_argument('-t',       '--template_MRI_name',  default=None,  type=str,  help='Template MRI name')
parser.add_argument('-td',      '--template_MRI_dir',   default=None,  type=str,  help='Parent directory of the template MRI folder')
parser.add_argument('-fids',    '--fiducial_file',      default=None,  type=str,  help='Fiducial file of the template MRI')
parser.add_argument('-paloc',   '--preauri_loc',        default=None,  type=str,  help='LPA/RPA location considered during the head digitization')
parser.add_argument('-nctrl',   '--nmax_Ctrl',          default=300,   type=int,  help='Number of maximum control points.')
parser.add_argument('-densify', '--dense_hsp',          action='store_true',      help='densify HSP?')
parser.add_argument('-v',       '--verbose',            action='store_true',      help='verbose mode or not?')
parser.add_argument('-o',       '--open_report',        action='store_true',      help='open report or not when completed?')
args = parser.parse_args()

#%% for test
args.pseudo_MRI_name    = 'ICBM2009c_asym_nlin_Amit_test1'
args.pseudo_MRI_dir     = None
args.template_MRI_name  = 'ICBM2009c_asym_nlin'
args.template_MRI_dir   = '/net/qnap/data/rd/ChildBrain/FS_SUBS_DIR/sub_fs732/subjects/'
args.headshape          = '/net/qnap/data/rd/ChildBrain/DATA/Biomag_Amit/jaiswal_amit/181211/aj_vis_01_raw_tsss.fif'
args.preauri_loc        ='CrusHelix'
args.dense_hsp          = False
args.show_good_hsps_idx = False
args.verbose            = True
args.nmax_Ctrl          = 200

#%%
from configuration_file import check_set_input_config
args  = check_set_input_config(args)

#%% Load modules
from os.path import join
from os import makedirs
from datetime import datetime
from mne import Report
from utils.meginpy.pseudoMRI import pseudomriengine

#%% Set an HTML report
report_dir  = join(args.pseudo_MRI_dir, args.pseudo_MRI_name, "report")
makedirs(report_dir, exist_ok=True)
date_str    = datetime.now().strftime("-%Y%m%d-%H%M")
args.report      = Report(title=f'pseudo-MRI generation report for subject: {args.pseudo_MRI_name}')
args.report_file = join(report_dir, f"pseudoMRI_report{date_str}.html")
args.report.add_sys_info(title='Analysis platform Info.')
args.report.add_html(args, title='All parameters', tags=('parameters',))
args.report.save(fname=args.report_file, open_browser=True, 
                 overwrite=True, sort_content=False, verbose=args.verbose)

#%% Run pseudomriengine
pseudomriengine(args.pseudo_MRI_name, args.pseudo_MRI_dir, args.headshape, 
                args.template_MRI_name, args.template_MRI_dir, args.def_fiducial_file, 
                dense_hsp=args.dense_hsp, mirror_hsps=args.mirror_hsps, 
                template_headsurf=args.template_headsurf, 
                dense_surf=args.dense_surf,   z_thres=args.z_thres, n_jobs=args.n_jobs, 
                destHSPsShiftInwrd=args.destHSPsShiftInwrd, Wreg_est=args.Wreg_est, 
                Wreg_apply= args.Wreg_apply, wtol=args.wtol, warp_anatomy=args.warp_anatomy, 
                which_mri=args.which_mri, blocksize=args.blocksize, write2format=args.write2format, 
                toplot=args.toplot, toooplot=args.toooplot, pyplot_fsize=args.pyplot_fsize, 
                save_pmri_plot=args.save_pmri_plot, plot_zoom_in=args.plot_zoom_in, 
                plot_nslice=args.plot_nslice, plot_tol=args.plot_tol, 
                plot_side_leave=args.plot_side_leave, plot_lw=args.plot_lw,  
                plot_titlecolor=args.plot_titlecolor, plot_titlefsize=args.plot_titlefsize, 
                dig_reject_min_max=args.dig_reject_min_max, 
                use_hpi=args.use_hpi, show_good_hsps_idx=args.show_good_hsps_idx, 
                rem_good_pts_idx=args.rem_good_pts_idx,  
                nmax_Ctrl=args.nmax_Ctrl,
                report=args.report, report_file=args.report_file, args=args)

    
    