#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 18:33:36 2024

@author: Amit Jaiswal, Megin Oy, Espoo, Finland  
        <amit.jaiswal@megin.fi> <amit.jaiswal@aalto.fi>
USAGE:
"""
import numpy as np
from nibabel import load as nib_load
# import sys
import mne
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontManager
# import meginpy
from mne.utils import logger, verbose, warn # check_fname
from time import sleep
from os.path import join, exists, dirname, islink                 # split, splitext, isfile
from os import makedirs , getcwd,  chdir, symlink, remove, cpu_count       # remove, mkdir, 
from mne.io.constants import FIFF
from scipy.spatial import distance
from mayavi.mlab import figure, points3d, orientation_axes, text3d
from scipy import linalg, stats
from copy import deepcopy
import IsoScore

#=========================== find closest node location and index  ========================================================= 
def find_closest_node(node, nodes): # function to find closest points
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index], closest_index

#=========================== find closest node location, index and distance ================================================ 
def find_closest_node_dist(node, nodes, multipy2dist=1): # function to find closest points
    closest_index = distance.cdist([node], nodes).argmin()
    closest_node = nodes[closest_index]
    diff = np.sqrt(np.sum(np.square(node-closest_node))) * multipy2dist # multipy2dist is to convert the unit m,cm,mm etc
    return closest_node, closest_index, diff




#===================================find and plot isotrack points ==========================================================
@verbose
def find_plot_isotraks(headshape_file, fiducial_output_file=None, toplot=False, newfig=True, scale_factor=.001, helmet=True, 
                       coord_frame='head', color=None, mode='sphere', opacity=0.4, figname=None, return_out=True, 
                       get_all_in_one=False, plot_axis_ori=True, axis=True, mergeEEGwithEXTRA=False,fig3d=None,  verbose=None):
    # Read file contents..............
    try:
        info = mne.io.read_info(headshape_file)
    except Exception as e:
        print(e)
        digs_pts, coord_frame = mne.io.read_fiducials(headshape_file, verbose=None)
        info = dict()
        info['dig'] = digs_pts
    # Find HPI, extras, and cardinal points locations...............
    hpi_loc = np.array([
        d['r'] for d in (info['dig'] or [])
        if (d['kind'] == FIFF.FIFFV_POINT_HPI and
            d['coord_frame'] == FIFF.FIFFV_COORD_HEAD)])
    ext_loc = np.array([
        d['r'] for d in (info['dig'] or [])
        if (d['kind'] == FIFF.FIFFV_POINT_EXTRA and
            d['coord_frame'] == FIFF.FIFFV_COORD_HEAD)])
    eeg_loc = np.array([
        d['r'] for d in (info['dig'] or [])
        if (d['kind'] == FIFF.FIFFV_POINT_EEG and
            d['coord_frame'] == FIFF.FIFFV_COORD_HEAD)])
    if mergeEEGwithEXTRA or ext_loc.shape[0]==0:
        print('\n#HSPs=%d, merging EEG locations with HSPs '%ext_loc.shape[0])
        ext_loc = np.vstack(( ext_loc.reshape(ext_loc.shape[0], 3), eeg_loc ))
    car_loc = mne.viz._3d._fiducial_coords(info['dig'], FIFF.FIFFV_COORD_HEAD)
    # Transform from head coords if necessary..................
    if coord_frame=='meg' and 'dev_head_t' in info:
        for loc in (hpi_loc, ext_loc, car_loc):
            loc[:] = mne.transforms.apply_trans(mne.transforms.invert_transform(info['dev_head_t']), loc)
    # elif coord_frame == 'mri':
    #     for loc in (hpi_loc, ext_loc, car_loc):
    #         loc[:] = mne.transforms.apply_trans(head_mri_t, loc)
    if len(car_loc) == len(ext_loc) == len(hpi_loc) == 0:
        warn('Digitization points not found. Cannot plot digitization.')
        return # check if it works
    if not fiducial_output_file is None:
        fid = open(fiducial_output_file, 'w')
        for ii in range(3):
            fid.writelines('%f\t%f\t%f\n'   %tuple(car_loc[ii,:]))   
            print('%.2f\t%.2f\t%.2f'   %tuple(car_loc[ii,:]*1000))
        fid.close()
        print('The cardinal points (in meter) were written in file: \n%s\n'%fiducial_output_file)
    if toplot:
        if newfig:
            fig3d = figure(figure=figname, bgcolor=(1,1,1), fgcolor=None, engine=None, size=(1200, 900))
            if axis:
                points3d(0,0,0, mode='axes', scale_factor=.1, color=(0,0,0))
        points3d(car_loc[:,0], car_loc[:,1], car_loc[:,2], scale_factor=scale_factor*8, opacity=opacity, mode=mode, 
                 color=tuple(np.random.random([1,3]).tolist()[0])  if color==None else color)#(0,0.7,0))
        points3d(hpi_loc[:,0], hpi_loc[:,1], hpi_loc[:,2], scale_factor=scale_factor*6, opacity=opacity, mode=mode,  
                 color=tuple(np.random.random([1,3]).tolist()[0])  if color==None else color)#(0,0.7,0))
        points3d(ext_loc[:,0], ext_loc[:,1], ext_loc[:,2], scale_factor=scale_factor*4, opacity=opacity,  mode=mode, 
                 color=tuple(np.random.random([1,3]).tolist()[0])  if color==None else color)#(0,0.7,0))
        if plot_axis_ori:
            orientation_axes()
    if helmet:
        helmet_surf = mne.surface.get_meg_helmet_surf(info) if 'chs' in info else None
    if return_out:
        if helmet:
            return car_loc, hpi_loc, ext_loc, helmet_surf, fig3d
        else:
            if get_all_in_one:
                return np.vstack(( car_loc, np.vstack((hpi_loc, ext_loc)) ))
            else:
                return car_loc, hpi_loc, ext_loc, fig3d
            
def get_ras_to_neuromag_trans_from_fiducial_file(fiducial_file):
    fiducials =  mne.io.read_fiducials(fiducial_file)
    fid_labels = ['LPA','Nas','RPA']
    fiducials_m = {}
    for ii in range(3):
        fiducials_m[fid_labels[ii]] = fiducials[0][ii]['r']
    trans_ras2neuromag = mne.transforms.get_ras_to_neuromag_trans(fiducials_m['Nas'].flatten(), 
                                                                  fiducials_m['LPA'].flatten(), 
                                                                  fiducials_m['RPA'].flatten())
    trans_neuromag2ras = linalg.inv(deepcopy(trans_ras2neuromag))
    return trans_ras2neuromag, trans_neuromag2ras

#%% get fiducials in different formats    
def get_fiducial_LNR(fiducial_file):
    fids_ras = mne.io.read_fiducials(fiducial_file)[0]
    labels = ['', 'LPA','Nasion','RPA']
    # fids_LNR = np.vstack((fids_LNR, dict([d for d in fids_ras if d['ident'] == fiff_id][0].items())['r']))
    fids_LNR, fids_LNR_dict = np.empty((0,3)), dict()
    for fiff_id in [1,2,3]:
        for d in fids_ras:
            if d['ident'] == fiff_id:
                fids_LNR = np.vstack((fids_LNR, dict(d.items())['r']))
                fids_LNR_dict[labels[fiff_id]] = dict(d.items())['r']
    return fids_ras, fids_LNR, fids_LNR_dict

#%% Find the best surface multiplier to make the surface fitting inside the digitization points
def find_uniform_scaler_for_minimum_dist(points1, points2, n_iter=200, mode='dist_sum'):
    points1_rphitheta = mne.transforms._cart_to_sph(points1)
    points2_rphitheta = mne.transforms._cart_to_sph(points2)
    points1_r = deepcopy(points1_rphitheta[:,0])
    points2_r = deepcopy(points2_rphitheta[:,0])
    
    dist_total, multipliers = [], []
    if mode=='dist_sum':
        mult = 1
        for ii in range(n_iter): # increasing multiplyer
            mult *=0.99
            dist_total.append(( points1_r - (deepcopy(points2_r)*mult)).sum())
            multipliers.append(mult)
        mult = 1
        for ii in range(n_iter):# decreasing multiplyer
            mult *=1.01
            dist_total.append(( points1_r - (deepcopy(points2_r)*mult)).sum())
            multipliers.append(mult)
        dist_total = np.array(np.abs(dist_total))
        mindist_idx = dist_total.argmin()
        mult_mindist = multipliers[mindist_idx]
        
    elif mode=='dist_mean':
        mult = 1
        for ii in range(n_iter): # increasing multiplyer
            mult *=0.99
            dist_total.append(( points1_r - (deepcopy(points2_r)*mult)).mean())
            multipliers.append(mult)
        mult = 1
        for ii in range(n_iter):# decreasing multiplyer
            mult *=1.01
            dist_total.append(( points1_r - (deepcopy(points2_r)*mult)).mean())
            multipliers.append(mult)
        dist_total = np.array(np.abs(dist_total))
        mindist_idx = dist_total.argmin()
        mult_mindist = multipliers[mindist_idx]

    elif mode=='ttest':
        mult = 1
        for ii in range(n_iter): # increasing multiplyer
            mult *=0.99
            dist_total.append(( points1_r - (deepcopy(points2_r)*mult)))
            multipliers.append(mult)
        mult = 1
        for ii in range(n_iter):# decreasing multiplyer
            mult *=1.01
            dist_total.append(( points1_r - (deepcopy(points2_r)*mult)))
            multipliers.append(mult)
        # dist_total = np.array(np.abs(dist_total))
        # mindist_idx = dist_total.argmin()
        # mult_mindist = multipliers[mindist_idx]
        
        dist_total = np.array(np.abs(dist_total)).T
        tstat_all, t_pvalue_all = [], [] 
        for ii in range(len(multipliers)):
            ref_dist = np.random.uniform(low=0.0, high=1, size=(len(dist_total[:,ii]),))
            tstat, t_pvalue = stats.ttest_ind(ref_dist, dist_total[:,ii])
            tstat_all.append(tstat)
            t_pvalue_all.append(t_pvalue)
        best_similar_idx = np.array(t_pvalue).argmax()
        mult_mindist = multipliers[best_similar_idx]
    
    return mult_mindist
        

#%% Find good quality HSPs that are abobe a threshold point and not too far from the surface
def find_good_HSPs(hpis_hsps, surf2warp, fiducial_file, above_thrs=0.02, min_rej_percent=2, max_rej_percent=10, 
                   toplot=False,show_good_hsps_idx=False, return_scene=False):
    
    n_min_rej_hsp = np.ceil(hpis_hsps.shape[0]*min_rej_percent/100).astype(int)
    n_max_rej_hsp = np.ceil(hpis_hsps.shape[0]*max_rej_percent/100).astype(int)

    trans_ras2neuromag, trans_neuromag2ras = get_ras_to_neuromag_trans_from_fiducial_file(fiducial_file)
    fids_ras, fids_LNR, fids_LNR_dict = get_fiducial_LNR(fiducial_file)
    hpis_hsps_ras= mne.transforms.apply_trans(trans_neuromag2ras, deepcopy(hpis_hsps), move=True) # convert in RAS 
    zlim         = fids_LNR[:,2].min()-above_thrs 
    point_set1   = deepcopy(hpis_hsps_ras[hpis_hsps_ras[:,2] >= zlim, :])   # Limit hpis_hsps_ras above zlim
    print('\nDiscarding %d points below the threshold on Z-axis.'%((hpis_hsps_ras[:,2] < zlim)*1).sum())
    surf_pos     = surf2warp['rr'][surf2warp['rr'][:,2] >= zlim, :]         # Limit headsurf above zlim
    point_set2   = surf_pos[mne.surface._DistanceQuery(surf_pos).query(point_set1)[1], :]
    mult_mindist = find_uniform_scaler_for_minimum_dist(point_set1, point_set2, n_iter=200, mode='dist_sum')
    point_set2_scaled = point_set2 * mult_mindist
    if toplot:
        fig3d = meginpy.viz.myMlabTriagularMesh(surf2warp['rr'], surf2warp['tris'], toplot=True, newfig=True, representation='s', color=(1,.8,0))
        meginpy.viz.myMlabPoint3d(surf_pos, newfig=False, scale_factor=0.0005, color=(1,.5,0)) 
        meginpy.viz.myMlabPoint3d(point_set2, newfig=False, scale_factor=0.003, color=(1,.6,0)) 
        meginpy.viz.myMlabPoint3d(point_set2_scaled, newfig=False, scale_factor=0.003, color=(1,0,0)) 
        meginpy.viz.myMlabPoint3d(point_set1, newfig=False, scale_factor=0.0035, color=(0,1,1), opacity=.5)
    point_set1_rphitheta        = mne.transforms._cart_to_sph(point_set1)
    point_set2_rphitheta        = mne.transforms._cart_to_sph(point_set2)
    outward_hsp_idx             = np.where(point_set1_rphitheta[:,0] > point_set2_rphitheta[:,0])[0]
    point_set2_scaled_rphitheta = mne.transforms._cart_to_sph(point_set2_scaled)
    
    # Find the hpis_hsps_ras outwards to the surface
    # closest_vert_pos_hpi_hsp_rphitheta = _cart_to_sph(closest_vert_pos_hpi_hsp)
    # digpoints_hpi_hsp_rphitheta        = _cart_to_sph(digpoints_hpi_hsp)
    # outward_hpi_hsp_idx = np.where(point_set1_rphitheta[:,0] > point_set2_scaled_rphitheta[:,0])[0]
    # alternatively use : outward_hpi_hsp_idx = mne.surface._points_outside_surface(digpoints_hpi_hsp, surf2warp, n_jobs=12, verbose=True)
    outward_hsp_percentage = (len(outward_hsp_idx)*100) / len(point_set1)
    all_dists = np.abs(point_set1_rphitheta[:,0] - point_set2_scaled_rphitheta[:,0])*1000 # in mm
    
    if outward_hsp_percentage==0 and all_dists.max()<20.0: # when all points are inside the surface and the head is not too small for the template 
        reject_idx = np.array([], int)
    
    else:
        # if outward_hsp_percentage < 5:
        """ find outliers """
        Q1      = np.percentile(all_dists, 25, interpolation = 'midpoint')
        Q3      = np.percentile(all_dists, 75, interpolation = 'midpoint')
        IQR     = Q3 - Q1
        extremes_idx = np.where(all_dists > (Q3+3.0*IQR))[0]
        outlrs_indx  = np.where(all_dists > (Q3+1.5*IQR))[0]
        rej_idx      = np.array(extremes_idx.tolist() + outlrs_indx.tolist())
        
        if len(rej_idx) == 0:
            reject_idx = all_dists.argsort(axis=None)[::-1][:n_min_rej_hsp]
            # reject_idx = np.intersect1d( outward_hsp_idx, reject_idx)
        
        elif len(rej_idx) > 0 and len(rej_idx) < n_min_rej_hsp:
            reject_idx1 = rej_idx
            reject_idx2 = all_dists.argsort(axis=None)[::-1][:n_min_rej_hsp]
            # reject_idx2 = np.intersect1d( outward_hsp_idx, reject_idx2)
            reject_idx  = np.union1d(reject_idx1, reject_idx2).astype(int)
        
        elif len(extremes_idx) >= n_max_rej_hsp:
            reject_idx = extremes_idx
        elif len(rej_idx) >= n_max_rej_hsp:
            reject_idx = np.array(extremes_idx.tolist() + outlrs_indx.tolist())
        # elif (len(extremes_idx) + len(outlrs_indx)) < n_max_rej_hsp:
        #     reject_idx = np.array(extremes_idx.tolist() + outlrs_indx.tolist())
        
        # elif (len(extremes_idx) + len(outlrs_indx)) < n_min_rej_hsp:                
        #     reject_idx = np.array(extremes_idx.tolist() + outlrs_indx.tolist())
        #     n_rem_idx = n_min_rej_hsp-len(reject_idx)
        #     if n_rem_idx > 0:
        #         rem_dist_array = np.delete(deepcopy(all_dists).flatten(), reject_idx)
        
        #         reject_idx2 = all_dists.argsort(axis=None)[::-1][:n_min_rej_hsp]
        
        #         reject_idx.tolist().append( all_dists.argsort(axis=None)[::-1][:n_min_rej_hsp])
        
        #     all_dists.argsort(axis=None)[::-1][:n_min_rej_hsp]
        # elif len(extremes_idx) < n_min_rej_hsp and :
            # reject_idx = all_dists.argsort(axis=None)[::-1][:n_min_rej_hsp]
        else:
            reject_idx = all_dists.argsort(axis=None)[::-1][:n_min_rej_hsp]
    print('Discarding %d points using HSPs-to-Surface distance-check.'%len(reject_idx))
    print('Total no. of discarded points= %d '%(((hpis_hsps_ras[:,2] < zlim)*1).sum() + len(reject_idx)))

    reject_idx = np.unique(reject_idx.astype(int))
    good_hpis_hsps_ras = np.delete(point_set1, reject_idx, axis=0)
    if toplot:
        meginpy.viz.myMlabPoint3d(good_hpis_hsps_ras, newfig=False, scale_factor=0.003, color=(0,1,0), mode='cube', opacity=.5)
        if show_good_hsps_idx:
            for i_pnt, pnt in enumerate(good_hpis_hsps_ras): 
                text3d(pnt[0], pnt[1], pnt[2]+pnt[2]*.05, '%s'%i_pnt, **dict(color=(0,0,0), scale=0.003))
    print('Total no. of good points= %d\n'%len(good_hpis_hsps_ras))
    good_hpis_hsps = mne.transforms.apply_trans(trans_ras2neuromag, deepcopy(good_hpis_hsps_ras), move=True) # convert back to NM 
    if return_scene:
        return good_hpis_hsps, fig3d  
    else:
        return good_hpis_hsps
    
    
#%% A grand single module to generate pseudoMRI for a given subject 
def pseudomriengine(pseudo_subject, pseudo_subjects_dir, isotrak, template, templates_dir, fiducial_file,  
                    dense_hsp=False, mirror_hsps=True, template_headsurf='auto', dense_surf=True, z_thres=0.02, 
                    destHSPsShiftInwrd=0.0025, Wreg_est=0.000005, Wreg_apply= 0.00005, wtol=1e-06,
                    warp_anatomy=True, which_mri='all', blocksize=500000, write2format=['.mgz'], n_jobs=1, 
                    toplot=True, toooplot=True, pyplot_fsize=12, save_pmri_plot=True,
                    plot_zoom_in='12%', plot_nslice=16, plot_tol=3, plot_side_leave='25%', plot_lw=1.5,  
                    plot_titlecolor=(.8,.9,.2), plot_titlefsize=18, dig_reject_min_max=[2, 10], 
                    use_hpi=True, show_good_hsps_idx=True, rem_good_pts_idx=[],  nmax_Ctrl= 150,
                    report=None, report_file=None):
    if isinstance(isotrak, str):
        isotrak_file = isotrak
        if exists(isotrak_file):
            print(pseudo_subject, isotrak_file)
        # else:
        #     raise IndexError
        fids, hpis, hsps, _    = find_plot_isotraks(isotrak_file, toplot=toplot, helmet=False)
    elif 'fids' in isotrak and 'hpis' in isotrak and 'hsps' in isotrak:
        fids = isotrak['fids']
        hpis = isotrak['hpis']
        hsps = isotrak['hsps']
    else:
        raise Exception('Contact developers to provide integration for you device.')
    all_hsps            = np.vstack((fids, np.vstack((hpis, hsps))))
    hpis_hsps           = np.vstack((hpis, hsps)) if use_hpi else hsps
        
    above_thrs = z_thres if dense_surf else 0.0
    trans_ras2neuromag, trans_neuromag2ras = get_ras_to_neuromag_trans_from_fiducial_file(fiducial_file)
    fids_ras, fids_LNR, fids_LNR_dict = get_fiducial_LNR(fiducial_file)
    
    if isinstance(template_headsurf, str):
        if template_headsurf=='dense_scalp':
            templ_surf_fname  = '%s/%s/bem/%s-head-dense.fif'%(templates_dir, template, template)
            template_headsurf = mne.read_bem_surfaces(templ_surf_fname)[0]
            # FIX IT: the dense surface has pits on the bottom surface which introduces incorrectness while selection the closest points
        elif template_headsurf in ['scalp', 'outer_skin', 'auto']:
            templ_surf_fname  = '%s/%s/bem/watershed/%s_outer_skin_surface'%(templates_dir, template, template)
            template_headsurf = mne.surface.read_surface(templ_surf_fname, return_dict=True)[2]
            template_headsurf['rr'] /= 1000  # covert to meter    
            template_headsurf['rr'] *= 0.975 # shrink inward to reduce mismatch (FIX it by checking the mean distance for points above fids) 
        else:
            raise Exception('template_headsurf mush be dense_scalp or scalp or outer_skin.')

        
    surf2warp = deepcopy(template_headsurf)
    # meginpy.viz.myMlabPoint3d(all_hsps, newfig=1, scale_factor=0.005, color=(0,1, 0))
    # meginpy.viz.myMlabPoint3d(hpis_hsps, newfig=0, scale_factor=0.003, color=(1,0, 0))
    # meginpy.viz.myMlabPoint3d(surf2warp['rr'], newfig=0, scale_factor=0.0003, color=(0,1, 0))
    hpis_hsps_good = find_good_HSPs(hpis_hsps, surf2warp, fiducial_file, above_thrs=above_thrs, 
                                              min_rej_percent=isotrak_reject_min_max[0], 
                                              max_rej_percent=isotrak_reject_min_max[1], toplot=toplot, 
                                              show_good_hsps_idx=show_good_hsps_idx)
    if len(rem_good_pts_idx)>0:
        hpis_hsps_good = np.delete(hpis_hsps_good, np.array(rem_good_pts_idx), axis=0)
        meginpy.viz.myMlabPoint3d( mne.transforms.apply_trans(trans_neuromag2ras, deepcopy(hpis_hsps_good), move=True),
                                  toplot=toplot, newfinmax_Ctrlg=False, scale_factor=0.010, color=(0,1,0), mode='cone')
        print('Total no. of good points= %d'%hpis_hsps_good.shape[0])
   
    hpis_hsps_above_thrs= np.where(hpis_hsps_good[:,2]>-above_thrs)[0] # doesn't matter but check it again
    hpis_hsps2warp      = hpis_hsps_good[hpis_hsps_above_thrs, :]
    iso_val = IsoScore.IsoScore(hpis_hsps2warp.T) # Uniformilty
    total_good_points = hpis_hsps2warp.shape[0]+3
    print('\nNo. of HSPs = %s \nNo. of HSPs used = %s \nIso_score = %.2f'%(all_hsps.shape[0], total_good_points, iso_val))
    print('HPI coils locations were not used.') if not use_hpi else None
    
    estimate_dense_HSPs = True if (total_good_points<50 or iso_val<0.80) and dense_hsp else False
    
    if estimate_dense_HSPs:
        print('\nEstimating dense HSPs...')
        # FIX IT: check if len(all_hsps)<50 and non-uniformly distributed, 
        #         then read dense_headshape_file (if exist) OR estimate here a dense hsp file using ----
        digpoints_dense, _, _, _ = apply_warp_to_find_dense_headshape(isotrak_file, outer_skinP_file=None, transfileP=None, 
                                                out_headshape_fname=isotrak_file.replace('.fif', '_dense_headshape.txt'), 
                                                toplot=toplot, toplot_final=False, below_thres=0.005, max_out_percent=5,
                                                ds_warped_mesh=3, TSPTrans_reg=1e-5, use_sphere_fit=True, verbose=True)
        hpis_hsps2warp_orig = deepcopy(hpis_hsps2warp)
        hpis_hsps2warp      = deepcopy(digpoints_dense)
    else:
        hpis_hsps2warp_orig = np.empty((0,3))
        hpis_hsps2warp      = hpis_hsps2warp
    
    # In most of the case head is symetric from left to write (ref ???, check literatures or validate with available MRI head surfaces)
    # so the concatenating the mirror HSP left <-> right would add extra points and make the HSP denser (TBV= to be validated)
    apply_mirroring = mirror_hsps and hpis_hsps2warp.shape[0]<40
    if apply_mirroring:
        print('\nNOTE: Mirroring HSPs to make it denser...')
        hpis_hsps2warp_true    = deepcopy(hpis_hsps2warp)
        hpis_hsps2warp_reflect = deepcopy(hpis_hsps2warp) * (-1,1,1)
        hpis_hsps2warp = np.vstack((hpis_hsps2warp_true, hpis_hsps2warp_reflect))
    
    # to avoid toooo dense points, downsample to 100-300 points only
    # nmax_Ctrl      = 150
    nmax_Ctrl      = min(nmax_Ctrl-(3 + hpis_hsps2warp_orig.shape[0]), hpis_hsps2warp.shape[0]) # 3 fids would be necessarily available, so max required are nmax_Ctrl-3
                                                                                                # Also, while using estimated dense HSPs, the original HSPs will be concatenated later 
    selidx         = np.linspace(0, hpis_hsps2warp.shape[0]-1, nmax_Ctrl, dtype=int)
    hpis_hsps2warp = hpis_hsps2warp[selidx, :]
    
    hpis_hsps2warp = np.vstack((hpis_hsps2warp_orig, hpis_hsps2warp)) # concatenating the original HSPs
    
    # concatenating the LPA, Nasian, and RPA
    all_hsps2warp  = np.vstack((fids, hpis_hsps2warp))
    
    all_hsps       = mne.transforms.apply_trans(trans_neuromag2ras, all_hsps, move=True)
    all_hsps2warp  = mne.transforms.apply_trans(trans_neuromag2ras, all_hsps2warp, move=True)
        
    digpoints = deepcopy(all_hsps2warp)
    iso_score_final = IsoScore.IsoScore(deepcopy(all_hsps2warp).T if all_hsps2warp.shape[0]==3 else deepcopy(all_hsps2warp))
    print(('\nThe final number of HSP used for warping = %d \niso_score = %.2f'%(digpoints.shape[0], iso_score_final)).upper())
    
    #% % Find warp tranform -------------------------------------------------------------------------------------------------
    # The three srcCtrl points for isotrak fids would be three MRI surface fids (from fiducial file)
    fids_ras, fids_LNR, fids_LNR_dict = get_fiducial_LNR(fiducial_file)
    srcCtrl_fids = deepcopy(fids_LNR)
    srcCtrl_fids_dist = utils.get_all_to_all_point_dist(srcCtrl_fids, digpoints[:3], toplot=toplot)
    
    # find closed points for rest of the HSPs and compute distances
    mesh_pos       = deepcopy(surf2warp['rr']) 
    above_thrs_idx = np.where(mesh_pos[:,2] > (fids_LNR[:,2].min()-above_thrs))[0]
    mesh_pos       = mesh_pos[above_thrs_idx, :]
    if toooplot:
        meginpy.viz.myMlabTriagularMesh(surf2warp['rr'], surf2warp['tris'], toplot=True, newfig=1, representation='s', opacity=0.2, axis=True)
        meginpy.viz.myMlabPoint3d(mesh_pos,   newfig=0, scale_factor=0.0003, color=(0,1, 0))
        meginpy.viz.myMlabPoint3d(srcCtrl_fids, newfig=0, scale_factor=0.005, color=(0,0, 1))
        meginpy.viz.myMlabPoint3d(digpoints[:3], newfig=0, scale_factor=0.005, color=(0,1, 1))
        meginpy.viz.myMlabPoint3d(digpoints,   newfig=0, scale_factor=0.005, color=(0.5,1, 1))
        meginpy.viz.myMlabPoint3d(all_hsps,  newfig=0, scale_factor=0.002, color=(1,0, 0))
    closest_vert_pos_rest, closest_vert_idx_rest, closest_vert_dist_rest = np.empty((0,3)), np.empty((0,1), dtype=int), np.empty((0,1))
    for ii in range(3, digpoints.shape[0]):
        closest_vert_pos, idx, closest_vert_dist = find_closest_node_dist(digpoints[ii,:], mesh_pos,  multipy2dist=1000)
        closest_vert_pos_rest  = np.vstack((closest_vert_pos_rest, closest_vert_pos))
        closest_vert_idx_rest  = np.vstack((closest_vert_idx_rest, idx))
        closest_vert_dist_rest = np.vstack((closest_vert_dist_rest, closest_vert_dist))
        
    # Append all for fids and hpi_hsp
    closest_vert_pos_all   = np.vstack((srcCtrl_fids, closest_vert_pos_rest))
    closest_vert_dist_all = np.hstack((srcCtrl_fids_dist, closest_vert_dist_rest.flatten())).T
    # meginpy.viz.myMlabPoint3d(closest_vert_pos_all, newfig=0, scale_factor=0.002, color=(0,0, 0))
    print('\nDestCtrl to SrcCtrl min, mean, max dist. = %.2f mm, %.2f mm, %.2f mm\n'\
          %(closest_vert_dist_all.min(),closest_vert_dist_all.mean(), closest_vert_dist_all.max()))
    # ax, _ = meginpy.viz.violin_plus_box_plot(closest_vert_dist_all, fig_face='#d0ecf4', ax_face='#d0ecf4', vioW=.8, boxW=.2)
    # IQR, outlrs_indx, outlrs_prcnt, \
    #     extremes_idx, extremes_prcnt = find_outliers_extreme_percentage(closest_vert_dist_all, percentile=[25,75],whis_outliers=1.5, 
    #                                                                     whis_extreme=3.0,res_round_by=2, toprint=False)
    srcCtrl  = deepcopy(closest_vert_pos_all) # source control points (p)
    destCtrl = deepcopy(digpoints)            # destination contrl points (q)
    if toooplot:
        figg = meginpy.viz.myMlabTriagularMesh(surf2warp['rr'], surf2warp['tris'], toplot=True, newfig=True, representation='s', color=(.751,.75,0.75), opacity=.3, axis=False, lw=.0000001)
        meginpy.viz.myMlabPoint3d(srcCtrl, newfig=False, scale_factor=0.004, color=(1,0,0), axis=False) 
        meginpy.viz.myMlabPoint3d(destCtrl, newfig=False, scale_factor=0.004, color=(0,1,0), axis=False)
        figg.scene.light_manager.light_mode = "vtk"
        figg.scene.light_manager.number_of_lights = 5
    print('Computing the warping transform using linear solver with %d source ' \
          'and %d destination control points (with regularization=%s)...'%(srcCtrl.shape[0], destCtrl.shape[0], Wreg_est))
    # warp_reg=0.000005
    # for warp_reg in [50000000000000, 500000000, 50000, 500, 50, 5,.5,.05,.005,.0005,.00005,.000005,.0000005,.000000005,.0000000005]:    
    if destHSPsShiftInwrd > 0: # integrated on 08/04/2022
        destCtrl_new = shift_destCtrl_inward_to_maintain_realistic_scalp2hsp_dist(deepcopy(destCtrl), 
                                                                                  trans_ras2neuromag, trans_neuromag2ras, 
                                                                                  destHSPsShiftInwrd=destHSPsShiftInwrd, toplot=False)
        meginpy.viz.myMlabPoint3d(destCtrl_new, toplot=toooplot, newfig=False, scale_factor=0.004, color=(0,0,0))
    else:
        destCtrl_new = deepcopy(destCtrl)
    
    W, A, e, Wreg_est2 = bst_warp_transform(srcCtrl, destCtrl_new, reg=Wreg_est, comment='Estimating Wtran for surfs..', 
                                            wtol=wtol, return_reg=True) # Note that this W, A, e are for RAS coordsys
    print(np.mean(np.abs(deepcopy(W)), axis=0), '\n\n', A, '\n\n', e)
    # print(warp_reg, e)
    
    #% % Warp the template head surface and check the distance between surface and HSPs -------------------------------------
    mesh_pos = surf2warp['rr']
    warpedSurf_verts = bst_warp_lm(deepcopy(mesh_pos), A, W, srcCtrl) + deepcopy(mesh_pos)
    warpedSurf       = deepcopy(surf2warp)
    warpedSurf['rr'] = deepcopy(warpedSurf_verts)
    if toooplot:
        meginpy.viz.myMlabTriagularMesh(surf2warp['rr'], surf2warp['tris'], toplot=True, newfig=True, representation='s', color=(1,.5,0), opacity=0.2)
        meginpy.viz.myMlabPoint3d(srcCtrl, newfig=False, scale_factor=0.002, color=(1,.5,0), opacity=.9) 
        meginpy.viz.myMlabTriagularMesh(warpedSurf['rr'], warpedSurf['tris'], toplot=True, newfig=False, representation='s', color=(0,1,0), opacity=0.3)
        meginpy.viz.myMlabPoint3d(destCtrl, newfig=False, scale_factor=0.004, color=(0,1,0), opacity=.9)
        # figg =meginpy.viz.myMlabTriagularMesh(warpedSurf['rr'], warpedSurf['tris'], toplot=True, newfig=True, representation='s', color=(.751,.75,0.75), opacity=.3, axis=False, lw=.0000001)
        # meginpy.viz.myMlabPoint3d(srcCtrl, newfig=False, scale_factor=0.004, color=(1,0,0), axis=False) 
        # meginpy.viz.myMlabPoint3d(destCtrl, newfig=False, scale_factor=0.004, color=(0,1,0), axis=False)  
        figg.scene.light_manager.light_mode = "vtk"
        figg.scene.light_manager.number_of_lights = 5
        
    closest_vert_pos_all2, closest_vert_dist_all2, mesh_pos2 = np.empty((0,3)), np.empty((0,1)),  deepcopy(warpedSurf['rr'])
    for ii in range(digpoints.shape[0]):
        closest_vert_pos_all2  = np.vstack((closest_vert_pos_all2, find_closest_node_dist(digpoints[ii,:], mesh_pos2,  multipy2dist=1000)[0]))
        closest_vert_dist_all2 = np.vstack((closest_vert_dist_all2, find_closest_node_dist(digpoints[ii,:], mesh_pos2,  multipy2dist=1000)[2]))
    print('\nIsotrak to warped-surface distance min, mean, max dist. = %.2f mm, %.2f mm, %.2f mm\n'\
          %(closest_vert_dist_all2.min(),closest_vert_dist_all2.mean(), closest_vert_dist_all2.max()))
    del warpedSurf_verts, closest_vert_pos_all2, closest_vert_dist_all2, mesh_pos2 #, warpedSurf
    
    #% % Create subject's directory to write warped results + Write the warping transform -----------------------------------
    subject_directory = '%s/%s/'%(pseudo_subjects_dir, pseudo_subject)
    makedirs(subject_directory, exist_ok=True)
    makedirs(subject_directory + 'mri', exist_ok=True)
    makedirs(subject_directory + 'bem', exist_ok=True)
    makedirs(subject_directory + 'surf', exist_ok=True)
    makedirs(subject_directory + 'label', exist_ok=True)
    makedirs(subject_directory + 'src', exist_ok=True)
    makedirs(subject_directory + 'coreg', exist_ok=True)
    makedirs(subject_directory + 'warped_figs', exist_ok=True)
    misc_comments = 'HSPs mirrored=%s\t'%str(apply_mirroring) + \
                    'HSPs densified=%s\t'%str(estimate_dense_HSPs)
    config_fname = write_mri_warping_config(templates_dir, isotrak_file.split('/', maxsplit=7)[-1], template, 
                             pseudo_subjects_dir, pseudo_subject, srcCtrl, destCtrl, W, A, e, 
                             coord_frame='MRI (surface RAS)',  misc_comments=misc_comments, return_fname=True)
    add_line_to_file(config_fname, "Regularization & Bending energy:\n")
    add_line_to_file(config_fname, "Reguln applied to estimate Wtrans for surface \t= %s\n"%Wreg_est2)
    add_line_to_file(config_fname, "Bending energy in estimating Wtrans for surface = %s\n"%e)
        
    #% % Find source mri, bems, src, etc. paths
    src_paths = mne.coreg._find_mri_paths(subject=template, skip_fiducials=False, subjects_dir=templates_dir)
    
    #% % Warp fiducials and compute new transform and write files -----------------------------------------------------------
    mne.utils.logger.info("Warping fiducials %s -> %s and computing new transform", template, pseudo_subject)
    
    # Get the template MRI fiducials >> Warp fiducials >> Write the warped fiducial file (in RAS coordsys)...
    template_fiducials =  mne.io.read_fiducials(fiducial_file)
    fid_labels         = ['LPA','Nas','RPA']
    template_fiducials_m = dict()
    template_fiducials_m_warped = dict()
    
    for fiff_id in [1,2,3]:
        for d in template_fiducials[0]:
            if d['ident'] == fiff_id:
                template_fiducials_m[fid_labels[fiff_id-1]]        = dict(d.items())['r']
                template_fiducials_m_warped[fid_labels[fiff_id-1]] = (bst_warp_lm( deepcopy(dict(d.items())['r']).reshape(1,3), \
                                                                            A, W, srcCtrl ) + deepcopy(dict(d.items())['r']).reshape(1,3)).flatten()
    
    warped_fids2write = deepcopy(template_fiducials) # just to copy the fiducial structure
    for fiff_id in [1,2,3]:
        for d in warped_fids2write[0]:
            if d['ident'] == fiff_id:
                warped_fids2write[0][fiff_id-1]['r']  = template_fiducials_m_warped[fid_labels[fiff_id-1]]  
    fiducial_file_warped = '%s/bem/%s-fiducials.fif'%(subject_directory, pseudo_subject)
    mne.io.write_fiducials(fiducial_file_warped, warped_fids2write[0], warped_fids2write[1], verbose=None)
    print('\nfiducial file for pMRI is written:  %s'%fiducial_file_warped)  
    
    # Compute and write the new trans for warped surfaces
    trans_ras2neuromag_warped, trans_neuromag2ras_warped = get_ras_to_neuromag_trans_from_fiducial_file(fiducial_file_warped)
    # trans_temp = mne.read_trans(  glob2.glob('%s/%s/**/*-trans.fif'%(templates_dir, template))[0]  ) # to find the trans structure
    # trans_temp['trans'] = trans_neuromag2ras_warped
    trans_temp = mne.transforms.Transform(4, 5, trans_neuromag2ras_warped)
    trans_file_warped = '%s/coreg/%s-trans.fif'%(subject_directory, pseudo_subject)
    mne.write_trans(trans_file_warped, trans_temp)
    print('\ntrans file for pMRI is written:  %s'%trans_file_warped)  
    
    #% % Warp bem and brain surfaces and write files ------------------------------------------------------------------------
    mne.utils.logger.info("Warping bem and brain surfaces:  %s -> %s", template, pseudo_subject)
    # Read different surfaces of the template reconstructed by Freesurfer 
    SurfaceFile_names, Surfaces, Surface_scaling = bst_get_SurfaceFile_v2(templates_dir, template, 
                                                                          read_files=True, read_metadata=False)
    
    # Apply warping on the surfaces using the warping transform >> write the warped surface (in RAS coordsys)
    figname_final ='R> P:%s'%(pseudo_subject)
    SurfWarped = dict()
    meginpy.viz.myMlabPoint3d(destCtrl, scale_factor=0.005, figname=figname_final, toplot=toplot, newfig=True) 
    meginpy.viz.myMlabPoint3d(np.vstack(list(template_fiducials_m_warped.values())), scale_factor=0.005, mode='cube', 
                      figname=figname_final, toplot=toplot, newfig=False)
    for surf in list(SurfaceFile_names.keys()):
        print('\nWarping %s surface...'%surf)
        Surface = Surfaces[surf]
        SurfNew = dict()
        SurfNew['Faces']    = Surface['Faces'] if 'Faces' in Surface else Surface['tris']
        Surface_verts       = Surface['Vertices'] if 'Vertices' in Surface else Surface['rr']
        SurfNew['Vertices'] = bst_warp_lm(deepcopy(Surface_verts), A, W, srcCtrl) + deepcopy(Surface_verts)
        del Surface_verts#, Surface_verts_nm
        if surf in ['/bem/watershed/%s_inner_skull_surface'%template, '/bem/watershed/%s_outer_skin_surface'%template, 
                    '/bem/watershed/%s_brain_surface'%template, #'/surf/lh.pial', '/surf/rh.pial'#, 
                    '/surf/lh.white', '/surf/rh.white', '/bem/%s-head-dense.fif'%template,
                    ]:
            meginpy.viz.myMlabTriagularMesh(SurfNew['Vertices'], SurfNew['Faces'], toplot=toplot, newfig=False, 
                                            figname=figname_final, representation='surface', opacity=0.3)
            mlab.view(azimuth=0, elevation=90) if toplot else None
        
        SurfNew_fname = '%s%s'%(subject_directory, surf.replace(template, pseudo_subject))
        print('Writing the warped surface file as: \n%s'%SurfNew_fname)
        makedirs( utils.dirname2(SurfNew_fname), exist_ok=True)
        if SurfNew_fname.endswith('.fif'):
            surf_temp = mne.read_bem_surfaces(SurfaceFile_names[surf])
            surf_temp[0]['rr']    = SurfNew['Vertices']
            surf_temp[0]['nn']    = [] 
            surf_temp[0]['sigma'] = 0.33
            mne.write_bem_surfaces(SurfNew_fname, surf_temp, overwrite=True, verbose=True)
            del surf_temp
        else:
            mne.surface.write_surface(SurfNew_fname, SurfNew['Vertices']*Surface_scaling[surf], SurfNew['Faces'], 
                                      create_stamp='', volume_info=None, file_format='auto', overwrite=True)
        SurfWarped[surf] = SurfNew
        del Surface, SurfNew
        
    # Make bem surface soft links
    try:
        utils.make_surfaces_soft_links3(pseudo_subjects_dir, pseudo_subject, surfs_from='mri2surf')
    except Exception as exerrr:
        utils.make_surfaces_soft_links(pseudo_subjects_dir, pseudo_subject)
    
    # copy/duplicate the files that does not require warping  (CONFIRM IT)
    print('Copying sphere and curv files...%s >> %s'%(template, pseudo_subject))
    for src_name in src_paths['duplicate']:
        srcfile = src_name.format(subjects_dir=templates_dir, subject=template)
        dstfile = src_name.format(subjects_dir=pseudo_subjects_dir, subject=pseudo_subject)
        copyfile(srcfile, dstfile)
        
    #% % Warp source spaces--------------------------------------------------------------------------------------------------
    mne.utils.logger.info("Warping source space:  %s -> %s", template, pseudo_subject)
    
    for src_name in src_paths['src']:
        print(src_name)
        src = src_name.format(subjects_dir=templates_dir, subject=template)
        dst = src_name.format(subjects_dir=pseudo_subjects_dir, subject=pseudo_subject)
    
        # read and warp the source space [in m]
        sss = mne.source_space.read_source_spaces(src)
        add_dist = False
        for ss in sss:
            ss['subject_his_id'] = pseudo_subject
            # ss['rr'] *= scale
            verts_pos =  deepcopy(ss['rr'])
            ss['rr'] = bst_warp_lm(deepcopy(verts_pos), A, W, srcCtrl) + deepcopy(verts_pos)
            del verts_pos
        if add_dist:
            dist_limit = np.inf
            mne.utils.logger.info("Recomputing distances, this might take a while")
            mne.source_space.add_source_space_distances(sss, dist_limit, n_jobs=n_jobs)
    
        mne.source_space.write_source_spaces(dst, sss, overwrite=True, verbose=True)
    
    #% % Warp label ---------------------------------------------------------------------------------------------------------
    mne.utils.logger.info("Warping labels:  %s -> %s", template, pseudo_subject)
    meginpy.viz.myMlabTriagularMesh(warpedSurf['rr'], warpedSurf['tris'], toplot=toplot, newfig=True, representation='s', opacity=0.2)
    
    # find labels path....
    lbl_dir = join(templates_dir, template, 'label')
    pattern = None
    if pattern is None:
        label_paths = []
        for dirpath, _, filenames in os.walk(lbl_dir):
            rel_dir = os.path.relpath(dirpath, lbl_dir)
            for filename in fnmatch.filter(filenames, '*.label'):
                thispath = join(rel_dir, filename)
                label_paths.append(thispath)
    else:
        label_paths = [os.path.relpath(thispath, lbl_dir) for thispath in iglob(pattern)]
    
    # apply warping ....
    src_root = join(templates_dir, template, 'label')
    dst_root = join(subject_directory, 'label')
    
    # warp labels
    for fname in label_paths:
        dst = join(dst_root, fname)
    
        dirname = os.path.dirname(dst)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    
        src = os.path.join(src_root, fname)
        l_old = mne.label.read_label(src)
        # pos = l_old.pos * scale
        print('warping... %s.label'%l_old.name)
        pos = bst_warp_lm(deepcopy(l_old.pos), A, W, srcCtrl) + deepcopy(l_old.pos)
        meginpy.viz.myMlabPoint3d(deepcopy(pos), scale_factor=0.0001,  newfig=False, toplot=toplot)
        l_new = mne.label.Label(l_old.vertices, pos, l_old.values, l_old.hemi,
                      l_old.comment, subject=pseudo_subject, name=l_old.name,
                      filename=l_old.filename.replace(templates_dir, pseudo_subjects_dir).replace(template, pseudo_subject))
        l_new.save(dst)
    
    # Copy .annot file and .ctab, as they are do not have any position info so no warping is required
    annot_list = glob2.glob('%s/%s/label/*.annot'%(templates_dir, template))
    ctab_list  = glob2.glob('%s/%s/label/*.ctab'%(templates_dir, template))
    annot_ctab_list = annot_list + ctab_list
    for annot_ctab_file in annot_ctab_list:
        copyfile(annot_ctab_file, annot_ctab_file.replace(templates_dir, pseudo_subjects_dir).replace(template, pseudo_subject))
    
    #% % Warp anatomy -------------------------------------------------------------------------------------------------------
    meginpy.viz.arrange_pyplot_font_etc(f_size=pyplot_fsize)
    if warp_anatomy:
        print('\n')
        mne.utils.logger.info("Warping anatomy:  %s -> %s", template, pseudo_subject)
        if which_mri=='all':
            mris2warp = src_paths['mri']  
        elif all([isinstance(item, int) for item in which_mri]):
            mris2warp = [src_paths['mri'][i_mri] for i_mri in which_mri]
        elif all([isinstance(item, str) for item in which_mri]):
            mris2warp = []
            for item, mrif in product(which_mri,src_paths['mri']):
                mris2warp.append(mrif) if mrif.endswith(item) else None
        for src_mri_fname in mris2warp:  
            print('\nwarping... %s ---> for %s'%(src_mri_fname, pseudo_subject))
            warped_mri_fname = src_mri_fname.replace( templates_dir, pseudo_subjects_dir ).replace( template, pseudo_subject )
            wdata, miscs = apply_warp_to_anatomy(deepcopy(srcCtrl), deepcopy(destCtrl), mridata=None, t1_fname=src_mri_fname, 
                                                           Torig=None, block_size=blocksize, reg=Wreg_apply, reg_mode=2, 
                                                           toplot_final=False, 
                                                           warped_surf=None, n_jobs=1, resample=False, rs_voxel_sizes=[.5,.5,.5])
            add_line_to_file(config_fname, "***%s***\n"%warped_mri_fname.replace(pseudo_subjects_dir, ''))
            add_line_to_file(config_fname, "Reguln applied to estimate Wtrans for vox \t= %s\n"%miscs['regWtransVox'])
            add_line_to_file(config_fname, "Bending energy in estimating Wtrans for vox \t= %s\n"%miscs['e_vox'])
            add_line_to_file(config_fname, "Reguln applied to estimate inv Wtrans for vox \t= %s\n"%miscs['reginvWtransVox'])
            add_line_to_file(config_fname, "Bending energy in estimating inv Wtrans for vox = %s\n"%miscs['e_vox_inv'])

            if not wdata is None:
                if save_pseudomri_plot:
                    # plot the warped MRI and check alignment with warped surface and destCtrl (HSPs) >> save for cross-checking later
                    meginpy.viz.plot_surf_anat_alignment_v2(wdata, miscs['Torig'], destCtrl, warpedSurf, 
                                                 title= warped_mri_fname.replace( pseudo_subjects_dir, '' ), zoom_in=plot_zoom_in,
                                                 nslice=plot_nslice, tol=plot_tol, side_leave=plot_side_leave, lw=plot_lw,  
                                                 titlecolor=plot_titlecolor, titlefsize=plot_titlefsize)
                    plt.show();     plt.pause(1);     plt.get_current_fig_manager().full_screen_toggle(); plt.pause(.5)
                    plt.subplots_adjust(top=0.999,bottom=0.025,left=0.1,right=0.9,hspace=0.0,wspace=-0.0);   plt.pause(.5)
                    plt.savefig('%s/%s/warped_figs/%s-%s.png'%(pseudo_subjects_dir, pseudo_subject, pseudo_subject, 
                                                               warped_mri_fname.split('/')[-1][:-4]),
                                dpi=None, facecolor='w', edgecolor='w', orientation='portrait', format=None,
                                transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None); plt.pause(.5)
                
                # Save the warped MRI file
                makedirs(os.path.dirname(warped_mri_fname), exist_ok=True)
                warped_mri_affine = deepcopy(miscs['mriobj'].affine)
                warped_mri_header = deepcopy(miscs['mriobj'].header)
                # write2format = ['.mgz'] #, '.nii']
                for out_format in write2format:
                    if out_format=='.mgz':
                        mri_warped = nib.freesurfer.mghformat.MGHImage(deepcopy(wdata), warped_mri_affine, warped_mri_header)
                    elif out_format=='.nii':
                        mri_warped = nib.Nifti1Image(deepcopy(wdata), warped_mri_affine, warped_mri_header)
                    # zooms = np.array(mri_warped.header.get_zooms())
                    # zooms[[0, 2, 1]] *= np.array([1,1,1])
                    # mri_warped.header.set_zooms(zooms)
                    # mri_warped._affine = mri_warped.header.get_affine()  
                    nib.save(mri_warped, warped_mri_fname.replace('.mgz', out_format))
                    del mri_warped
                plt.close()
    return print('\nCheck results in -> %s/%s'%(pseudo_subjects_dir, pseudo_subject), '\ne=%s'%e)