#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 18:33:36 2022

@author: Amit Jaiswal, Megin Oy, Espoo, Finland  
        <amit.jaiswal@megin.fi> <amit.jaiswal@aalto.fi>
USAGE: meginpy.pseudoMRI: Implementation of the pseudo-MRI module.
"""
import numpy as np
import mne
from matplotlib import pyplot as plt
from mne.utils import verbose, warn 
from os.path import join, exists, dirname, relpath
from os import makedirs,  walk      
from mne.io.constants import FIFF
from scipy.spatial import distance
from scipy import linalg, stats
from copy import deepcopy
from IsoScore import IsoScore
from . import viz
from . import utils
from shutil import  copyfile
import fnmatch
from glob import iglob
import glob2
from itertools import product
from tqdm import tqdm
import nibabel as nib
from nibabel.processing import resample_to_output
import configparser
from mayavi.mlab import figure, points3d, orientation_axes, text3d, gcf, close

def find_closest_node(node, nodes): # function to find closest points
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index], closest_index

def find_closest_node_dist(node, nodes, multipy2dist=1): # function to find closest points
    closest_index = distance.cdist([node], nodes).argmin()
    closest_node = nodes[closest_index]
    diff = np.sqrt(np.sum(np.square(node-closest_node))) * multipy2dist 
    return closest_node, closest_index, diff

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

def get_fiducial_LNR(fiducial_file):
    fids_ras = mne.io.read_fiducials(fiducial_file)[0]
    labels = ['', 'LPA','Nasion','RPA']
    fids_LNR, fids_LNR_dict = np.empty((0,3)), dict()
    for fiff_id in [1,2,3]:
        for d in fids_ras:
            if d['ident'] == fiff_id:
                fids_LNR = np.vstack((fids_LNR, dict(d.items())['r']))
                fids_LNR_dict[labels[fiff_id]] = dict(d.items())['r']
    return fids_ras, fids_LNR, fids_LNR_dict

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
        
def find_good_HSPs(hpis_hsps, surf2warp, fiducial_file, above_thrs=0.02, 
                   min_rej_percent=2, max_rej_percent=10, 
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
        fig3d = viz.myMlabTriagularMesh(surf2warp['rr'], surf2warp['tris'], toplot=True, newfig=True, representation='s', color=(1,.8,0))
        viz.myMlabPoint3d(surf_pos, newfig=False, scale_factor=0.0005, color=(1,.5,0)) 
        viz.myMlabPoint3d(point_set2, newfig=False, scale_factor=0.003, color=(1,.6,0)) 
        viz.myMlabPoint3d(point_set2_scaled, newfig=False, scale_factor=0.003, color=(1,0,0)) 
        viz.myMlabPoint3d(point_set1, newfig=False, scale_factor=0.0035, color=(0,1,1), opacity=.5)
    point_set1_rphitheta        = mne.transforms._cart_to_sph(point_set1)
    point_set2_rphitheta        = mne.transforms._cart_to_sph(point_set2)
    outward_hsp_idx             = np.where(point_set1_rphitheta[:,0] > point_set2_rphitheta[:,0])[0]
    point_set2_scaled_rphitheta = mne.transforms._cart_to_sph(point_set2_scaled)
    
    outward_hsp_percentage = (len(outward_hsp_idx)*100) / len(point_set1)
    all_dists = np.abs(point_set1_rphitheta[:,0] - point_set2_scaled_rphitheta[:,0])*1000 # in mm
    
    if outward_hsp_percentage==0 and all_dists.max()<20.0:
        reject_idx = np.array([], int)
    else:
        """ find outliers """
        Q1      = np.percentile(all_dists, 25, interpolation = 'midpoint')
        Q3      = np.percentile(all_dists, 75, interpolation = 'midpoint')
        IQR     = Q3 - Q1
        extremes_idx = np.where(all_dists > (Q3+3.0*IQR))[0]
        outlrs_indx  = np.where(all_dists > (Q3+1.5*IQR))[0]
        rej_idx      = np.array(extremes_idx.tolist() + outlrs_indx.tolist())
        
        if len(rej_idx) == 0:
            reject_idx = all_dists.argsort(axis=None)[::-1][:n_min_rej_hsp]
        
        elif len(rej_idx) > 0 and len(rej_idx) < n_min_rej_hsp:
            reject_idx1 = rej_idx
            reject_idx2 = all_dists.argsort(axis=None)[::-1][:n_min_rej_hsp]
            reject_idx  = np.union1d(reject_idx1, reject_idx2).astype(int)
        
        elif len(extremes_idx) >= n_max_rej_hsp:
            reject_idx = extremes_idx
        elif len(rej_idx) >= n_max_rej_hsp:
            reject_idx = np.array(extremes_idx.tolist() + outlrs_indx.tolist())
        else:
            reject_idx = all_dists.argsort(axis=None)[::-1][:n_min_rej_hsp]
    print('Discarding %d points using HSPs-to-Surface distance-check.'%len(reject_idx))
    print('Total no. of discarded points= %d '%(((hpis_hsps_ras[:,2] < zlim)*1).sum() + len(reject_idx)))

    reject_idx = np.unique(reject_idx.astype(int))
    good_hpis_hsps_ras = np.delete(point_set1, reject_idx, axis=0)
    if toplot:
        viz.myMlabPoint3d(good_hpis_hsps_ras, newfig=False, scale_factor=0.003, color=(0,1,0), mode='cube', opacity=.5)
        if show_good_hsps_idx:
            for i_pnt, pnt in enumerate(good_hpis_hsps_ras): 
                text3d(pnt[0], pnt[1], pnt[2]+pnt[2]*.05, '%s'%i_pnt, **dict(color=(0,0,0), scale=0.003))
    print('Total no. of good points= %d\n'%len(good_hpis_hsps_ras))
    good_hpis_hsps = mne.transforms.apply_trans(trans_ras2neuromag, deepcopy(good_hpis_hsps_ras), move=True) # convert back to NM 
    if return_scene:
        return good_hpis_hsps, fig3d  
    else:
        return good_hpis_hsps
    
def shift_destCtrl_inward_to_maintain_realistic_scalp2hsp_dist(destCtrl, trans_ras2neuromag, 
                                                               trans_neuromag2ras, destHSPsShiftInwrd=0.0025, toplot=True):
    # In general, the skin surface stays just below the digitized points which is 1-3 mm in case when we digitize the HPI ...
    # or headshape due to thickness of coil and headshape. But for ideal warping situation, the warped surface goes through ...
    # all the digitization points which means the warped surface is slightly outward than it is supposed to be.
    # To overcome this problem, we assume the destCtrl points slightly inwards (three cardinal points 0-2 mm inward and other HSPs 0-5 mm inward)
    cardinal_shift, hsps_shift = min(0.002, destHSPsShiftInwrd/2), min(0.006, destHSPsShiftInwrd)
    print('\nShifting cardinals and HSPs inward by %smm and %smm for computing warping transform.\n'%(cardinal_shift*1000, 
                                                                                                      hsps_shift*1000))
    destCtrl2 = mne.transforms.apply_trans(trans_ras2neuromag, deepcopy(destCtrl), move=True)
    destCtrl2_rphitheta = mne.transforms._cart_to_sph(deepcopy(destCtrl2))          #  (rad, azimuth, polar)
    destCtrl2_rphitheta[:3,0]  -= cardinal_shift        # maximum inward shift limited to 2 mm for cardinals
    destCtrl2_rphitheta[3:,0]  -= hsps_shift            # maximum inward shift limited to 5 mm for HSPs
    destCtrl3           = mne.transforms._sph_to_cart(deepcopy(destCtrl2_rphitheta)) # convert back to cart.
    destCtrl4 = mne.transforms.apply_trans(trans_neuromag2ras, deepcopy(destCtrl3), move=True)
    if toplot:
        viz.myMlabPoint3d(destCtrl,    newfig=1, scale_factor=0.003, color=(1,0, 0))
        viz.myMlabPoint3d(destCtrl2,   newfig=0, scale_factor=0.003, color=(0,1, 0))
        viz.myMlabPoint3d(destCtrl3,   newfig=0, scale_factor=0.003, color=(0,0, 1))
        viz.myMlabPoint3d(destCtrl4,   newfig=0, scale_factor=0.003, color=(0,0, 0))
        
    return destCtrl4

def my_warp3d_trans(p, q, reg=1e-3, reg_mode=2, comment='', wtol=1e-06, return_reg=False):
    N = p.shape[0]
    px = np.tile(p[:,0].reshape((p[:,0].shape[0], 1)), (1, N))
    py = np.tile(p[:,1].reshape((p[:,1].shape[0], 1)), (1, N))
    pz = np.tile(p[:,2].reshape((p[:,2].shape[0], 1)), (1, N))
    K = np.sqrt((px - px.T)**2 + (py - py.T)**2 + (pz - pz.T)**2)
    P = np.hstack(( p, np.ones((N,1)) ))
    L = np.vstack([np.hstack([K, P]),
                   np.hstack([P.T, np.zeros((4, 4))])])
    D = np.concatenate((q - p, np.zeros((4, 3))), axis=0)
    if reg_mode==1:
        try:
            H = linalg.solve(L,D, check_finite=True, assume_a='gen', transposed=False)    
        except linalg.LinAlgError: 
            L += reg * np.eye(L.shape[0])
            H = linalg.solve(L,D, check_finite=True, assume_a='gen', transposed=False)
    else:        
        try:
            H = linalg.solve(L,D, check_finite=True, assume_a='gen', transposed=False)
        except linalg.LinAlgError:
            print('Using regularization = %s'%reg)
            L += reg * np.eye(L.shape[0])
            H = linalg.solve(L,D, check_finite=True, assume_a='gen', transposed=False)
        while any((np.dot(L, H) - D).flatten() >= wtol): # changed on 13/05/2022
            print('Using regularization = %s'%reg)
            L += reg * np.eye(L.shape[0])
            H = linalg.solve(L,D, check_finite=True, assume_a='gen', transposed=False)
            reg *= 10
    if (np.isnan(H)).any():
        H = linalg.pinv(L) * D
    W = H[:N,:]
    A = H[N:, :]
    e = np.sum(np.diag(np.dot(W.T, np.dot(K, W))))
    print('\n%s'%comment + ' Bending energy = %s'%e + ' (reg = %s)\n'%reg )
    if return_reg:
        return W, A, e, reg
    else:
        return W, A, e

def my_warp_src(r, A, W, p):
    rw = np.dot(r, A[:3,:3])
    rw = rw + A[3,:]
    n_p = p.shape[0]
    U = np.sqrt( ((np.tile(r[:,0].reshape((r[:,0].shape[0],1)), (1, n_p)) - p[:,0].T) ** 2) + 
                  ((np.tile(r[:,1].reshape((r[:,1].shape[0],1)), (1, n_p)) - p[:,1].T) ** 2) +   
                  ((np.tile(r[:,2].reshape((r[:,2].shape[0],1)), (1, n_p)) - p[:,2].T) ** 2) )    
    rw = rw + np.dot(U, W)   
    return rw

def add_line_to_file(fname, line):
    with open(fname, 'a') as fileID:
        fileID.writelines(line)
    fileID.close()
    
def write_mri_warping_config(subjects_dir_from, isotrak_fname, subject_from, 
                             subjects_dir_to, subject_to, 
                             srcCtrl, destCtrl, W, A, e, coord_frame='MRI (surface RAS)', 
                             misc_comments='', fname=None, return_fname=False):
    """
    Write the cfg file describing a output pseudoMRI.
    ----------
    subjects_dir_from : TYPE
        DESCRIPTION.
    isotrak_fname : TYPE
        DESCRIPTION.
    subject_from : TYPE
        DESCRIPTION.
    subjects_dir_to : TYPE
        DESCRIPTION.
    subject_to : TYPE
        DESCRIPTION.
    srcCtrl : TYPE
        DESCRIPTION.
    destCtrl : TYPE
        DESCRIPTION.
    W : TYPE
        DESCRIPTION.
    A : TYPE
        DESCRIPTION.
    e : TYPE
        bending energy.
    coord_frame : TYPE, optional
        DESCRIPTION. The default is 'MRI (surface RAS)'.
    misc_comments : TYPE, optional
        DESCRIPTION. The default is ''.
    fname : TYPE, optional
        DESCRIPTION. The default is None.
    """
    config = configparser.RawConfigParser()
    configparser.ConfigParser.optionxform=str
    
    config.add_section("MRI Warping")
    config.set("MRI Warping", 'template_dir',       subjects_dir_from)
    config.set("MRI Warping", 'template_name',      subject_from)
    config.set("MRI Warping", 'isotrak_fname',      isotrak_fname)
    config.set("MRI Warping", 'subject_to',         subject_to)
    config.set("MRI Warping", 'No. of ctrl points', str(len(destCtrl)))
    config.set("MRI Warping", 'coord_frame',        str(coord_frame))
    
    config.set("MRI Warping", 'misc. comments',     str(misc_comments))
    
    config.set("MRI Warping", 'destCtrl',           str(destCtrl))
    config.set("MRI Warping", 'srcCtrl',            str(srcCtrl))
    
    config.set("MRI Warping", 'e',                  str(e))
    config.set("MRI Warping", 'W',                  str(W))
    config.set("MRI Warping", 'A',                  str(A))
    
    config.set("MRI Warping", 'coord_frame',        str(coord_frame))

    if fname is None:
        fname = '%s/%s/pseudoMRI_processing_parameters.cfg'%(subjects_dir_to, subject_to)
    config.set("MRI Warping", 'version', '1')
    with open(fname, 'w') as fileID:
        config.write(fileID)
    print('\npseudoMRI configs are written in file %s'%fname)
    if return_fname:
        return fname
    
#%%
def my_get_SurfaceFile_v2(subjects_dir, subject, read_files=True, read_metadata=False):
    SurfaceFile_names = dict()
    for surf in ['/bem/%s-head.fif'%subject, '/bem/%s-head-dense.fif'%subject, '/bem/watershed/%s_inner_skull_surface'%subject, 
                 '/bem/watershed/%s_outer_skull_surface'%subject, '/bem/watershed/%s_outer_skin_surface'%subject, 
                 '/bem/watershed/%s_brain_surface'%subject,
                 '/bem/watershed/synthstriped_brain.surf', '/bem/watershed/synthstriped_inner_skull.surf',
                 '/surf/lh.orig', '/surf/rh.orig', '/surf/lh.pial', '/surf/rh.pial', '/surf/lh.white', 
                 '/surf/rh.white', '/surf/lh.inflated', '/surf/rh.inflated', 
                 '/surf/lh.smoothwm', '/surf/rh.smoothwm', '/surf/lh.seghead']:
        if exists(join(subjects_dir, subject) + surf):
            SurfaceFile_names[surf] = join(subjects_dir, subject) + surf
            print(surf)
            # Surfaces = dict()
    if read_files:
        Surfaces = dict()
        Surface_scaling = dict()
        for surf in list(SurfaceFile_names.keys()):
            surface_file = SurfaceFile_names[surf]
            try:
                surface_mesh = mne.surface.read_surface(surface_file, read_metadata=read_metadata, return_dict=True, 
                                                        file_format='auto', verbose=None)[2]
                scaling_factor = 1000.
                surface_mesh['rr'] /= scaling_factor # convert to m
            except ValueError as valuerr: # this is reading dense and meadium skin surface
                print(valuerr)
                surface_mesh = mne.read_bem_surfaces(surface_file)[0]
                scaling_factor = 1.
                surface_mesh['rr'] /= scaling_factor
            Surfaces[surf]        = surface_mesh     
            Surface_scaling[surf] = scaling_factor
            del surf, scaling_factor, surface_mesh
    return SurfaceFile_names, Surfaces, Surface_scaling

#%% warp anatomy----------------------------------------------------------------------
def apply_warp_to_anatomy(srcPts, destPts, mridata, t1_fname=None, Torig=None, block_size=1000000, 
                          reg=0.00005, reg_mode=1, toplot_final=False, warped_surf=None, n_jobs=1, resample=False, 
                          rs_voxel_sizes=[.5,.5,.5]):
    """
    Parameters
    ----------
    srcPts : np.array Nx3
        Source control points (from surface), in freesurfer surface RAS coordsys.
    destPts : np.array Nx3
        Destination control points (from dig points), in freesurfer surface RAS coordsys.
    mridata : np.array
        Data from MRI object
    t1_fname : str, optional. The default is None.
        mri filename in mgz format  e.g. '../subjects/mni_152_pMRI/mri/orig/001.mgz'. 
    Torig : np.array 4x4, The default is None.
        Transform from voxels to the FreeSurfer MRI surface RAS coordinate frame, 
        known as Torig in the FreeSurfer documentation, and vox2ras_tkr in nibabel.
    block_size : TYPE, optional
        Block size for each loop. The default is 1000000.
    toplot_final : TYPE, optional
        DESCRIPTION. The default is False.
    warped_surf : TYPE, optional
        DESCRIPTION. The default is None.
    Returns
    -------
    newCube : np.array
        warped mridata (anatomy).
    t1 : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.
    Torig : TYPE
        DESCRIPTION.
    Torig_inv : TYPE
        DESCRIPTION.
    """
    if mridata is None or Torig is None:
        print('Reading T1 data from %s'%t1_fname)
        t1 = nib.load(t1_fname); print(t1)
        if resample:
            print('\nNote: Applying resampling....')
            t1_res = resample_to_output(deepcopy(t1), voxel_sizes=rs_voxel_sizes, 
                                        out_class=nib.freesurfer.mghformat.MGHImage)
            del t1
            t1  = t1_res
            print(t1)
        data = np.asarray(t1.dataobj)
        mgh = nib.MGHImage(t1.dataobj, t1.affine)
        vox2ras_tkr = mgh.header.get_vox2ras_tkr()
        Torig = vox2ras_tkr 
        vox2ras_tkr_inv = np.linalg.inv(vox2ras_tkr)
        Torig_inv = vox2ras_tkr_inv
        miscs = dict()
        miscs['mriobj']        = t1
        miscs['Torig']         = Torig
        miscs['Torig_inv']     = Torig_inv
        miscs['original_anat'] = data
    else:
        data = deepcopy(mridata)
        miscs = dict()
    print('\nData dimensions = %s'%str(data.shape))
    print('Using %d vox/it for warping, increasing block_size would fasten the process but requires more memory.'%block_size)
    srcPts_vox  = mne.transforms.apply_trans(np.linalg.inv(Torig), srcPts*1000)
    destPts_vox = mne.transforms.apply_trans(np.linalg.inv(Torig), destPts*1000)
    W_vox, A_vox, e_vox, regWtransVox  = my_warp3d_trans(srcPts_vox, destPts_vox, reg=reg, reg_mode=reg_mode, 
                                                            comment='Estimating Wtrans for vox..', return_reg=True)
    W_vox_inv, A_vox_inv, e_vox_inv, reginvWtransVox = my_warp3d_trans(destPts_vox, srcPts_vox, reg=reg, reg_mode=reg_mode, 
                                                            comment='Estimating inv. Wtrans for vox..', return_reg=True)
    miscs['e_vox']     = e_vox
    miscs['e_vox_inv'] = e_vox_inv
    miscs['regWtransVox'] = regWtransVox
    miscs['reginvWtransVox'] = reginvWtransVox
    utils.tic()
    if len(data.shape) > 3:
        print('====> Omitting, not implemented for 4D data.')
        newCube = None
    else: 
        sizeMri = data.shape
        newCube = np.ones(sizeMri)
        nVoxels = data.size     
        BLOCK_SIZE = block_size 
        nBlocks = int(np.ceil(nVoxels / BLOCK_SIZE))
        ix0 = 0
        for i in tqdm(range(nBlocks)):
            ix1 = min(ix0 - 0 + BLOCK_SIZE, nVoxels)
            xv, yv, zv = np.unravel_index(np.arange(ix0,ix1), sizeMri, order='F')
            rv = np.vstack(( np.vstack((xv, yv)), zv)).T
            if n_jobs==1:
                rv_inv = my_warp_src(rv, A_vox_inv, W_vox_inv, destPts_vox) + rv
            else:
                print('Parallel processing is not supported yet; stay tuned.')
                rv_inv = my_warp_src(rv, A_vox_inv, W_vox_inv, destPts_vox) + rv
            rv_inv = np.round(rv_inv)
            iOutside = np.where( np.sum( np.logical_or((rv_inv < 0), 
                                                       (rv_inv > np.tile(np.array(sizeMri)-1, 
                                                                         (rv_inv.shape[0], 1)))), axis=1) > 0)[0]
            rv_inv[iOutside,:] = 1
            rv_inv = rv_inv.astype(int)
            ix_inv = utils.sub2ind_matlab(sizeMri, rv_inv[:,0], rv_inv[:,1], v3=rv_inv[:,2]).astype(int)
            newCube = newCube.flatten('F')
            newCube[np.arange(ix0,ix1)] =  data.copy().flatten('F')[ix_inv]
            newCube = newCube.reshape(sizeMri, order='F')
            newCube = newCube.flatten('F')
            newCube[iOutside] = 0
            newCube = newCube.reshape(sizeMri, order='F')
            ix0 = ix1# + 1;
        del rv
        print('\n'); 
    utils.toc()
    if toplot_final and not warped_surf is None and not newCube is None:
        print('\nFIX IT for plotting')
    return newCube, miscs

#%% A grand single module to generate pseudoMRI for a given subject 
def pseudomriengine(pseudo_subject, pseudo_subjects_dir, isotrak, template, templates_dir, fiducial_file,  
                    dense_hsp=False, mirror_hsps=True, template_headsurf='auto', dense_surf=True, z_thres=0.02, 
                    destHSPsShiftInwrd=0.0025, Wreg_est=0.000005, Wreg_apply= 0.00005, wtol=1e-06,
                    warp_anatomy=True, which_mri='all', blocksize=500000, write2format=['.mgz'], n_jobs=1, 
                    toplot=True, toooplot=True, pyplot_fsize=12,
                    plot_zoom_in='12%', plot_nslice=16, plot_tol=3, plot_side_leave='25%', plot_lw=1.5,  
                    plot_titlecolor=(.8,.9,.2), plot_titlefsize=18, dig_reject_min_max=[2, 10], 
                    use_hpi=True, show_good_hsps_idx=True, rem_good_pts_idx=[],  nmax_Ctrl= 150,
                    report=None, report_file=None, args=None):
    if isinstance(isotrak, str):
        isotrak_file = isotrak
        if exists(isotrak_file):
            print(pseudo_subject, isotrak_file)
        # else:
        #     raise IndexError
        fids, hpis, hsps, fig3d    = find_plot_isotraks(isotrak_file, toplot=toplot, helmet=False)
    elif 'fids' in isotrak and 'hpis' in isotrak and 'hsps' in isotrak:
        fids = isotrak['fids']
        hpis = isotrak['hpis']
        hsps = isotrak['hsps']
    else:
        raise Exception('Contact developers to provide integration for you device.')
    all_hsps            = np.vstack((fids, np.vstack((hpis, hsps))))
    hpis_hsps           = np.vstack((hpis, hsps)) if use_hpi else hsps
    
    if report!=None and report_file!=None:
        viz.get_put_snaps(report, report_file, title='Original headshape points', section='Headshape', tags=('hsps',), 
              caption=f"No. of digitized points = {len(all_hsps)} ({len(fids)} fids, {len(hpis)} HPIs, {len(hsps)} hsps)", 
              add2report=True, save=True, fig3d=fig3d,  image_format='png', replace=True, **args.snap_config)
        close(fig3d); del fig3d
    
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
    hpis_hsps_good = find_good_HSPs(hpis_hsps, surf2warp, fiducial_file, above_thrs=above_thrs, 
                                    min_rej_percent=dig_reject_min_max[0], 
                                    max_rej_percent=dig_reject_min_max[1], toplot=toplot, 
                                    show_good_hsps_idx=show_good_hsps_idx)
    if len(rem_good_pts_idx)>0:
        hpis_hsps_good = np.delete(hpis_hsps_good, np.array(rem_good_pts_idx), axis=0)
        viz.myMlabPoint3d( mne.transforms.apply_trans(trans_neuromag2ras, deepcopy(hpis_hsps_good), move=True),
                                  toplot=toplot, newfinmax_Ctrlg=False, scale_factor=0.010, color=(0,1,0), mode='cone')
        print('Total no. of good points= %d'%hpis_hsps_good.shape[0])
    if report!=None and report_file!=None:
        viz.get_put_snaps(report, report_file, title='Good headshape points', section='Headshape', tags=('hsps',), 
              caption=f"No. of good headshape points = {len(hpis_hsps_good)} ", 
              add2report=True, save=True, fig3d=gcf(),  image_format='png', replace=True, **args.snap_config)
        close(gcf())
   
    hpis_hsps_above_thrs= np.where(hpis_hsps_good[:,2]>-above_thrs)[0] # doesn't matter but check it again
    hpis_hsps2warp      = hpis_hsps_good[hpis_hsps_above_thrs, :]
    iso_val = IsoScore.IsoScore(hpis_hsps2warp.T) # Uniformilty
    total_good_points = hpis_hsps2warp.shape[0]+3
    print('\nNo. of HSPs = %s \nNo. of HSPs used = %s \nIso_score = %.2f'%(all_hsps.shape[0], total_good_points, iso_val))
    print('HPI coils locations were not used.') if not use_hpi else None
    
    hpis_hsps2warp_orig = np.empty((0,3))
    hpis_hsps2warp      = hpis_hsps2warp
    
    apply_mirroring = mirror_hsps and hpis_hsps2warp.shape[0]<40
    if apply_mirroring:
        print('\nNOTE: Mirroring HSPs to make it denser...')
        hpis_hsps2warp_true    = deepcopy(hpis_hsps2warp)
        hpis_hsps2warp_reflect = deepcopy(hpis_hsps2warp) * (-1,1,1)
        hpis_hsps2warp = np.vstack((hpis_hsps2warp_true, hpis_hsps2warp_reflect))
    
    # to avoid too dense points, downsample to 100-300 points only
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
    
    #% % Compute warping transform
    # The three srcCtrl points for isotrak fids would be three MRI surface fids (from fiducial file)
    fids_ras, fids_LNR, fids_LNR_dict = get_fiducial_LNR(fiducial_file)
    srcCtrl_fids = deepcopy(fids_LNR)
    srcCtrl_fids_dist = utils.get_all_to_all_point_dist(srcCtrl_fids, digpoints[:3], toplot=toplot)
    
    # find closed points for rest of the HSPs and compute distances
    mesh_pos       = deepcopy(surf2warp['rr']) 
    above_thrs_idx = np.where(mesh_pos[:,2] > (fids_LNR[:,2].min()-above_thrs))[0]
    mesh_pos       = mesh_pos[above_thrs_idx, :]
    if toooplot:
        viz.myMlabTriagularMesh(surf2warp['rr'], surf2warp['tris'], representation='s', opacity=0.2, axis=True, color=(.7,.7,.7))
        viz.myMlabPoint3d(mesh_pos,     newfig=0, scale_factor=0.0003,  color=(0,1, 0))
        viz.myMlabPoint3d(srcCtrl_fids, newfig=0, scale_factor=0.005,   color=(0,0, 1))
        viz.myMlabPoint3d(digpoints[:3],newfig=0, scale_factor=0.005,   color=(0,1, 1))
        viz.myMlabPoint3d(digpoints,    newfig=0, scale_factor=0.005,   color=(0.5,1, 1))
        viz.myMlabPoint3d(all_hsps,     newfig=0, scale_factor=0.002,   color=(1,0, 0))
    if report!=None and report_file!=None:
        viz.get_put_snaps(report, report_file, title='All aligned before detecting control points', section='Control points', tags=('warping',), 
              caption="Headshape points and template scalp surface aligned before identifying the control point pairs", 
              add2report=True, save=True, fig3d=gcf(),  image_format='png', replace=True, **args.snap_config)
        close(gcf())
    
    closest_vert_pos_rest, closest_vert_idx_rest, closest_vert_dist_rest = np.empty((0,3)), np.empty((0,1), dtype=int), np.empty((0,1))
    for ii in range(3, digpoints.shape[0]):
        closest_vert_pos, idx, closest_vert_dist = find_closest_node_dist(digpoints[ii,:], mesh_pos,  multipy2dist=1000)
        closest_vert_pos_rest  = np.vstack((closest_vert_pos_rest, closest_vert_pos))
        closest_vert_idx_rest  = np.vstack((closest_vert_idx_rest, idx))
        closest_vert_dist_rest = np.vstack((closest_vert_dist_rest, closest_vert_dist))
        
    # Append all for fids and hpi_hsp
    closest_vert_pos_all   = np.vstack((srcCtrl_fids, closest_vert_pos_rest))
    closest_vert_dist_all = np.hstack((srcCtrl_fids_dist, closest_vert_dist_rest.flatten())).T
    print('\nDestCtrl to SrcCtrl min, mean, max dist. = %.2f mm, %.2f mm, %.2f mm\n'\
          %(closest_vert_dist_all.min(),closest_vert_dist_all.mean(), closest_vert_dist_all.max()))
    srcCtrl  = deepcopy(closest_vert_pos_all) # source control points (p)
    destCtrl = deepcopy(digpoints)            # destination contrl points (q)
    if toooplot:
        figg = viz.myMlabTriagularMesh(surf2warp['rr'], surf2warp['tris'], toplot=True, newfig=True, representation='s', color=(.7,.7,.7), opacity=.3, axis=False)
        viz.myMlabPoint3d(srcCtrl, newfig=False, scale_factor=0.004, color=(1,0,0), axis=False) 
        viz.myMlabPoint3d(destCtrl, newfig=False, scale_factor=0.004, color=(0,1,0), axis=False)
        if report!=None and report_file!=None:
            viz.get_put_snaps(report, report_file, title='Control points pairs', section='Control points', tags=('warping',), 
                  caption=f" {len(srcCtrl)} control point pairs (red: source (scalp), green: destination (digitization))", 
                  add2report=True, save=True, fig3d=figg,  image_format='png', replace=True, **args.snap_config)
            close(figg); del figg
    print(f'Computing the warping transform using linear solver with {srcCtrl.shape[0]} source ' \
          f'and {destCtrl.shape[0]} destination control points (with regularization = {Wreg_est}).')

    if destHSPsShiftInwrd > 0: 
        destCtrl_new = shift_destCtrl_inward_to_maintain_realistic_scalp2hsp_dist(deepcopy(destCtrl), 
                                                                                  trans_ras2neuromag, trans_neuromag2ras, 
                                                                                  destHSPsShiftInwrd=destHSPsShiftInwrd, toplot=False)
        viz.myMlabPoint3d(destCtrl,     toplot=toooplot, newfig=True,  scale_factor=0.002, color=(0,0,1))
        viz.myMlabPoint3d(destCtrl_new, toplot=toooplot, newfig=False, scale_factor=0.002, color=(0,1,0))
        if report!=None and report_file!=None:
            viz.get_put_snaps(report, report_file, title='Inward shifted hsps', section='Headshape', tags=('hsps',), 
                  caption=f"Blue: original, Green: inward shifted {destHSPsShiftInwrd * 1000} mm", 
                  add2report=True, save=True, fig3d=gcf(),  image_format='png', replace=True, **args.snap_config)
            close(gcf())
    else:
        destCtrl_new = deepcopy(destCtrl)
    
    W, A, e, Wreg_est2 = my_warp3d_trans(srcCtrl, destCtrl_new, reg=Wreg_est, comment='Estimating Wtran for surfs..', 
                                            wtol=wtol, return_reg=True) # Note that this W, A, e are for RAS coordsys
    print(np.mean(np.abs(deepcopy(W)), axis=0), '\n\n', A, '\n\n', e)
    
    #% % Warp the template head surface and check the distance between surface and HSPs 
    mesh_pos = surf2warp['rr']
    warpedSurf_verts = my_warp_src(deepcopy(mesh_pos), A, W, srcCtrl) + deepcopy(mesh_pos)
    warpedSurf       = deepcopy(surf2warp)
    warpedSurf['rr'] = deepcopy(warpedSurf_verts)
    if toooplot:
        viz.myMlabTriagularMesh(surf2warp['rr'], surf2warp['tris'], representation='s', color=(.7,.7,.7), opacity=0.2)
        viz.myMlabPoint3d(srcCtrl, newfig=False, scale_factor=0.002, color=(1,0,0), opacity=.8) 
        viz.myMlabTriagularMesh(warpedSurf['rr'], warpedSurf['tris'], newfig=False, representation='s', color=(0,1,0), opacity=0.2)
        viz.myMlabPoint3d(destCtrl, newfig=False, scale_factor=0.003, color=(0,1,0), opacity=.8)
        gcf().scene.light_manager.light_mode = "vtk"
        gcf().scene.light_manager.number_of_lights = 5
        if report!=None and report_file!=None:
            viz.get_put_snaps(report, report_file, title='Warped scalp surface', section='Warped surfaces', tags=('warping',), 
                  caption="Warped scalp surface (green)", 
                  add2report=True, save=True, fig3d=gcf(),  image_format='png', replace=True, **args.snap_config)
            close(gcf())
        
    closest_vert_pos_all2, closest_vert_dist_all2, mesh_pos2 = np.empty((0,3)), np.empty((0,1)),  deepcopy(warpedSurf['rr'])
    for ii in range(digpoints.shape[0]):
        closest_vert_pos_all2  = np.vstack((closest_vert_pos_all2, find_closest_node_dist(digpoints[ii,:], mesh_pos2,  multipy2dist=1000)[0]))
        closest_vert_dist_all2 = np.vstack((closest_vert_dist_all2, find_closest_node_dist(digpoints[ii,:], mesh_pos2,  multipy2dist=1000)[2]))
    print('\nIsotrak to warped-surface distance min, mean, max dist. = %.2f mm, %.2f mm, %.2f mm\n'\
          %(closest_vert_dist_all2.min(),closest_vert_dist_all2.mean(), closest_vert_dist_all2.max()))
    del warpedSurf_verts, closest_vert_pos_all2, closest_vert_dist_all2, mesh_pos2 
    
    #% % Create subject's directory to write warped results + Write the warping transform 
    subject_directory = f'{pseudo_subjects_dir}/{pseudo_subject}/'
    makedirs(subject_directory, exist_ok=True)
    for dir_ in ['mri', 'bem', 'surf', 'label', 'src', 'coreg', 'warped_figs']:
        makedirs(subject_directory + dir_, exist_ok=True)
    misc_comments = 'HSPs mirrored = %s\t'%str(apply_mirroring)
    config_fname = write_mri_warping_config(templates_dir, isotrak_file.split('/', maxsplit=7)[-1], template, 
                             pseudo_subjects_dir, pseudo_subject, srcCtrl, destCtrl, W, A, e, 
                             coord_frame='MRI (surface RAS)',  misc_comments=misc_comments, return_fname=True)
    add_line_to_file(config_fname, "Regularization & Bending energy:\n")
    add_line_to_file(config_fname, "Reguln. applied to estimate Wtrans for surface \t= %s\n"%Wreg_est2)
    add_line_to_file(config_fname, "Bending energy in estimating Wtrans for surface = %s\n"%e)
        
    #% % Find source mri, bems, src, etc. paths
    src_paths = mne.coreg._find_mri_paths(subject=template, skip_fiducials=False, subjects_dir=templates_dir)
    
    #% % Warp fiducials and compute new transform and write files 
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
                template_fiducials_m_warped[fid_labels[fiff_id-1]] = (my_warp_src( deepcopy(dict(d.items())['r']).reshape(1,3), \
                                                                            A, W, srcCtrl ) + deepcopy(dict(d.items())['r']).reshape(1,3)).flatten()
    
    warped_fids2write = deepcopy(template_fiducials) # just to copy the fiducial structure
    for fiff_id in [1,2,3]:
        for d in warped_fids2write[0]:
            if d['ident'] == fiff_id:
                warped_fids2write[0][fiff_id-1]['r']  = template_fiducials_m_warped[fid_labels[fiff_id-1]]  
    fiducial_file_warped = '%s/bem/%s-fiducials.fif'%(subject_directory, pseudo_subject)
    mne.io.write_fiducials(fiducial_file_warped, warped_fids2write[0], warped_fids2write[1], verbose=args.verbose)
    print('\nfiducial file for pMRI was written as %s'%fiducial_file_warped)  
    
    # Compute and write the new trans for warped surfaces
    trans_ras2neuromag_warped, trans_neuromag2ras_warped = get_ras_to_neuromag_trans_from_fiducial_file(fiducial_file_warped)
    trans_temp = mne.transforms.Transform(4, 5, trans_neuromag2ras_warped)
    trans_file_warped = '%s/coreg/%s-trans.fif'%(subject_directory, pseudo_subject)
    mne.write_trans(trans_file_warped, trans_temp)
    print('\nTrans file for pMRI is written:  %s'%trans_file_warped)  
    
    #% % Warp headmodel and brain surfaces and write files 
    mne.utils.logger.info("Warping headmodel and brain surfaces:  %s -> %s", template, pseudo_subject)
    # Read different surfaces of the template reconstructed by Freesurfer 
    SurfaceFile_names, Surfaces, Surface_scaling = my_get_SurfaceFile_v2(templates_dir, template, 
                                                                          read_files=True, read_metadata=False)
    
    # Warp these surfaces and write in RAS coordsys system
    figname_final ='R> P:%s'%(pseudo_subject)
    SurfWarped = dict()
    viz.myMlabPoint3d(destCtrl, scale_factor=0.005, figname=figname_final, toplot=toplot, newfig=True) 
    viz.myMlabPoint3d(np.vstack(list(template_fiducials_m_warped.values())), scale_factor=0.005, mode='cube', 
                      figname=figname_final, toplot=toplot, newfig=False)
    for surf in list(SurfaceFile_names.keys()):
        print('\nWarping %s surface...'%surf)
        Surface = Surfaces[surf]
        SurfNew = dict()
        SurfNew['Faces']    = Surface['Faces'] if 'Faces' in Surface else Surface['tris']
        Surface_verts       = Surface['Vertices'] if 'Vertices' in Surface else Surface['rr']
        SurfNew['Vertices'] = my_warp_src(deepcopy(Surface_verts), A, W, srcCtrl) + deepcopy(Surface_verts)
        del Surface_verts
        if surf in [f'/bem/watershed/{template}_inner_skull_surface', 
                    f'/bem/watershed/{template}_outer_skin_surface', 
                    f'/bem/watershed/{template}_brain_surface',
                    '/surf/lh.white', '/surf/rh.white', 
                    f'/bem/{template}-head-dense.fif',
                    ]:
            viz.myMlabTriagularMesh(SurfNew['Vertices'], SurfNew['Faces'], toplot=toplot, newfig=False, 
                                            figname=figname_final, representation='surface', opacity=0.2)        
        SurfNew_fname = '%s%s'%(subject_directory, surf.replace(template, pseudo_subject))
        print('Writing the warped surface file as: \n%s'%SurfNew_fname)
        makedirs( utils.dirname2(SurfNew_fname), exist_ok=True)
        if SurfNew_fname.endswith('.fif'):
            surf_temp = mne.read_bem_surfaces(SurfaceFile_names[surf])
            surf_temp[0]['rr']    = SurfNew['Vertices']
            surf_temp[0]['nn']    = [] 
            surf_temp[0]['sigma'] = 0.33
            mne.write_bem_surfaces(SurfNew_fname, surf_temp, overwrite=True, verbose=args.verbose)
            del surf_temp
        else:
            mne.surface.write_surface(SurfNew_fname, SurfNew['Vertices']*Surface_scaling[surf], SurfNew['Faces'], 
                                      create_stamp='', volume_info=None, file_format='auto', overwrite=True)
        SurfWarped[surf] = SurfNew
        del Surface, SurfNew
    if report!=None and report_file!=None:
        viz.get_put_snaps(report, report_file, title='pseudo-MRI surfaces', section='Warped surfaces', tags=('pseudoMRI',), 
              caption="Warped head and cortical surface", 
              add2report=True, save=True, fig3d=gcf(),  image_format='png', replace=True, **args.snap_config)
        close(gcf())
    # Make bem surface soft links
    try:
        utils.make_surfaces_soft_links3(pseudo_subjects_dir, pseudo_subject, surfs_from='mri2surf')
    except Exception as exerrr:
        print(exerrr)
        utils.make_surfaces_soft_links(pseudo_subjects_dir, pseudo_subject)
    
    # copy/duplicate the files that does not require warping  (CONFIRM IT)
    print('Copying sphere and curv files...%s >> %s'%(template, pseudo_subject))
    for src_name in src_paths['duplicate']:
        srcfile = src_name.format(subjects_dir=templates_dir, subject=template)
        dstfile = src_name.format(subjects_dir=pseudo_subjects_dir, subject=pseudo_subject)
        copyfile(srcfile, dstfile)
        
    #% % Warp source spaces
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
            ss['rr'] = my_warp_src(deepcopy(verts_pos), A, W, srcCtrl) + deepcopy(verts_pos)
            del verts_pos
        if add_dist:
            dist_limit = np.inf
            mne.utils.logger.info("Recomputing distances, this might take a while")
            mne.source_space.add_source_space_distances(sss, dist_limit, n_jobs=n_jobs)
        mne.source_space.write_source_spaces(dst, sss, overwrite=True, verbose=args.verbose)
    
    #% % Warp labels
    mne.utils.logger.info("Warping labels:  %s -> %s", template, pseudo_subject)
    lbl_dir = join(templates_dir, template, 'label')
    pattern = None
    if pattern is None:
        label_paths = []
        for dirpath, _, filenames in walk(lbl_dir):
            rel_dir = relpath(dirpath, lbl_dir)
            for filename in fnmatch.filter(filenames, '*.label'):
                thispath = join(rel_dir, filename)
                label_paths.append(thispath)
    else:
        label_paths = [relpath(thispath, lbl_dir) for thispath in iglob(pattern)]
    # apply warping ....
    src_root = join(templates_dir, template, 'label')
    dst_root = join(subject_directory, 'label')
    # warp labels
    for fname in label_paths:
        dst = join(dst_root, fname)
        dirname_ = dirname(dst)
        if not exists(dirname_):
            makedirs(dirname_)
        src = join(src_root, fname)
        l_old = mne.label.read_label(src)
        # pos = l_old.pos * scale
        print('warping... %s.label'%l_old.name)
        pos = my_warp_src(deepcopy(l_old.pos), A, W, srcCtrl) + deepcopy(l_old.pos)
        # viz.myMlabPoint3d(deepcopy(pos), scale_factor=0.0001,  newfig=False, toplot=toplot)
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
    
    #% % Warp anatomy
    # viz.arrange_pyplot_font_etc(f_size=pyplot_fsize)
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
        mris2warp = [fname_ for  fname_ in mris2warp if not 'rawavg.mgz' in fname_] # discard rawavg.mgz bcoz it's in native coords 
        for src_mri_fname in mris2warp:  
            print('\nwarping... %s for %s'%(src_mri_fname, pseudo_subject))
            warped_mri_fname = src_mri_fname.replace( templates_dir, pseudo_subjects_dir ).replace( template, pseudo_subject )
            wdata, miscs = apply_warp_to_anatomy(deepcopy(srcCtrl), deepcopy(destCtrl), mridata=None, t1_fname=src_mri_fname, 
                                                           Torig=None, block_size=blocksize, reg=Wreg_apply, reg_mode=2, 
                                                           toplot_final=False, warped_surf=None, n_jobs=1, 
                                                           resample=False, rs_voxel_sizes=[.5,.5,.5])            
            add_line_to_file(config_fname, "***%s***\n"%warped_mri_fname.replace(pseudo_subjects_dir, ''))
            add_line_to_file(config_fname, "Reguln. applied to estimate Wtrans for vox \t= %s\n"%miscs['regWtransVox'])
            add_line_to_file(config_fname, "Bending energy in estimating Wtrans for vox \t= %s\n"%miscs['e_vox'])
            add_line_to_file(config_fname, "Reguln applied to estimate inv Wtrans for vox \t= %s\n"%miscs['reginvWtransVox'])
            add_line_to_file(config_fname, "Bending energy in estimating inv Wtrans for vox = %s\n"%miscs['e_vox_inv'])
            if wdata is not None:
                fig = viz.plot_surf_anat_alignment_v4(wdata, miscs['Torig'], destCtrl, warpedSurf, zoom_in=plot_zoom_in,
                                                title= warped_mri_fname.replace( pseudo_subjects_dir, '' ), 
                                                nslice=plot_nslice, tol=plot_tol, side_leave=plot_side_leave, lw=plot_lw,  
                                                titlecolor=plot_titlecolor, titlefsize=plot_titlefsize)
                plt.show();     plt.pause(.5);     plt.get_current_fig_manager().full_screen_toggle(); plt.pause(.5)
                plt.subplots_adjust(top=0.999,bottom=0.025,left=0.1,right=0.9,hspace=0.0,wspace=-0.0); plt.pause(.5)
                if report!=None and report_file!=None:
                    plt.suptitle('')
                    report.add_figure(fig, title=f"pseudo-MRI's {src_mri_fname.split('/')[-1]}",
                          caption=join(f"../{pseudo_subject}/mri/{src_mri_fname.split('/')[-1]}"),
                          image_format=None, tags=('pseudoMRI',), replace=True, section='Warped anatomy')
                    report.save(fname=report_file, open_browser=False, overwrite=True, sort_content=False, verbose=args.verbose)
                # Save the warped MRI file
                makedirs(dirname(warped_mri_fname), exist_ok=True)
                warped_mri_affine = deepcopy(miscs['mriobj'].affine)
                warped_mri_header = deepcopy(miscs['mriobj'].header)
                for out_format in write2format:
                    if out_format=='.mgz':
                        mri_warped = nib.freesurfer.mghformat.MGHImage(deepcopy(wdata), warped_mri_affine, warped_mri_header)
                    elif out_format=='.nii':
                        mri_warped = nib.Nifti1Image(deepcopy(wdata), warped_mri_affine, warped_mri_header)
                    nib.save(mri_warped, warped_mri_fname.replace('.mgz', out_format))
                    del mri_warped
                plt.close()
    return print('\nCheck results in -> %s/%s'%(pseudo_subjects_dir, pseudo_subject), '\ne=%s'%e)
#%%