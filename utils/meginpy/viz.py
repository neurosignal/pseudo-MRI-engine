#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 22:27:11 2019
Author: Amit Jaiswal <amit.jaiswal@megin.fi>, <amit.jaiswal@aalto.fi>
meginpy.viz: This is part of MEGINPY platform employing tools for visualization.

"""
import mne
from mne.transforms import read_trans, invert_transform, apply_trans
from mayavi.mlab import figure, points3d, triangular_mesh, orientation_axes, view, screenshot
from mayavi.mlab import gcf as m_gcf
from mne.utils import verbose
from copy import deepcopy
from numpy.random import random
from numpy import unravel_index
import matplotlib.pyplot as plt
from time import sleep
from matplotlib.pyplot import close
import numpy as np

#%%=========================== read and plot surface =========================================== 
@verbose
def read_plot_surface(surface_file, transfile=None, toplot=False, newfig=True, figname=None, 
                      representation='wireframe', opacity=0.1, axis=True, fname2writeAsBEMsurface=None, 
                      surfID=4, surfSigma=0.33, color=None, transtoapply=None, plot_axis_ori=True, 
                      verbose=None, return_scene=False):
    representation='wireframe' if representation in ['wireframe', 'w'] else 'surface'
    try:
        surface_mesh = mne.surface.read_surface(surface_file, read_metadata=False, return_dict=True, 
                                                file_format='auto', verbose=None)[2]
        surface_mesh.update(coord_frame=mne.io.constants.FIFF.FIFFV_COORD_MRI)
        surface_mesh['rr'] /= 1000. # convert to m
    except ValueError as valuerr: # this is reading dense and meadium skin surface
        print(valuerr)
        surface_mesh = mne.read_bem_surfaces(surface_file)[0]
  
    if not transfile is None:
        surface_mesh = mne.transform_surface_to(surface_mesh, 'head', 
                                                mne.transforms.read_trans(transfile), copy=True) 
    if transfile is None and not transtoapply is None:
        surface_mesh['rr'] = mne.transforms.apply_trans(transtoapply, deepcopy(surface_mesh['rr']))
    if not fname2writeAsBEMsurface is None:
        print('\nWriting: %s\n'%fname2writeAsBEMsurface)
        surface_mesh['id']    = surfID      # just to fake it as bem surface corresponding to given ID
        surface_mesh['sigma'] = surfSigma   # just to fake it
        mne.write_bem_surfaces(fname2writeAsBEMsurface,[surface_mesh], verbose=None)
    fig3d  = None
    if toplot:
        if newfig:
            fig3d = figure(figure=figname, bgcolor=(1,1,1), fgcolor=None, engine=None, size=(800, 700))
        if axis:
            points3d(0,0,0, mode='axes', scale_factor=surface_mesh['rr'].std(), color=(0,0,0))
        triangular_mesh(surface_mesh['rr'][:,0], surface_mesh['rr'][:,1], surface_mesh['rr'][:,2], 
                             surface_mesh['tris'],representation=representation, mode='sphere', 
                             opacity=opacity, scale_factor=.001, 
                             color=tuple(random([1,3]).tolist()[0]) if color is None else color)   
        if plot_axis_ori:
            orientation_axes()
    out = [surface_mesh, fig3d] if return_scene else surface_mesh
    return out

#%%=========================== Get snapshots of 3D scene =========================================== 
def get_snapped_figure_from_3d(fig_3d, figsize=(10,6), coordsys='RAS', tight=False, nsnap=6, titlepad=0,
                               top=1.0,bottom=0.0,left=0.0,right=1.0,hspace=-0.1,wspace=-0.1):    
    nrows, ncols = [1, 4] if nsnap==4 else [1, 3] if nsnap==3 else [2, 3]
    fig_2d, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if coordsys in ['RAS', 'ras','head']:
        if nsnap==3:
            azm, elev = [0, 90, 0], [-90, -90, 360]
            view_names= ['sagittal','coronal','axial']
    row_idxs, col_idxs = unravel_index(range(nsnap), (nrows, ncols))
    if fig_3d is None:  fig_3d = m_gcf()
    for ii in range(nsnap):
        print(f'{view_names[ii].upper()}')
        view(azimuth=azm[ii],elevation=elev[ii], figure=fig_3d)
        sleep(.25)
        fig_3d.scene._lift()
        scrnshot = screenshot(antialiased=True, figure=fig_3d)
        if nrows==1:
            axes[ii].imshow(scrnshot)
            axes[ii].set_axis_off()
            axes[ii].set_title(view_names[ii].upper(), y=1.0, pad=titlepad)
        else:
            axes[row_idxs[ii], col_idxs[ii]].imshow(scrnshot)
            axes[row_idxs[ii], col_idxs[ii]].set_axis_off()
            axes[row_idxs[ii], col_idxs[ii]].set_title(view_names[ii].upper())
    plt.pause(.2)
    plt.subplots_adjust(top=top,bottom=bottom,left=left,right=right,hspace=hspace,wspace=wspace)
    plt.tight_layout() if tight else None
    view(azimuth=azm[0],elevation=elev[0]); sleep(.5)
    sleep(.5)
    return fig_2d               

#===========================   ================================================================== 
def myMlabPoint3d(vertices=None, toplot=True, newfig=True, scale_factor=.008, opacity=0.5, mode='sphere', 
                  color=None, figname=None, axis=True, plot_axis_ori=True, transfile=None):
    if not transfile is None:
        vertices = apply_trans(invert_transform(read_trans(transfile))['trans'], vertices)
    fig3d = None
    if toplot:
        if newfig:
            fig3d = figure(figure=figname, bgcolor=(1,1,1), fgcolor=None, engine=None, size=(1200, 900))
        else:
            fig3d = m_gcf()
        if axis and not vertices is None:
            points3d(0,0,0, mode='axes', scale_factor=vertices.std(), color=(0,0,0))
        if not vertices is None:
            if vertices.shape==(3,):
                vertices = vertices.reshape([1,3])
            points3d(vertices[:,0], vertices[:,1], vertices[:,2], scale_factor=scale_factor, opacity=opacity, mode=mode,
                          color=tuple(random([1,3]).tolist()[0]) if color==None else color)#(0,0.7,0))
        if plot_axis_ori:
            orientation_axes()
    return fig3d

#===========================   ================================================================== 
def myMlabTriagularMesh(vertices=None, tris=None, toplot=True, newfig=True, representation='wireframe', scale_factor=.008, 
                       opacity=0.5, mode='sphere', color=None, figname=None, unit='m', axis=False, transfile=None, bgcolor=(1,1,1), fig=None):
    representation='surface' if representation in ['s', 'surf', 'surface'] else 'wireframe'
    if not transfile is None:
        vertices = apply_trans(invert_transform(read_trans(transfile))['trans'], vertices)
    if toplot:
        if newfig:
            figure(figure=figname, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1200, 900))
        if axis:
            points3d(0,0,0, mode='axes', scale_factor=vertices.std() if unit=='m' else 100, color=(0,0,0))
        if not vertices is None:
            clr = tuple(random([1,3]).tolist()[0]) if color==None else color
            if tris is None:
                points3d(vertices[:,0], vertices[:,1], vertices[:,2], 
                                 scale_factor=scale_factor, opacity=opacity, mode=mode, color=clr)
            else:
                triangular_mesh(vertices[:,0], vertices[:,1], vertices[:,2], tris, representation=representation,
                                     scale_factor=scale_factor, opacity=opacity, mode=mode, color=clr)#(0,0.7,0))
        orientation_axes()
        
#%%
def tight_equate_axis_lim(ax, apply_on='xy'):
    ax = np.array(ax) if not isinstance(ax, np.ndarray) else ax
    for ii in range(len(apply_on)):
        ax_ = apply_on[ii]
        lim1, lim2 = [], []
        for ii in range(len(ax.flatten())):
            lim1.append( eval(f'ax.flatten()[ii].get_{ax_}lim()[0]'))
            lim2.append( eval(f'ax.flatten()[ii].get_{ax_}lim()[1]'))
        for ii in range(len(ax.flatten())):
            eval( f'ax.flatten()[ii].set_{ax_}lim( min(lim1), max(lim2))')

#%%
def get_plot_dig_sensors(inst, fig=None, newfig=True):
    chs = np.array([inst.info['chs'][ii]['loc'] for ii in range(len(inst.info['chs']))])
    dig = np.array([inst.info['dig'][ii]['r'] for ii in range(len(inst.info['dig']))])
    fig = myMlabPoint3d(apply_trans(inst.info['dev_head_t'], chs[:,:3]), scale_factor=.006, color=(0,0,1))
    myMlabPoint3d(dig, scale_factor=.01, newfig=False, color=(1,0,0))
    if dig.shape[0]>3:  myMlabPoint3d(dig, scale_factor=.003, newfig=False, color=(1,0,1))
    return fig      

#%% Take snaps of final smooth surface and put in report 
def get_put_snaps(repMNE, f_RepMNE, title, section, tags, caption, add2report=True,
                  save=True, fig3d=None, image_format='png', replace=False, verb=True, 
                  open_browser=False, figsize=(10, 6), coordsys='RAS', tight=False, nsnap=6, 
                  top=1.0, bottom=0.0, left=0.0, right=1.0, hspace=-0.1, wspace=-0.1, titlepad=0):
    if add2report:
        print('Adding 3D snaps to report...')
        if fig3d is None: fig3d = m_gcf()
        fig2d = get_snapped_figure_from_3d(fig3d, figsize=figsize, coordsys=coordsys, tight=tight, 
                                               nsnap=nsnap,  top=top, bottom=bottom, left=left, right=right, 
                                               hspace=hspace, wspace=wspace, titlepad=titlepad)
        repMNE.add_figure(fig2d, title=title, section=section, image_format=image_format, 
                          tags=tags, replace=replace, caption=caption)
        close(fig=fig2d); del fig2d, fig3d
        if save:
            repMNE.save(fname=f_RepMNE, open_browser=open_browser, 
                        overwrite=True, sort_content=False, verbose=verb)
            print(f'Snaps added to report: {f_RepMNE}')
            
#%% Plot to check surface and digitization points alignment with anatomy (MRI)
def plot_surf_anat_alignment_v2(wdata, Torig, destCtrl, warpedSurf, title=None, nslice=8, tol=2, side_leave='25%', 
                                lw=1.5, titlecolor=(.8,.9,.2), titlefsize=18, cmap='gist_gray_r', zoom_in='10%',
                                viewmodes = ['yz', 'xy', 'xz']):
    zoom_in = -int(zoom_in[:-1])/100
    side_leave = int(side_leave.replace('%', ''))/100
    surf_rr_vox = apply_trans(np.linalg.inv(Torig), deepcopy(warpedSurf['rr'])*1000)
    destCtrl_vox  = apply_trans(np.linalg.inv(Torig), deepcopy(destCtrl)*1000)
    slice_idx =   np.linspace(int(wdata.shape[0]*side_leave), wdata.shape[0] - int(wdata.shape[0]*side_leave), nslice, dtype=int)
    fig = plt.figure(num=title)
    plt.suptitle(str(title))
    nrow, ncol, nplot = len(viewmodes)*2, nslice//2, 0
    for iview in range(len(viewmodes)):
        viewmode = viewmodes[iview]
        for jj in range(nslice):
            nplot += 1
            fig.add_subplot(nrow, ncol, nplot)
            if viewmode=='yz':
                plt.imshow(wdata[slice_idx[jj],:,:], cmap=cmap)
                plt.tricontour(surf_rr_vox[:, 2], surf_rr_vox[:, 1], warpedSurf['tris'], surf_rr_vox[:, 0], 
                                       levels=[slice_idx[jj]],  colors='r', linewidths=lw,   zorder=1)
                destCtrl_vox2 = np.empty((0,2))
                for ii in range(destCtrl_vox.shape[0]):
                    tol = tol # slice
                    if destCtrl_vox[ii,0] < slice_idx[jj]+tol and destCtrl_vox[ii,0] > slice_idx[jj]-tol:
                        destCtrl_vox2 = np.vstack((destCtrl_vox2, destCtrl_vox[ii,1:]))
                plt.scatter(destCtrl_vox2[:, 1], destCtrl_vox2[:, 0], color=(1,.5,0), marker='o', edgecolors='k')
                plt.text(wdata[slice_idx[jj],:,:].shape[0]/2, wdata[slice_idx[jj],:,:].shape[1]/2, str(slice_idx[jj]), 
                         **dict(ha='center', va='center', color=titlecolor, fontsize=titlefsize))
                plt.margins(zoom_in)
                plt.box(on=False);   plt.xticks([]);    plt.yticks([])
            elif viewmode=='xy':
                plt.imshow(wdata[:,slice_idx[jj],:], cmap=cmap)
                plt.tricontour(surf_rr_vox[:, 2], surf_rr_vox[:, 0], warpedSurf['tris'], surf_rr_vox[:, 1], 
                                       levels=[slice_idx[jj]],  colors='r', linewidths=lw,   zorder=1)
                destCtrl_vox2 = np.empty((0,2))
                for ii in range(destCtrl_vox.shape[0]):
                    tol = tol # slice
                    if destCtrl_vox[ii,1] < slice_idx[jj]+tol and destCtrl_vox[ii,1] > slice_idx[jj]-tol:
                        destCtrl_vox2 = np.vstack((destCtrl_vox2, destCtrl_vox[ii,[0,2]]))
                plt.scatter(destCtrl_vox2[:, 1], destCtrl_vox2[:, 0], color=(1,.5,0), marker='o', edgecolors='k')
                plt.text(wdata[slice_idx[jj],:,:].shape[0]/2, wdata[slice_idx[jj],:,:].shape[1]/2, str(slice_idx[jj]), 
                         **dict(ha='center', va='center', color=titlecolor, fontsize=titlefsize))
                plt.margins(zoom_in) # 10% zoom in
                plt.box(on=False);   plt.xticks([]);    plt.yticks([])
            elif viewmode=='xz':
                plt.imshow(wdata[:,:,slice_idx[jj]].T, cmap=cmap)
                plt.tricontour(surf_rr_vox[:, 0], surf_rr_vox[:, 1], warpedSurf['tris'], surf_rr_vox[:, 2], 
                                       levels=[slice_idx[jj]],  colors='r', linewidths=lw,   zorder=1)
                destCtrl_vox2 = np.empty((0,2))
                for ii in range(destCtrl_vox.shape[0]):
                    tol = tol # slice
                    if destCtrl_vox[ii,2] < slice_idx[jj]+tol and destCtrl_vox[ii,2] > slice_idx[jj]-tol:
                        destCtrl_vox2 = np.vstack((destCtrl_vox2, destCtrl_vox[ii,[0,1]]))
                plt.scatter(destCtrl_vox2[:, 0], destCtrl_vox2[:, 1], color=(1,.5,0), marker='o', edgecolors='k')
                plt.text(wdata[slice_idx[jj],:,:].shape[0]/2, wdata[slice_idx[jj],:,:].shape[1]/2, str(slice_idx[jj]), 
                         **dict(ha='center', va='center', color=titlecolor, fontsize=titlefsize))
                plt.margins(zoom_in) # 10% zoom in
                plt.box(on=False);   plt.xticks([]);    plt.yticks([])
    return fig