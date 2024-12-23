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
from mayavi.mlab import close as mclose
from mayavi.mlab import savefig as msavefig
from mayavi.mlab import quiver3d
from mne.utils import verbose
from copy import deepcopy
from numpy.random import random
from numpy import degrees, arctan2
from numpy import unravel_index, linspace, unique
import matplotlib.pyplot as plt
from time import sleep
from os import path
from mayavi.mlab import gcf as m_gcf
from matplotlib.pyplot import close
from nilearn.plotting import view_img

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
            # view_names= ['left','posterior','superior']
            view_names= ['sagittal','coronal','axial'] # changed in Dec 2024
        elif nsnap==4:
            azm, elev = [0, 0,  90, 0], [-90, 90, -90, 360]
            view_names= ['left','right','posterior','superior']
        else:
            azm, elev = [0, 0, 90, 0, 0, 90], [-90, 90, 90, 360, 180, -90]
            view_names= ['left','right','anterior','superior','inferior', 'posterior']
    elif coordsys in ['VOX', 'vox']:
        if nsnap==3:
            azm, elev = [-180, 180, 90], [-270, 180, -90]
            # view_names= ['left','posterior','superior']
            view_names= ['sagittal','coronal','axial'] # changed in Dec 2024
        elif nsnap==4:
            azm, elev = [-180, 180, 180, 90], [-270, 270, 180, -90]
            view_names= ['left','right','posterior','superior']
        else:
            azm, elev = [-180, 180, 180, 90, 90, 180], [-270, 270, 360, -90, 90, 180]
            view_names= ['left','right','anterior','superior','inferior', 'posterior']
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
        
#%%===========================  plot sensor and source time series ============================ 
def plot_sensor_source_time_series(raw, stc_raw, 
                                   n_verts = '10%', figwidth=15, figheight=None, show=False):
    print(f"Plotting time-series from {n_verts} of {raw._data.shape[0]} "
          f"sensos and {stc_raw.data.shape[0]} sources.")
    ch_types = unique(raw.info.get_channel_types()).tolist()
    fig, ax = plt.subplots(len(ch_types)+1, 1, sharex=True, 
                **dict(figsize=(figwidth, 3*(len(ch_types)+1) \
                                if figheight is None else figheight)))
    for ii, ch in enumerate(ch_types):
        raw_ch = raw.copy().pick(picks=ch)
        chidx= linspace(0, raw_ch._data.shape[0]-1, 
                           round(raw_ch._data.shape[0]*float(n_verts[:-1])/100), dtype=int)
        ax[ii].plot(raw_ch.times, raw_ch._data[chidx, :].T.mean(1), linewidth=.25)
        ax[ii].fill_between(raw_ch.times, raw_ch._data[chidx, :].T.mean(1) + \
                            raw_ch._data[chidx, :].T.std(1),
                    raw_ch._data[chidx, :].T.mean(1) - \
                        raw_ch._data[chidx, :].T.std(1), color='k', alpha=.2)
        ax[ii].legend([f'Sensor data ({ch.upper()}s)'], loc=1); 
        del raw_ch
    chidx= linspace(0, stc_raw._data.shape[0]-1, 
                       round(stc_raw._data.shape[0]*float(n_verts[:-1])/100), dtype=int)
    ax[-1].plot(stc_raw.times, stc_raw._data[chidx, :].T.mean(1), linewidth=.25)
    ax[-1].fill_between(stc_raw.times, stc_raw._data[chidx, :].T.mean(1) + \
                        stc_raw._data[chidx, :].T.std(1),
                       stc_raw._data[chidx, :].T.mean(1) - \
                           stc_raw._data[chidx, :].T.std(1), color='k', alpha=.2)
    ax[-1].legend(['Source data'], loc=1)
    ax[-1].set_xlabel('Time (s)')
    fig.tight_layout();     fig.subplots_adjust(hspace=0.001)
    plt.show() if show else None
    return fig

#%%=========================== x x x x x x x x x x =========================================== 
def label_line(line, label, x, y, color='0.5', size=12):
    """Add a label to a line, at the proper angle.

    Arguments
    ---------
    line : matplotlib.lines.Line2D object,
    label : str
    x : float
        x-position to place center of text (in data coordinated
    y : float
        y-position to place center of text (in data coordinates)
    color : str
    size : float
    """
    xdata, ydata = line.get_data()
    x1 = xdata[0]
    x2 = xdata[-1]
    y1 = ydata[0]
    y2 = ydata[-1]

    ax = line.get_axes()
    text = ax.annotate(label, xy=(x, y), xytext=(-10, 0),
                       textcoords='offset points',
                       size=size, color=color,
                       horizontalalignment='left',
                       verticalalignment='bottom')

    sp1 = ax.transData.transform_point((x1, y1))
    sp2 = ax.transData.transform_point((x2, y2))

    rise = (sp2[1] - sp1[1])
    run = (sp2[0] - sp1[0])

    slope_degrees = degrees(arctan2(rise, run))
    text.set_rotation(slope_degrees)
    return text

#%% 
# Taken from https://stackoverflow.com/questions/18780198/how-to-rotate-matplotlib-annotation-to-match-a-line
# Example: 
# kk = np.array([[3.08,	7.17], [2.99,	6.25], [4.1, 8.8]])
# kkk = np.array([[2022,	2030], [2022,	2030], [2022,	2032]])
# cagr = [12.8, 8.9, 8.9]
# kk_lab = ['Maximize market research','Verified market research','Market research future']

# plt.figure()
# for ii in range(3):
#     line, = plt.plot(kkk[ii], kk[ii], '-o', linewidth=3, label=kk_lab[ii]) 
#     line_annotate(f'CAGR = {cagr[ii]}', line, kkk[ii].sum()/2)
# plt.xticks([2022, 2030], [2022, 2030])
# plt.xlim(2021.8, 2032.2); plt.xlabel('Year', labelpad=-2), plt.ylabel('Market size (billian USD)'); 
# plt.legend(); plt.grid(alpha=.2)

import numpy as np
from matplotlib.text import Annotation
from matplotlib.transforms import Affine2D


class LineAnnotation(Annotation):
    """A sloped annotation to *line* at position *x* with *text*
    Optionally an arrow pointing from the text to the graph at *x* can be drawn.
    Usage
    -----
    fig, ax = subplots()
    x = linspace(0, 2*pi)
    line, = ax.plot(x, sin(x))
    ax.add_artist(LineAnnotation("text", line, 1.5))
    """

    def __init__(
        self, text, line, x, xytext=(0, 5), textcoords="offset points", **kwargs
    ):
        """Annotate the point at *x* of the graph *line* with text *text*.

        By default, the text is displayed with the same rotation as the slope of the
        graph at a relative position *xytext* above it (perpendicularly above).

        An arrow pointing from the text to the annotated point *xy* can
        be added by defining *arrowprops*.

        Parameters
        ----------
        text : str
            The text of the annotation.
        line : Line2D
            Matplotlib line object to annotate
        x : float
            The point *x* to annotate. y is calculated from the points on the line.
        xytext : (float, float), default: (0, 5)
            The position *(x, y)* relative to the point *x* on the *line* to place the
            text at. The coordinate system is determined by *textcoords*.
        **kwargs
            Additional keyword arguments are passed on to `Annotation`.

        See also
        --------
        `Annotation`
        `line_annotate`
        """
        assert textcoords.startswith(
            "offset "
        ), "*textcoords* must be 'offset points' or 'offset pixels'"

        self.line = line
        self.xytext = xytext

        # Determine points of line immediately to the left and right of x
        xs, ys = line.get_data()

        def neighbours(x, xs, ys, try_invert=True):
            inds, = np.where((xs <= x)[:-1] & (xs > x)[1:])
            if len(inds) == 0:
                assert try_invert, "line must cross x"
                return neighbours(x, xs[::-1], ys[::-1], try_invert=False)

            i = inds[0]
            return np.asarray([(xs[i], ys[i]), (xs[i+1], ys[i+1])])
        
        self.neighbours = n1, n2 = neighbours(x, xs, ys)
        
        # Calculate y by interpolating neighbouring points
        y = n1[1] + ((x - n1[0]) * (n2[1] - n1[1]) / (n2[0] - n1[0]))

        kwargs = {
            "horizontalalignment": "center",
            "rotation_mode": "anchor",
            **kwargs,
        }
        super().__init__(text, (x, y), xytext=xytext, textcoords=textcoords, **kwargs)

    def get_rotation(self):
        """Determines angle of the slope of the neighbours in display coordinate system
        """
        transData = self.line.get_transform()
        dx, dy = np.diff(transData.transform(self.neighbours), axis=0).squeeze()
        return np.rad2deg(np.arctan2(dy, dx))

    def update_positions(self, renderer):
        """Updates relative position of annotation text
        Note
        ----
        Called during annotation `draw` call
        """
        xytext = Affine2D().rotate_deg(self.get_rotation()).transform(self.xytext)
        self.set_position(xytext)
        super().update_positions(renderer)


def line_annotate(text, line, x, *args, **kwargs):
    """Add a sloped annotation to *line* at position *x* with *text*

    Optionally an arrow pointing from the text to the graph at *x* can be drawn.

    Usage
    -----
    x = linspace(0, 2*pi)
    line, = ax.plot(x, sin(x))
    line_annotate("sin(x)", line, 1.5)

    See also
    --------
    `LineAnnotation`
    `plt.annotate`
    """
    ax = line.axes
    a = LineAnnotation(text, line, x, *args, **kwargs)
    if "clip_on" in kwargs:
        a.set_clip_path(ax.patch)
    ax.add_artist(a)
    return a



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

#%%=========================== Get snapshots of 3D scene =========================================== 
def get_snapped_figure_from_3d(fig_3d, figsize=(10,6), coordsys='RAS', tight=False, nsnap=6, titlepad=0,
                               top=1.0,bottom=0.0,left=0.0,right=1.0,hspace=-0.1,wspace=-0.1):    
    nrows, ncols = [1, 4] if nsnap==4 else [1, 3] if nsnap==3 else [2, 3]
    fig_2d, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if coordsys in ['RAS', 'ras','head']:
        if nsnap==3:
            azm, elev = [0, 90, 0], [-90, -90, 360]
            # view_names= ['left','posterior','superior']
            view_names= ['sagittal','coronal','axial'] # changed in Dec 2024
        elif nsnap==4:
            azm, elev = [0, 0,  90, 0], [-90, 90, -90, 360]
            view_names= ['left','right','posterior','superior']
        else:
            azm, elev = [0, 0, 90, 0, 0, 90], [-90, 90, 90, 360, 180, -90]
            view_names= ['left','right','anterior','superior','inferior', 'posterior']
    elif coordsys in ['VOX', 'vox']:
        if nsnap==3:
            azm, elev = [-180, 180, 90], [-270, 180, -90]
            # view_names= ['left','posterior','superior']
            view_names= ['sagittal','coronal','axial'] # changed in Dec 2024
        elif nsnap==4:
            azm, elev = [-180, 180, 180, 90], [-270, 270, 180, -90]
            view_names= ['left','right','posterior','superior']
        else:
            azm, elev = [-180, 180, 180, 90, 90, 180], [-270, 270, 360, -90, 90, 180]
            view_names= ['left','right','anterior','superior','inferior', 'posterior']
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