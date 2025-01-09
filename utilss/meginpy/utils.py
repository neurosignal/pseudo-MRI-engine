#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 14:56:36 2022
@author: Amit Jaiswal, Megin Oy, Espoo, Finland  
        <amit.jaiswal@megin.fi> <amit.jaiswal@aalto.fi>
USAGE: meginpy.utils: This is part of MEGINPY platform employing utility tools.

"""
from time import time
from os.path import exists, dirname
from os import remove, symlink
import numpy as np
from matplotlib.pyplot import (figure, violinplot, xticks, 
                               ylabel, scatter, legend)

def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time()
    
def toc():
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set") 
        
def dirname2(fname):
    return dirname(fname) + '/'

def make_surfaces_soft_links(subjects_dir, subject): # make the soft links of surfaces, if don't exist already ....
    for bemsurf in ['brain','inner_skull','outer_skull','outer_skin']:  
        bemsurf_link = '%s/%s/bem/%s.surf'%(subjects_dir, subject, bemsurf)
        bemsurf_file = '%s/%s/bem/watershed/%s_%s_surface'%(subjects_dir, subject, subject, bemsurf)
        if not exists(bemsurf_link):
            try:
                symlink(bemsurf_file, bemsurf_link)
            except FileExistsError as feerr:
                print(feerr)
                remove(bemsurf_link)
                symlink(bemsurf_file, bemsurf_link)
        else:
            try:
                cprint(bemsurf_link + ' already exits...', fclr=11)
            except NameError as nmerr:
                print(nmerr)
                print(bemsurf_link + ' already exits...')
          
def make_surfaces_soft_links3(subjects_dir, subject, surfs_from='mri2surf'):
    for bemsurf in ["brain", "inner_skull", "outer_skull", "outer_skin"]:
        bemsurf_link = f"{subjects_dir}/{subject}/bem/{bemsurf}.surf"
        if bemsurf in  ["brain", "inner_skull"]:
            if surfs_from=='mri2surf':
                bemsurf_file = f"{subjects_dir}/{subject}/bem/watershed/synthstriped_{bemsurf}.surf"
            else:
                bemsurf_file = f"{subjects_dir}/{subject}/bem/watershed/{subject}_{bemsurf}_surface"
        else:
            bemsurf_file = f"{subjects_dir}/{subject}/bem/watershed/{subject}_{bemsurf}_surface"
        try:
            symlink(bemsurf_file, bemsurf_link)
        except FileExistsError as feerr:
            print(feerr)
            print(bemsurf_link + " already exits. \nRemoving the old and writing new link.")
            remove(bemsurf_link)
            symlink(bemsurf_file, bemsurf_link)
    
def get_all_to_all_point_dist(vertices1, vertices2, toplot=True, unit='mm', unit_multiplier=1000, return_fig=False):
    if vertices1.shape!=vertices2.shape:
        # raise Exception('vertices1 and vertices2 are not of the same size.')
        print('\n\nNOTE: vertices1 and vertices2 are not of the same size.')
        print('len(vertices1) = %d \nlen(vertices2) = %d'%(vertices1.shape[0], vertices2.shape[0]))
        vertices11 = vertices1 if vertices1.shape[0] < vertices2.shape[0] else vertices2
        vertices22 = vertices1 if vertices1.shape[0] > vertices2.shape[0] else vertices2
        one2one_dists = []
        for ii in range(vertices11.shape[0]):
            one2one_dists.append( np.sqrt(np.sum(np.square(vertices11[ii,:] - vertices22[ii,:]))) )
        one2one_dists = np.array(one2one_dists)*unit_multiplier
    else:
        one2one_dists = []
        for ii in range(vertices1.shape[0]):
            one2one_dists.append( np.sqrt(np.sum(np.square(vertices1[ii,:] - vertices2[ii,:]))) )
        one2one_dists = np.array(one2one_dists)*unit_multiplier

    if toplot:
        fig = figure() 
        vio = violinplot(one2one_dists, showmeans=True)
        xticks([])
        vio['cmeans'].set_color(np.array([[1,0,0, 0.99]]))
        vio['cmeans'].set_linewidth(0)
        ylabel('Distance between two point sets (%s)'%unit)
        scatter(1, one2one_dists.mean(), s=50, c='crimson', marker='o', zorder=3, 
                    label='mean = %.f %s'%(one2one_dists.mean(), unit))
        legend()
    if return_fig:
        return one2one_dists, fig
    else:
        return one2one_dists

def sub2ind_matlab(siz, v1, v2, v3=None):
    lensiz = len(siz)
    if lensiz < 2:
        raise 'MATLAB:sub2ind:InvalidSize'
    numOfIndInput = 2 if v3 is None else 3
    if np.sum(np.where(np.min(v1.flatten()) < 1)[0]) or np.sum(np.where(np.max(v1.flatten()) > siz[0])[0]):
        raise 'MATLAB:sub2ind:IndexOutOfRange'
    ndx = np.double(v1)
    s = v1.shape[0]
    if numOfIndInput >= 2:
        if not s==v2.shape[0]:
            raise 'MATLAB:sub2ind:SubscriptVectorSize'
        if np.sum(np.where(np.min(v2.flatten()) < 1)[0]) or np.sum(np.where(np.max(v2.flatten()) > siz[1])[0]):
            raise 'MATLAB:sub2ind:IndexOutOfRange'
        ndx = ndx + (np.double(v2) - 1)*siz[0]
    if numOfIndInput ==3:
        k = np.cumprod(siz)
        v = v3
        if not s==v.shape[0]:
            raise 'MATLAB:sub2ind:SubscriptVectorSize'
        if np.sum(np.where(np.min(v.flatten()) < 1)[0]) or np.sum(np.where(np.max(v.flatten()) > siz[2])[0]):
            raise 'MATLAB:sub2ind:IndexOutOfRange'
        ndx = ndx + (np.double(v)-1)*k[1]
    return ndx

