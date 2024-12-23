"""
Created on Wed Jan 17 14:56:36 2024

@author: Amit Jaiswal, Megin Oy, Espoo, Finland  
        <amit.jaiswal@megin.fi> <amit.jaiswal@aalto.fi>
USAGE:
"""
# fmt: off
from time import time
# import os
try:
    from collections import MutableMapping
except Exception:
    from collections.abc import MutableMapping
from paramiko import SSHClient, AutoAddPolicy, Transport, SFTPClient
# from os import path, chdir
# import shutil
import subprocess
# from paramiko import SSHClient
from time import sleep
from mne.utils import get_subjects_dir, verbose
from mne.transforms import read_ras_mni_t, combine_transforms, invert_transform, apply_trans
try:
    from mne.source_space import _read_mri_info
except ImportError as ierr:
    from mne._freesurfer import _read_mri_info
import os.path as op
from os.path import exists, dirname, join, isfile, isdir
from colored import fg, bg, attr
from IPython.display import display
from os import remove, mkdir, makedirs, cpu_count, symlink, listdir, walk
from os.path import split, splitext, exists, join, dirname, isfile, islink, getsize
import numpy as np
from copy import deepcopy
from matplotlib.pyplot import figure,  violinplot,   xticks, ylabel, scatter, legend, imshow, clim, box
# import paramiko
from re import search, findall
import sympy
from numpy import arctan2, sqrt
import numexpr as ne
from mne.surface import _DistanceQuery
from mne import read_trans
from shutil import copyfile
from scipy.spatial import distance
import mne
# from .pseudoMRI import find_closest_node_dist
"""
Utils loaded from meginpy
"""

def tic():
    # Homemade version of matlab tic and toc functions
    global startTime_for_tictoc
    startTime_for_tictoc = time()
def toc():
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set") 
        
        
def all_files_exists(all_fls):
    cnt=1;
    for fl in all_fls:
        if not exists(fl):
            print('\n%s does not exist.\n'%fl)
            cnt *=0
    return bool(cnt)

#===========================  ================================================================== 
def run_using_ssh(command, username="amit", password="XXXX", host="xxxx.xxxx.local", port=22, todisp=False):
    """
    Run unix command on another system using ssh. The function was written for running sim_raw on palmu system.
    Parameters
    ----------
    command : string, for example: "mkdir ~/Documents/kkkkkkk"
        DESCRIPTION. The default is sim_cmd.
    username : TYPE, optional
        DESCRIPTION. The default is "amit".
    password : TYPE, optional
        DESCRIPTION. The default is "XXXX".
    host : hostname, for example:  "pamlu.megan.local"
        DESCRIPTION. The default is "xxxx.xxxx.local".
    port : TYPE, optional
        DESCRIPTION. The default is 22.
    Returns
    -------
    None.
    """
    # pip install pycrypto, paramiko
    print('\nRunning command %s on %s\n'%(command, host))
    ssh = SSHClient()
    ssh.set_missing_host_key_policy(AutoAddPolicy())
    ssh.connect(host, port, username, password)
    stdin, stdout, stderr = ssh.exec_command(command)
    lines = stdout.readlines()
    if todisp:
        display(lines)
        
#===========================  ================================================================== 
def run_via_ssh(command, hostname="xxxxx.xxxxx.local", port=22, username='xxxx', password='xxxxxxxx', todisplay=True):
    ssh = SSHClient()
    ssh.set_missing_host_key_policy(AutoAddPolicy())
    ssh.connect(hostname=hostname, port=port, username=username, password=password)
    stdin, stdout, stderr = ssh.exec_command(command)
    lines = stdout.readlines()
    if todisplay:
        return display(lines)
        
#===========================  ================================================================== 
def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def dirname2(fname):
    return dirname(fname) + '/'

        
#===========================  ================================================================== 
class MySFTPClient(SFTPClient):
    def put_dir(self, source, target):
        ''' Uploads the contents of the source directory to the target path. The
            target directory needs to exists. All subdirectories in source are 
            created under target.
        '''
        for item in listdir(source):
            if isfile(join(source, item)):
                self.put(join(source, item), '%s/%s' % (target, item))
            else:
                if not islink(join(source, item)): # added to not sending link
                    print('Sending: ' + join(source, item))
                    self.mkdir('%s/%s' % (target, item), ignore_existing=True)
                    self.put_dir(join(source, item), '%s/%s' % (target, item))

    def mkdir(self, path, mode=511, ignore_existing=False):
        ''' Augments mkdir by adding an option to not fail if the folder exists  '''
        try:
            super(MySFTPClient, self).mkdir(path, mode)
        except IOError:
            if ignore_existing:
                pass
            else:
                raise
                     
#===========================  ==================================================================    
def n_all_files(APP_FOLDER='.'):
    totalDir, totalFiles = 0,0
    for base, dirs, files in walk(APP_FOLDER):
        # print('Searching in : ',base)
        for directories in dirs:
            totalDir += 1
        for Files in files:
            totalFiles += 1
    return totalFiles
               
def get_dir_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in walk(start_path):
        for f in filenames:
            fp = join(dirpath, f)
            # skip if it is symbolic link
            if not islink(fp):
                total_size += getsize(fp)
    return total_size

        
#===========================  ================================================================== 
#% % For sending folders to triton for processing with vestal on tritol cluster. 
def transfer_dirs_amor_palmu(bucket_remote, bucket_local, hostname, username, password, port=22):
    # bucket_remote      = '/m/nbe/scratch/tbi-meg/amit-work/Simdataplaceholder/'
    # bucket_local         = '/net/qnap/data/rd/tbi_test/SimuWorks/ANALYSIS/3142/mybucket/'
    """
    To send whole directory to triton when required.
    Parameters
    ----------
    bucket_remote : TYPE
        DESCRIPTION.
    bucket_local : TYPE
        DESCRIPTION.
    hostname : TYPE
        DESCRIPTION.
    username : TYPE
        DESCRIPTION.
    password : TYPE
        DESCRIPTION.
    port : TYPE, optional
        DESCRIPTION. The default is 22.

    Returns
    -------
    None.

    """
    ssh = SSHClient()
    transport = Transport((hostname, port))
    transport.connect(username=username, password=password)
        
    makedirs(bucket_local, exist_ok=True)    
    ii=1
    while ii==1:
        # sftp = ssh.open_sftp()
        sftp = MySFTPClient.from_transport(transport)
        try:                        # try to read the 'data2transfer.csv' in client's bucket.
            sftp.get(bucket_remote+'data2transfer.csv',  bucket_local+'data2transfer.csv')
            fid = open(bucket_local + 'data2transfer.csv')
            all_in_file = fid.readlines()[0].split(',')
            whichdir     = all_in_file[0]
            whichdir_size = int(all_in_file[1])
            fid.close()
            if whichdir_size==1:   # transfer only when it's 1 in size cell of data2transfer.csv
                totalsize = get_dir_size(whichdir)
                totalfiles = n_all_files(whichdir)
                print('Trying to get %s \n(total %d files, total size %.2f MB)'%(whichdir, totalfiles, totalsize/1024/1024))
                fid = open(bucket_local + 'data2transfer.csv', 'w')
                fid.writelines('%s,' %(bucket_remote + whichdir.split('/')[-1]))
                fid.writelines('%d,' %totalsize)
                fid.writelines('%d' %totalfiles)
                fid.close()
                print('\nWait! sending (%.2f MB, %d files) file: \n%s to the client.\n'%(totalsize/1024/1024, totalfiles, whichdir))
                sftp.put(bucket_local+'data2transfer.csv', bucket_remote+'data2transfer.csv')
    
                sftp.mkdir(bucket_remote + whichdir.split('/')[-1], ignore_existing=True)
                sftp.put_dir(whichdir, bucket_remote + whichdir.split('/')[-1])
                sftp.close()
                print('\nFile sent:%s \n'%whichdir)
        except FileNotFoundError:   # print the message in the rest time when not sending anything.
            print('No demand by client, bucket is empty.')
        sleep(1)
        sftp.close()
    ssh.close()
    
        
#===========================  ================================================================== 
def read_talxfm(subject, subjects_dir=None, verbose=None):
    """Compute MRI-to-MNI transform from FreeSurfer talairach.xfm file.

    Parameters
    ----------
    %(subject)s
    %(subjects_dir)s
    %(verbose)s

    Returns
    -------
    mri_mni_t : instance of Transform
        The affine transformation from MRI to MNI space for the subject.
    """
    # Adapted from freesurfer m-files. Altered to deal with Norig
    # and Torig correctly
    subjects_dir = get_subjects_dir(subjects_dir)
    # Setup the RAS to MNI transform
    ras_mni_t = read_ras_mni_t(subject, subjects_dir)
    ras_mni_t['trans'][:3, 3] /= 1000.  # mm->m

    # We want to get from Freesurfer surface RAS ('mri') to MNI ('mni_tal').
    # This file only gives us RAS (non-zero origin) ('ras') to MNI ('mni_tal').
    # Se we need to get the ras->mri transform from the MRI headers.

    # To do this, we get Norig and Torig
    # (i.e. vox_ras_t and vox_mri_t, respectively)
    path = op.join(subjects_dir, subject, 'mri', 'orig.mgz')
    if not op.isfile(path):
        path = op.join(subjects_dir, subject, 'mri', 'T1.mgz')
    if not op.isfile(path):
        raise IOError('mri not found: %s' % path)
    _, _, mri_ras_t, _, _ = _read_mri_info(path)
    mri_mni_t = combine_transforms(mri_ras_t, ras_mni_t, 'mri', 'mni_tal')
    return mri_mni_t
        
#===========================  ================================================================== 
def cprint(this_str, prefix='', fclr=None, bgclr=None, attr_type='reset'):
    """
    Colored printing in IPython console
    ----------
    this_str : TYPE
        DESCRIPTION.
    prefix : TYPE, optional
        DESCRIPTION. The default is ''.
    fclr : TYPE, optional
        DESCRIPTION. The default is None.
    bgclr : TYPE, optional
        DESCRIPTION. The default is None.
    attr_type : TYPE, optional
        DESCRIPTION. The default is 'reset'.

    Returns
    -------
    Colored text.

    """
    if fclr==None and bgclr==None and prefix=='':
        print(this_str) # as default print of python
    else:
        if fclr==None and bgclr==None:
            clr = fg('default') + bg('default')
        elif fclr==None and not (bgclr==None):
            clr = fg('default') + bg(bgclr)
        elif not (fclr==None) and bgclr==None:
            clr = fg(fclr) + bg('default')
        elif not (fclr==None) and not (bgclr==None):
            clr = fg(fclr) + bg(bgclr)
        try:
            print(clr + prefix + this_str + attr(attr_type))
        except Exception as errrr:
            display(errrr)
            print('\n')
            print(this_str) # as default print of python

#===========================  ================================================================== 
def make_surfaces_soft_links(subjects_dir, subject): # make the soft links of surfaces, if don't exist already ....
    for bemsurf in ['brain','inner_skull','outer_skull','outer_skin']:  
        bemsurf_link = '%s/%s/bem/%s.surf'%(subjects_dir, subject, bemsurf)
        bemsurf_file = '%s/%s/bem/watershed/%s_%s_surface'%(subjects_dir, subject, subject, bemsurf)
        if not exists(bemsurf_link):
            try:
                symlink(bemsurf_file, bemsurf_link)
            except FileExistsError as feerr:
                remove(bemsurf_link)
                symlink(bemsurf_file, bemsurf_link)
        else:
            try:
                cprint(bemsurf_link + ' already exits...', fclr=11)
            except NameError as nmerr:
                print(bemsurf_link + ' already exits...')
        
#===========================  ==================================================================
def make_surfaces_soft_links2(subjects_dir, subject): 
    for bemsurf in ['brain','inner_skull','outer_skull','outer_skin']:  
        bemsurf_link = '%s/%s/bem/%s.surf'%(subjects_dir, subject, bemsurf)
        bemsurf_file = '%s/%s/bem/watershed/%s_%s_surface'%(subjects_dir, subject, subject, bemsurf)
        try:
            symlink(bemsurf_file, bemsurf_link)
        except FileExistsError as feerr:
            display(feerr)
            print(bemsurf_link + ' already exits. \nRemove the old and write new link.')
            remove(bemsurf_link)
            symlink(bemsurf_file, bemsurf_link)

# def make_surfaces_soft_links3(subjects_dir, subject):
#     for bemsurf in ["brain", "inner_skull", "outer_skull", "outer_skin"]:
#         bemsurf_link = "%s/%s/bem/%s.surf" % (subjects_dir, subject, bemsurf)
#         if bemsurf in  ["brain", "inner_skull"]:
#             bemsurf_file = "%s/%s/bem/watershed/synthstriped_%s.surf" % (
#                 subjects_dir,  subject, bemsurf, )
#         else:
#             bemsurf_file = "%s/%s/bem/watershed/%s_%s_surface" % (
#                 subjects_dir,  subject, subject,  bemsurf, )
#         try:
#             symlink(bemsurf_file, bemsurf_link)
#         except FileExistsError as feerr:
#             display(feerr)
#             print(bemsurf_link + " already exits. \nRemove the old and write new link.")
#             remove(bemsurf_link)
#             symlink(bemsurf_file, bemsurf_link)
            
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
            display(feerr)
            print(bemsurf_link + " already exits. \nRemoving the old and writing new link.")
            remove(bemsurf_link)
            symlink(bemsurf_file, bemsurf_link)
            
#===========================  ==================================================================
def copy_surfaces_to_bem_dir(subjects_dir, subject): 
    for bemsurf in ['brain','inner_skull','outer_skull','outer_skin']:  
        bemsurf_link = '%s/%s/bem/%s.surf'%(subjects_dir, subject, bemsurf)
        bemsurf_file = '%s/%s/bem/watershed/%s_%s_surface'%(subjects_dir, subject, subject, bemsurf)
        try:
            # symlink(bemsurf_file, bemsurf_link)
            copyfile(bemsurf_file, bemsurf_link)
        except FileExistsError as feerr:
            display(feerr)
            print(bemsurf_link + ' already exits. \nRemove the old and write new link.')
            remove(bemsurf_link)
            copyfile(bemsurf_file, bemsurf_link)                
#===========================  =========================================================================
def find_closet_match(test_str, list2check, str2skip=''):
    test_str = test_str.replace(str2skip, '')
    scores = {}
    for ii in list2check:
        ii = ii.replace(str2skip, '');    # print(ii)
        cnt = 0
        if len(test_str)<=len(ii):
            str1, str2 = test_str, ii
        else:
            str1, str2 = ii, test_str
        for jj in range(len(str1)):
            cnt += 1 if str1[jj]==str2[jj] else 0
        scores[ii] = cnt
    scores_values        = np.array(list(scores.values()))
    closest_match_idx    = np.argsort(scores_values, axis=0, kind='quicksort')[-1]
    closest_match_values = scores_values[closest_match_idx]
    closest_match        = np.array(list(scores.keys()))[closest_match_idx]
    print('\n%s\n%s'%(closest_match, closest_match_idx))
    return closest_match, closest_match_idx, closest_match_values

#========================================================== ==================================================
def remove_list_elements_with_patterns(patterns, my_list):
    patterns = [patterns] if not isinstance(patterns, list) else patterns
    new_list = deepcopy(my_list)
    rem_idx  = []
    for pattern in patterns:
        rem_idx.append([my_list.index(i) for i in my_list if pattern in i])
    rem_idx = sum(rem_idx, [])
    rem_idx = np.flip(np.sort(np.unique(rem_idx))).tolist()
    for rem_indxx in rem_idx:
        new_list.pop(rem_indxx)
    return new_list

#%%===========================================================================================================
def fileparts(fname):
    fdir    = dirname(fname)
    fdir2   = dirname(fname) + '/'
    dfname  = fname.split('/')[-1].split('.')[0]  
    dfname2 = fname.split('/')[-1]
    ext     = fname.split('/')[-1].split('.')[-1]
    ext2    = '.' + fname.split('/')[-1].split('.')[-1]
    return fdir, fdir2, dfname, dfname2, ext, ext2

#%%
def parent_dir(fnameORdirname):
    if not fnameORdirname.endswith('/'):
        par_dir = dirname2(fnameORdirname)
    else:
        while fnameORdirname.endswith('/'):
            fnameORdirname = fnameORdirname[:-1]
        par_dir = dirname2(fnameORdirname)
    return par_dir

#%% Get all to all point distance
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

#%% Compute all at once (very fast)
def get_closest_points_distance_for_all_vertices(vertices1, vertices2, toplot=True, unit='mm', unit_multiplier=1000):
    if vertices1.shape>=vertices2.shape:
        closest_pts_dist_all   = _DistanceQuery(vertices2).query(vertices1)[0]
    else:
        closest_pts_dist_all   = _DistanceQuery(vertices1).query(vertices2)[0]
    if toplot:
        figure() 
        vio = violinplot(deepcopy(closest_pts_dist_all)*unit_multiplier, showmeans=True)
        xticks([])
        vio['cmeans'].set_color(np.array([[1,0,0, 0.99]]))
        vio['cmeans'].set_linewidth(0)
        ylabel('Distance between two point sets (%s)'%unit)
        scatter(1, closest_pts_dist_all.mean(), s=50, c='crimson', marker='o', zorder=3, 
                    label='mean = %.2f %s'%(closest_pts_dist_all.mean()*unit_multiplier, unit))
        legend()
    return closest_pts_dist_all            
            
#%% Computing distance one by one and reducing the number of vertices by pushing the current to Inf 
def get_closest_points_distance_for_all_vertices_v2(vertices1, vertices2, toplot=True, unit='mm', unit_multiplier=1000):
    if vertices1.shape!=vertices2.shape:
        # raise Exception('vertices1 and vertices2 are not of the same size.')
        print('\n\nNOTE: vertices1 and vertices2 are not of the same size.')
        print('len(vertices1) = %d \nlen(vertices2) = %d'%(vertices1.shape[0], vertices2.shape[0]))
        vertices11 = deepcopy(vertices1) if vertices1.shape[0] < vertices2.shape[0] else deepcopy(vertices2)
        vertices22 = deepcopy(vertices1) if vertices1.shape[0] > vertices2.shape[0] else deepcopy(vertices2)
        print("Wait, it's computing distance for %d points...."%vertices11.shape[0])
        closest_pts_dist_all = []
        for ii in range(vertices11.shape[0]):
            pos, idx, distt  = find_closest_node_dist(vertices11[ii,:], vertices22)
            closest_pts_dist_all.append( distt )
            vertices22[idx,:] = np.array([1.e+20,1.e+20,1.e+20])
        closest_pts_dist_all = np.array(closest_pts_dist_all)
    else:
        vertices11 = deepcopy(vertices1)
        vertices22 = deepcopy(vertices2)        
        print("Wait, it's computing distance for %d points...."%vertices11.shape[0])
        closest_pts_dist_all = []
        for ii in range(vertices11.shape[0]):
            pos, idx, distt  = find_closest_node_dist(vertices11[ii,:], vertices22)
            closest_pts_dist_all.append( distt )
            vertices22[idx,:] = np.array([1.e+20,1.e+20,1.e+20])
        closest_pts_dist_all = np.array(closest_pts_dist_all)
    if toplot:
        figure() 
        vio = violinplot(closest_pts_dist_all*unit_multiplier, showmeans=True)
        xticks([])
        vio['cmeans'].set_color(np.array([[1,0,0, 0.99]]))
        vio['cmeans'].set_linewidth(0)
        ylabel('Distance between two point sets (%s)'%unit)
        scatter(1, closest_pts_dist_all.mean(), s=50, c='crimson', marker='o', zorder=3, 
                    label='mean = %.f %s'%(closest_pts_dist_all.mean(), unit))
        legend()
    return closest_pts_dist_all  
#%%

def find_sum(str1):
    return sum(map(int, findall('\d+', str1)))

#%%
def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

#%%
def appendSpherical_np(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,5] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew

#%%
def asSpherical(xyz):
    #takes list xyz (single coord)
    x       = xyz[0]
    y       = xyz[1]
    z       = xyz[2]
    r       =  np.sqrt(x*x + y*y + z*z)
    theta   =  sympy.acos(z/r)*180/ np.pi #to degrees
    phi     =  sympy.atan2(y,x)*180/ np.pi
    return [r,theta,phi]

#%%
def asCartesian(rthetaphi):
    #takes list rthetaphi (single coord)
    r       = rthetaphi[0]
    theta   = rthetaphi[1]* np.pi/180 # to radian
    phi     = rthetaphi[2]* np.pi/180
    x = r * sympy.sin( theta ) * sympy.cos( phi )
    y = r * sympy.sin( theta ) * sympy.sin( phi )
    z = r * sympy.cos( theta )
    return [x,y,z]

#%%
def cart2sph(x,y,z, ceval=ne.evaluate):
    """ x, y, z :  ndarray coordinates
        ceval: backend to use: 
              - eval :  pure Numpy
              - numexpr.evaluate:  Numexpr """
    azimuth = ceval('arctan2(y,x)')
    xy2 = ceval('x**2 + y**2')
    elevation = ceval('arctan2(z, sqrt(xy2))')
    r = eval('sqrt(xy2 + z**2)')
    return azimuth, elevation, r   

#%% find outliers and extremes percentage
def find_outliers_extreme_percentage(this_data, percentile=[25,75], whis_outliers=1.5, whis_extreme=3.0, res_round_by=2, toprint=True):
    Q1, Q3= np.percentile(this_data,percentile)
    IQR = Q3 - Q1
    if toprint:
        print('\nUsing..\tpercentile= %s, \twhis_outliers= %.1f, \twhis_extreme= %.1f\n'%(str(percentile), whis_outliers, whis_extreme))
    # for outliers....................
    lower_bound1 = Q1 -(whis_outliers * IQR) 
    upper_bound1 = Q3 +(whis_outliers * IQR)
    outlrs_indx= np.where(this_data>upper_bound1)[0]
    outlrs_percentage = (len(outlrs_indx)*100)/len(this_data)
    # if toprint:
    #     print('\nLower_bound1=%.4f, Upper_bound1=%.4f'%(lower_bound1, upper_bound1)) 
    #     print('Outliers indices=', outlrs_indx)
    #     print('Outliers percentage=%.2f'%outlrs_percentage)
    # for extremes..................
    lower_bound2 = Q1 -(whis_extreme * IQR) 
    upper_bound2 = Q3 +(whis_extreme * IQR)
    extremes_idx= np.where(this_data>upper_bound2)[0]
    extremes_percentage = (len(extremes_idx)*100)/len(this_data)
    if toprint:
        # print('\nLower_bound2=%.4f, Upper_bound2=%.4f'%(lower_bound2, upper_bound2)) 
        # print('Extreme indices=', extremes_idx)
        # print('Extreme percentage=%.2f\n'%extremes_percentage)
        print('\nOutliers indices & %% = [%s: %s], \tExtreme indices & %% = [%s: %s]' %(str(list(extremes_idx)), str(round(extremes_percentage,2)),
                                                                                        str(list(outlrs_indx)), str(round(outlrs_percentage,2))))
    return round(IQR, res_round_by), outlrs_indx, round(outlrs_percentage,res_round_by), extremes_idx, round(extremes_percentage,res_round_by)

#%%
def get_IQROutlExtPercnt(this_data, percentile=[25,75], whis_outliers=1.5, whis_extreme=3.0, res_round_by=4, toprint=True):
    Q1, Q3= np.percentile(this_data,percentile)
    IQR = Q3 - Q1
    if toprint:
        print('\nUsing..\tpercentile= %s, \twhis_outliers= %.1f, \twhis_extreme= %.1f\n'%(str(percentile), whis_outliers, whis_extreme))
    # for outliers....................
    lower_bound1 = Q1 -(whis_outliers * IQR) 
    upper_bound1 = Q3 +(whis_outliers * IQR)
    outlrs_indx= np.where(this_data>upper_bound1)[0]
    outlrs_percentage = (len(outlrs_indx)*100)/len(this_data)

    # for extremes..................
    lower_bound2 = Q1 -(whis_extreme * IQR) 
    upper_bound2 = Q3 +(whis_extreme * IQR)
    extremes_idx= np.where(this_data>upper_bound2)[0]
    extremes_percentage = (len(extremes_idx)*100)/len(this_data)
    if toprint: print(f'Out%%, Ext%% = {outlrs_percentage}, {extremes_percentage}')
    return round(IQR,res_round_by), round(outlrs_percentage,res_round_by), round(extremes_percentage,res_round_by)

#%% converted from matlab sub2ind
def sub2ind_matlab(siz, v1, v2, v3=None):
    # %SUB2IND Linear index from multiple subscripts.
    # %   SUB2IND is used to determine the equivalent single index
    # %   corresponding to a given set of subscript values.
    # %
    # %   IND = SUB2IND(SIZ,I,J) returns the linear index equivalent to the
    # %   row and column subscripts in the arrays I and J for a matrix of
    # %   size SIZ. 
    # %
    # %   IND = SUB2IND(SIZ,I1,I2,...,IN) returns the linear index
    # %   equivalent to the N subscripts in the arrays I1,I2,...,IN for an
    # %   array of size SIZ.
    # %
    # %   I1,I2,...,IN must have the same size, and IND will have the same size
    # %   as I1,I2,...,IN. For an array A, if IND = SUB2IND(SIZE(A),I1,...,IN)),
    # %   then A(IND(k))=A(I1(k),...,IN(k)) for all k.
    # %
    # %   Class support for inputs I,J: 
    # %      float: double, single
    # %      integer: uint8, int8, uint16, int16, uint32, int32, uint64, int64
    # %
    # %   See also IND2SUB.
    
    # %   Copyright 1984-2015 The MathWorks, Inc.
    
    # siz = siz
    lensiz = len(siz)
    if lensiz < 2:
        raise 'MATLAB:sub2ind:InvalidSize'
    
    numOfIndInput = 2 if v3 is None else 3
    # if lensiz < numOfIndInput: #FIX it
    #     # %Adjust for trailing singleton dimensions
    #     siz = [siz, np.ones((1,numOfIndInput-lensiz))];
    # elif lensiz > numOfIndInput:
    #     # %Adjust for linear indexing on last element
    #     siz = [siz(1:numOfIndInput-1), prod(siz(numOfIndInput:end))];
    
    if np.sum(np.where(np.min(v1.flatten()) < 1)[0]) or np.sum(np.where(np.max(v1.flatten()) > siz[0])[0]):
        # %Verify subscripts are within range
        raise 'MATLAB:sub2ind:IndexOutOfRange'
    
    ndx = np.double(v1)
    s = v1.shape[0]
    if numOfIndInput >= 2:
        if not s==v2.shape[0]:
            # %Verify sizes of subscripts
            raise 'MATLAB:sub2ind:SubscriptVectorSize'
        # if any(min(v2(:)) < 1) || any(max(v2(:)) > siz(2)):
        if np.sum(np.where(np.min(v2.flatten()) < 1)[0]) or np.sum(np.where(np.max(v2.flatten()) > siz[1])[0]):
            # %Verify subscripts are within range
            raise 'MATLAB:sub2ind:IndexOutOfRange'
        # %Compute linear indices
        ndx = ndx + (np.double(v2) - 1)*siz[0]
        
    if numOfIndInput ==3:
        # %Compute linear indices
        k = np.cumprod(siz)
        # for i in range(2,numOfIndInput):
        v = v3
        # % %Input checking
        if not s==v.shape[0]:
            # %Verify sizes of subscripts
            raise 'MATLAB:sub2ind:SubscriptVectorSize'
        # if (any(min(v(:)) < 1)) || (any(max(v(:)) > siz(i))):
        if np.sum(np.where(np.min(v.flatten()) < 1)[0]) or np.sum(np.where(np.max(v.flatten()) > siz[2])[0]):
            # %Verify subscripts are within range
            raise 'MATLAB:sub2ind:IndexOutOfRange'
        ndx = ndx + (np.double(v)-1)*k[1]
    return ndx

#%%
class Defineself(object):
    def __init__(self):
        self.name = 'amit'
        return self

#%%
def get_one2all_point_distances(vertices1, vertices2):
    vertices11 = deepcopy(vertices1)
    vertices22 = deepcopy(vertices2)
    
    if vertices11.shape>vertices22.shape:
        raise Exception("vertices1's shape must be 1x3.")
    else:
        closest_pts_dist_all = []
        for ii in range(vertices22.shape[0]):
            closest_pts_dist_all.append( np.linalg.norm(vertices11 - vertices22[ii,:].reshape(1,3)) )
    closest_pts_dist_all = np.array(closest_pts_dist_all)
    return closest_pts_dist_all 

#%% Get all to all distance matrix
def get_all_to_all_point_dist_matrix(vertices1, vertices2=None, toplot=False, unit_multiplier=1000):
    if not vertices2 is None and vertices1.shape[0] != vertices2.shape[0]:
        print('Both point sets are not of equal length.')
        raise IndexError
    
    if vertices2 is None:
        vertices2 = deepcopy(vertices1)
    one2one_dists = np.zeros((vertices1.shape[0], vertices1.shape[0]))
    for ii in range(vertices1.shape[0]):
        for jj in range(vertices1.shape[0]):
            one2one_dists[ii, jj] = np.sqrt(np.sum(np.square(vertices1[ii,:] - vertices2[jj,:])))
    one2one_dists *= unit_multiplier
    if toplot:
        figure() 
        imshow(np.tril(one2one_dists), cmap='bwr')
        clim(-one2one_dists.max(), one2one_dists.max())
        box(on=False)
    return one2one_dists
        
#%%
def get_trans_and_invert(transfile):
    T_ras2nm = invert_transform(read_trans(transfile))['trans']
    T_nm2ras = read_trans(transfile)['trans']
    return T_ras2nm, T_nm2ras

#%%
def closest_vert(node, nodes): # class to find closest points
    closest_index = distance.cdist([node], nodes).argmin()
    return closest_index, nodes[closest_index]

#%% Finding closest vertices for all given vertices
def get_closest_points_for_all(points, point_cloud):
    points = points.reshape(points.shape[0],3)
    closest_indices  = []
    closest_vertices = np.empty((0,3), float)
    for ii in range(points.shape[0]):
        closest_index = distance.cdist([points[ii,:]], point_cloud).argmin()
        closest_vertx = point_cloud[closest_index,:]
        closest_indices.append( closest_index )
        closest_vertices = np.vstack(( closest_vertices, closest_vertx ))
    return np.array(closest_indices), closest_vertices

#%% Finding closest vertices for all given vertices
def get_closest_points_for_all_v2(points, point_cloud_):
    point_cloud = deepcopy(point_cloud_)
    points = points.reshape(points.shape[0],3)
    closest_indices  = []
    closest_vertices = np.empty((0,3), float)
    for ii in range(points.shape[0]):
        closest_index = distance.cdist([points[ii,:]], point_cloud).argmin()
        closest_vertx = point_cloud[closest_index,:]
        point_cloud   = np.delete(point_cloud, closest_index, axis=0)
        closest_indices.append( closest_index )
        closest_vertices = np.vstack(( closest_vertices, closest_vertx ))
    return np.array(closest_indices), closest_vertices
#%%
def get_left_right_sensor_indices(info, include_mid_chs_in_left=False, include_mid_chs_in_right=False, out_coordsys='head'):
    indices, chans_location = {}, {}
    locs      = [info['chs'][ii]['loc'] for ii in range(len(info['chs']))]
    chan_locs = apply_trans(info['dev_head_t']['trans'], np.array(locs)[:,:3])
    indices['lh'] = np.where(chan_locs[:,0]<=0)[0] if include_mid_chs_in_left  else np.where(chan_locs[:,0]<0)[0]
    indices['rh'] = np.where(chan_locs[:,0]>=0)[0] if include_mid_chs_in_right else np.where(chan_locs[:,0]>0)[0]
    chans_location['lh'], chans_location['rh'] = chan_locs[indices['lh']], chan_locs[indices['rh']]
    if 'dev' in out_coordsys:
        chans_location['lh'] = apply_trans(invert_transform(info['dev_head_t'])['trans'], chans_location['lh'], move=True)
        chans_location['rh'] = apply_trans(invert_transform(info['dev_head_t'])['trans'], chans_location['rh'], move=True)
    return indices, chans_location 

#%%
def convert_mri_to_fiff(input_fname, out_niifile=None, out_fiffile=None, delete_nii=True, 
                        username="amit", password="XXXX", host="xxxx.xxxx.local", port=22, 
                        mri_convert='/home/amit2/0SOFTWAREs/freesurfer7/freesurfer/bin/mri_convert', todisp=False):
    if out_niifile is None:
        out_niifile =  input_fname.replace(input_fname[-4:], '.nii')
    if out_fiffile is None:
        out_fiffile =  input_fname.replace(input_fname[-4:], '.fif')
    
    command = '%s %s %s'%(mri_convert, input_fname, out_niifile)
    print(subprocess.check_output(command, shell=True))
    command2 = '/neuro/bin/util/nifti2fiff %s %s'%(out_niifile, out_fiffile)
    run_using_ssh(command2, username=username, password=password, host=host, port=port, todisp=True)
    if delete_nii:
        remove(out_niifile)
    return 
    
#%% Read points from .pts file
def read_3dpoints_from_pts(filename, toplot=True, points_mode='3d'):
    data1 = open(filename, 'r').read().split()
    data1= [float(i) for i in data1]
    data1 = np.array(data1)
    points_dim=3 if points_mode=='3d' else 2
    data1 = data1.reshape(data1.shape[0]//points_dim, points_dim)
    return data1

#%% Read points from .pts file
def write_3dpoints_from_pts(filename, points3d):
    fid = open(filename, 'w')
    for ii in range(points3d.shape[0]):
        fid.writelines('%s\t%s\t%s\n' %(points3d[ii,0], points3d[ii,1], points3d[ii,2]))
    fid.close()

#%% Get channel montage
def get_channel_montage(info, meg=True, eeg=False):
    picks = mne.pick_types(info, meg=meg, eeg=eeg)
    info2 = info.copy().pick_channels([info['ch_names'][ii] for ii in picks], ordered=False)
    montage = mne.channels.make_standard_montage('biosemi64')           
 

#%% Add CSV heading
def add_column_heading(dirname, fname, ii, col_heading=None, iicond=0, mode='a+', close=True):
    resfile, fid = '%s/%s'%(dirname, fname), None
    fid = open(resfile, mode)
    if ii==iicond and col_heading is not None:
        fid.writelines(col_heading)
        fid.writelines('\n')
    fid.close() if close else None
    print(resfile) if op.exists(resfile) else None
    return resfile, fid

#%% Change (increase or decrease) a value by given percentage         
def change_value(value, raise_by):
    assert isinstance(raise_by, str)
    assert raise_by.endswith('%')
    raise_by = raise_by[:-1]
    if raise_by.startswith('-'):
        raise_by = raise_by[1:]
        raised_value = value - value * float(raise_by)/100
    else:
        raised_value = value + value * float(raise_by)/100
    return raised_value
#%%
def min_mean_max(array, ax=0, n=1):
    return array.min(ax).round(n), array.mean(ax).round(n), array.max(ax).round(n)

#%%
def has_module(module_name):
    """Determine if nibabel is installed.

    Returns
    -------
    has : bool
        True if the user has nibabel.
    """
    try:
        exec('import %s'%module_name)
    except ImportError:
        return False
    else:
        return True
    
def find_closest_node_dist(node, nodes, multipy2dist=1): # function to find closest points
    closest_index = distance.cdist([node], nodes).argmin()
    closest_node = nodes[closest_index]
    diff = np.sqrt(np.sum(np.square(node-closest_node))) * multipy2dist # multipy2dist is to convert the unit m,cm,mm etc
    return closest_node, closest_index, diff

#%%
def closest_n_verts(node, nodesss, nverts): # to find n closest points
    nodes = deepcopy(nodesss)
    tol   = np.abs(nodesss).max()*10e10
    closest_indexs = []
    for ii in range(nverts):
        closest_index = distance.cdist([node], nodes).argmin()
        closest_indexs.append(closest_index)
        nodes[closest_index,:] = tol
    return closest_indexs, nodesss[closest_indexs,:]

#%%
# from scipy.spatial import distance
def find_closest_node(node, nodes): # function to find closest points
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index], closest_index

#%% remove directory name upto some level
def rmdirname(filename, level=0):
    dir_name = ''
    if level>0:
        dir_name = filename
        level = min(level, len(dir_name.split('/'))-2)
        level = len(dir_name.split('/'))-level
        # print(level)
        # while len(dir_name.split('/'))>2 and level>0:
        while level>1:
            # print(level)
            dir_name = dirname(dir_name)
            level -= 1
    return filename.replace(dir_name, '')

#%%
def color_norm_red_blue(x):
    normx = (x-np.min(x))/(np.max(x)-np.min(x))
    clrs = []
    for ii in normx:  clrs.append(((ii, 0, 1-ii)))
    return clrs
#%%
def normx(x):
    normx = (x-np.min(x))/(np.max(x)-np.min(x))
    return normx
#%%
def norm_low_high(x, lower, upper):
    normx = (x-np.min(x))/(np.max(x)-np.min(x))
    l_norm = [lower + (upper - lower) * k for k in normx]
    return l_norm
#%%
def get_same_length(data1, data2):
    idx1 = np.linspace(0, len(data1)-1, min(len(data1), len(data2)), dtype=int)
    idx2 = np.linspace(0, len(data2)-1, min(len(data1), len(data2)), dtype=int)
    data1_new = data1[idx1]
    data2_new = data2[idx2]
    return data1_new, data2_new

#%%
