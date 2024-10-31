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
import meginpy
from mne.utils import logger, verbose, warn # check_fname
from time import sleep
from os.path import join, exists, dirname, islink                 # split, splitext, isfile
from os import makedirs , getcwd,  chdir, symlink, remove, cpu_count       # remove, mkdir, 
from mne.io.constants import FIFF
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
