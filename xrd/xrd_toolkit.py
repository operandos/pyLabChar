# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:46:44 2024

@author: James Nohl
"""

import matplotlib.pyplot as plt
import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import pymatgen
import zipfile as zf
from xml.etree import ElementTree

def get_ras_dat(ras_file):
    fp = open(ras_file,'r',encoding='latin-1')
    lines = []
    cond = True
    while cond:
        line = fp.readline()[:-1]
        lines.append(line)
        if line == r'*RAS_INT_END':
            cond = False
            break
    
    dat_x=[]
    dat_y=[]
    dat_i=[]
    r=len(lines)
    for i,l in enumerate(lines):
        if l==r"*RAS_INT_START":
            r=i
        if l==r"*RAS_INT_END":
            break
        if i > r:
            dat_x.append(float(l.split()[0]))
            dat_y.append(float(l.split()[1]))
            dat_i.append(i)
    
    dat_x=np.array(dat_x)
    dat_y=np.array(dat_y)
    dat_i=np.array(dat_i)
    return lines, dat_i, dat_x, dat_y

def get_bruker_dat(brml_file):
    dat = zf.ZipFile(brml_file, mode = 'r')
    data = dat.open(r'Experiment0/RawData0.xml')
    data_tree = ElementTree.parse(data)
    points = data_tree.findall('.//Datum')
    x = np.array([float(p.text.split(',')[2]) for p in points])
    y = np.array([int(p.text.split(',')[4]) for p in points])
    return x, y

def plot_xrd_file(file):
    if '.ras' in file:
        lines,i,x,y = get_ras_dat(file)
    elif '.brml' in file:
        x,y = get_bruker_dat(file)
    plt.scatter(x, y, marker='+')
    plt.show()