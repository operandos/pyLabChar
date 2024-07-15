# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 10:07:51 2024

@author: James Nohl
"""

import numpy as np
import os
import matplotlib.pyplot as plt

import peak_fit as pf

def load_txt(file:str, zapper=True, thresh=2000):
    dat = np.loadtxt(file)
    x, y = dat[:,0], dat[:,1]
    if zapper:
        x, y = zap(x,y, thresh=2000)
    return x, y

def average_dat(file_list:list):
    dat={}
    for file in file_list:
        x,y = np.loadtxt(file)

def zap(x,y, thresh=2000):
    y_absdiff = np.abs(np.diff(y, prepend=y[0]))
    dis = np.argwhere(y_absdiff>thresh)
    if len(dis)>0:
        rem = np.arange(np.min(dis),np.max(dis)+1,1)
        x = np.delete(x, rem)
        y = np.delete(y, rem)
    return x, y
        
def plot(file, plot=True, axes=True, title=None, y_label=False, norm=False, xlim=None):
    name = os.path.split(file)[1].split('.txt')[0]
    x,y = load_txt(file)
    x,y = zap(x,y)
    if xlim is not None:
        rows = np.where((x>=xlim[0])&(x<=xlim[1]))[0]
        x = x[rows]
        y = y[rows]
    if norm:
        y = y/np.max(y)
    elif type(norm) is float:
        y = y/np.max(y[np.where(x<norm)[0]])
    plt.plot(x,y, label=name)
    if axes:
        Raman_axes(y_label=y_label)
    if plot:
        plt.legend()
        plt.show()

def Raman_axes(y_label=False):
    plt.xlabel('Raman shift [$cm^{-1}$]')
    plt.ylabel('Raman intensity [arb.u.]')
    if y_label is False:
        plt.gca().yaxis.set_tick_params(labelleft=False)
        plt.gca().set_yticks([])

def fit_HOPG(file, title,
             init_fit=False):
    x,y = pfm.load_txt(file)
    mod, params = pfm.model({
     'G': {'centres': [1550], 'type': 'lorentz'}},
    c_bound =100, amps=[100], sigma = 40, a_bound = 200)
    res = pfm.fit(x,y,mod,params,xlim=[800,2000])
    if title is None:
        title = 'Raman plot with fit'
    pfm.plot_result(res, x, y, xlim=[800,2000], init_fit=init_fit, 
                    title=title,
                    colors=['yellowgreen'])
    pfm.print_result(res, x, y)

def fit_t_aC(file, 
             title = None,
             init_fit=False):
    x,y = pfm.load_txt(file)
    mod, params = pfm.model({'aC': {'centres': [1200], 'type': 'gauss'},
     'Dia': {'centres': [1330], 'type': 'lorentz'},
     'D': {'centres': [1380], 'type': 'gauss'},
     'G': {'centres': [1550], 'type': 'gauss'}},
    c_bound =50, amps=[3,50,5,8], sigma = 35, a_bound = 100)
    res = pfm.fit(x,y,mod,params,xlim=[800,2000])
    if title is None:
        title = 'Raman plot with fit'
    pfm.plot_result(res, x, y, xlim=[800,2000], init_fit=init_fit, title=title, colors=['darkorange','darkorange','khaki','powderblue','yellowgreen'])
    pfm.print_result(res, x, y)

def fit_LFP(file, 
            title = None,
            init_fit=False):
    x,y = pfm.load_txt(file)
    mod, params = pfm.model({
            'PO4': {'centres': [950],  'type': 'lorentz'},
            'aC':  {'centres': [1200], 'type': 'gauss'},
            'Dia': {'centres': [1330], 'type': 'lorentz'},
            'Dis': {'centres': [1380], 'type': 'gauss'},
            'G':   {'centres': [1550], 'type': 'lorentz'}
                         },
    c_bound =50, amps=[3,50,5,8], sigma = 35, a_bound = 100)
    res = pfm.fit(x,y,mod,params,xlim=[800,2000])
    if title is None:
        title = 'Raman plot with fit'
    pfm.plot_result(res, x, y, xlim=[800,2000], init_fit=init_fit, title=title, colors=['tomato','darkorange','darkorange','khaki','powderblue','yellowgreen'])
    pfm.print_result(res, x, y)