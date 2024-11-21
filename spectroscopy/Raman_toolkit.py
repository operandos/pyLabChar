# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 10:07:51 2024

@author: James Nohl
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import uniform_filter1d as uf1d
from scipy.signal import find_peaks as fp

import peak_fit as pf

def load_txt(file:str, zapper=True, thresh=2000):
    dat = np.loadtxt(file)
    x, y = dat[:,0], dat[:,1]
    if zapper:
        x, y = zap(x,y, thresh=2000)
    return x, y

def average_dat(file_list:list, norm=True, **kwargs):
    dat=pd.DataFrame()
    for file in file_list:
        name = os.path.split(file)[1].split('.txt')[0]
        x,y = load_txt(file, **kwargs)
        if norm is True:
            y = y/np.max(y)
        dat[name] = y
    return dat
        

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

def knot_mod(x,y, prom=0.1, knot_cut=True, window=None, **kwargs):
    
    plt.scatter(x,y, s=8, c='#99002299', edgecolor='none', label='data',zorder=-1, alpha=0.3)
    
    if window is not None:
        y = uf1d(y, window, **kwargs)
    
    plt.plot(x,y)

    troughs = fp(y*-1+1,width=5)
    ysort = np.argsort(np.append(x[troughs[0]],[x[0],x[-1]]))
    knot_vals = np.sort(np.append(x[troughs[0]],[x[0],x[-1]]))
    labs = np.arange(0,len(knot_vals),1)
    y_vals = np.append(y[troughs[0]],[y[0],y[-1]])[ysort]
    plt.scatter(knot_vals,y_vals)
    for lab,xp,yp in zip(labs, knot_vals,y_vals):
        plt.annotate(lab, [xp,yp], fontsize=12)
    plt.show()
    if knot_cut is True:
        try:
            knot_cut = []

            while True:
                knot_cut.append(int(input()))

        # if the input is not-integer, just print the list
        except:
            print(knot_cut)
    if knot_cut is None:
        knot_cut=[]
    
    knot_vals=np.delete(knot_vals, knot_cut)
    
    return knot_vals

def fit_mod(x,y, mod, params, window=10, knot_vals = [202, 250, 750, 800, 890, 1220, 1750, 1900, 2000], 
            show_bkg_knots=True, pprint=True, tech='Raman', title=None, save_path=False,
            **kwargs,
            ):
    #mod, params = pf.model(peaks=peaks, bkg=False
    #                       )
    if type(knot_vals) is list:
        knot_vals=np.array(knot_vals)
    mod += pf.models.SplineModel(xknots=knot_vals, prefix='bkg_')
    #bkg = lf.models.SplineModel(xknots=knot_vals, prefix='bkg_')
    params.update(mod.components[-1].guess(uf1d(y, size=5*window), x))
    for k in list(params.keys()):
        if 'bkg_' in k:
            params[k].vary=False
    if type(window) is int:
        y = uf1d(y, window)
    result = pf.fit(x, y, mod, params, xlim=None, norm=False)
    comp_val = result.eval_components()
    if show_bkg_knots:
        kvy=[]
        for p in params:
            if 'bkg_' in p:
                kvy.append(params[p].value)
        plt.plot(x, comp_val['bkg_'])
        plt.plot(x,uf1d(y, size=5*window), zorder=0)
        plt.scatter(result.components[-1].xknots, kvy)
        plt.scatter(x,y, s=8, c='#99002299', edgecolor='none', label='data',zorder=-1, alpha=0.3)
        if title is not None:
            plt.title(title)
        if save_path is not False:
            plt.savefig(rf"{save_path}\bkg_plots\{title}.png", dpi=400, transparent=True)
        plt.show()
    pf.plot_result(result, x, y, tech='Raman', title = title, show=False, **kwargs)
    if type(save_path) is str:
        plt.savefig(rf"{save_path}\{title}.png", dpi=400, transparent=True)
    if title:
        print(title)
    if pprint:
        pf.print_result(result)
    plt.show()
    return result

def fit_HOPG(file, title,
             init_fit=False):
    x,y = pf.load_txt(file)
    mod, params = pf.model({
     'G': {'centres': [1550], 'type': 'lorentz'}},
    c_bound =100, amps=[100], sigma = 40, a_bound = 200)
    res = pf.fit(x,y,mod,params,xlim=[800,2000])
    if title is None:
        title = 'Raman plot with fit'
    pf.plot_result(res, x, y, xlim=[800,2000], init_fit=init_fit, 
                    title=title,
                    colors=['yellowgreen'])
    pf.print_result(res, x, y)

def fit_t_aC(file, 
             title = None,
             init_fit=False):
    x,y = load_txt(file)
    mod, params = pf.model({
     'aC':  {'type'   : 'gauss',
             'center' : (1200,True)},
     'Dia': {'type'   : 'lorentz',
             'center' : (1330,True)},
     'D':   {'type'   : 'gauss',
             'center' : (1380,True)},
     'G':   {'type': 'gauss',
             'center': (1550,True)}})
    res = pf.fit(x,y,mod,params,xlim=[800,2000])
    if title is None:
        title = 'Raman plot with fit'
    pf.plot_result(res, x, y, xlim=[800,2000], init_fit=init_fit, title=title, colors=['darkorange','darkorange','khaki','powderblue','yellowgreen'])
    pf.print_result(res)

def fit_LFP(file, 
            title = None,
            init_fit=False):
    x,y = load_txt(file)
    mod, params = pf.model({
            'PO4': {'type'  : 'lorentz',
                    'center': (950,True,900,1000),
                    'amp'   : (1,True,0),
                    'sigma' : (2,True,0,200)
                    },
            'aC':  {'center': (1200,True,1150,1300),
                    'amp'   : (1,True,0,40),
                    'sigma' : (10,True,0,200)
                    },
            #'Dia': {'centres': [1330], 'type': 'lorentz'},
            'D':   {#'type'  : 'lorentz',
                    'center': (1380,True,1300,1450),
                    'amp'   : (5,True,0),
                    'sigma' : (50,True,1,200)
                    },
            'G':   {'type'  : 'lorentz',
                    'center': (1600,True,1500,1680),
                    'amp'   : (5,True,0),
                    'sigma' : (50,True,0,200)
                    }
            })
    #c_bound =50, amps=[3,50,5,8], sigma = 35, a_bound = 100)
    res = pf.fit(x,y,mod,params,xlim=[800,2000])
    if title is None:
        title = 'Raman plot with fit'
    pf.plot_result(res, x, y, xlim=[800,2000], init_fit=init_fit, title=title, colors=['tomato','darkorange','darkorange','powderblue','yellowgreen'], tech='Raman')
    pf.print_result(res)
