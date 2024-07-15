# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:43:46 2024

@author: James Nohl
"""

from lmfit import models, Parameters
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson as simp
import matplotlib.cm as cm
from cycler import cycler

def model(peaks:dict, amps=None, a_bound=1, c_bound=0.4, sigma=None, a_scale=None, bkg=True):
    """
    Generate lmfit model and parameters objects
    Parameters
    ----------
    peaks : dict
        Provide in format:
            peaks = {'comp':{'type'  : 'lorentz' or 'gauss'
                             'center': (VALUE,VARY,MIN,MAX,EXPR,BRUTE_STEP),
                             'amp'   : (),
                             'sigma' : ()
                            }
                    }
        OR
            peaks={'comp':{'centers':[1.5,4.2],
                   'type'='gauss'}
                  }                          
    amps : TYPE, optional
        DESCRIPTION. The default is None.
    a_bound : TYPE, optional
        DESCRIPTION. The default is 1.
    c_bound : TYPE, optional
        DESCRIPTION. The default is 0.4.
    sigma : TYPE, optional
        DESCRIPTION. The default is None.
    a_scale : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    mod : TYPE
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.

    """
    mod=None
    m = {}
    params=Parameters()
    if amps is None:
        amps = [1]*len(peaks)
    for comp, a in zip(peaks, amps):
        if 'type' not in peaks[comp].keys():
            peaks[comp]['type']='gauss'
        name = comp
        if peaks[comp]['type']=='gauss':
            m[name] = models.GaussianModel(prefix=f'{name}_')
        elif peaks[comp]['type']=='lorentz':
            m[name] = models.LorentzianModel(prefix=f'{name}_')
        params.update(m[name].make_params())
        
        ### amplitude ###
        for k in peaks[comp]:
            if k == 'type':
                continue
            if type(peaks[comp][k]) is tuple:
                if k == 'amp':
                    kname = 'amplitude'
                else:
                    kname=k
                params.add_many((f'{name}_{kname}',)+peaks[comp][k])
            else:
                if 'amp' in peaks[comp].keys():
                    a = peaks[comp]['amp']
                if type(a_scale)==int or type(a_scale)==float or type(a_scale)==np.float64:
                    a = a*a_scale
                params[f'{name}_amplitude'].set(a,min=0.001, max=a_bound)
                ### center ###
                if  'center' in peaks[comp].keys():
                    if type(peaks[comp]['center']) is list:
                        if len(peaks[comp]['center']) == 3:
                            no, cmin, cmax = peaks[comp]['center']
                        if len(peaks[comp]['center']) == 2:
                            no, cmax = peaks[comp]['center']
                            cmin = no-c_bound
                    if type(peaks[comp]['center']) is dict:
                        if 'val' in peaks[comp]['center']:
                            no = peaks[comp]['center']['val']
                        if 'min' in peaks[comp]['center']:
                            cmin = peaks[comp]['center']['min']
                        if 'max' in peaks[comp]['center']:
                            cmax = peaks[comp]['center']['max']
                    if type(peaks[comp]['center']) is int or type(peaks[comp]['center']) is float:
                        no = peaks[comp]['center']
                        cmin = no-c_bound
                        cmax = no+c_bound
                    params[f'{name}_center'].set(no, min=cmin, max=no+cmax)
                
                ### sigma ###
                if sigma is None:
                    params[f'{name}_sigma'].set(2)
                else:
                    params[f'{name}_sigma'].set(sigma, min=0.0001, max=sigma*3)
        ### background ###
        if bkg:
            comp = 'bkg'
            m[comp] = models.LinearModel(prefix='bkg_')
            params.update(m[comp].make_params(bkg_slope=0.00001, bkg_intercept=-0.01))
            #params[f'{comp}'].set(bkg_slope=-0.2, bkg_intercept=0)
    for comp in peaks:
        ### expr ###
        if 'expr' in peaks[comp].keys():
            for k in list(peaks[comp]['expr'].keys()):
                params.add(f'{name}_{k}', expr = peaks[comp]['expr'][k])
    for comp in m:
        if mod is None:
            mod = m[comp]
        else:
            mod += m[comp]
    return mod, params

def fit(x,y, model, params, xlim=[0,7], norm=True):
    """
    Using lmfit model and parameters objects, fit x, y series of data.

    Parameters
    ----------
    x : np.array
        DESCRIPTION.
    y : np.array
        DESCRIPTION.
    model : lmfit.model.CompositeModel
        DESCRIPTION.
    params : lmfit.parameter.Parameters
        DESCRIPTION.
    xlim : list, optional
        DESCRIPTION. The default is [0,7].
    norm : bool, optional
        Wether to normalise data before fitting. The default is True.

    Returns
    -------
    result : TYPE
        DESCRIPTION.

    """
    if type(xlim) is list and len(xlim)==2:
        rows = np.where((x>=xlim[0])&(x<=xlim[1]))[0]
    else:
        rows = np.arange(0,len(x),1)
    if norm is True:
        y = (y-np.min(y[rows]))/(np.max(y[rows])-np.min(y[rows]))
    result = model.fit(y[rows], params, x=x[rows])
    return result

def plot_result(result, x, y, xlim=[0,7], norm=True, y_label=False, init_fit=False, leg_loc=None, title=None, show=True, colors=None, alpha=0.3):
    if type(xlim) is list and len(xlim)==2:
        rows = np.where((x>=xlim[0])&(x<=xlim[1]))[0]
    else:
        rows = np.arange(0,len(x),1)
    comps = result.eval_components(x=x[rows])
    #dely = result.eval_uncertainty(sigma=3)
    if norm is True:
        y = (y-np.min(y[rows]))/(np.max(y[rows])-np.min(y[rows]))
    plt.scatter(x[rows], y[rows], s=8, c='#99002299', edgecolor='none', alpha=0.8, label='data')
    plt.plot(x[rows], result.best_fit, '-', label=f'best fit\n$R^2$={round(result.rsquared,3)}', color='k')
    if init_fit:
        plt.plot(x[rows], result.init_fit, '--', label='initial fit')
    #plt.fill_between(x, result.best_fit-dely, result.best_fit+dely,
    #                        color="#8A8A8A", label=r'3-$\sigma$ band')
    bg = False
    for b in ['bg_','bkg_','background_']:
        if b in comps:
            background = comps[b]
            bg = True
            plt.plot(x[rows], background, '--', c='k', label=r'bkg')
    if bg is False:
        background = np.zeros(x[rows].shape)
    plt.gca().set_prop_cycle(cycler('color', cm.Pastel2.colors))
    if colors is None:
        colors = cycler('color', cm.Pastel2.colors)
    for c,comp in zip(colors,comps):
        if any(b in comp for b in ['bg_','bkg_','background_']):
            continue
        #plt.plot(x, comps[comp], label=comp)
        labs = comp.split('_')
        if len(labs)==3:
            lab = f'{labs[0]}_{labs[1]}'
        elif len(labs)==2:
            lab = labs[0]
        if type(c) is dict:
            c = c['color']
        plt.fill_between(x[rows],
                            background,
                            comps[comp] + background,
                            color = c, alpha = alpha,
                            label = lab)
    plt.xlabel('Raman shift [$cm^{-1}$]')
    plt.ylabel('Raman intensity [arb.u.]')
    if title is not None:
        plt.title(title)
    else:
        plt.title('data, best-fit, and fit components')
    plt.legend(bbox_to_anchor=leg_loc)
    if y_label is False:    
        plt.gca().yaxis.set_tick_params(labelleft=False)
        plt.gca().set_yticks([])
    if show:
        plt.show()
    
def print_result(result, tab=True):
    comps=[]
    for val in result.best_values:
        comp = val.rsplit('_',1)[0]
        comps.append(comp)
    comps = list(set(comps))
    comps.sort()
    pts = result.eval_components()
    if tab:
        print('\tcenter\tamp\tsigma\theight\tFWHM\tArea')#\tType')
    for i,comp in enumerate(comps):
        if 'bkg' in comp or 'back' in comp:
            continue
        else:
            amp=result.best_values[f'{comp}_amplitude']
            cen=result.best_values[f'{comp}_center']
            sig=result.best_values[f'{comp}_sigma']
            heig = 0.3989423*amp/max(1e-15, sig)
            fwhm = 2.3548200*sig
            area = simp(pts[f'{comp}_'])
            #mod = result.components[i]._name # doesn't work
            if tab:
                print(f'{comp}\t{round(cen,2)}\t{round(amp,2)}\t{round(sig,2)}\t{round(heig,2)}\t{round(fwhm,2)}\t{round(area,2)}')#\t{mod}')