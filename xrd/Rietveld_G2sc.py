# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:16:43 2024

@author: James Nohl
"""

import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import lines
from cycler import cycler
import numpy as np
import glob
import os
from scipy.signal import find_peaks
import pymatgen
from pymatgen.analysis.diffraction.xrd import XRDCalculator

import xrd_toolkit as xtk

drive = sys.path[-1].split('\\Users\\')[0]
user = sys.path[-1].split('Users\\')[1].split('\\')[0]

sys.path.insert(0,rf"{drive}\Users\{user}\gsas2full\GSASII") # needed to "find" GSAS-II modules
import GSASIIscriptable as G2sc

def HistStats(gpx):
    '''prints profile rfactors for all histograms'''
    print(u"*** profile Rwp, "+os.path.split(gpx.filename)[1])
    for hist in gpx.histograms():
        print("\t{:20s}: {:.2f}".format(hist.name,hist.get_wR()))
    print("")
    gpx.save()

def CellStats(gpx, key = 'unit cell', pprint=True):
    cell_dict={}
    for p in gpx.data['Phases']:
        if type(gpx.data['Phases'][p]) is type(None):
            continue
        print(p)
        cell_dict[p]={}
        if key == 'unit cell':
            cell_dat = gpx.data['Phases'][p]['General']['Cell']
            if pprint:
                print('a\tb\tc\tvol')
                print(cell_dat[1],'\t',cell_dat[2],'\t',cell_dat[3],'\t',cell_dat[7])
            cell_dict[p]['unit_cell']=cell_dat
    if pprint is False:
        return cell_dict

def CompStats(gpx, pprint=True):
    for p in gpx.data['Phases']:
        if type(gpx.data['Phases'][p]) is type(None):
            continue
        print(p)
        comp = gpx.phase(p).composition
        #if pprint:
        #    if 'Li' in list(comp.keys()):
        #        print(comp['Li']/3,'\t',comp['Ni']/3,'\t',comp['O']/3)
    if pprint is False:
        return comp

def PhaseStats(gpx, key='fraction'):
    for p in gpx.data['Phases']:
        if type(gpx.data['Phases'][p]) is type(None):
            continue
        print(p)
        print(gpx.phase(p)['Histograms'][list(gpx.phase(p)['Histograms'].keys())[0]]['Scale'][0])

def tab_stats(gpx):
    print('Rwp')
    for hist in gpx.histograms():
        print(hist.get_wR())
    print('unit cell')
    CellStats(gpx, key = 'unit cell', pprint=True)
    print('composition')
    CompStats(gpx, pprint=True)
    print('phase_fract')
    PhaseStats(gpx, key='fraction')

def get_bkg_pts(gpx, hist):
    return np.array(gpx.histogram(hist).Background[1]['FixedPoints'])

def get_hist(gpx, hist):
    x   = gpx.histogram(hist).get('data')[1][0]
    dat = gpx.histogram(hist).get('data')[1][1]
    mod = gpx.histogram(hist).get('data')[1][3]
    bkg = gpx.histogram(hist).get('data')[1][4]
    res = gpx.histogram(hist).get('data')[1][5]
    wR  = gpx.histogram(hist).get_wR()
    return x, dat, mod, bkg, res, wR
    
def plot_gpx(gpx, hist=None, refs=None, bkg_plot=False, xlim=None, plot=True, savepath=None):
    if type(gpx) is str:
        gpx  = G2sc.G2Project(gpx)
    if hist is None:
        hist = gpx.histograms()[0]
    x, dat, mod, bkg, res, wR = get_hist(gpx, hist)
    if xlim is not None:
        rowx = np.where((x.data>=xlim[0])&(x.data<=xlim[1]))[0]
        x = x.data[rowx]
        dat = dat[rowx]
        mod = mod[rowx]
        bkg = bkg[rowx]
        res = res[rowx]
    xmin = np.min(x)
    xmax = np.max(x)
    if bkg_plot is False:
        y = dat-bkg
        y1 = mod-bkg
    else:
        y = dat
        y1 = mod
    fig, (ax0,ax1) = plt.subplots(2,1, gridspec_kw={'height_ratios': [4,1]})
    ax0.scatter(x, y, marker='+',s=7, color=[0,0,0,0.5], label='data')
    if bkg_plot:
        ax0.plot(x, bkg, c='r', label='bkg')
        ax0.plot(x, y1, label='fit')
    else:
        ax0.plot(x, y1, label=f'fit, Rwp= {round(wR,2)}')
    handles, _  = ax0.get_legend_handles_labels()
    ymin, ymax = ax0.get_ylim()
    if refs is not None:
        xrd_sim = XRDCalculator()
        colors = cycler('color', np.vstack(((0,0,0),cm.Dark2.colors)))
        if type(refs) is str:
            refs=[refs]
        for i,(ref,c) in enumerate(zip(refs,colors)):
            if type(ref) is str:
                name = os.path.split(ref)[1].split('.')[0]
                struct = pymatgen.core.Structure.from_file(ref)
                pattern = xrd_sim.get_pattern(struct)
            else:
                name='something else from structures class'
                pattern = xrd_sim.get_pattern(ref)
            rows= np.where((pattern.x>=xmin)&(pattern.x<=xmax))[0]
            ax0.vlines(pattern.x[rows],(-i*ymax/50),((-i*ymax/50)-(16*ymax/300)), label=name, color=c['color'])
        _, labels = ax0.get_legend_handles_labels()
        for c in colors:
            vertical_line = lines.Line2D([], [],  marker='|', linestyle='None', color=c['color'],
                                  markersize=10, markeredgewidth=1.5)
            handles.append(vertical_line)
        ax0.legend(handles, labels)
    else:
        ax0.legend()
    if refs is not None:
        ax0.set_ylim(bottom=(-i*ymax/50)-(16*ymax/300))
    ax0.set_xticklabels([])
    ax0.set_ylabel('Intensity [arb.u.]')
    ax0.yaxis.set_tick_params(labelleft=False)
    ax0.set_yticks([])
    ax1.plot(x, res, color='0.5', zorder=0, linewidth=1)
    ax1.set_ylabel('residual')
    ax1.yaxis.set_tick_params(labelleft=False)
    ax1.set_yticks([])
    ax1.set_facecolor("none")
    plt.xlabel('2-theta')
    plt.subplots_adjust(hspace=0)
    if savepath is not None:
        plt.savefig(savepath, dpi=400, transparent=True, bbox_inches='tight')
    if plot:
        plt.show()

def exclusion_regions(ras_file, cif_to_remove, hicut=60):
    """Get exclusion regions around reflections in the input file """
    lines,i,x,y = xtk.get_ras_dat(ras_file)
    
    xrd_sim = XRDCalculator()
    
    struct_remove = pymatgen.core.Structure.from_file(cif_to_remove)
    pattern_remove = xrd_sim.get_pattern(struct_remove)
    
    peaks_hicut = pattern_remove.x[np.where((pattern_remove.x>hicut)
                                            &(pattern_remove.x<=np.max(x)))[0]]
    
    bound = 0.1
    bound2 = 3
    #cut_row = {}
    exclusion_list=[]
    for i,val in enumerate(peaks_hicut):
        ### find dat peak in region of sim peak
        vrows = np.where((x>=val-bound)&(x<=val+bound))[0]
        pv = find_peaks(y[vrows])[0]
        if len(pv) == 1:
            pv = pv[0]
            val_c = x[vrows][pv]
            cut_row = np.where((x>=val_c-0.5)&(x<=val_c+0.5))[0]
        else:
            val_c=None
            cut_row = np.where((x>=peaks_hicut[i]-0.5)&(x<=peaks_hicut[i]+0.5))[0]

        pltrows = np.where((x>=val-bound2)&(x<=val+bound2))[0]
        plt.scatter(x[pltrows], y[pltrows], marker='+')
        plt.scatter(x[cut_row], y[cut_row], marker='+', label='cut_rows')
        if val_c is not None:
            plt.scatter(val_c, y[vrows][pv],    marker='o', label='id_cif_peak')
        plt.legend()
        plt.show()
        print('cif peak : ',val_c, '.......low : ',np.min(x[cut_row]), '...high : ',np.max(x[cut_row]))
        exclusion_list.append([np.min(x[cut_row]),np.max(x[cut_row])])
    return exclusion_list

def refine(dat_file, prm_file, phs_file, cifs=None, 
            logLam=8, coeffs=10, limits=None, exclusions=None, stop=999,):

    filename = os.path.split(dat_file.split('.')[0])[1]
    
    folder = os.path.split(dat_file)[0]
    gpx = G2sc.G2Project(newgpx=rf"{folder}\{filename}.gpx")
    gpx.data['Controls']['data']['max cyc'] = 20
    
    hist = gpx.add_powder_histogram(dat_file, prm_file)
    
    if limits is None:
        limits = hist.get('Limits')
    if exclusions is not None:
        limits.extend(exclusions)
    
    ### background
    stepcount=0
    gpx.save(rf'{folder}\step{stepcount}.gpx')
    hist.calc_autobkg(logLam=logLam)
    hist.fit_fixed_points()
    hist.setHistEntryValue(['Sample Parameters','Scale'], [1.0, False])
    hist.set_refinements({
                        #"Limits": limits,
                        "Background": {"no. coeffs": coeffs, "refine": True},
                        "Sample Parameters": ["DisplaceX", "DisplaceY"]#, "Shift"]
                        })
    hist.fit_fixed_points()
    plot_gpx(gpx, hist, bkg_plot=True, savepath=rf'{folder}\step0.png')
    
    ###
    stepcount+=1
    if stepcount >= stop:
        return
    gpx.save(rf'{folder}\step{stepcount}_AddPhase.gpx')
    hist.set_refinements({
                        "Background": {"no. coeffs": coeffs, "refine": False}
                        })
    ### add phase(s)
    if type(phs_file) is str:
        phs_file = [phs_file]
    phs={}
    for p in phs_file:
        phs_name = os.path.split(p)[1].split('.cif')[0]
        phs[phs_name]= gpx.add_phase(p, phasename=phs_name, histograms=[hist])
    
    ### phase fraction
    stepcount+=1
    if stepcount >= stop:
        return
    gpx.save(rf'{folder}\step{stepcount}_PhaseFract.gpx')
    for p in phs:
        phs[p].set_HAP_refinements({"Scale":True})
    gpx.do_refinements([{}]) # refine after setting
    HistStats(gpx)
    plot_gpx(gpx, hist)    
    
    ### unit cell
    stepcount+=1
    if stepcount >= stop:
        return
    gpx.save(rf'{folder}\step{stepcount}_UnitCell.gpx')
    #for p in phs:
    #    phs[p].set_HAP_refinements({"Scale":True})#, histograms='all')
    refdict1 = {"set": {"Cell": True}} # set the cell flag (for all phases)
    gpx.set_refinement(refdict1)
    gpx.do_refinements([{}])
    HistStats(gpx)
    plot_gpx(gpx, hist)
    
    ### microstrain
    stepcount+=1
    if stepcount >= stop:
        return
    gpx.save(rf'{folder}\step{stepcount}.gpx')
    refdict2 = {"set": {"Mustrain": {"type":"isotropic","refine":True}#,
                        #"Size":{"type":"isotropic","refine":True},
                        }}
    
    gpx.set_refinement(refdict2,histogram=[hist])
    gpx.do_refinements([{}]) # refine after setting
    HistStats(gpx)
    plot_gpx(gpx, hist)

    
    ### instrument parameters
    stepcount+=1
    if stepcount >= stop:
        return
    #print(f"\t\t~~~step{stepcount}~~~\n###### instrument parameters ######")
    gpx.save(rf'{folder}\step{stepcount}.gpx')
    hist.set_refinements({"Instrument Parameters": ["U", "V", "W"]})
    #phs0.set_refinements({"Atoms":{"all":"XU"}})
    gpx.do_refinements([{}]) # refine after setting
    HistStats(gpx)
    refs_list = phs_file
    if cifs is not None:
        refs_list.extend(cifs)
    plot_gpx(gpx, hist, refs=refs_list,
             savepath=rf'{folder}\step{stepcount}.png')
    for p in phs:
        print(gpx.phase(phs[p]).composition)

    ### site occupancy
    stepcount+=1
    if stepcount >= stop:
        return
    gpx.save(rf'{folder}\step{stepcount}_SiteOcc.gpx')
    gpx.phase(0).setPhaseEntryValue(keylist=['Atoms',2,6], newvalue=1.0)# set Ni2 site occupancy to 1
    gpx.add_EqnConstr(1.0,['0::Afrac:0','0::Afrac:1'], [1,1])
    gpx.phase(0).set_refinements({"Atoms":{"Li1":"F","Ni1":"F"}})
    #phs[list(phs.keys())[0]].set_refinements({"Atoms":{"Li1":"F","Ni1":"F"}})
    gpx.do_refinements([{}]) # refine after setting
    plot_gpx(gpx, hist, refs=refs_list,
             savepath=rf'{folder}\step{stepcount}.png')

    for p in phs:
        print(gpx.phase(phs[p]).composition)
    tab_stats(gpx)

def batch_plot(gpx_files:list, xlim=None, refs=[], names=None, title=None):
    if names is None:
        names=[]
        for f in gpx_files:
            names.append(os.path.split(f.split('.gpx')[0])[1])
    for f,name in zip(gpx_files, names):
        gpx  = G2sc.G2Project(f)
        hist = gpx.histograms()[0]
        x, dat, mod, bkg, res, wR = get_hist(gpx, hist)
        if xlim is not None:
            rows = np.where((x>=xlim[0])&(x<=xlim[1]))[0]
        else:
            rows = np.arange(0,len(x),1)
        y  = (dat-bkg)[rows]
        y1 = (mod-bkg)[rows]
        xp = x[rows]
        plt.plot(xp, y, label=name)
    ax = plt.gca()
    handles, _  = ax.get_legend_handles_labels()
    ymin, ymax = ax.get_ylim()
    if len(refs) > 0:
        xrd_sim = XRDCalculator()
        colors = cycler('color', np.vstack(((0,0,0),cm.Dark2.colors)))
        for i,(ref,c) in enumerate(zip(refs,colors)):
            if type(ref) is str:
                name = os.path.split(ref)[1].split('.')[0]
                struct = pymatgen.core.Structure.from_file(ref)
                pattern = xrd_sim.get_pattern(struct)
            else:
                name='something else from structures class'
                pattern = xrd_sim.get_pattern(ref)
            rows= np.where((pattern.x>=xlim[0])&(pattern.x<=xlim[1]))[0]
            plt.vlines(pattern.x[rows],(-i*ymax/50)-600,((-i*ymax/50)-(16*ymax/300)), label=name, color=c['color'])
        _, labels = ax.get_legend_handles_labels()
        for c in colors:
            vertical_line = lines.Line2D([], [],  marker='|', linestyle='None', color=c['color'],
                                  markersize=10, markeredgewidth=1.5)
            handles.append(vertical_line)
    plt.legend(handles, labels)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_yticks([])
    plt.ylabel('Intensity [arb.u.]')
    plt.xlabel('2-theta')
    if title is None:
        title='XRD profiles'
    plt.title(title)
    plt.show()

def load_gpx(file):
    gpx  = G2sc.G2Project(file)
    hist = gpx.histograms()[0]
    return gpx, hist

def places(place):
    if place == 1:
        return 'st'
    elif place == 2:
        return 'nd'
    elif place == 3:
        return 'rd'
    else:
        return 'th'