# Last significant edit: 05/01/2025
import numpy as np
import scipy as sps
import glob
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import os
import sys
import time
import pickle as pk
import TRplot_04282025 as trp
from ipywidgets import interactive,interact,fixed
import pandas as pd
from astropy.io import ascii
from astropy.table import Table
from lmfit.models import PolynomialModel, LorentzianModel, LinearModel, ConstantModel, GaussianModel, VoigtModel
from lmfit import models, Model, Parameters, minimize, report_fit
import string
from matplotlib import rc
from matplotlib import pyplot as plt
import h5py as hdf
import glob
from operator import itemgetter as getter


def f_params(p,vary,bounds):
    params = Parameters()
    for j,n in enumerate(list(p.keys())):
        if p[n] == 0:
            p[n] = .00000001
        mn = bounds[n][0]/p[n]; mx = bounds[n][1]/p[n];
        params.add(n,value = 1,vary = vary[n],min = mn,max = mx)
    return params
    
def O_nLL_Poisson(func,params,p_p,x,I):
    ppp = np.array(params)*np.array(list(p_p.values()))
    pp = dict(zip(np.array(list(p_p.keys())),ppp))
    mI = func(x,pp)
    return np.sum(-I*np.log(mI)+mI)

def f_voigt(x,p={}):
    if len(p)==0:
        xrange = x[-1]-x[0]
        p = {
            'A':1,
            'cen':(xrange)/2,
            'sig':xrange/10,
            'gam':xrange/10,
            'off':0.000001  }
    pp = {
        'amplitude':p['A'],
        'center':p['cen'],
        'sigma':p['sig'],
        'gamma':p['gam']  }
    model = VoigtModel()
    params = f_params(pp)
    voigt = model.eval(params,x=x)+p['off']
    return voigt
    
def f_gaussian(x,p={}):
    if len(p)==0:
        xrange = x[-1]-x[0]
        p = {
            'cen':(xrange)/2,
            'A':1,
            'sig':xrange/10,
            'off':0.000001
        }
    gaussian = p['A']*np.exp(-(x-p['cen'])**2/(2*p['sig']**2))/np.sqrt(2*np.pi*p['sig']**2)+p['off']
    return gaussian
    
def f_gaussian_asymoffset(x,p={}): # for this function, A is defined as peak height, whereas normally it is the area under the peak. I.e. the functions are normalized to return 1 at the center.
    if len(p)==0:
        xrange = x[-1]-x[0]
        p = {
            'cen':(xrange)/2,
            'A':1,
            'sig':xrange/10,
            'off':0.000001,
            'offR':0.000001
        }
    lx = len(x)
    Left = np.zeros(lx); Left[x<p['cen']] = 1
    Rigt = np.zeros(lx); Rigt[x>=p['cen']] = 1
    Lgaussian =                       p['A']*np.exp(-(x-p['cen'])**2/(2*p['sig']**2))+p['off']
    Rgaussian = (p['A']+p['off']-p['offR'])*np.exp(-(x-p['cen'])**2/(2*p['sig']**2))+p['offR']
    return Left*Lgaussian+Rigt*Rgaussian
    
def f_lorentzian(x,p={}):
    if len(p)==0:
        xrange = x[-1]-x[0]
        p = {
            'cen':(xrange)/2,
            'A':1,
            'sig':xrange/10,
            'off':0.000001
        }
    lorentzian = (p['sig']/(2*np.pi))*p['A']/((x-p['cen'])**2+(p['sig']/2)**2) + p['off']
    return lorentzian
    
def guessp_gaussian(x,y,p={}):
    ymaxind = np.argmax(y)
    yminind = np.argmin(y)
    cen = x[ymaxind]+.001
    ymax = y[ymaxind]
    cond = y>ymax/2
    FWHM = np.abs(x[cond][0]-x[cond][-1])
    sig = FWHM/(np.sqrt(8*np.log(2)))
    A = ymax*FWHM
    bounds,vary = {},{}
    if len(p)==0:
        p['sig']      = sig        ;  p['cen']      = cen        ; p['A']    = A      ; p['off']=y[yminind]
    bounds['sig'] = [0.0000001,np.abs(x[0]-x[-1])*4] ;  bounds['cen'] = [x[0],x[-1]]  ; bounds['A'] = [0.0000001,ymax*sig*100] ; bounds['off'] = [0.0000001,ymax]
    vary['sig']   = 1                              ;  vary['cen']   = 1             ; vary['A']   = 1                        ; vary['off'] = 1
    # params = f_params(p,bounds,vary)
    return p,bounds,vary#,params
    
def O_nLL1D(params,func1D,p_p,x,I,offset=10000):
    ppp = np.array(params)*np.array(list(p_p.values()))
    pp = dict(zip(np.array(list(p_p.keys())),ppp))
    mI = func1D(x,pp)
    nLL = np.sum(-I*np.log(mI)+mI)+offset
    return nLL
    
def checkparams(func1D,x,y,p,plot=True):
    mx = np.linspace(x[0],x[-1],1000)
    my = func1D(mx,p=p)
    if plot:
        trp.trplot([x,mx],[y,my],index = [0,1],markersize=5,linestyle=['','-'],marker=['o',''])
    return mx,my

def fit1D(func1D,x,y,p,vary,bounds,plot=True,printbest=True,offset=0,method='least_sq',o={'verbose':True,'xtol':2.3e-16,'gtol':2.3e-16,'ftol':2.3e-16}):
    params = f_params(p,vary,bounds)
    if method == 'least_sq':
        result = minimize(O_nLL1D,params,method=method,args=(func1D,p,x,y),kws={'offset':offset},verbose=o['verbose'],xtol=o['xtol'],gtol=o['gtol'])
    fitp = dict(zip(np.array(list(p.keys())),np.array(result.params)*np.array(list(p.values()))))
    mx,my = checkparams(func1D,x,y,fitp,plot=plot)
    bestnLL = O_nLL1D(params,func1D,fitp,x,y,offset=offset)
    if printbest:
        print('success = '+str(result.success))
        for k,key in enumerate(fitp.keys()):
            try:
                print('%s = %f +/- %f'%(key,result.params[key].value*p[key],result.params[key].stderr*p[key]))
            except:
                print('%s = %f +/- %s'%(key,result.params[key].value*p[key],'NOTHING'))
        print('best nLL = %f'%bestnLL)
    return result,fitp,mx,my,bestnLL

def plotnLL1D(func1D,p_p,vary,bounds,x,y,prange_p = {},points=20,plot=False,offset=0):
    if len(prange_p)==0:
        prange_p=bounds
    nLL_pv = {}; fitp_pvp = {}
    for pkey,p in p_p.items():
        if vary[pkey]==1:
            fitp_pvp[pkey] = np.zeros((points,len(p_p)))
            nLL_pv[pkey] = np.zeros(points)
            pp_p = copy.deepcopy(p_p)
            vvary = copy.deepcopy(vary) # reset all vary to 1
            vvary[pkey] = 0
            # print(vvary)
            prange = prange_p[pkey]
            parray = np.linspace(prange[0],prange[1],points)
            for v,pv in enumerate(parray):
                pp_p[pkey] = pv
                result,fitp,mx,my,bestnLL = fit1D(func1D,x,y,pp_p,vvary,bounds,plot=False,printbest=False,offset=offset,o={'verbose':False,'xtol':1e-24,'gtol':1e-24})
                nLL_pv[pkey][v] = bestnLL
                fitp_pvp[pkey][v] = list(fitp.values())
            trp.trplot(parray,nLL_pv[pkey],xlabel=pkey,ylabel='nLL')
        else:
            pass    
    return nLL_pv, fitp_pvp


