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
import trPlot as trp
from ipywidgets import interactive,interact,fixed
import pandas as pd
from astropy.io import ascii
from astropy.table import Table
from lmfit.models import PolynomialModel
import string
from matplotlib import rc
from matplotlib import pyplot as plt
import h5py as hdf
import glob
import xarray as xr
from trGeneral import *
from trFit import *


DataDir = './trData/XAS/'

####
def f_RIXShdftods(hdffile):
    hdfobj = hdf.File(hdffile)
    ds = xr.Dataset()
    data = hdfobj['data']
    meta = hdfobj['meta']
    keys = list(data.keys())
    ds = ds.assign_coords(E=np.transpose(data['E'][:,0]))
    ds = ds.assign_coords(E_l=('E',np.transpose(data['E_l'][:,0])))
    ds = ds.assign_coords(E_r=('E',np.transpose(data['E_r'][:,0])))
    for key in meta.keys():
        value = np.transpose(meta[key])[0]
        ds.attrs[key]=value
    ds.attrs['norm_factor'] = ds.attrs['count_time'].sum()/ds.attrs['points_per_pixel'][0]
    for key in ['rixs','rixs_l','rixs_r']:
        ds = ds.assign({key:('E',np.transpose(data[key][:,0]))})
        rawcounts = ds[key]*ds.attrs['norm_factor']
        ds = ds.assign({key+'_rawcounts':('E',rawcounts.data)})
        ds.attrs['norm_factor2_'+key] = float(ds[key].sum())
    return ds
def f_XAShdftods(hdffile):
    hdfobj = hdf.File(hdffile)
    ds = xr.Dataset()
    ds = ds.assign_coords(energy=hdfobj['data']['pgm_en'][:,0])
    ds = ds.assign_coords(scan=[0])
    for k,dkey in enumerate(hdfobj['data'].keys()):
        temp = np.array(hdfobj['data'][dkey][:,0])
        ds = ds.assign({dkey:('energy',temp)})
    for k,mkey in enumerate(hdfobj['meta'].keys()):
        if mkey=='pol':
            temp = [np.array(hdfobj['meta'][mkey])[0].decode()]
        else:
            temp = np.array(hdfobj['meta'][mkey])[0]
        ds = ds.assign({mkey:('scan',temp)})
    return ds
def setDataDir(DataDirr):
    global DataDir
    DataDir=DataDirr
def getscanfiles(scan_nums):
    ls = len(scan_nums)
    AllFiles = glob.glob(DataDir+'*.hdf'); lf = len(AllFiles[0])
    scanfiles = np.array(np.repeat('',ls),dtype = '<U%i'%(lf+10))
    for s,scan in enumerate(scan_nums):
        for f,file in enumerate(AllFiles):
            if file.find(str(scan))!=(-1):
                scanfiles[s] = file
        if scanfiles[s] == '':
            print('cannot find scan %i'%scan)
    return scanfiles

def f_D_RIXS_fromSIX(scan_nums, svar = [], scanvar='energy', I='rixs', Erange = [-8000,18000], binsize=4, save=False, 
                     metavars = ['T_cryo','energy','pol','norm_I0','points_per_pixel','th','tth','scan'], addvars = {}):
    scanfiles = getscanfiles(scan_nums)
    ls = len(scan_nums)

    dscheck = f_hdftods(scanfiles[0])
    scanvarinmeta = len(np.where(np.array(list(dscheck.attrs.keys()))==scanvar)[0])>0
    if scanvarinmeta:
        svar = np.arange(ls)*1.0
    elif len(svar)==0:
        raise Exception('Need to input scan variable array: svar')

    bins = np.arange(Erange[0],Erange[1],binsize*1.)
    Eloss = bins[:-1]+binsize/2.
    
    metvars = {}
    for k,key in enumerate(metavars):
        metvars[key] = np.array([])
    
    nf,nf2= np.arange(ls)*1., np.arange(ls)*1.
    II,cc,ee = {},{},{}
    for f,file in enumerate(scanfiles):
        ds = f_RIXShdftods(file)
        if scanvarinmeta:
            svar[f] = ds.attrs[scanvar][0].round(1)
        nf[f] = ds.attrs['norm_factor']
        for k,key in enumerate(metavars):
            if key=='pol':
                temp = ds.attrs[key].decode()
            else:
                temp = ds.attrs[key][0]
            metvars[key] = np.append(metvars[key],temp)
        dsinrange = ds.where((ds['E']>Erange[0])&(ds['E']<Erange[1]),drop=True)
        ds_binned = dsinrange.groupby_bins('E',bins = bins).mean()
    
        II[f] = np.array(ds_binned[detector])
        nf2[f] = np.sum(II[f])
        II[f]/=nf2[f]
        cc[f] = np.array(ds_binned[detector+'_rawcounts'])
    
    for k,key in enumerate(metavars):
        metvars[key] = ('scanvar',metvars[key])
    
    II = np.array(list(II.values()))
    cc = np.array(list(cc.values()))
    
    D = xr.Dataset()
    D = D.assign_coords(scanvar=svar)
    D = D.assign_coords(Eloss=Eloss)
    D = D.assign({'counts':(('scanvar','Eloss'),cc),
                  'I':(('scanvar','Eloss'),II),
                  'norm_factor':('scanvar',nf),
                  'norm_factor2':('scanvar',nf2)
                  })
    D = D.assign(metvars)
    D = D.assign(addvars)
    if save:
        D.to_netcdf('./D_%i_%i.nc'%(scan_nums[0],scan_nums[-1]))
    return D,dscheck
    
def f_D_XAS_fromSIX(scan_nums, save=False, addvars = {}):
    scanfiles = getscanfiles(scan_nums)
    ls = len(scan_nums)
    svar = np.arange(ls)
    D = f_XAShdftods(scanfiles[0])
    for f,file in enumerate(scanfiles[1:]):
        ds = f_XAShdftods(file)
        ds['scan'] = [f+1]
        D = xr.concat([D,ds],dim='scan')
    D = D.assign(addvars)
    if save:
        D.to_netcdf('./D_%i_%i.nc'%(scan_nums[0],scan_nums[-1]))
    return D