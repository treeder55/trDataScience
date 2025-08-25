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
from lmfit.models import PolynomialModel
import string
from matplotlib import rc
from matplotlib import pyplot as plt
import h5py as hdf
import glob
from trGeneral import *
from trFit import *

def load_IE_ce(hdffile,Elim=[-4000,11000],det='rixs'):
    hdfobj = hdf.File(hdffile)
    hdfdata = hdfobj['data']
    hdfmeta = hdfobj['meta']
    IE_ce = {}
    for m,(okey,ckey) in enumerate(zip(['E',det],['E_e','I_e'])):
        IE_ce[ckey] = np.transpose(np.array(hdfdata[okey]))[0]
    normfactor = np.sum(hdfmeta['count_time'])/hdfmeta['points_per_pixel'][0][0]
    IE_ce['c_e'] = IE_ce['I_e']*normfactor
    cond = (IE_ce['E_e']>-4000)&(IE_ce['E_e']<11000)
    for c,ckey in enumerate(IE_ce.keys()):
        IE_ce[ckey] = IE_ce[ckey][cond]
    return IE_ce, normfactor
def load_M_c(hdfpath_q,Q_q,Elim=[-4000,11000],det='rixs'):
    le = np.inf
    for q,(qkey,hdfpath) in enumerate(zip(hdfpath_q.keys(),hdfpath_q.values())):
        IE_ce, normfactor = load_IE_ce(hdfpath,Elim=Elim)
        le = np.min([le,len(IE_ce['E_e'])])
    lq = len(Q_q); le = int(le);
    M_c = {
        'Q_q':Q_q,
        'E_qe':np.zeros((lq,le)),
        'I_qe':np.zeros((lq,le)),
        'c_qe':np.zeros((lq,le)),
        'normfactor_q':np.zeros(lq),
        'normfactor2_q':np.zeros(lq)  }
    normfactor_q = np.zeros(lq)
    for q,(qkey,hdfpath) in enumerate(zip(hdfpath_q.keys(),hdfpath_q.values())):
        IE_ce, normfactor = load_IE_ce(hdfpath,Elim=Elim)
        M_c['normfactor_q'][q] = normfactor
        M_c['normfactor2_q'][q] = np.sum(IE_ce['I_e'])
        M_c['E_qe'][q][:le] = IE_ce['E_e'][:le]/1000 # dropping data points at the edge to ensure a regular 2D grid of data.
        M_c['I_qe'][q][:le] = IE_ce['I_e'][:le] #
        M_c['c_qe'][q][:le] = IE_ce['c_e'][:le]
    M_c['normfactor2_q'] = M_c['normfactor2_q']/np.average(M_c['normfactor2_q'])
    M_c['I_qe'] = M_c['I_qe']/np.transpose([M_c['normfactor2_q']])
    M_c['c_qe'] = M_c['c_qe']/np.transpose([M_c['normfactor2_q']])
    return M_c
def f_hdfpath_z(reductionfolder,ppp=4):
    reductionfolder = './trData/'+reductionfolder
    metadata_z = np.load(reductionfolder+'metadata.npy',allow_pickle=True).item()
    zkeys_z = np.array(np.delete(np.array(list(metadata_z.keys())),0),dtype=int)
    hdfpath_z = {}
    for z,zkey in enumerate(zkeys_z):
        hdfpath_z[zkey] = glob.glob(reductionfolder+('ppp%i_%i/*.hdf'%(ppp,zkey)))[0]
    return hdfpath_z, metadata_z
def f_metadata_Qlq(l_l, z_Qq, metadata_z):
    metadata_Qlq = {}  # Q is experimental tuning parameter, e.g. frequency, that was actually tuned for the set of spectra. l (label) is the experimental parameters that are kept track of. Q is always an entry in q.
    lkeys_l = list(l_l.keys())
    Qkeys_Q = np.array(list(z_Qq.keys())) # Q is the index
    for Q,Qkey in enumerate(Qkeys_Q):
        # print(Qkey)
        Qtrunc = Qkey[:1]
        z_q = z_Qq[Qkey]
        lq = len(z_q)
        zkeys_q = np.array(['z' + str(z) for z in z_q])
        metadata_Qlq[Qkey] = {'z':zkeys_q}
        for l,lkey in enumerate(lkeys_l):
            # print(qkey)
            metadata_Qlq[Qkey][lkey] = np.zeros(lq)
            if lkey == 'p':
                metadata_Qlq[Qkey][lkey] = np.array(np.zeros(lq),dtype=object)
            for q,z in enumerate(z_q):
                # print(zkey)
                metadat = metadata_z[z][l_l[lkey]]
                try:
                    metadat = float(metadat)
                except:
                    pass
                metadata_Qlq[Qkey][lkey][q] = metadat
    return metadata_Qlq
def f_M_Qpc(reductionfolder, z_Qq, l_l={},Elim=[-4000,11000],sort=True,D_Q=2,det_Q=[],pkeys_p=['LH','LV'],ppp=4):
    if len(l_l)==0:
        l_l = dict(zip(['s','p','A','f','Q','a','T','E'], np.arange(8)))
        #{'s':[0,1],'p':[2,4],'A':[5,7],'f':[8,14],'Q':[15,18]} # spot, polarization, amplitude, repitition rate. This string gives the labels of the meta data values. Any Q is always a subset of l, sometimes with an additional number if there are multiple scans over the same parameter
    hdfpath_z, metadata_z = f_hdfpath_z(reductionfolder,ppp=ppp)
    metadata_Qlq = f_metadata_Qlq(l_l, z_Qq, metadata_z)
    M_Qpc = {} # c is the data column label, so ckeys_c = ['E_d','I_d']
    metadata_Qplq = {}
    Qkeys_Q = list(z_Qq.keys())
    if len(det_Q)==0:
        det_Q = dict(zip(Qkeys_Q,np.repeat('rixs',len(Qkeys_Q))))
    if D_Q==2:
        D_Q = dict(zip(Qkeys_Q,np.repeat(2,len(Qkeys_Q))))
    for Q,Qkey in enumerate(Qkeys_Q):
        M_Qpc[Qkey] = {}
        metadata_Qplq[Qkey] = {}
        for p,pkey in enumerate(pkeys_p):
            metadata_Qplq[Qkey][pkey] = {}
            metadata_lq = metadata_Qlq[Qkey]
            p_q = metadata_lq['p']
            cond = (p_q==pkey)
            Q_q = metadata_lq[Qkey[:1]][cond]
            # print('(hdfpath_z,z_Qq[Qkey][cond]) = '+str(hdfpath_z)+' '+str(z_Qq[Qkey][cond]))
            hdfpath_q = dictcasttodict(hdfpath_z,z_Qq[Qkey][cond])#np.array([z for z in z_Qq[Qkey][cond]]))
            M_c = load_M_c(hdfpath_q,Q_q,Elim=Elim,det=det_Q[Qkey])
            if D_Q[Qkey]==3:
                T_q = metadata_lq[Qkey[1]][cond]
                M_c['T_q'] = T_q
            M_Qpc[Qkey][pkey] = M_c
            for l,lkey in enumerate(metadata_Qlq[Qkey].keys()):
                metadata_Qplq[Qkey][pkey][lkey] = metadata_Qlq[Qkey][lkey][cond]
    if sort:
        M_Qpc, metadata_Qplq = sort_M_Qpc(M_Qpc,metadata_Qplq)
    return M_Qpc, metadata_Qplq
def sort_M_Qpc(M_Qpc, metadata_Qplq): ################### Need to get sorting figured out. 
    for Q,Qkey in enumerate(M_Qpc.keys()):
        for p,pkey in enumerate(M_Qpc[Qkey].keys()):
            Q_q = M_Qpc[Qkey][pkey]['Q_q']
            try:
                T_q = M_Qpc[Qkey][pkey]['T_q']
                sortedind = f_sortedind2D(T_q,Q_q) # T_q is first here on purpose.
            except:
                sortedind = np.argsort(Q_q)
            for l,lkey in enumerate(metadata_Qplq[Qkey][pkey].keys()):
                l_q = metadata_Qplq[Qkey][pkey][lkey]
                metadata_Qplq[Qkey][pkey][lkey] = l_q[sortedind]
            for c,ckey in enumerate(M_Qpc[Qkey][pkey].keys()):
                M = M_Qpc[Qkey][pkey][ckey]
                M_Qpc[Qkey][pkey][ckey] = M[sortedind]
    return M_Qpc, metadata_Qplq
def f_sortedind2D(a_i,b_i): # repeats a-values with sorted b-values, then next a-value and sorted b-values.
    la = len(a_i)
    ua = np.sort(np.unique(a_i))
    ind = np.arange(la)
    sortedind = np.zeros(la,dtype=int)
    count1,count2 = 0,0
    for i,a in enumerate(ua):
        cond = (a_i>(a-.01))&(a_i<(a+0.01))
        subind = ind[cond]
        subb = b_i[cond]
        subsortedind = np.argsort(subb)
        count2 += np.sum(cond)
        sortedind[count1:count2] = subind[subsortedind]
        count1 += np.sum(cond)
    return sortedind
def centerelastic(M_Qpc,xlim=[-.07,.03],plot=True,bounds=[],offset=100000,function='gaussian'):
    for Q,Qkey in enumerate(M_Qpc.keys()):
        print(Qkey)
        for p,pkey in enumerate(M_Qpc[Qkey].keys()):
            print(pkey)
            E_qe = M_Qpc[Qkey][pkey]['E_qe']
            I_qe = M_Qpc[Qkey][pkey]['c_qe']
            for q in range(len(I_qe)):
                print(q)
                xcond = (E_qe[q]>xlim[0])&(E_qe[q]<xlim[1])
                x = E_qe[q][xcond]
                y = I_qe[q][xcond]
                p_p,bounds,vary = guessp_gaussian(x,y)
                if len(bounds)==0:
                    bounds['cen'] = [-.1,.1]; bounds['A']=[.000001,100000]; 
                    bounds['off'] = [0.0000001,1000]; bounds['sig'];
                # mx,my = checkparams(f_gaussian,x,y,p_p)
                if function == 'gaussian':
                    result,fitp,mx,my,bestnLL = fit1D(f_gaussian,x,y,p_p,vary,bounds=bounds,plot=plot,offset=offset)
                elif function == 'gaussian_asymoffset':
                    p_p['offR']=p_p['off']; bounds['offR']=bounds['off']; vary['offR'] = vary['off']
                    result,fitp,mx,my,bestnLL = fit1D(f_gaussian_asymoffset,x,y,p_p,vary,bounds=bounds,plot=plot,offset=offset)
                elif function == 'lorentzian':
                    result,fitp,mx,my,bestnLL = fit1D(f_lorentzian,x,y,p_p,vary,bounds=bounds,plot=plot,offset=offset)
                print(result.params)
                M_Qpc[Qkey][pkey]['E_qe'][q] = E_qe[q]-fitp['cen']
    return M_Qpc
    
def plot_M_c(M_c,xlim,ylim,off=0,title='',label='',colors=0,markersize=1,plotsize=[1.1,1.3],linestyle='',save=False,savename='noname.jpg',yscale='linear'):
    Q_q = M_c['Q_q']; E_qe = M_c['E_qe']; I_qe = M_c['I_qe']; c_qe = M_c['c_qe']; 
    normfactor_q = M_c['normfactor_q']; 
    lq = len(Q_q); le = len(E_qe[0])
    Q_qe = np.transpose(np.repeat([Q_q],le,axis=0))
    x=E_qe
    y=I_qe
    e=np.sqrt(c_qe)/np.transpose([normfactor_q])
    trp.errorbar(x,y,e,off=off,xlim=xlim,ylim=ylim,title=title,index=np.arange(lq),label=label,
                 linestyle=linestyle,plotsize=plotsize,xlabel='E (eV)',ylabel='I (arb.)',colors=colors, 
                 markersize=markersize,save=save,savename=savename,yscale=yscale)
def sumalongx(x,y,n): # M is a matrix with 2 columns: x, y. n is the number of points to be averaged together
    lx = len(x)
    newlend, remainder = np.divmod(lx,n)
    if remainder == 0:
        x = np.average(np.reshape(x,(newlend,n)),axis=1)
        y = np.sum(np.reshape(y,(newlend,n)),axis=1)
    else:
        x = np.average(np.reshape(x[:-remainder],(newlend,n)),axis=1)
        y = np.sum(np.reshape(y[:-remainder],(newlend,n)),axis=1)
    return x,y
def Eaveraging(E_qe,c_qe,n):
    lq = len(E_qe)
    le = len(E_qe[0])
    newlend, remainder = np.divmod(le,n)
    avgE_qe = np.zeros((lq,newlend))
    sumc_qe = np.zeros((lq,newlend))
    for q in range(len(E_qe)):
        avgE_qe[q],sumc_qe[q] = sumalongx(E_qe[q],c_qe[q],n)
    return avgE_qe,sumc_qe
def f_avgM_pc(M_pc,n):
    avgM_pc = {}
    for p,pkey in enumerate(M_pc.keys()):
        M_c = M_pc[pkey]
        avgM_pc[key] = f_avgM_c(M_c,n)
    return avgM_pc
def f_avgM_c(M_c,n):
    if n==0:
        avgM_c = M_c
    else:
        avgM_c = copy.deepcopy(M_c)
        E_qe = M_c['E_qe']; c_qe = M_c['c_qe'];  normfactor_q = M_c['normfactor_q']
        E_qe, c_qe = Eaveraging(E_qe,c_qe,n)
        avgM_c['E_qe'] = E_qe
        avgM_c['c_qe'] = c_qe
        avgM_c['I_qe'] = c_qe/np.transpose([normfactor_q])
    return avgM_c
    
def f_savemetadata(metadata,reductionfolder,update=False):
    if update:
        metadata_old = np.load('./trData/%s/metadata.py'%reductionfolder,allow_pickle=True).item()
        for key in metadata_old.keys():
            metadata[key] = metadata_old[key]
    f_makefolder('./trData/'+reductionfolder)
    np.save('./trData/%s/metadata'%reductionfolder,metadata)
def f_save_M_Qpc(reductionfolder,M_Qpc,metadata_Qplq,update=False,overwrite=True):
    if update:
        oldM_Qpc,oldmetadata_Qplq = np.load('./trData/%sM_Qpc.npy'%reductionfolder,allow_pickle=True).item()
        oldQkeys = list(oldM_Qpc.keys())
        newQkeys = list(M_Qpc.keys())
        for oldQkey in oldQkeys:
            if overwrite: #overwrite = True means you will overwrite the data in the saved file, so below you do not overwrite the incoming new data with old data. It is a little confusing.
                if oldQkey in newQkeys:
                    pass
                else:
                    M_Qpc[oldQkey] = oldM_Qpc[oldQkey]
                    metadata_Qplq[oldQkey] = oldmetadata_Qplq[oldQkey]
            else:
                M_Qpc[oldQkey] = oldM_Qpc[oldQkey]
                metadata_Qplq[oldQkey] = oldmetadata_Qplq[oldQkey]
    np.save('./trData/%sM_Qpc'%reductionfolder,(M_Qpc,metadata_Qplq))