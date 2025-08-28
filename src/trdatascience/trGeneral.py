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
from operator import itemgetter as getter
from datetime import date

def f_imports(file = 'imports.txt'):
    with open(file,'r') as f:
        imports = f.read()
    return imports

def dictcast(di,keys):
    keys = np.array(keys)
    try:
        out = np.array(list(getter(*keys)(di)))
    except:
        out = np.array(list(getter(*keys)(di)), dtype=object)
    return out
    
def dictcasttodict(di,keys):
    return dict(zip(keys,dictcast(di,keys)))

def f_combinedictionaries(dict1,dict2,overwrite=True): # if overwrite=True, if dict2 has index that matches dict1, it will overwrite the dict1 entry with the dict2 entry. if overwrite=False, it will disregard the dict2 entry.
    keys1 = dict1.keys()
    keys2 = dict2.keys()
    if overwrite:
        combdict = dict1
        for i, key2 in enumerate(keys2):
            combdict[key2] = dict2[key2]
    else:
        combdict = dict2
        for j, key1 in enumerate(keys1):
            combdict[key1] = dict1[key1]
    return combdict

def getdate():
    today = date.today()
    datestr = today.isoformat()
    datestr = datestr[5:]+datestr[:5]
    datestr = datestr.replace('-','')
    return datestr

def concatenatestringsalongaxis1(strings_sq): # 2d array of strings, must be regular array (each array must have same length
    flippedstrings = np.flip(strings_sq,axis=0)
    string_q = np.array(np.repeat('',len(strings_sq[0])),dtype='<U20')
    for i,strr in enumerate(flippedstrings):
        string_q = np.char.add(strr,string_q)
    return string_q

def f_makefolder(foldername):
    try:
        os.mkdir(foldername)
    except:
        print(foldername + ' already made')