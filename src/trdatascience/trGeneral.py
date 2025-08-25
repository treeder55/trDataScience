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