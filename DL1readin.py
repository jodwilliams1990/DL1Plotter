#CODE TO READ IN DL1 FILES AND PLOT - data read in
from CHECLabPy.core.io import DL1Reader
import numpy as np
from matplotlib import pyplot as plt
import scipy.fftpack
import os
import re
import seaborn as sns
from scipy.stats import norm

# HOW MANY EVENTS, HOW MANY DATASETS ####################################################
def howmanyh5datasets(onlyone,maxdatain,datainlist,pathdir):     
    dataincounta=-1
    if onlyone==0:
        stronly=0                               #OUTPUT
        for datain in range (0,maxdatain):
            str=datainlist[datain]            #OUTPUT
            strtest=str.endswith("h5")
            if strtest == True:
                dataincounta=dataincounta+1     #OUTPUT
    if onlyone==1:
        dataincounta=0                          #OUTPUT
        str="data_Run030_dl1.h5"            #OUTPUT    
        stronly="data_Run030_dl1.h5"            #OUTPUT
    openfirst=0
    datain=0
    while openfirst==0:
        strtest=datainlist[datain].endswith("h5")
        if strtest == True:
            path=(pathdir +'/'+datainlist[datain])
            readert = DL1Reader(path)            #OUTPUT
            ievt = readert.select_column('iev').values            #OUTPUT
            maxievt=max(ievt)            #OUTPUT
            openfirst=1
        else:
            datain=datain+1                      
    return {'onlyone':onlyone, 'dataincounta':dataincounta, 'str':str, 'maxievt':maxievt,'ievt':ievt,'readert':readert,'stronly':stronly}

def howmanytxtfiles(onlyone,maxdatain,datainlist):     
    dataincountb=-1
    stronly2=0
    for datain in range (0,maxdatain):
        str=datainlist[datain]
        if onlyone==1:
            str="data_Run029_runlog.txt"
            stronly2="data_Run029_runlog.txt"           
        strtest=str.endswith("txt")
        if strtest == True:
            dataincountb=dataincountb+1
    if onlyone==1:
        dataincountb=0
    return {'dataincountb':dataincountb, 'stronly2':stronly2}


