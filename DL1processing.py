from CHECLabPy.core.io import DL1Reader
import numpy as np
from matplotlib import pyplot as plt
import scipy.fftpack
import os
import re

def datafft(ievcountend,ievcountstart,image):
    N = ievcountend-ievcountstart                # Number of samplepoints
    T = 1.0/8000                                 # sample spacing
    x = np.linspace(0.0, N*T, N)
    y = image[7,0,:]
    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N/2)                
    fig, ax = plt.subplots()
    ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
# DATA ARRAY ALLOCATION ####################################################
def textallocation(i,da,correc,textinputtype,textreadin,textin,a):
    if i==0:
        textreadin[i,textin]=float(a[-4:-1])         #0
    if i==1:
        textreadin[1,textin]=float(a[-5:-1])         #1
    if i==2+correc and correc==0 and textinputtype==2:
        textreadin[2,textin]=float(a[11:-4])         #2
    if i==4+correc:
        hasht=[x.start() for x in re.finditer('\t',a)]
        textreadin[3,textin]=float(a[:hasht[0]])            #3
        textreadin[4,textin]=float(a[hasht[0]:hasht[1]])         #4
        textreadin[5,textin]=float(a[hasht[1]:hasht[2]])         #5
        textreadin[6,textin]=float(a[hasht[2]:hasht[3]])         #6
        textreadin[7,textin]=float(a[hasht[3]:hasht[4]])         #7
        textreadin[8,textin]=float(a[hasht[4]:hasht[5]])         #8                                                         
    if i>4+correc and i<21+correc:
        textreadin[i+da,textin]=float(a[8:-1])  #9-25
    if i>20+correc and i<85+correc:
        textreadin[i+da,textin]=float(a[-4:-1])  #26-89
    if i>84+correc:
        textreadin[i+da,textin]=float(a[-4:])    #90                      
    i=i+1 
    return textreadin
def firsteventdataallocation(n_cols,n_rows,alldata,dataincount,image):
    for x in range (0,n_cols):
        for y in range (0,n_rows):
            alldata[x,y,0,dataincount]=image[x,y]   
    return alldata
def alleventsdataallocation(image,maxievt,n_cols,n_rows,alldata,dataincount,alldatapix1):
    dim3=len(image[0][0])
    if dim3>maxievt:
        dim3=maxievt
    for x in range (0,n_cols):
        for y in range (0,n_rows):
            for ievrange in range (0,dim3):#(ievcountend-ievcountstart)):
                alldata[x,y,ievrange,dataincount]=image[x,y,ievrange]                           
    for y in range (0,n_rows):      #x=1
        for ievrange in range (0,dim3):#(ievcountend-ievcountstart)):
            alldatapix1[y,ievrange,dataincount]=image[1,y,ievrange]
    aeda=1
    return {'aeda':aeda,'alldata':alldata,'alldatapix1':alldatapix1}