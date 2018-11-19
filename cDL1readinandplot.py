#CODE TO READ IN DL1 FILES AND PLOT
from CHECLabPy.core.io import DL1Reader
#from IPython.display import display
import numpy as np
from matplotlib import pyplot as plt
import os
import DL1readin
import DL1processing
import DL1plotting

pathdir = r'C:\Users\Jamie Williams\Desktop\New folder\NSB200PE'
datainlist=(os.listdir(pathdir))
maxdatain = len(datainlist)
plt.ioff()

# READ IN DATA ####################################################
def loadmultiplecolumns():
    pixel, charge,t_event = reader.select_columns(['pixel', 'charge', 't_pulse'])
    charge = charge.values # Convert from Pandas Series to numpy array
    pixel = pixel.values # Convert from Pandas Series to numpy array
    return {'charge':charge, 'pixel':pixel}
def loadmultiplerows():
    for row in reader.iterate_over_rows():
        break     # keep in for first row only
    row
    return row
def readevents():
    for df in reader.iterate_over_rows():
        break     # keep in for first event only
    df
    return df
def chargehistreadin():
    charge = reader.select_column('charge',start=0, stop=nrows) #only bring in the data we want
    amp_pulse = reader.select_column('amp_pulse',start=0, stop=nrows) #only bring in the data we want
    waveform_rms = reader.select_column('waveform_rms',start=0, stop=nrows) #only bring in the data we want
    waveform_mean = reader.select_column('waveform_mean',start=0, stop=nrows) #only bring in the data we want
    charge = charge.values # Convert from Pandas Series to numpy array
    amp_pulse = amp_pulse.values # Convert from Pandas Series to numpy array
    waveform_rms = waveform_rms.values # Convert from Pandas Series to numpy array
    waveform_mean = waveform_mean.values # Convert from Pandas Series to numpy array
    print(dataincount)
    dim3=len(charge)
    if dim3>len(chargecounttot):
        dim3=len(chargecounttot)
    for count2 in range (0,dim3):
        chargecounttot[count2,dataincount-1]=charge[count2]
        cctok=1
        amppulsecounttot[count2,dataincount-1]=amp_pulse[count2]
        waveformrmscounttot[count2,dataincount-1]=waveform_rms[count2]
        waveformmeancounttot[count2,dataincount-1]=waveform_mean[count2] 
    return {'charge':charge, 'amp_pulse':amp_pulse,'waveform_rms':waveform_rms,'waveform_mean':waveform_mean} 
def rowandcolumnextractor(datain):
    row = m['row'].values
    col = m['col'].values
    n_rows = m.metadata['n_rows']
    n_cols = m.metadata['n_columns']
    return {'row':row, 'col':col,'n_rows':n_rows,'n_cols':n_cols}
########################################################################################  
# END OF DEFINITIONS ###################################################################  
########################################################################################  

# READ IN INITIAL DATA TO GENERATE ARRAY SIZES ################################ 
howmanyh5datasets=DL1readin.howmanyh5datasets(0,maxdatain,datainlist,pathdir) 
onlyone=howmanyh5datasets['onlyone']    #print(onlyone)
dataincounta=howmanyh5datasets['dataincounta']
str=howmanyh5datasets['str']
maxievt=howmanyh5datasets['maxievt']
ievt=howmanyh5datasets['ievt']
readert=howmanyh5datasets['readert']
stronly=howmanyh5datasets['stronly']
howmanytxtfiles=DL1readin.howmanytxtfiles(0,maxdatain,datainlist)
dataincountb=howmanytxtfiles['dataincountb']
stronly2=howmanytxtfiles['stronly2']
###############################################################################        

# INITIALISATION  #############################################################
h5in=1                  #1  #READ IN H5 FILES? 
readinonly=0            #4  #read in h5 file - columns specified in the loop (~line 65) 
readinrow=0             #4  #read through each row, for each of the 64 pixels
readthroughevents=0     #4  #read through each of the events
plot2=0                 #4  #Read in and plot charge [or other parameter] histogram for a specific 
plotcharge=0            #4  #look at charge 
plotwfm=1               #4  #plot waveform mean (or other parameter) for each pixel, event and dataset
iterate=10             #5  #read in more than just the first event [1000000] or [1] - may change in loop
done=1000000           #5  #read in first H5 file [1] or multiple [1000000] kill the loop through all of the datasets after the first read in text file - may change in loop
plotongraph=0           #6  #plot on camera image (for each event, each data run unless iterate/done say otherwise)
ffton=0                 #4  #Fourier Transform of pixel data [MORE WORK TO DO]
diffNSB=0               #4  #need to create NSBcount=0 outside
NSBcount=1              #5  #While file is the data allocated to? Corresponding to which PE?
PEcount=0               #5  #While file is the data allocated to? Corresponding to which PE?
if diffNSB==1 and NSBcount==0:
    NSBPE=np.zeros((9,10))
    NSB=[40,80,125,250,400,500,700,800,1000]
    PEm=[0,100,200,300,500,700,900,1100,1300,1500]
txtin=1#0#1     # Read Text Files in associated with each dataset
textinputtype=2     # Parameters to read in text file successfully
sm=0        # single module [1] or full camera [default=0] - will change to single module if 8x8 pixel found
plotchargehistogram=0
PCAPE=0
cctok=0
max_ok=0
ampchargewaveformcomp=0
tempdatain=0
gmsme=0
aeda=0
meangmok=0 #super pixel calculation
meangmspgmok=0

###############################################################################        
# READ IN H5 FILE #############################################################
if h5in==1:
    alldata=np.zeros((8,8,maxievt,dataincounta+1))
    alldatapix1=np.zeros((8,maxievt,dataincounta+1))
    chargecounttot=np.zeros(((maxievt+1)*64,dataincountb+1))  # Number of events x number of datasets x number of pixels
    amppulsecounttot=np.zeros(((maxievt+1)*64,dataincountb+1))
    waveformrmscounttot=np.zeros(((maxievt+1)*64,dataincountb+1))
    waveformmeancounttot=np.zeros(((maxievt+1)*64,dataincountb+1))
    dataincount=-1
## Read in each file
    for datain in range (0,maxdatain):
        str=datainlist[datain]
        strtest=str.endswith("h5")
### If the file is a h5 file
        if strtest == True and iterate>0 and done>0:# and str==stronly:
            dataincount=dataincount+1
            path=(pathdir +'/'+datainlist[datain]) #selectfile
            reader = DL1Reader(path)
            #print(reader.metadata)
            #print(m)
            a=reader.load_entire_table()
            reader.load_entire_table()
            iev = reader.select_column('iev').values  # other variables that can be imported - t_cpu_ns, t_cpu_sec, t_tack, first_cell_id, baseline_start_mean, baseline_start_rms, baseline_end_mean, baseline_end_rms, baseline_subtracted, t_event, charge, t_pulse, amp_pulse, fwhm, tr, waveform_mean, waveform_rms, saturation_coeff                       
            miniev=min(iev)
            maxiev=max(iev)
            nrows=reader.n_rows        # number of rows of data in the dl1 file         
            if readinonly==1:          #### Load multiple columns with the select_columns method   
                loadmultiplecolumns()
            if readinrow==1:           #### Load each row for each of the 64 pixels   
                loadmultiplerows()            
            if readthroughevents==1:   #### Load each event for 1 pixel
                readevents()
            if plot2==1:               #### Charge Histogram. X is integ(V(t)dt) in mVns against count on Y               
                chargehistreadin()      #Read in column and histogram
                ampchargewaveformcomp=1
                plotchargehistogram=1
            if plotcharge==1: # PLOT ON THE CAMERA. #### Plot Charge [or other parameter] against pixels in the camera
                m = reader.mapping
                #print (m)   #prints all of the experimental setup data
                #print(reader.columns)   #prints all available data types
                charge = reader.select_column('charge').values    # other variables that can be imported - t_cpu_ns, t_cpu_sec, t_tack, first_cell_id, baseline_start_mean, baseline_start_rms, baseline_end_mean, baseline_end_rms, baseline_subtracted, t_event, charge, t_pulse, amp_pulse, fwhm, tr, waveform_mean, waveform_rms, saturation_coeff                       
                iev = reader.select_column('iev').values
                if iterate ==1:         ##### Read in only the first event
                    ievcount=10
                    charge = charge[iev == ievcount]  #changes event number. Can iterate through?
                    #DO NOT CHANGE ****************************************************    
                    rowandcolumnextracto=rowandcolumnextractor()
                    n_rows=rowandcolumnextracto['n_rows']
                    n_cols=rowandcolumnextracto['n_cols']
                    col=rowandcolumnextracto['col']
                    row=rowandcolumnextracto['row']
                    if n_rows==8:
                        sm=1
                    image = np.ma.zeros((n_rows, n_cols))  
                    image[row, col] = charge               # charge from each pixel. Brings in data correspnding to TM (row, col) for each SP. Can extract for each SP at each trigger. 
                    if sm==0:
                        DL1plotting.imagemasking2d(image)
                    fig = plt.figure(figsize=(10, 10))     # size of plot
                    ax = fig.add_subplot(111)              # size of overall image
                    im = ax.imshow(image, origin='lower')
                    fig.colorbar(im)
                    # Arrow start and end ***************************************************
                    DL1plotting.addanarrow()
                    plt.show()
                if iterate >1:  # READS THROUGH EVERY EVENT (FOR ALL H5 FILES UNLESS "DONE" SAYS OTHERWISE) ##############        
                    charget=charge    
                    ievcountstart=0         # READ IN WHICH EVENTS (Start)
                    ievcountend = maxiev    # READ IN WHICH EVENTS (End)
                    rowandcolumnextracto=rowandcolumnextractor()
                    n_rows=rowandcolumnextracto['n_rows']
                    n_cols=rowandcolumnextracto['n_cols']
                    col=rowandcolumnextracto['col']
                    row=rowandcolumnextracto['row']
                    if n_rows==8:           # SINGLE MODULE OR FULL CAMERA? 
                        sm=1
                    image = np.ma.zeros((n_rows, n_cols,ievcountend-ievcountstart))
                    for ievcount in range (ievcountstart, ievcountend):                    
                        charge = charget[iev == ievcount]  #changes event number. Can iterate through?
                        #DO NOT CHANGE ****************************************************                  
                        image[row, col,ievcount-ievcountstart] = charge               # charge from each pixel. Brings in data correspnding to TM (row, col) for each SP. Can extract for each SP at each trigger. 
                        if sm==0:
                            DL1plotting.imagemasking3d(image)
                        if plotongraph==1:
                            fig = plt.figure(figsize=(10, 10))     # size of plot
                            ax = fig.add_subplot(111)              # size of overall image
                            im = ax.imshow(image[:,:,ievcount-ievcountstart], origin='lower')
                            fig.colorbar(im)
                            DL1plotting.addanarrow()
                            plt.show
#### Plot waveform mean (or other camera parameter) for each pixel, dataset, event
            if plotwfm==1: # PLOT ON THE CAMERA. 
                m = reader.mapping
                #print (m)   #prints all of the experimental setup data
                #print(reader.columns)   #prints all available data types
                waveform_mean = reader.select_column('amp_pulse').values       # other variables that can be imported - t_cpu_ns, t_cpu_sec, t_tack, first_cell_id, baseline_start_mean, baseline_start_rms, baseline_end_mean, baseline_end_rms, baseline_subtracted, t_event, charge, t_pulse, amp_pulse, fwhm, tr, waveform_mean, waveform_rms, saturation_coeff                       
                iev = reader.select_column('iev').values
                if iterate ==1:     ##### Read in only the first event
                    ievcount=10
                    waveform_mean = waveform_mean[iev == ievcount]  #changes event number. Can iterate through?
                    #DO NOT CHANGE ****************************************************    
                    rowandcolumnextracto=[]
                    rowandcolumnextracto=rowandcolumnextractor(datain)
                    n_rows=rowandcolumnextracto['n_rows']
                    n_cols=rowandcolumnextracto['n_cols']
                    col=rowandcolumnextracto['col']
                    row=rowandcolumnextracto['row']
                    if n_rows==8:
                        sm=1
                    image = np.ma.zeros((n_rows, n_cols))  
                    image[row, col] = waveform_mean               # charge from each pixel. Brings in data correspnding to TM (row, col) for each SP. Can extract for each SP at each trigger. 
                    if plotongraph==1: ###### Plot data on graph
                        if sm==0:
                            DL1plotting.imagemasking2d(image)
                        fig = plt.figure(figsize=(10, 10))     # size of plot
                        ax = fig.add_subplot(111)              # size of overall image
                        im = ax.imshow(image, origin='lower')
                        fig.colorbar(im)
                        DL1plotting.addanarrow(image)
                        plt.show()
##### Read in all events
                if iterate >1:          
                    waveform_meant=waveform_mean    
                    ievcountstart=0
                    ievcountend = maxiev
                    rowandcolumnextracto=rowandcolumnextractor(datain)
                    n_rows=rowandcolumnextracto['n_rows']
                    n_cols=rowandcolumnextracto['n_cols']
                    col=rowandcolumnextracto['col']
                    row=rowandcolumnextracto['row']
                    if n_rows==8:
                        sm=1
                    image = np.ma.zeros((n_rows, n_cols,ievcountend-ievcountstart))
                    for ievcount in range (ievcountstart, ievcountend):
                        waveform_mean = waveform_meant[iev == ievcount]  #changes event number. Can iterate through?
                        #DO NOT CHANGE ****************************************************                  
                        image[row, col,ievcount-ievcountstart] = waveform_mean               # charge from each pixel. Brings in data correspnding to TM (row, col) for each SP. Can extract for each SP at each trigger. 
                        if sm==0:
                            DL1plotting.imagemasking3d(ievcount,ievcountstart,image)
###### Plot data on graph 
                        if plotongraph==1:
                            fig = plt.figure(figsize=(10, 10))     # size of plot
                            ax = fig.add_subplot(111)              # size of overall image
                            im = ax.imshow(image[:,:,ievcount-ievcountstart], origin='lower')
                            fig.colorbar(im)
                            # Arrow start and end ***************************************************
                            DL1plotting.addanarrow(m,ax,image)
                            plt.show                            
###############################################################################
## FOURIER TRANSFORM OF PIXEL DATA ############################################
            if ffton==1:
                DL1processing.datafft(ievcountend,ievcountstart,image)
                plt.show()
###############################################################################
## ALLOCATE DATA TO ARRAY PER DATASET (first event only) ######################
            if iterate ==1:                
                alldata=DL1processing.firsteventdataallocation(n_cols,n_rows,alldata,dataincount,image)             
###############################################################################
# ALLOCATE DATA TO ARRAY PER DATASET (all events) #############################
            if iterate>1 and plotwfm==1:
                aeda1=DL1processing.alleventsdataallocation(image,maxievt,n_cols,n_rows,alldata,dataincount,alldatapix1)
                aeda=aeda1['aeda']
                alldata=aeda1['alldata']
                alldatapix1=aeda1['alldatapix1']
###############################################################################
## ALLOCATE DATA TO ARRAY PER DATASET (NSB Data in different sets) ############
            if diffNSB==1:
                NSBPE[NSBcount,PEcount]=np.mean(alldata[:,:,:,PEcount])
                PEcount=PEcount+1
        done=done-1
NSBcount=NSBcount+1
###############################################################################
#### -----> END OF H5 FILES <----- ############################################
###############################################################################

## READ IN ASSOCIATED TEXT FILES  #################################################################
if txtin==1:
    textreadin=np.zeros((100,dataincountb+1))
    textin=0
    for datain in range (0,maxdatain):
        str=datainlist[datain]
        strtest=str.endswith("txt")
        if strtest == True:# and stronly2==str:
            path=(pathdir +'/'+datainlist[datain])
            file=open(path,"r")
            i=0
            textinputtype=1
            for row in file:
                a=row
                da=4
                if textinputtype==1:                #WORKING FOR SET 2
                    correc=-1
                    textreadin=DL1processing.textallocation(i,da,correc,textinputtype,textreadin,textin,a)
                    i=i+1  
                if textinputtype==2:                #WORKING FOR SET 4
                    PCAPE=1
                    correc=0
                    textreadin=DL1processing.textallocation(i,da,correc,textinputtype,textreadin,textin,a)
                    i=i+1
            textin=textin+1                      
    tempdatain=1
if aeda==1:
    gaindict=DL1plotting.gainmatchingsigmameanevent(alldata)
    sigmagm=gaindict['sigmagm']
    meangm=gaindict['meangm']
    gmsme=gaindict['gmsme']

# GRAPH PLOTTING ########]]]7######################################################   
PEamp=1
picplot=1
if plotwfm==1 and txtin==1 and iterate>1 and done>0 and PEamp==1 and textinputtype==2: #parameter yes, textfile in yes, all events, all datasets, correct textfile read in
    DL1plotting.AmplitudeagainstPE(alldatapix1,alldata,textreadin)['f0'].show
if plot2==1 and plotchargehistogram==1: ######################################## If the charge distribution is read in and it produces data that can be plot
    CDP=DL1plotting.ChargeDistPlot(charge,onlyone,chargecounttot)
    CDP['f1'].show
    CDP['f1'].clear
    maxcharge=CDP['maxcharge']
    max_ok=CDP['max_ok']                 
if textinputtype==2 and PCAPE==1 and cctok==1 and max_ok==1: ################### If PE read in and ok to be plot
    DL1plotting.PeakChargeAgainstPE(textreadin,chargecounttot,maxcharge).show()
    DL1plotting.ChargeAgainstPE(chargecounttot,alldata,maxcharge).show()
if NSBcount==9 and diffNSB==1: ################################################ OK
    #picplot=0
    NSBAPE=DL1plotting.NSBAmpPE(NSBPE,NSB,1,PEm)
    NSBMGM=DL1plotting.NSBmeangm(NSB,meangm)
    if picplot==1:
        NSBAPE['f4'].show()
        NSBMGM['f20'].show
        NSBMGM.clear
if gmsme==1:#sigmaineachpixela==1: ############################################ OK
    #picplot=0
    vmax=len(sigmagm[0][0])
    for v in range (1,vmax):
        print('Sigma for Dataset ',v)
        sgie=DL1plotting.sigmagmineachpixel(sigmagm,v)
        if picplot==1:
            sgie['f12'].show()
            sgie.clear
if gmsme==1:#meangmineachpixel==1: ############################################ OK
    vmax=len(sigmagm[0][0])
    #picplot=0
    for v in range (1,vmax):
        print('Mean for Dataset ',v)
        mgie=DL1plotting.meangmineachpixel(meangm,v)
        if picplot==1:
            mgie['f13'].show()
            mgie.clear
if meangmok==1 and gmsme==1: ################################################## OK
    vmax=len(sigmagm[0][0])
    meangmsp=DL1plotting.meangmsp1(meangm,vmax)
    for v in range (0,vmax): 
        print('Mean Superpixels for Dataset ',v) 
        DL1plotting.meangmsp2(meangmsp,v)['f14'].show()
    meangmspgmok=1
if meangmspgmok==1: ########################################################### OK
    gm=200
    for v in range (0,vmax): 
        print('Mean Superpixels for Dataset ',v, ' for gain matching at ',gm, 'PE' ) 
        DL1plotting.meangmspgm(meangmsp,gm,v)['f15'].show() 
if plot2==1 and ampchargewaveformcomp==1: ##################################### OK
    for v in range (1,5):
        DL1plotting.ampchargewaveformcom(chargecounttot,amppulsecounttot,waveformrmscounttot,waveformmeancounttot,v)['f16'].show()                 
if txtin==1 and gmsme==1 :#and Tempmeangmok==1: ############################## OK
    for v in range (1,5):
        DL1plotting.Tempmeangm(textreadin,meangm,chargecounttot,v)['f21'].show()