#CODE TO READ IN DL1 FILES AND PLOT
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import norm

plt.ioff()

def NSBAmpPE(pathdir,NSBPE,NSB,v,PEm):
    f4=plt.figure(4)
    plotpiccount=0
    readinarray=[0,1,2,3,4,5,7,8]
    NSBt=[40,80,125,250,400,500,800,1000]    
    NSBPEt=np.zeros((8,10))
    if v==1:
        for l in range (0,8):
            for l2 in range (0,10):
                NSBPEt[l,l2]=NSBPE[readinarray[l],l2]
        for j in range (0,4):
            plt.plot(NSB,NSBPE[:,j])
        for j in range (4,8):
            plt.plot(NSBt,NSBPEt[:,j])
        plt.xlabel('NSB (MHz)')
        plt.ylabel('Amplitude (mV)')
        plt.title('PE versus amplitude for different PE')
    if v==2:
        for k in range (0,8):
            plt.plot(PEm,NSBPEt[k,:],legend='%d Mhz' %NSBt[k])
        plt.xlabel('PE')
        plt.ylabel('Amplitude (mV)')        
        plt.title('PE versus amplitude for different NSB')
    if v==3:
        f4=sns.heatmap(NSBPEt)
        plt.xlabel("increasing PE")
        plt.ylabel("Decreasing NSB")
        plt.title('NSB, PE, amplitude')
    plt.savefig(pathdir+ r'\Plots\PEAgainstCharge_' + str(plotpiccount) + '_' + str(v) +'.png')
    return {'f4':f4, 'v':v}
def PeakChargeAgainstPE(pathdir,textreadin,chargecounttot,maxcharge):    #-------------Peak charge against PE
    plotpiccount=0
    f2=plt.figure(2)
    plt.plot(textreadin[2,0:(len(chargecounttot[0]))],maxcharge[0:(len(chargecounttot[0]))])
    plt.xlabel('PE')
    plt.ylabel('Max Charge (mVns)')
    plt.title('PE against Peak Charge')
    plt.savefig(pathdir+ r'\Plots\PEAgainstCharge_' + str(plotpiccount) + '.png')
    return f2
def ChargeAgainstPE(pathdir,chargecounttot,alldata,maxcharge):    #--------------------Charge vs Amplitude
    plotpiccount=0
    f3=plt.figure(3)
    for d in range (0,len(chargecounttot[0])):
        plt.scatter(alldata[1,1,d,0:59],maxcharge[0:59])
    plt.xlabel('Amplitude (mV)')
    plt.ylabel('Peak Charge (mVns)')
    plt.title('Amplitude against Peak Charge')
    plt.savefig(pathdir+ r'\Plots\AmplitudeAgainstCharge_' + str(plotpiccount) + '.png')
    return f3
def gainmatchingsigmameanevent(alldata):    #for each pixel
    dim=len(alldata[0][0][0])
    #dim2=len(alldata)
    sigmagm=np.zeros((8,8,dim))
    meangm=np.zeros((8,8,dim))
    for zp in range (0,dim):
        for yp in range (0,8):
            for xp in range (0,8):
                (mu,sigma)=norm.fit(alldata[xp,yp,:,zp])
                meangm[xp,yp,zp]=mu
                sigmagm[xp,yp,zp]=sigma                      
    gmsme=1
    return {'sigmagm':sigmagm, 'meangm':meangm,'gmsme':gmsme} 
def AmplitudeagainstPE(alldatapix1,alldata,textreadin):  #---------------------Amplitude against PE
    plotpic=1#0
    plotpiccount=0    
    numberofeventstoplot=len(alldatapix1[0]) #1 or all?
    f0=plt.figure(0)
    f0.clear()
    for plotcount in range (0,numberofeventstoplot):
        plt.scatter(textreadin[2,0:59],alldata[1,1,plotcount,0:59])
    plt.plot(textreadin[2,0:59],textreadin[2,0:59]) #1mV/PE line
    plt.xlabel('PE',fontsize=14)
    plt.ylabel('Amplitude(mV)',fontsize=14)
    plt.title('Amplitude against PE',fontsize=14)    
    plt.savefig(r'C:\Users\Jamie Williams\Desktop\New folder\d2018-10-04-TM-NSB\Plots\AmplitudeAgainstPE_' + str(plotpiccount) + '.png')
    if plotpic==1:
        plt.show()
    plt.close()
    return {'f0':f0, 'noeventplot':numberofeventstoplot}
def Tempmeangm(textreadin,meangm,chargecounttot,v):   #------------------------Pixel amplitude against temperature
    f21=plt.figure(21) 
    f21.clear()
    count3lim=len(meangm[0]) 
    plotpic=1#0
    plotpiccount=0
    if v==1:
        for i in range (0,count3lim):
            for j in range (0,count3lim):
                plt.scatter(textreadin[3,:],meangm[i,j,:])
                plt.xlabel('Temp (Celsius)',fontsize=14)
                plt.ylabel('Amplitude (mV)',fontsize=14)
                plt.title('Primary Temp vs Average Pixel Amplitude',fontsize=24)
    if v==2:  
        for i in range (0,count3lim):
            for j in range (0,count3lim):
                plt.scatter(textreadin[4,:],meangm[i,j,:])
                plt.xlabel('Temp (Celsius)',fontsize=14)
                plt.ylabel('Amplitude (mV)',fontsize=14)
                plt.title('Aux Temp vs Average Pixel Amplitude',fontsize=24)
    if v==3:  
        for i in range (0,count3lim):
            for j in range (0,count3lim):
                plt.scatter(textreadin[5,0:5],meangm[i,j,0:5])
                plt.scatter(textreadin[5,6:7],meangm[i,j,7:8])
                plt.xlabel('Temp (Celsius)',fontsize=14)
                plt.ylabel('Amplitude (mV)',fontsize=14)
                plt.title('PSU Temp vs Average Pixel Amplitude',fontsize=24)
    if v==4:  
        for i in range (0,count3lim):
            for j in range (0,count3lim):
                plt.scatter(textreadin[6,0:5],meangm[i,j,0:5])
                plt.scatter(textreadin[6,6:7],meangm[i,j,7:8])
                plt.xlabel('Temp (Celsius)',fontsize=14)
                plt.ylabel('Amplitude (mV)',fontsize=14)
                plt.title('Mean Pixel Amplitude against Si Temperature',fontsize=14)
    plt.savefig(r'C:\Users\Jamie Williams\Desktop\New folder\d2018-10-04-TM-NSB\Plots\TempAmp_'+ str(v) +'_' + str(plotpiccount) + '.png')
    if plotpic==1:
        plt.show()
    plt.close()
    return {'f21':f21, 'count3lim':count3lim}
def ampchargewaveformcom(chargecounttot,amppulsecounttot,waveformrmscounttot,waveformmeancounttot,v):
    f16=plt.figure(16)
    f16.clear()
    count3lim=len(chargecounttot[0])
    if v==1:    
        print('Amplitude vs Charge')
        for count3 in range (0,count3lim):
            plt.scatter(chargecounttot[0:55000,count3],amppulsecounttot[0:55000,count3])
            plt.xlabel("Charge (mVns)")
            plt.ylabel("Amplitude(mV)")
    if v==2:
        print('Amplitude vs Waveform RMS')
        for count3 in range (0,count3lim):
            plt.scatter(waveformrmscounttot[0:55000,count3],amppulsecounttot[0:55000,count3])
            plt.xlabel('Waveform RMS (mV)')
            plt.ylabel('Amplitude(mV)')
    if v==3:
        print('Amplitude vs Waveform Mean')
        for count3 in range (0,count3lim):
            plt.scatter(waveformmeancounttot[0:55000,count3],amppulsecounttot[0:55000,count3])
            plt.xlabel('Waveform Mean (mV)')
            plt.ylabel('Amplitude(mV)')
    if v==4:
        print('Waveform Mean vs Waveform RMS')
        for count3 in range (0,count3lim):
            plt.scatter(waveformmeancounttot[0:55000,count3],waveformrmscounttot[0:55000,count3])
            plt.xlabel('Waveform Mean (mV)')
            plt.ylabel('Waveform RMS (mV)')
    plt.show()
    return {'f16':f16, 'count3lim':count3lim}
def ChargeDistPlot(charge,onlyone,chargecounttot):    #------------------------Charge distribution
    f1=plt.figure(1)
    hist=[]
    edges=[]
    gainmatching=0
    plotpic=1#0
    plotpiccount=0
    if onlyone==1 and gainmatching ==0:
        hist, edges, _ = plt.hist(chargecounttot[0:len(charge),0], bins=250)
        between = (edges[1:] + edges[:-1]) / 2
        max_ = between[hist.argmax()]
        maxcharge=max_
    if onlyone==0:
        maxcharge=np.zeros((len(chargecounttot[0])))   
        for d in range (0,len(chargecounttot[0])):
            hist, edges, _ = plt.hist(chargecounttot[:,d], bins=250)
            between = (edges[1:] + edges[:-1]) / 2
            max_ = between[hist.argmax()]
            maxcharge[d]=max_
            max_ok=1         
    plt.xlabel('Charge (mVns)',fontsize=14)
    plt.ylabel('Counts',fontsize=14)
    plt.title('Charge Histogram',fontsize=24)
    plt.savefig(r'C:\Users\Jamie Williams\Desktop\New folder\d2018-10-04-TM-NSB\Plots\ChargeHistogram_'+ str(plotpiccount) + '.png')
    if plotpic==1:
        plt.show()
    return {'f1':f1,'maxcharge':maxcharge,'max_ok':max_ok}
def sigmagmineachpixel(pathdir,sigmagm,v): #-------------------------------------------Variance in each gain matched pixel
    plotpic=1#0
    plt.ioff()
    f12 = plt.figure(figsize=(10, 10))     # size of plot
    ax = f12.add_subplot(111)              # size of overall image
    im = ax.imshow(sigmagm[:,:,v], origin='lower')
    plt.colorbar(im)
    complete=1    
    plt.title('Gain Match Sigma in Each Pixel for Event %d' % v , fontsize=14)
    plt.savefig(pathdir+ r'\Plots\GMSigmaeachpixel_'+ str(v) + '.png')
    if plotpic==1:
        f12.show()
    plt.close()
    return {'f12':f12,'complete':complete} 
def meangmspgm(meangmsp,mvpe,h):   #-------------------------------------------Difference in gain matching from expected
    plotpic=1#0
    f15 = plt.figure(figsize=(10, 10))     # size of plot
    ax = f15.add_subplot(111)              # size of overall image
    im = ax.imshow((meangmsp[:,:,h]-mvpe), origin='lower')
    plt.colorbar(im)
    plt.title('Difference from Expected Gain Match in each pixel for Event %d' % h , fontsize=14)
    plt.savefig(r'C:\Users\Jamie Williams\Desktop\New folder\d2018-10-04-TM-NSB\Plots\MeanGMeachpixelDifffromGM_'+ str(h) + '.png')
    if plotpic==1:
        f15.show()
    plt.close()
    return {'f15':f15, 'h':h}
def NSBmeangm(pathdir,NSB,meangm):    #------------------------------------------------Mean gain matched amplitude against NSB 
    plotpiccount=0
    plotpic=1#0
    f20=plt.figure(20)
    f20.clear()    
    for i in range (0,8):
        for j in range (0,8):
            plt.scatter(NSB,meangm[i,j,:])
    plt.xlabel('NSB',fontsize=14)
    plt.ylabel('Amplitude (mV)',fontsize=14)
    complete=1
    plt.title('Amplitude against NSB',fontsize=24)
    plt.savefig(pathdir+ r'\Plots\NSBMeanAmp_'+ str(plotpiccount) + '.png')
    if plotpic==1:
        plt.show()
    return {'f20':f20,'complete':complete}
def meangmineachpixel(pathdir,meangm,v):      #----------------------------------------Gain Matched Amplitude for Each Pixel
    plotpic=1#0
    plt.ioff()
    f13 = plt.figure(figsize=(10, 10))     # size of plot
    ax = f13.add_subplot(111)              # size of overall image
    im = ax.imshow(meangm[:,:,v], origin='lower')
    plt.colorbar(im) 
    complete=1
    plt.title('Mean Amplitude in Each Pixel for Event %d' % v , fontsize=14)
    plt.savefig(pathdir + r'\Plots\MeanGMeachpixel_'+ str(v) + '.png')
    if plotpic==1:
        f13.show()
    plt.close()
    return {'f13':f13, 'complete':complete}
def meangmsp1(meangm,noevents):      #-----------------------------------------Gain Matched Amplitude for Each Super Pixel
    meangmsp=np.zeros((4,4,noevents))
    for h in range (0,noevents):
        for i in range (0,4):
            for j in range (0,4):
                fromx = 2*i
                tox = (2*i)+1
                fromy = 2*j
                toy=(2*j)+1
                meangmsp[i,j,h]=0.25*(meangm[fromx,fromy,h]+meangm[fromx,toy,h]+meangm[tox,fromy,h]+meangm[tox,toy,h])
    return meangmsp
def meangmsp2(pathdir,meangmsp,h):      #----------------------------------------------Gain Matched Amplitude for Each Super Pixel
    plotpic=1#0
    f14 = plt.figure(figsize=(10, 10))     # size of plot
    f14.clear()
    ax = f14.add_subplot(111)              # size of overall image
    im = ax.imshow(meangmsp[:,:,h], origin='lower')
    plt.colorbar(im)
    plt.title('Mean Gain Matched Super Pixel Amplitude for Event %d' % h , fontsize=14)
    plt.savefig(pathdir + r'\Plots\MeanGMsuperpixel_'+ str(h) + '.png')
    if plotpic==1:
        f14.show()
    plt.close()
    return {'f14':f14, 'h':h}
def imagemasking2d(image):   #-------------------------------------------------Masking the camera, 2D
    image[0:8, 40:48] = np.ma.masked       # Ensure no events where there are no TMs
    image[0:8, 0:8] = np.ma.masked
    image[40:48, 0:8] = np.ma.masked
    image[40:48, 40:48] = np.ma.masked
    return image
def imagemasking3d(ievcount,ievcountstart,image):   # -------------------------Masking the camera, 3D         
    image[0:8, 40:48,ievcount-ievcountstart] = np.ma.masked   # Ensure no events where there are no TMs
    image[0:8, 0:8,ievcount-ievcountstart] = np.ma.masked
    image[40:48, 0:8,ievcount-ievcountstart] = np.ma.masked
    image[40:48, 40:48,ievcount-ievcountstart] = np.ma.masked
    return image
def addanarrow(m,ax,image):    # Arrow start and end ************************************
    axl = m.metadata['fOTUpCol_l']          # base x
    ayl = m.metadata['fOTUpRow_l']          # base y
    adx = m.metadata['fOTUpCol_u'] - axl    # arrowhead x
    ady = m.metadata['fOTUpRow_u'] - ayl    # arrowhead y
    ax.arrow(axl, ayl, adx, ady, head_width=1, head_length=1, fc='r', ec='r')
    text = "ON-Telescope UP"
    ax.text(axl, ayl, text, fontsize=8, color='r', ha='center', va='bottom')
    return image