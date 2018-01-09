# -*- coding: utf-8 -*-
"""
Created on Mon Jan 04 11:18:34 2016

@author: dng5
"""

import IgorBin
import Generic
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

PIXIS_background_filename ='Y:/Data/2017/April/12/PIXIS_12Apr2017_10003.ibw'
bgndImage=IgorBin.LoadIBW(PIXIS_background_filename)['Data']
PIXIS_background1=np.array(bgndImage[:,0:bgndImage.shape[1]/2], dtype=float)
PIXIS_background2=np.array(bgndImage[:,bgndImage.shape[1]/2:], dtype=float)
IsatCounts = 235.3
imagingTime=40.0

def zeroNansInfs(x):
    if x!=x:
        return 0
    elif np.isinf(x):
        return 0
    else:
        return x
zeroNansInfsVector = np.vectorize(zeroNansInfs, otypes=[np.float])

def processIBW(filename, angle=-41,IsatCounts=IsatCounts,imagingTime=imagingTime,bgnd1=PIXIS_background1,bgnd2=PIXIS_background2):
    """"
    Uses IgorBin.LoadIBW to load igor binary data into python, 
    calculates the optical depth and rotates according to specified angle.

    filename : the filename including path to open
    
    returns a dictionary:
        {"OptDepth": od, "rotOD": rotOD, "Note": Note}
    
    NoteString is the note from Igor
    od is a numpy array of the optical depth
    rotOD is a numpy array of the rotated optical depth
    """
    outDict={}
    filedict=IgorBin.LoadIBW(filename)
    camera = filename.split("/")[-1].split("_")[0]
    
    if camera == 'PIXIS':
        image = np.array(filedict['Data']*1.0)
        Raw1=np.zeros_like(PIXIS_background1)
        Raw1= image[:,0:image.shape[1]/2]-bgnd1
        outDict['Raw1']=Raw1
        Raw2= image[:,image.shape[1]/2:]-bgnd2
        outDict['Raw2']=np.array(Raw2, dtype='float')
        od = -np.log(Raw1.astype(float)/Raw2.astype(float))
        od = zeroNansInfsVector(od)
        odCorr = od+(Raw2-Raw1)/(IsatCounts*imagingTime)
        rotOD=ndimage.interpolation.rotate(od,angle)
        rotODcorr=ndimage.interpolation.rotate(odCorr,angle)
        rotRaw1=ndimage.interpolation.rotate(Raw1,angle)
        rotRaw2=ndimage.interpolation.rotate(Raw2,angle)
        outDict['OptDepth']=od
        outDict['rotOD']=rotOD
        outDict['Note']=filedict['Note']
        outDict['odCorr']=odCorr
        outDict['rotODcorr']=rotODcorr
        outDict['rotRaw1']=rotRaw1
        outDict['rotRaw2']=rotRaw2
    elif camera == 'Flea3':
        image = np.array(filedict['Data']*1.0)
        numImages=int(image.shape[1]/488)-2
        bgnd = image[:,-488:]
        Raw2 = image[:,-488*2:-488]-bgnd
        outDict['RawProbe']=Raw2
        for i in range(numImages):
            Raw1=image[:,i*488:(i+1)*488]-bgnd
            od = -np.log(Raw1.astype(float)/Raw2.astype(float))
            od = zeroNansInfsVector(od)
            rotOD=ndimage.interpolation.rotate(od,angle)
            outDict['od'+str(i+1)]=od
            outDict['Raw'+str(i+1)]=Raw1
            outDict['rotOD'+str(i+1)]=rotOD
        outDict['Note']=filedict['Note']
    else:
        print "Not a PIXIS or Flea3 file. Amend readIgor.py to handle this camera type"
        
    return outDict
    
def getIndexedWaves(infoString):
    """
    Extracts indexed waves and values from igor "Note" string
    
    infoString : Igor header string in usual lab format
    
    returns a dictionary, waveDict, with as many entries as there are indexed
            waves, in the format {"IndexedWave":IndexedValue}
    """
    
    waveDict = {}
    nameString = Generic.StringByKey('nan', 'IndexedWaves', infoString, '=', '\n')
    nameArray=np.array(nameString.split(';'))
    numWaves = nameArray.size
    
    valueString = Generic.StringByKey('nan', 'IndexedValues', infoString, '=', '\n')
    valueArray=np.array(valueString.split(';'))
    
    for i in np.arange(numWaves-1):
        waveDict[nameArray[i]]=np.float(valueArray[i])
        
    return waveDict
    
def countMultipleROIs(filename,roiList,background=True,bgndLoc="left",roiB=0,showImage=True,vmin=-0.2,vmax=2,weight=np.array([1.0,1.0,1.0])):
    """
    Counts the total number, as well as fraction, in multiple regions of interest
    from an optical depth.
    
    filename : the filename including path to open
    roiList : an array or list of list of regions of interest, in the format
                [[xmin1,xmax1,ymin1,ymax1],[xmin2,xmax2,ymin2,ymax2],...]
            There must be at least 2 rois.
    background : True (default) to perform a background subtraction for each roi
                 False not to perform a background subtraction
    bgndLoc : location of the background roi to be subtracted from each roi. 
                Options are: 'left' (default), 'right', 'top', 'bottom', or 'custom"
                if 'custom' is chosen, the roiB input should be a list of background
                rois of the same length as roiList
    roiB : if 'custom' is selected for bgndLoc, a list of background rois for each main roi,
            in the same order and format as roiList
    showImage : if True (default), will display an image of the optical depth with
                rectangles around the rois and dashed rectangles around the background
                rois (if any), with vmin and vmax as the colorbar limits
    vmin : if showImage is True, the minimum od value for the colormap
    vmax : if showImage is True, the maximum od value for the colormap                
    outputs a tuple of counts, fractions, outDict
        counts : list of the same length as the number of rois. number of counts in each roi, in order
        fractions : list of the same length as the number of rois. fraction of the total in each roi, in order
        outDict : the output of processIBW
    """
    outDict=processIBW(filename)
    od = outDict['rotOD']
    roiList=np.array(roiList)
    numRois = np.size(roiList,axis=0)
    counts=np.zeros(numRois)
    fractions=np.zeros(numRois)
    probe=np.zeros(numRois)
    
    if showImage:
        figure=plt.figure()
        panel=figure.add_subplot(1,1,1)
        panel.imshow(od,vmin=vmin,vmax=vmax)
        xmin,xmax=panel.get_xlim()
    
    for i in np.arange(numRois):
        roi=roiList[i,:]
        odLocal = od[roi[0]:roi[1],roi[2]:roi[3]]
        counts[i]=np.sum(odLocal)
        probe[i]=np.average(outDict['Raw2'][roi[0]:roi[1],roi[2]:roi[3]])
        
        if showImage:
            panel.axhspan(roi[0],roi[1],xmin=(roi[2]-xmin)/(xmax-xmin),xmax=(roi[3]-xmin)/(xmax-xmin),ls='solid', fill=False)

        if background:
            if bgndLoc=="left":
                roiB=np.array([roi[0],roi[1],2*roi[2]-roi[3],roi[2]]) 
            elif bgndLoc=="right":
                roiB=np.array([roi[0],roi[1],roi[3],2*roi[3]-roi[2]]) 
            elif bgndLoc=="bottom":
                roiB=np.array([roi[1],2*roi[1]-roi[0],roi[2],roi[3]])
            elif bgndLoc=="top":
                roiB=np.array([2*roi[0]-roi[1],roi[0],roi[2],roi[3]])
            elif bgndLoc=="custom":
                roiB=roiB[i,:]
            else:
                print "Background subtraction requested, but undefined backround roi"
                
            if showImage:
                panel.axhspan(roiB[0],roiB[1],xmin=(roiB[2]-xmin)/(xmax-xmin),xmax=(roiB[3]-xmin)/(xmax-xmin), ls='dashed', fill=False)
            
            bgndCounts=np.sum(od[roiB[0]:roiB[1],roiB[2]:roiB[3]])
            counts[i]-=bgndCounts
 #   countsW=weight*counts
    total=np.sum(counts)
    fractions = counts/total
    
    
    
    return (counts, fractions, outDict,probe)
    
def showImage(fileroot,filenum,vmin=-0.2,vmax=2.0):
    filename = fileroot + "_"+ str(filenum).zfill(4) + ".ibw"
    outDict=processIBW(filename)
    od = outDict['rotOD']
    
    if showImage:
        figure=plt.figure()
        panel=figure.add_subplot(1,1,1)
        panel.imshow(od,vmin=vmin,vmax=vmax)
        xmin,xmax=panel.get_xlim()
    
    
    return (od, outDict)

def batchCountMultipleROIs(fileroot,filestart,filestop,roiList,background=True,bgndLoc="left",roiB=0,showImage=False,weight=np.array([1.0,1.0,1.0])):
    numRois = np.size(roiList,axis=0)    
    filerange = np.arange(filestart,filestop+1)    
    
    
    filename = fileroot + "_"+ str(filestart).zfill(4) + ".ibw"
    outDict = processIBW(filename)
    infoString=outDict["Note"]
    waveDict=getIndexedWaves(infoString)
    waveDict=waveDict.fromkeys(waveDict.keys(),[])
        
    
    
    counts=np.zeros((filerange.size,numRois))
    fractions=np.zeros((filerange.size,numRois))
    probeAvg=np.zeros((filerange.size,numRois))
    i=0
    for filenum in filerange:
        filename = fileroot + "_"+ str(filenum).zfill(4) + ".ibw"
        counts[i,:], fractions[i,:], outDict,probeAvg[i,:] = countMultipleROIs(filename,roiList,background=background,bgndLoc=bgndLoc,roiB=roiB,showImage=showImage,weight=weight)
        infoString=outDict["Note"]
            
        waveDictLocal=getIndexedWaves(infoString)
        for k in waveDict.iterkeys():
            
            waveDict[k]=np.append(waveDict[k],waveDictLocal[k])
        i+=1    
    return counts, fractions, waveDict, probeAvg
    
def plotFracsByKey(fractions,waveDict,keyName,xlabel='',ylabel='Fractional Population'):
    if xlabel=='':
        xlabel=keyName
    figure=plt.figure()
    panel=figure.add_subplot(1,1,1)
    for i in range(fractions.shape[1]):
        panel.plot(waveDict[keyName],fractions[:,i],'o',label='frac'+str(i))#,'bo', label='mF=+1')
    panel.legend()
    panel.set_xlabel(xlabel)
    panel.set_ylabel(ylabel)
    return
    
def countUwaveNums(fileroot,filelist,roi):
    counts1=np.zeros(filelist.size)
    counts2=np.zeros(filelist.size)
    imbal=np.zeros(filelist.size)
    signalGood=np.zeros(filelist.size)
    probeCounts=np.zeros(filelist.size)
    roiB=np.array([roi[0],roi[1],2*roi[2]-roi[3],roi[2]])
    for ind, filenum in enumerate(filelist):
        filename=fileroot+"_"+ str(filenum).zfill(4) + ".ibw"
        dictFlea =processIBW(filename, angle=-38)
        probe=np.average(dictFlea['RawProbe'][roi[0]:roi[1],roi[2]:roi[3]])
        od1=np.sum(dictFlea['od1'][roi[0]:roi[1],roi[2]:roi[3]])
        od2=np.sum(dictFlea['od2'][roi[0]:roi[1],roi[2]:roi[3]])
        od1B=np.sum(dictFlea['od1'][roiB[0]:roiB[1],roiB[2]:roiB[3]])        
        od2B=np.sum(dictFlea['od2'][roiB[0]:roiB[1],roiB[2]:roiB[3]])   
        probeCounts[ind]=probe
        counts1[ind]=od1-od1B
        counts2[ind]=od2-od2B
        imbal[ind]=(counts1[ind]-counts2[ind])/(counts1[ind]+counts2[ind])
        signalGood[ind]=((counts1[ind]>0) & (counts2[ind]>0))
    return counts1,counts2,imbal,signalGood,probeCounts

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        