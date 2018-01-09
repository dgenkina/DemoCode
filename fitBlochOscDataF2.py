# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 16:41:25 2016

@author: dng5
"""

import readIgor
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import scipy.ndimage as snd
import matplotlib.gridspec as gridspec
import time



def getRoi(array, image, xCent,yCent,weightArray=1.0,updateWeights=False,r=16,eps=5,draw=True,color=(255,0,0),horizontalStretch=True):
    '''This function and outputs the total optical depth in the given region of interest (roi),
    as well as the summed optical depth in the top, bottom, left and right
    halves of the roi individually to aid in centering the roi on the atom cloud.
    It also places a marker on the image where the chosen roi is and updates weights (if 
    updateWeights is true) for probe image fitting. 
    
    
    array: the optical depth array 
    image: the image object created from the optical depth, to update roi locations   
    xCent: initial guess for x center of the roi
    yCent: initial guess for y center of the roi
    weightArray: If updateWeights is true, this array is the shape of optical depth array, 
        initially all ones. Things inside the roi are turned to zeros, so that 
        the atoms are not counted towards to probe image fit.
    updateWeights: bool controlling if weightArray is updated
    r: radious of roi ellipse
    eps: elipticity of roi ellipse
    draw: bool controlling whether to add the outline of the roi to the image
    color: if draw is true, rgb color of the outline of the roi added to image
    horizontalStretch: if True, roi is elliptical to be elongated in the horizontal. 
        If False, it is elongated in the vertical.
    
    Returns:
    counts: total summed optical depth inside the roi
    countsL: summed optical depth in the left half of the roi
    countsR: summed optical depth in the right half of the roi
    countsT: summed optical depth in the top half of the roi
    countsB: summed optical depth in the bottom half of the roi
    weightArray: if updateWeights is true, weightArray with the inside of the 
        roi set to 0'''
    
    ylen=array.shape[0]
    xlen=array.shape[1]
    if horizontalStretch:
        bbox=(xCent-r,yCent-np.int(np.sqrt(r*r-eps*eps)),xCent+r,yCent+np.int(np.sqrt(r*r-eps*eps)))
    else:
        bbox=(xCent-np.int(np.sqrt(r*r-eps*eps)),yCent-r,xCent+np.int(np.sqrt(r*r-eps*eps)),yCent+r)
    
    y,x = np.ogrid[-yCent:ylen-yCent, -xCent:xlen-xCent]
    if horizontalStretch:
        xloc1=x-eps
        yloc1=y
        xloc2=x+eps
        yloc2=y
    else:
        xloc1=x
        yloc1=y-eps
        xloc2=x
        yloc2=y+eps
    
    mask = np.sqrt((xloc1)**2.0 + yloc1**2.0) +np.sqrt((xloc2)**2.0 + yloc2**2.0) <= 2.0*r
    if updateWeights:
        weightArray[mask]=0
    maskL = ((mask) & (x<0))
    maskR = ((mask) & (x>0))
    maskT=((mask) & (y<0))
    maskB=((mask) & (y>0))

    counts=np.sum(array[mask])
    countsL=np.sum(array[maskL])
    countsR=np.sum(array[maskR])
    countsT=np.sum(array[maskT])
    countsB=np.sum(array[maskB])
    
    if draw: 
        draw = ImageDraw.Draw(image)
        draw.ellipse(bbox,outline=color)
    return (counts, countsL, countsR, countsT, countsB, weightArray)
    
kLdist=63.0


def getProbeReconstruction(fileroot, filelistProbes, roi,b1, b2, angle=0):
    '''This function creates a matrix with file number along one axis and a 
    flattened version of the probe array along the other axis. It also calculates
    the transpose of that matrix.
    
    fileroot: the file path excluding file number for the files to use
    filelistProbes: the list of file numbers to be used for probe reconstruction
    roi: the region of interest within the probe image to be used
    b1: background array for the atom images
    b2: background array for the probe images
    angle: the angle to rotate the image before taking the roi
    
    Returns:
    probeMatrix: the matrix of probes
    probeMatrixT: the transpose of the matrix of probes
    '''
    filename=fileroot+"_"+ str(filelistProbes[0]).zfill(4) + ".ibw"
    dict1 =readIgor.processIBW(filename, angle=angle,bgnd1=b1,bgnd2=b2)  
    rotProbe=snd.interpolation.rotate(dict1['Raw2'],angle)[roi[0]:roi[1],roi[2]:roi[3]]
    N=rotProbe.flatten().size
    print ''
    print 'Calculating probe matrix'    
    print 'number of pixels in roi = ' + str(N)
    t1=time.clock()
    probeMatrix=np.ones((N,filelistProbes.size+1))
    for ind, filenum in enumerate(filelistProbes):
        filename=fileroot+"_"+ str(filenum).zfill(4) + ".ibw"
        dict1 =readIgor.processIBW(filename, angle=angle,bgnd1=b1,bgnd2=b2)  
        rotProbe=dict1['rotRaw2'][roi[0]:roi[1],roi[2]:roi[3]]
        probeMatrix[:,ind+1]=rotProbe.flatten()
    probeMatrixT=probeMatrix.transpose()
    t2=time.clock()
    print 'constructed probe matrix in '+str(t2-t1)+' s'
    return probeMatrix, probeMatrixT


def processAndSave(fileroot,filelist,roi,key,b1, b2,plot=True,xlabel='Lattice pulse time [s]',angle=0, probeMatrix=1.0,probeMatrixT=1.0, fitProbes=False):
    '''This is the main function to run when processing a sequences of files.
    It takes files from a single scan and runs the blochOscFractionsV2
    function to processes the files in the sequece, then
    saves important properties of the entire seqence of files to a .npz file.
    
    The inputs are passed directly to the blochOscFractionsV2 function. They 
    are:
    
    fileroot: the file path excluding file number for the files to use
    filelist: the list of file numbers in the scan to be analyzed
    roi: the region of interest within the optical depth image to be used
    key: the name of the variable changed during the scan (the x axis for fractional population plots)
    b1: background array for the atom images
    b2: background array for the probe images
    angle: the angle to rotate the image before taking the roi 
    plot: if True, make plots of fractional populations vs scanned variable
        and create a plot containing the images from all the files in the scan,
        with different spin states separated vertically and diffferent images 
        (sorted in increasing order of the scanned variable) separated horizontally
    xlabel: if Plot is true, x axis label for the fractional population plot
    angle: angle the original image is rotated by before the roi is taken
    probeMatrix,probeMatrixT = if fitProbes is true, the matrix of probe images
        and its transpose to be used for probe reconstruction
    fitProbes: if True, apply the probe reconstruction procedure before calculating
        fractional populations, using the matrix of probes and its transpose
        
    The saved file has the name of the fileroot and the first and last file number
    in filelist, separated by a dash.
    '''    
    date=fileroot.split("/")[-1].split("_")[1]
    saveName=date+'_files_'+np.str(filelist[0])+'-'+str(filelist[-1])
    t1=time.clock()
    (numM2, numM, num0, numP, numP2,fractionM2, fractionM, fraction0, fractionP, fractionP2,waveDict,qlist,xCenters,ktot,bgndAvg,odFiltAll)=blochOscFractionsV2(fileroot,filelist,roi,key,b1, b2,plot=True,xlabel='Oscillation time [s]',angle=angle, probeMatrix=probeMatrix,probeMatrixT=probeMatrixT, fitProbes=fitProbes)        
    tlist=waveDict[key]
    np.savez(saveName,numM2=numM2, numM=numM, num0=num0, numP=numP, numP2=numP2,fractionM2=fractionM2, fractionM=fractionM, fraction0=fraction0, fractionP=fractionP, fractionP2=fractionP2,qlist=qlist,xCenters=xCenters,tlist=tlist,ktot=ktot,bgndAvg=bgndAvg,odFiltAll=odFiltAll)#np.arange(filelist.size))#
    
    t2=time.clock()
    print 'Processing took ' + str(t2-t1) +' s'
    return

IsatCounts = 235.3
imagingTime=40.0

   
def blochOscOneFileV2(fileroot,filenum, roi, b1, b2, angle=0, probeMatrix=1.0,probeMatrixT=1.0, fitProbes=False,  draw=True):
    '''This is the main function that processess a single file, from a 
    single realization of the experiment. It takes the raw .ibw file, sends it
    to readIgor.processIBW to extract an atom, probe and optical depth image,
    applies a gaussian filter to the optical depth (od) image to enhance signal to noise,
    finds the x and y positions of the mask of regions of interests (rois) that 
    centers the rois on the atoms, and counts the total od in each roi. 
    
    If fitProbes is true, it also performs a probe reconstruction routine:
    after finding the regions of interest where the various atom cloud are located,
    it applies a mask that masks the atoms away. Then, it uses the atom image
    with the actual atoms masked out and fits it to a linear superpositions of
    probe images from a range of probe shots (about 10-100 shots). This way, a
    version of the probe image that is as close as possible to what the atoms
    actually saw is reconstructed. The reconstructed probe image is then used
    to re-calculate the optical depth, and the counts in each roi are then taken
    from this od. 
    
    Inputs
    fileroot: the file path excluding file number for the files to use
    filenum: the number of the file to be analyzed
    roi: the large region of interest cointaining all atom clouds within the optical depth image to be used
    b1: background array for the atom images
    b2: background array for the probe images
    angle: the angle to rotate the image before taking the roi 
    plot: if True, make plots of fractional populations vs scanned variable
        and create a plot containing the images from all the files in the scan,
        with different spin states separated vertically and diffferent images 
        (sorted in increasing order of the scanned variable) separated horizontally
    xlabel: if Plot is true, x axis label for the fractional population plot
    angle: angle the original image is rotated by before the roi is taken
    probeMatrix,probeMatrixT = if fitProbes is true, the matrix of probe images
        and its transpose to be used for probe reconstruction
    fitProbes: if True, apply the probe reconstruction procedure before calculating
        fractional populations, using the matrix of probes and its transpose
    draw: if True, show image of the optical depth with the determined rois for
        each atom cloud and background rois used to estimate error. 
        
    Outputs
    counts: the total summed od in each of the atom cloud rois, 25 in total.
    odFiltered: the optical depth in the large roi, with gaussian filtering 
        applied. If fitProbes is true, this is the od obtained using the 
        reconstructed probe image. 
    xCent: the central position along x of the atom roi mask (also the central poition
        along x of the central mF=0 atom cloud)
    yCent: the central position along y of the atom roi mask (also the central poition
        along y of the central mF=0 atom cloud)
    ktot: the total momentum of the entire atomic distribution
    bgndAvg: the average counted od in the background roi images, taken next to
        each atom roi. This should be close to zero, and is used as an unccertainty
        in the counted od.
    dict1: a dictionary of all the data included in the original .ibw file
    '''     
    filename=fileroot+"_"+ str(filenum).zfill(4) + ".ibw"
    print ''
    print ''
    print 'Processing ' + filename
    dict1 =readIgor.processIBW(filename, angle=angle,bgnd1=b1,bgnd2=b2)
  
    odRoi1=dict1['rotODcorr'][roi[0]:roi[1],roi[2]:roi[3]]   
    atomsRoi=dict1['rotRaw1'][roi[0]:roi[1],roi[2]:roi[3]]  
    
    odFiltered=snd.filters.gaussian_filter(odRoi1,1.0)
##    
#    if draw:      
#        fig1=plt.figure()
#        pan1=fig1.add_subplot(1,1,1)
#        pan1.imshow(odFiltered,vmin=-0.15,vmax=0.4)
#        
    
    odFiltLin = np.sum(odFiltered,axis=0)
    peak2=np.max(odFiltLin[150:220])
    xGuess2=np.float( np.where(odFiltLin==peak2)[0])
    
    
    y1=kLdist  
    x1=55
    y2=20
    angle2=52
    x2=0

##    
#    if draw:
#        figure2=plt.figure()
#        panel2=figure2.add_subplot(1,1,1)
#        panel2.plot(odFiltLin,'bo')
  
    xCent=xGuess2

    
    roi2=np.array([600, 900, 560, 840])
    

    
    rotOD2=snd.interpolation.rotate(dict1['rotOD'],angle2)
    odRoi2 = rotOD2[roi2[0]:roi2[1],roi2[2]:roi2[3]]
    odFiltered2=snd.filters.gaussian_filter(odRoi2,1.0)
    
#    if draw:
#        fig2=plt.figure()
#        pan2=fig2.add_subplot(1,1,1)
#        pan2.imshow(odFiltered2 , vmin=-0.15, vmax=0.3)
        
        
    odFiltLin2 = np.sum(odFiltered2,axis=0)   
    peak2=np.max(odFiltLin2[130:190])
    xGuess2=np.float(np.where(odFiltLin2==peak2)[0])

    
#    if draw:     
#        figure3=plt.figure()
#        panel3=figure3.add_subplot(1,1,1)
#        panel3.plot(odFiltLin2,'bo')
##        

    yCent=-0.786*xCent+1.309*xGuess2+86.5
        
    #print xCent,xGuess2, yCent

    
    norm=mpl.colors.Normalize(vmin=-0.15,vmax=0.7)
    im = Image.fromarray(np.uint8(plt.cm.jet(norm(odFiltered))*255))
   

    offsets = np.array([[0,0],[-x2,y1],[x2,-y1],[-x2*2,y1*2],[x2*2,-y1*2],[-x1,-y2],[-x1-x2,-y2+y1],[-x1+x2,-y2-y1],[-x1-x2*2,-y2+y1*2],[-x1+x2*2,-y2-y1*2],[x1,y2],[x1-x2,y2+y1],[x1+x2,y2-y1],[x1-x2*2,y2+y1*2],[x1+x2*2,y2-y1*2],[-2*x1,-2*y2],[-2*x1-x2,-2*y2+y1],[-2*x1+x2,-2*y2-y1],[-2*x1-x2*2,-2*y2+y1*2],[-2*x1+x2*2,-2*y2-y1*2],[2*x1,2*y2],[2*x1-x2,2*y2+y1],[2*x1+x2,2*y2-y1],[2*x1-x2*2,2*y2+y1*2],[2*x1+x2*2,2*y2-y1*2]])#np.array([[0,0],[0,69],[0,-69],[-58,49],[-58,119],[-58,-21],[57,-47],[57,21],[57,-115]])
    r=12.0
    klist=offsets[:,1]/kLdist
    counts=np.zeros(offsets.shape[0])    
    for i in np.arange(offsets.shape[0]):
        offs=offsets[i]
        counts[i]= getRoi(odFiltered, im, xCent+offs[0],yCent+offs[1],r=r,draw=False)[0]
    #    print counts[i]
    
   # print np.max(counts)
    if np.max(counts)==0:
        maxInd=0
    else:
        maxInd=np.where(counts==np.max(counts))[0][0]
   # print maxInd

    allcounts=getRoi(odFiltered, im, xCent+offsets[maxInd,0],yCent+offsets[maxInd,1],r=r,draw=False)
   # print allcounts

    i=0
    while ((np.abs(allcounts[4]-allcounts[3])>2.0) & (i<20)):
        if (allcounts[4]-allcounts[3])>0:
            yCent=yCent+1

            allcounts=getRoi(odFiltered, im, xCent+offsets[maxInd,0],yCent+offsets[maxInd,1],r=r,draw=False)
        else: 
            yCent=yCent-1

            allcounts=getRoi(odFiltered, im, xCent+offsets[maxInd,0],yCent+offsets[maxInd,1],r=r,draw=False)
       #print 'yCent = %i' %(yCent)
        i=i+1

    i=0
    while ((np.abs(allcounts[2]-allcounts[1])>2.0) & (i<20)):
        if (allcounts[2]-allcounts[1])>0:
            xCent=xCent+1
     
            allcounts=getRoi(odFiltered, im, xCent+offsets[maxInd,0],yCent+offsets[maxInd,1],r=r,draw=False)
        else: 
            xCent=xCent-1

            allcounts=getRoi(odFiltered, im, xCent+offsets[maxInd,0],yCent+offsets[maxInd,1],r=r,draw=False)
        #print 'xCent = %i' %(xCent)
        i=i+1
        
    bgndAvg=0.0

    if fitProbes:
        weightArray=np.ones(odFiltered.shape)
    for i in np.arange(offsets.shape[0]):
        offs=offsets[i]
        

        count= getRoi(odFiltered, im, xCent+offs[0],yCent+offs[1],r=r, draw=draw)[0]

        if fitProbes:
            (count,cL,cR,cT,cB,weightArray)= getRoi(odFiltered, im, xCent+offs[0],yCent+offs[1],r=r,weightArray=weightArray,updateWeights=True, draw=False)
        
        bgnd=getRoi(odFiltered, im, xCent+offs[0]-2.0*r,yCent+offs[1],r=r, draw=draw,color=(0,0,0))[0]
        bgndAvg+=bgnd/counts.size
        counts[i]=count-bgnd
    print 'Background average no reconstruction = ' +str(bgndAvg)
    if fitProbes:
        bgndAvg2=0.0
        xTw=probeMatrixT*weightArray.flatten()
        rhs=np.dot(xTw,atomsRoi.flatten())
        lhs=np.dot(xTw,probeMatrix)
        beta=np.linalg.solve(lhs,rhs)
        newProbe=np.dot(probeMatrix,beta).reshape(atomsRoi.shape)
        newOd=-np.log(atomsRoi/newProbe)
        newOd = readIgor.zeroNansInfsVector(newOd)
        odFiltered = newOd+(newProbe-atomsRoi)/(IsatCounts*imagingTime)
        odFiltered=snd.filters.gaussian_filter(odFiltered,1.0)
        
        im2 = Image.fromarray(np.uint8(plt.cm.jet(norm(odFiltered))*255))
        for i in np.arange(offsets.shape[0]):
            offs=offsets[i]

            count= getRoi(odFiltered, im2, xCent+offs[0],yCent+offs[1],r=r, draw=True)[0]
            bgnd= getRoi(odFiltered, im2, xCent+offs[0]-2.0*r,yCent+offs[1],r=r, draw=True,color=(0,0,0))[0]
            bgndAvg2+=bgnd/counts.size
            counts[i]=count#-bgnd
        print 'Background average with reconstruction = ' +str(bgndAvg2)
    if draw:

        if fitProbes:
            fig4=plt.figure()
            pan4=fig4.add_subplot(1,1,1)
            pan4.imshow(im2)
            pan4.set_title(filename+'_reconProbe')
            fig4.show()
        else:
            fig4=plt.figure()
            pan4=fig4.add_subplot(1,1,1)
            pan4.imshow(im)
            pan4.set_title(filename)
            fig4.show()
       
        
    ktot=np.dot(counts,klist)/np.sum(counts)
        
    return counts, odFiltered, xCent,yCent, ktot, bgndAvg, dict1
    
def blochOscFractionsV2(fileroot,filelist,roi,key,b1, b2, angle=0,plot=True,xlabel='', probeMatrix=1.0,probeMatrixT=1.0, fitProbes=False):  
    '''This is the main function that processess a sequence of files in the 
    same scan. It sends each file to the blochOscOneFileV2 function, and 
    uses the output of that function to extract the fractional population in 
    each spin state from the counted optical depths (ods). It then
    stores the total counted ods and fractions for each spin state, as well 
    as the measured crystal momentum to arrays, each having the length of 
    filelist, such that each of these variables for the entire scan is easily acessible.
    
    Inputs
    fileroot: the file path excluding file number for the files to use
    filelist: the list of file numbers in the scan to be analyzed
    roi: the region of interest within the optical depth image to be used
    key: the name of the variable changed during the scan (the x axis for fractional population plots)
    b1: background array for the atom images
    b2: background array for the probe images
    angle: the angle to rotate the image before taking the roi 
    plot: if True, make plots of fractional populations vs scanned variable
        and create a plot containing the images from all the files in the scan,
        with different spin states separated vertically and diffferent images 
        (sorted in increasing order of the scanned variable) separated horizontally
    xlabel: if Plot is true, x axis label for the fractional population plot
    angle: angle the original image is rotated by before the roi is taken
    probeMatrix,probeMatrixT = if fitProbes is true, the matrix of probe images
        and its transpose to be used for probe reconstruction
    fitProbes: if True, apply the probe reconstruction procedure before calculating
        fractional populations, using the matrix of probes and its transpose
        
    Outputs
    numM2: total counted od in the mF=-2 spin state for each file in filelist
    numM: total counted od in the mF=-1 spin state for each file in filelist
    num0: total counted od in the mF=0 spin state for each file in filelist
    numP: total counted od in the mF=+1 spin state for each file in filelist
    numP2: total counted od in the mF=+2 spin state for each file in filelist
    fractionM2: fractional population in the mF=-2 spin state for each file in filelist
    fractionM: fractional population in the mF=-1 spin state for each file in filelist
    fraction0: fractional population in the mF=0 spin state for each file in filelist
    fractionP: fractional population in the mF=+1 spin state for each file in filelist
    fractionP2: fractional population in the mF=+2 spin state for each file in filelist
    waveDict: list of all indexed waves for all the files. waveDict[key] gives the list
        of values for the scanned variable in the sequence (x axis for fractional 
        populations plot)
    qlist: list of crystal momenta for each file in filelist, calculated based on 
        the central position along y of the central cloud of mF=0 atoms
    xCenters: central positions along x of the central cloud of mF=0 atoms for
        each file in filelist
    ktot: total momentum of the entire atomic distribution for each file in filelist
    bgndAvg: average summed od in the background rois
    odFiltAll: optical depth images of all the files in filelist
    ''' 
    num0=np.zeros(filelist.size)
    numM=np.zeros(filelist.size)
    numP=np.zeros(filelist.size)
    numM2=np.zeros(filelist.size) 
    numP2=np.zeros(filelist.size)
    fractionP2=np.zeros(filelist.size)
    fractionP=np.zeros(filelist.size)
    fraction0=np.zeros(filelist.size)
    fractionM=np.zeros(filelist.size)
    fractionM2=np.zeros(filelist.size)
    qlist=np.zeros(filelist.size)
    ktot=np.zeros(filelist.size)

    (counts, odFiltered, xCent,yCent,k,bgndAvg, dict1)=blochOscOneFileV2(fileroot,filelist[0], roi, b1, b2, angle=angle,draw=False,probeMatrix=probeMatrix,probeMatrixT=probeMatrixT, fitProbes=fitProbes)
    infoString=dict1["Note"]
    waveDict=readIgor.getIndexedWaves(infoString)
    waveDict=waveDict.fromkeys(waveDict.keys(),[])
    odFiltAll=np.zeros((filelist.size,odFiltered.shape[0],odFiltered.shape[1]))
    xCenters=np.zeros(filelist.size)
    bgndAvg=np.zeros(filelist.size)
    for ind, filenum in enumerate(filelist):

        (counts, odFiltered, xCent, yCent, ktot[ind],bgndAvg[ind],dict1)=blochOscOneFileV2(fileroot,filenum, roi,b1, b2, angle=angle, draw=False,probeMatrix=probeMatrix,probeMatrixT=probeMatrixT, fitProbes=fitProbes)
        
        xCenters[ind]=xCent
        qlist[ind]=2.0*(144.5-yCent)/kLdist

        infoString=dict1["Note"]
        waveDictLocal=readIgor.getIndexedWaves(infoString)
        for k in waveDict.iterkeys():
            
            waveDict[k]=np.append(waveDict[k],waveDictLocal[k])

        odFiltAll[ind]=odFiltered
        

        roiNum=counts.size/5
        num0[ind]=np.sum(counts[0:roiNum])
        numM[ind]=np.sum(counts[roiNum:2*roiNum])
        numP[ind]=np.sum(counts[roiNum*2:roiNum*3])
        numM2[ind]=np.sum(counts[roiNum*3:4*roiNum])
        numP2[ind]=np.sum(counts[roiNum*4:])
        
        
        total=num0[ind]+numM[ind]+numP[ind]+numM2[ind]+numP2[ind]
        
        fractionP2[ind]=numP2[ind]/total
        fractionP[ind]=numP[ind]/total
        fraction0[ind]=num0[ind]/total
        fractionM[ind]=numM[ind]/total
        fractionM2[ind]=numM2[ind]/total
#        print numM2[ind], numM[ind], num0[ind], numP[ind], numP2[ind]
        print 'Counted fractional populations: '
        print r'mF = 2: %.2f, mF = 1: %.2f, mF = 0: %.2f, mF = -1: %.2f, mF = -2: %.2f' %(fractionM2[ind], fractionM[ind], fraction0[ind], fractionP[ind], fractionP2[ind])

    if plot:
        figure3=plt.figure()
        panel3=figure3.add_subplot(1,1,1)
        if key == 'ind':
            xvals=range(filelist.size)
        else:
            xvals =waveDict[key]
            
        if xlabel=='':
            if key == 'ind':
                xlabel = 'shot number'
            else:
                xlabel = key

        panel3.plot(xvals,fractionP2,'co', label='mF=+2')
        panel3.plot(xvals,fractionP,'bo', label='mF=+1')
        panel3.plot(xvals,fraction0,'go', label='mF=0')
        panel3.plot(xvals,fractionM,'ro', label='mF=-1')
        panel3.plot(xvals,fractionM2,'mo', label='mF=-2')
        plt.legend()
        panel3.set_xlabel(xlabel)
        panel3.set_ylabel('Fractional population')
        figure3.show()
        
        
#        figure4=plt.figure()
#        pan4=figure4.add_subplot(1,1,1)
#        pan4.plot(xvals, qlist,'bo')
#        pan4.set_ylabel(r'Quasimomentum [$k_L$]')
#        pan4.set_xlabel(key)
        
        figure = plt.figure()
        gs = gridspec.GridSpec(5,filelist.size)
        gs.update(left=0.1, right=0.99, wspace=0.01, hspace=0)
        if key == 'ind':
            sort=range(filelist.size)
            tsorted=range(filelist.size)
        else:
            sort=np.argsort(waveDict[key])
            tsorted = waveDict[key][sort]

        w=30
        vmax=0.5
        for i in range(filelist.size):
            j=sort[i]

            panel1=figure.add_subplot(gs[i])
            panel1.imshow(odFiltAll[j][:,int(xCenters[j]-2*49-w):int(xCenters[j]-2*49+w)],vmin=-0.05, vmax=vmax)
            panel1.yaxis.set_ticks([])  
            panel1.xaxis.set_ticks([]) 
            if i==0:
                panel1.set_ylabel(r'mF=2')
            
            panel2=figure.add_subplot(gs[i+filelist.size])
            panel2.imshow(odFiltAll[j][:,int(xCenters[j]-49-w):int(xCenters[j]-49+w)],vmin=-0.05, vmax=vmax)
            panel2.yaxis.set_ticks([])  
            panel2.xaxis.set_ticks([])
            if i==0:
                panel2.set_ylabel(r'mF=1')
    
            panel3=figure.add_subplot(gs[i+2*filelist.size])
            panel3.imshow(odFiltAll[j][:,int(xCenters[j]-w):int(xCenters[j]+w)],vmin=-0.05, vmax=vmax)
            panel3.yaxis.set_ticks([])  
            panel3.xaxis.set_ticks([])
            if i==0:
                panel3.set_ylabel(r'mF=0')
            
            panel4=figure.add_subplot(gs[i+3*filelist.size])
            panel4.imshow(odFiltAll[j][:,int(xCenters[j]+49-w):int(xCenters[j]+49+w)],vmin=-0.05, vmax=vmax)
            panel4.yaxis.set_ticks([])  
            panel4.xaxis.set_ticks([])  
            if i==0:
                panel4.set_ylabel(r'mF=-1')
            
            panel5=figure.add_subplot(gs[i+4*filelist.size])
            panel5.imshow(odFiltAll[j][:,int(xCenters[j]+2*49-w):int(xCenters[j]+2*49+w)],vmin=-0.05, vmax=vmax)
            panel5.yaxis.set_ticks([])  
            panel5.xaxis.set_ticks([]) 
           # panel5.set_xlabel('%.2f' %(tsorted[i]*1e3))
            if i==0:
                panel5.set_ylabel(r'mF=-2')
        figure.show()
    
    return numM2, numM, num0, numP, numP2,fractionM2, fractionM, fraction0, fractionP, fractionP2,waveDict, qlist, xCenters,ktot,bgndAvg,odFiltAll
        
