# -*- coding: utf-8 -*-
"""
Created on Wed Jan 03 14:56:33 2018

@author: dng5
"""

import numpy as np
import IgorBin
import fitBlochOscDataF2
import fitSynDimPulsingAllParamsF2


PIXIS_background_filename ='Data/PIXIS_10Apr2017_10003.ibw'
bgndImage=IgorBin.LoadIBW(PIXIS_background_filename)['Data']
b1=np.array(bgndImage[:,0:bgndImage.shape[1]/2], dtype=float)
b2=np.array(bgndImage[:,bgndImage.shape[1]/2:], dtype=float)
    
roi=np.array([380, 690, 300,650])
fileroot = 'Data/PIXIS_12Jul2017'  #PIXIS fileroot
filelist=np.arange(4,34)
key='pulseDelay'
flux='pos'
date=fileroot.split("/")[-1].split("_")[1]
saveName=date+'_files_'+np.str(filelist[0])+'-'+str(filelist[-1])
m0=0
filelistProbes=np.arange(4,34)
fitProbes=True

angle=-41


kLdist=63.0

'''First, show one image with orders identified, without background reconstruction'''
filenum = 12
dict1=fitBlochOscDataF2.blochOscOneFileV2(fileroot,filenum, roi, b1, b2, angle=angle)
raw_input("Press Enter to continue...")


'''Then calculate the matrix of probe images from entire sequence of images'''
probeMatrix,probeMatrixT=fitBlochOscDataF2.getProbeReconstruction(fileroot, filelistProbes, roi, b1, b2, angle=angle)
raw_input("Press Enter to continue...")

'''And show the same file with background reconstruction implemented'''
dict1=fitBlochOscDataF2.blochOscOneFileV2(fileroot,filenum, roi, b1, b2,angle=angle, 
                                          probeMatrix=probeMatrix,probeMatrixT=probeMatrixT, fitProbes=True)
raw_input("Press Enter to continue...")

                                          
'''Then, identify and count fractions for all images in the sequence, and save to 
.npz file'''
fitBlochOscDataF2.processAndSave(fileroot,filelist,roi,key, b1, b2,angle=angle, plot=True,
                                 xlabel='Oscillation time [s]',probeMatrix=probeMatrix,probeMatrixT=probeMatrixT, fitProbes=True)
raw_input("Press Enter to continue...")

                                 
'''Fit the data to a model to get coupling strength and detuning'''
dataFile=np.load('12Jul2017_files_4-33.npz')
omegaGuess=0.5
deltaGuess=-0.03
fitSynDimPulsingAllParamsF2.fitAndPlot(dataFile,omegaGuess,deltaGuess)