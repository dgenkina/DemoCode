Hello Andres :))

	This is some of the code I used to analyze data for the project I talked about at Intel. 
It is written for python 2.7 and uses modlues numpy, scipy, matplotlib, re, struct,
PIL and time. They should be part of any standard install. Let me know if you have trouble
running this code.


EXPERIMENTAL BACKGROUND 

	This is beginning to end processing of a calibration scan - a set of data taken to 
calibrate the Raman coupling strength and detuning. A sample set of data is included in 
the Data subfolder. 
	The calibration is done by turning on the lattice 
adiabatically and then turning on the Raman coupling relatively quickly (here 300 us),
inducing Rabi oscillations. Both the Raman and lattice couplings are kept on for a varibale
amount of time t. Then the couplings are turned off and the atoms are imaged after 
a time-of flight procedure that separates different spin states along one axis (here in 
the horizontal) and different momentum states along another axis (here in the vertical). 
This data was taken in the F=2 manifold, so there are 5 different spin states.
	The data provided is 30 different realizations of this experiment, for 30 different values
of the time t. 
	The time-of-flight images are taken with absorption imaging: first, on resonant light is shined
on the atoms and the shadow is imaged on the camera. This is called the atom image. Then the same light
is shined directly at the camera, to approximate the light the atoms saw. This is called the probe 
image. Then the light is shut off completely and another image is taken to get a measure of the 
background light seen by the camera. The atom number is then propotional to the optical depth, 
given by (on a per pixel basis) od = - ln((atomImage - backgroundImage)/(probeImage - backgrounImage)).



RUNNING THE CODE

You can simply run the main.py script. What will happen when you run this script is:

1. The image taken at one of the times t will be processed and displayed. 
	-- You can change which file is used for this by changing filnum (it can be between 4 and 33). 
	-- The image is showing the optical depth (proportional to atom number) of the atoms
		after time of flight. 
	-- The red ellipses represent the regions of interest (rois) that the processing code
		has centered on the atoms. These are used to obtain the atom number. 
		More detail on how these rois are found is below.
	-- The black ellipses represent the refernce background regions. They are used to 
		get a measure of the error in the counted atom number. The average optical
		depth summed in a background image should be printed as output. The closer to 
		zero the better!

2. Then, the same image will be re-processed using a background cancellation technique that
   I call probe reconstruction (explained in more detail below). 
	-- First the code will calculate probe matrices that are necessary for probe reconstruction.
		These matrices are sized 30 (number of files in the set) by ~100 000 (number of 
		pixels of the image that is used to look for atoms), so this will take some time.
	-- Then, another image for the same file as in 1 will pop up. Except now, the optical 
		depth is calculated with the 'reconstructed' probe. 
	-- The background in the image should look flatter, and the average optical depth in the
		background rois should be significantly smaller.

3. Then, all 30 files will be processed as in 2, and a .npz file with the outcome will be created.
	-- This code takes images as in 2 and calculates the fractional population in each spin state
		for each file included
	-- The first plot that will pop up will be this calculated fractional population as a function
		of time t (see experimental detail). This is the Rabi oscillation
	-- The second plot will be a conglamorate of all the images in all the files, where each 
		image is cut into 5 slices with one spin state in each. So, each column of this 
		conglamorate image is a single image cut into spin state. The images are presented
		in ascending order in t in the horizontal direction.

4. Finally, the fractional population data will be fit to a model to extract a coupling strength omega
   and detuning delta.
	-- The image will show the same fractional populations as in 3, with the best fit from the
		theory in lines on top of the data dots. 
	-- The fitted values for omega and delta with uncertainties will be printed at the top of the 
		graph



DATA FORMAT AND BACKGROUND FILES

	For historical reasons, the data in our lab is saved in Igor binary (.ibw) format. The code for 
converting .ibw to a python dictionary is written by my adviser, Ian SPielman. This code is contained
in files Generic.py and IgorBin.py. The rest of the code included is mine. 
	The code that uses these dictionaries and calculates optical depths that are used in the 
images you will see is contained in the readIgor file.
	In our lab, the bakcground images needed for absorption images are taken once every few days and
are slightly different for the atom image and the probe image. These are called b1 and b2 in main.py and
taken from a separate file. 




FINDING ATOM ROIS

	The first two images you will see if you run main.py will include little red elliptical regions of 
interest (rois) around each atomic cloud. The relative positions of these clouds to each other 
remain fixed (they are determined by the time of flight and the flux of the effective 2-D lattice). The
overall position of these rois does not change in the calibration data included. However, it does
change in the actual experiment. Therefore, it was necessary to write code that finds the central 
position of these atoms in a realtively efficient manner. Most of the code in file fitBlochOscDataF2.py
in function BlochOscOneFileV2 is dedicated to this pursuit. The general idea is as follows:

	-- Define a list of central positions of the atoms relative to the position of the central 
		momentum peak of the mF=0 spin state
	-- Find an initial guess for the x and y locations of this peak:
		- Sum the optical depth image along the vertical direction, find the location of the
		central peak - this is the inital guess in x position
		- Rotate the image such that the atomic clouds that formed diagonal lines initially
		are now vertically on top of each other
		- Sum along the vertical axis in the rotated image, find the location of the central
		peak
		- Relate that location back to the y position in the unrotated image - this is the 
		initial guess in y position
		- Using the locations given by the initial guesses, find the ellipse that has the 
		largest number of atoms 
		- In this ellipse, calculate the number of atoms in the left and right halves. Shift
		x the center until the number in the left and right is equal
		- In the same ellipse, calculate the number of atoms in the top and bottom halves.  
		Shif the y center until the number in the top and bottom is equal. 
		- Since the relative positions between the rois are fixed, centering one atom cloud in 
		its roi will center all the atom clouds in their rois. 


BACKGROUND CANCELLATION WITH PROBE RECONSTRUCTION

	The images used in this experiment are absorption images, as explaind in the experimental 
background section. However, often the optical depth in regions where there are no atoms does not
actually equal to zero, in part because the probe light was actually slightly different from the light
the atoms saw. To correct for this effect, we use this probe reconstruction technique, which 
reconstructs a probe image that is as close as possible to the light seen by the atoms during the atom 
image. This is done as follows: 
	-- First, a matrix of probe images is created. This contains probe images, flattened into a 
		vector, from each of the 30 data files. Both the probe matrix and its transpose 
		are calculated. 
	-- For the atom image at hand, after finding the regions of interest where the various atom 
		cloud are located, a mask of rois is created that masks the atoms away. 
	-- Then, weighted least squares is used to fit the atom image to a  a linear superpositions of
		probe images from a range of probe shots (about 10-100 shots). The weight inside the 
		rois, where the atoms actually are, is zero and the weight everywhere else is one.
	-- This way, a version of the probe image that is as close as possible to what the atoms
    		actually saw is reconstructed. 
	-- The reconstructed probe image is then used to re-calculate the optical depth.



	