# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 15:18:23 2016

@author: dng5
"""

import numpy as np
from numpy import linalg as LA
from scipy import optimize
from scipy import linalg as sLA
import matplotlib.pyplot as plt
#import readIgor


'''Define global constants'''
hbar = 1.0545718e-34 # reduced Planck constant m^2 kg/s
mRb =1.44467e-25 #mass of rubidium in kg
lambdaR = 790e-9 # Raman wavelength in m
lambdaL = 1.064e-6 #lattice wavelength in m
Erecoil = (2.0*np.pi*hbar)**2.0/(2.0*mRb*lambdaL**2.0) #recoil energy

'''Define parameters of Raman coupling ramp-on'''
tau=0.0003*Erecoil/hbar
rampOnt=0.0001*Erecoil/hbar

'''Define color dictionary for plotting different spin states'''
cDict={}
cDict[-2]='m-'
cDict[-1]='r-'
cDict[0]='g-'
cDict[1]='b-'
cDict[2]='c-'


Flat=False #whether to use flat or consistent with Clebsch-Gordans Raman coupling


def Fz(S):
    '''z component of F angular momentum matrix, for spin S system'''
    a=np.arange(np.float(-S),np.float(S+1))
    F=np.diag(a)
    return F
    
def Fx(S):
    '''x component of F angular momentum matrix, for spin S system'''
    F=np.zeros((2*S+1,2*S+1))
    for i in range(int(2*S+1)):
        for j in range(int(2*S+1)):
            if np.abs(i-j)==1:
                F[i,j]=(1.0/2.0)*np.sqrt(S*(S+1)-(i-S)*(j-S))
    return F
    
def FxFlat(S):
    '''Flat ( uniform coupling)version of x component 
    of F angular momentum matrix, for spin S system'''
    F=np.zeros((2*S+1,2*S+1))
    for i in range(2*S+1):
        for j in range(2*S+1):
            if np.abs(i-j)==1:
                F[i,j]=1.0/np.sqrt(2)
    return F 

def RamanLatHam(k, omega, delta, epsilon, U, n, S, m0, c):
    '''Synthetic dimensions Hamiltonian, including Raman and lattice coupling.
    
    k: momentum (in units of lattice recoil momentum)
    omega: Raman coupling strength (in units of lattice recoil energy)
    delta: Raman detuning (in units of lattice recoil energy)
    epsilon: quadratic zeeman shifts (in units of lattice recoil energy)
    U: lattice depth (in units of lattice recoil energy)
    n: number of lattice momentum orders to include
    S: spin quantum number
    m0: initial spin state (only changes center of Brillouin zone)
    c: flux, kR/kL
    
    Returns H in block diagonal form, with each block corresponding to a 
    different lattice momentum order and within the block each row representing
    a different spin state
    '''
    Nlat=2*n+1
    Ntot=Nlat*(2*S+1)
    Kinetic=np.zeros(Ntot)

    for i in range(Ntot):
        spinI=np.float(i%(2*S+1)-S)
        latI=np.float(np.divide(i,2*S+1)-n)
        Kinetic[i]=(k-2.0*(spinI-m0)*c-2.0*latI)**2.0
    H=np.diag(Kinetic)
    H+=delta*sLA.block_diag(*[Fz(S)]*Nlat)
    H+=((-1.0)**(S+1))*epsilon*sLA.block_diag(*[Fz(S)**2.0]*Nlat)
    H+=(np.sqrt(2.0)/2.0)*omega*sLA.block_diag(*[Fx(S)]*Nlat)    
        
    for i in range(Ntot):
        for j in range(Ntot):
            if np.abs(i-j)==(2*S+1):
                H[i,j]=U/4.0         
    return H
    
def expRamp(t,tMax,start,stop,tau):
    '''Exponential ramp for Raman ramp-on. 
    t: current time
    tMax: total time of the exponential ramp
    start: value of ramped parameter at the beginning of ramp
    stop: value of ramped parameter at the end of ramp
    tau: timescale of ramp
    
    returns: value of ramped parameter at time t
    '''
    out=(stop-start*np.exp(tMax/tau)+(start-stop)*np.exp(t/tau))/(1-np.exp(tMax/tau))
    return out
    
def rampedOnPsi(k,omega,delta,epsilon,U,n,S,m0,c,rampOnt=0.0003*Erecoil/hbar,steps=50):
    '''Find the wavefuntion Psi after the Raman coupling strength is ramped on exponentially
    
    k: momentum (in units of lattice recoil momentum)
    omega: Raman coupling strength (in units of lattice recoil energy)
    delta: Raman detuning (in units of lattice recoil energy)
    epsilon: quadratic zeeman shifts (in units of lattice recoil energy)
    U: lattice depth (in units of lattice recoil energy)
    n: number of lattice momentum orders to include
    S: spin quantum number
    m0: initial spin state to load from
    c: flux, kR/kL
    rampOnt: time of the exponential ramp
    steps: number of steps in Trotter decomposition
    
    returns wavefunction psi after Raman is ramped on, in the same basis as the 
    Hamiltonian H
    '''
    tlist=np.linspace(0.0,rampOnt,steps)  
    dt=tlist[1]-tlist[0]
    Energy1, V1 = LA.eig(RamanLatHam(k, 0.0, 0.0, epsilon, U, n, S, m0,c))
    sort=np.argsort(Energy1)
    V1sorted=V1[:,sort]
    if c==0.0:
        psiLat=V1sorted[:,0].reshape(2*n+1,2*S+1).transpose()
        psi0=np.zeros(Energy1.size).reshape(2*n+1,2*S+1).transpose()
        psi0[m0+S]=psiLat[1]/np.sqrt(np.sum(psiLat[1]**2.0))
        psi0=psi0.transpose().flatten()
       
    else:
        psi0=V1sorted[:,0]
    
    for t in tlist:
        omegaLoc=expRamp(t,rampOnt,0.0,omega,tau)
        Energy,V=sLA.eigh(RamanLatHam(k, omegaLoc, delta, epsilon, U, n, S, m0,c))
        V=V+0.0j
        Vinv=np.conj(V.transpose())
        psi0=np.dot(Vinv,psi0)
        teo=np.diag(np.exp(-1.0j*Energy*dt))
        psi0=np.dot(teo,psi0)
        psi0=np.dot(V,psi0)
        
    return psi0

    
    
def propagateRLHamiltonian(t, k, omega, delta, epsilon, U, n,S,m0,c,**kwargs):
    '''Time evolves a wavefunction under the combined Raman and lattice Hamiltonian.
    Includes ramp on from zero Raman coupling to omega. 
    
    t: evolution time    
    k: momentum (in units of lattice recoil momentum)
    omega: Raman coupling strength (in units of lattice recoil energy)
    delta: Raman detuning (in units of lattice recoil energy)
    epsilon: quadratic zeeman shifts (in units of lattice recoil energy)
    U: lattice depth (in units of lattice recoil energy)
    n: number of lattice momentum orders to include
    S: spin quantum number
    m0: initial spin state to load from
    c: flux, kR/kL
    
    Calculates wavefunction after Raman ramp on and evolution time t in the 
    same basis as the Hamiltonian
    
    Outputs fractional population in each spin state as given by the evolved wavefunction
    '''
    t=np.array(t)    
    psi0=rampedOnPsi(k,omega,delta,epsilon,U,n,S,m0,c,**kwargs)

    H = RamanLatHam(k, omega, delta, epsilon, U, n, S, m0,c)
    Energy, V = LA.eig(H)
    
    mat=np.array([np.identity(Energy.size)]*t.size)


    V = V + 1j*0.0
    Vinv = np.conjugate(np.transpose(V))
    V=np.array([V]*t.size)
    Vinv=np.array([Vinv]*t.size)
    psi0=np.array([psi0]*t.size)

    aa=np.exp(-1j*np.outer(t, Energy))

    Uprop = np.einsum('ij,ijk->ijk',aa,mat)

    a = np.einsum('ijk,ik->ij',Vinv,psi0)
    b = np.einsum('ijk,ik->ij',Uprop,a)                                                           
    psi = np.einsum('ijk,ik->ij',V,b)
    
    pops=np.absolute(psi)**2.0                     

    latPops=np.sum(pops.reshape(t.size,2*n+1,2*S+1)[:,np.divide(n,2)-1:np.divide(n,2)+2,:],axis=2).flatten() 
    #populations in the -2k_L, 0, and +2k_L lattice sites, summed over spin sites,in time step blocks
    spinPops=np.sum(pops.reshape(t.size,2*n+1,2*S+1),axis=1).flatten() #[:,0]#
    #populations in each spin state, summed over lattice sites, in time step blocks 
    return spinPops
   

def fitAndPlot(dataFile, omegaGuess, deltaGuess, U=4.4,epsilon=0.0206,m0=0,S=2,c=1064.0/790.0,n=7,k=0.0):
    '''Reads in a processed data file containing fractional populations as a function
    of oscillation time, where the 1-D lattice is ramped on adiabatically and the 
    Raman coupling is turned on quickly. Fits the data to the prediction of 
    Hamiltonian evolution, with fitting parameters of Raman coupling strength (omega)
    and Raman detuning (delta).
    
    dataFile: the file containing data to fit to the model
    omegaGuess: initial guess for Raman coupling strength Omega
    deltaGuess: initial guess for Raman detuning delta
    
    k: momentum (in units of lattice recoil momentum)
    omega: Raman coupling strength (in units of lattice recoil energy)
    delta: Raman detuning (in units of lattice recoil energy)
    epsilon: quadratic zeeman shifts (in units of lattice recoil energy)
    U: lattice depth (in units of lattice recoil energy)
    n: number of lattice momentum orders to include
    S: spin quantum number
    m0: initial spin state (only changes center of Brillouin zone)
    c: flux, kR/kL
    
    Plots the data with the theory curve from the fit, with optimal fit 
    paramenters of Omega and delta in the title, with uncertinties from fit. 
    '''    
    
    print ''
    print ''
    print 'Fitting data to model... '
    
    def constrainedPropagateHam( k,epsilon,U, n,S,m0,c):
        return lambda t,omega,delta: np.array(propagateRLHamiltonian(t, k, omega, delta, epsilon, U,n,S,m0,c,rampOnt=rampOnt))
    HofT=constrainedPropagateHam( k,epsilon,U, n,S,m0,c)     
    
    tList=dataFile['tlist']
    sort=np.argsort(tList)
    tList=tList[sort]
    
    
    fractionP=dataFile['fractionP'][sort]
    fraction0=dataFile['fraction0'][sort]
    fractionM=dataFile['fractionM'][sort]
    
    fractions=np.append(np.append(fractionM,fraction0),fractionP)
    fractions=fractions.reshape(3,fraction0.size).transpose()
    
    if S==2:
        fractionP2=dataFile['fractionP2'][sort]
        fractionM2=dataFile['fractionM2'][sort]
        fractions=np.append(np.append(np.append(np.append(fractionM2,fractionM),fraction0),fractionP),fractionP2)
        fractions=fractions.reshape(5,fraction0.size).transpose()
    
    
    
    tRecoils = np.array(tList*Erecoil/hbar)
    fractions=np.array(fractions)
    
    popt,pcov = optimize.curve_fit(HofT,tRecoils,fractions.flatten(), p0=(omegaGuess,deltaGuess))
    pcov = np.sqrt(np.diag(pcov))
    
    tForFit=np.linspace(np.min(tRecoils),np.max(tRecoils),100)
    pops_fitted=HofT(tForFit,*popt)
    
    pops_fitted=pops_fitted.reshape(tForFit.size,2*S+1).transpose()
    
    figure=plt.figure()
    panel=figure.add_subplot(1,1,1)
    panel.set_title( r'$\Omega$ = %.3f $\pm$ %.3f $E_L/h$, $\delta$ = %.3f $\pm$ %.3f $E_L/h$' %(popt[0],pcov[0],popt[1],pcov[1]))#, U= '+str(np.round(popt[2],3))+r'$E_L$')
    
    panel.plot(tList*1.0e6,fractionP,'bo', label=r'$m_F$=+1')
    panel.plot(tList*1.0e6,fraction0,'go', label=r'$m_F$=0')
    panel.plot(tList*1.0e6,fractionM,'ro', label=r'$m_F$=-1')
    
    
    panel.plot(tForFit*hbar*1.0e6/Erecoil,pops_fitted[S],'g-')
    panel.plot(tForFit*hbar*1.0e6/Erecoil,pops_fitted[S+1],'b-')
    panel.plot(tForFit*hbar*1.0e6/Erecoil,pops_fitted[S-1],'r-')
    
    if S==2:
        panel.plot(tForFit*hbar*1.0e6/Erecoil,pops_fitted[S+2],'c-')
        panel.plot(tForFit*hbar*1.0e6/Erecoil,pops_fitted[S-2],'m-')
        panel.plot(tList*1.0e6,fractionM2,'mo', label=r'$m_F$=-2')
        panel.plot(tList*1.0e6,fractionP2,'co', label=r'$m_F$=+2')
    panel.set_xlabel('Lattice pulse time [us]')
    plt.legend()
    
    print r'Fitted values $\Omega$ = %.3f $\pm$ %.3f $E_L/h$, $\delta$ = %.3f $\pm$ %.3f $E_L/h$' %(popt[0],pcov[0],popt[1],pcov[1])
    return