import numpy as np
from scipy.special import jacobi, comb, factorial, gamma

# Conveniently splits the files' name
def param_inbetween(fname, del1, del2):
    return fname.split(del1)[-1].split(del2)[0]

# Extract parameters from files' name
def get_sim_params(fname):

    Q       =   int(param_inbetween(fname, "Q", "_"))
    SCHEME  =   str(param_inbetween(fname, "Q%d_"%Q, "_X"))
    # Q   =   int(param_inbetween(fname, "Q" ,  "_X"))
    nph     =   int(param_inbetween(fname, "X" ,   "R"))
    R       = float(param_inbetween(fname, "R" ,   "Y"))
    nth     =   int(param_inbetween(fname, "Y" , "_dt"))
    dt      = float(param_inbetween(fname, "dt", "_Ta"))
    tau     = float(param_inbetween(fname, "Ta", "rho"))
    V0      = float(param_inbetween(fname, "a" ,   "."))
    it      =   int(param_inbetween(fname, "." ,   " "))

    return Q, SCHEME, nph, nth, R, dt, tau, V0, it

# Collect polynomials in dumpfile as columns: t, F0, F1, F2, F3
def poly_collector(dumpfile, idx, t, amplF):
    
    dumpfile[idx,0] = t     # Actual time = it*dt
    for aidx in range(len(amplF)):
        dumpfile[idx,aidx+1] = amplF[aidx]

# Average over the domain
def poly_average(Y,nph,nth):
    W = Y.reshape((int(nph + 1), int(nth + 1)))
    W = W[:-1,:-1] # Cut off last theta and phi values
    return W.mean()


#######################################################################################

# Evaluate analytic solution for shear wave up to n-th mode

def shear_a(n): # Eq. (1.50) (Notice: no V)
    return (np.pi * ((2*n + 1)*(4*n + 3))**.5 / (4*(n + 1)**1.5) )* \
            ( ( 1./2**(2*n) ) * comb(2*n, n) )**2 

def shear_ampl_F(t,R,tau,n):
    return shear_a(n)*np.exp(- tau*4.*n*(n + 1.5)*(1./R**2) * t)

def shear_norm_F(n):
    return ( (2*n + 1)*(4*n + 3) / (4*(n + 1)) )**.5

def shear_F(th,n):
    return shear_norm_F(n) * jacobi(n,alpha=1,beta=-0.5)(np.cos(2*th))


def sol_shear(R,t,th,tau,n=20):
    sum_n = 0.0
    for idx in range(n+1):
        sum_n += (shear_ampl_F(t,R,tau,idx) * shear_F(th,idx)  )
        
    return np.sin(th) * sum_n

####################################################################################
# Evaluate analytic solution for sound wave up to n-th mode

def jacobi_neg(n,alpha,beta,z):  # Jacobi polynomials for alpha < -1
    sum_n = 0.0
    for m in range(n+1):
        sum_n += comb(n,m) * gamma(alpha + beta + n + m + 1) * (alpha + m + 1) \
                / gamma(alpha + m + 2)  * ((z - 1)*.5)**m
    return gamma(alpha + n + 1) / ( factorial(n) * gamma(alpha + beta + n + 1) ) * sum_n

def sound_norm_F(n):
    return (-1)**n * ( n*(4*n - 1) / (2*n - 1) )**.5

def sound_F(th,n):
    return sound_norm_F(n) * jacobi_neg(n,alpha=-1,beta=-0.5,z=np.cos(2*th))

def lambdasqn(n):
    return 2*n*(2*n-1)

def zetan(R, tau, n): # Eq. (1.65)
    return tau * (lambdasqn(n) - 1) / R**2
   
def alphan(R, tau, n): # Eq. (1.64+1.65)
    return np.sqrt(lambdasqn(n) - zetan(R, tau, n)**2)    

##############################################################################
def sound_ampl_F(t, R, tau, n, key_init=0): # Specify key_init=0,1 to switch to the desired amplitudes
    an = 0
    if key_init == 0:    
        if n == 1:
            an = 2/np.sqrt(3)

    elif key_init == 1:
        if n == 1:
            an = np.pi*((np.pi**2+3)/8./np.sqrt(3))
        elif n == 2:
            an = np.pi*(-np.sqrt(7)*(4*np.pi**2-33.)/256/np.sqrt(6))
        elif n == 3:
            an = np.pi*( np.sqrt(55)*(4*np.pi**2-37.)/2048/np.sqrt(3))
        
    return an * np.exp(- zetan(R, tau, n) * t) * ( np.cos( alphan(R, tau, n) * t ) -
                                                   (zetan(R, tau, n)/ alphan(R, 0, n) ) * np.sin( alphan(R, tau, n) * t ) )

def sol_sound(R, t, th, tau, n=1, key_init=0): # Solution for the key_init=1 benchmark   
    sum_n = 0.0
    for idx in range(1,n+1):
        sum_n += ( sound_ampl_F(t, R, tau, idx, key_init) * sound_F(th,idx)  )
    
    return (1./ np.sin(th)) * sum_n