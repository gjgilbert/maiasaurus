import numpy as np
import warnings
import scipy.optimize as op

pi = np.pi


#####

def H(p, normalize_output=True):
    '''
    Calculates Shannon information (in nats) from a probability vector

    p: vector of probabilities; will be normalized if not done so already
    normalize_output: boolean flag to normalize output to range (0,1); default=True

    -- returns
        Hout: Shannon information
    '''
    # check probabilities normalization
    if np.isclose(np.sum(p),1.0) != True:
        warnings.warn('Input probability vector was not normalized...fixing automatically')
        p = p/np.sum(p)

    # calculate entropy
    N = len(p)

    if normalize_output:
        K = 1/np.log(N)
    else:
        K = 1.0

    return -K*np.sum(p*np.log(p))



def D(p, normalize_output=True):
    '''
    Calculates disequliibrium from a probability vector

    p: vector of probabilities; will be normalized if not done so already
    normalize_output: boolean flag to normalize output to range (0,1); default=True

    -- returns
        Dout: disequilibrium
    '''
    # check probabilities normalization
    if np.isclose(np.sum(p),1.0) != True:
        warnings.warn('Input probability vector was not normalized...fixing automatically')
        p = p/np.sum(p)

    # calculate disequilibrium
    N = len(p)

    if normalize_output:
        K = N/(N-1)
    else:
        K = 1.0
    
    return K*np.sum((p-1/N)**2)



def C(p, normalize_output=True):
    '''
    Calculates LMC complexity from a probability vector

    p: vector of probabilities; will be normalized if not done so already
    normalize_output: boolean flag to normalize output to range (0,1); default=True

    Normalization determined from a power law fit to Anteneodo & Plastino (1996)
    -- returns
        Cout: LMC complexity
    '''
    # check probabilities normalization
    if np.isclose(np.sum(p),1.0) != True:
        warnings.warn('Input probability vector was not normalized...fixing automatically')
        p = p/np.sum(p)

    # calculate disequilibrium
    N = len(p)

    if normalize_output:
        K = 1/Cmax(N)
        
    else:
        K = 1.0

    return K * H(p, False)*D(p, False)



def ap9(p,N):
    '''
    Eq.9 from Anteneodo & Plastino (1996) for fixed n=1
    '''
    return (2-3*p+1/N)*np.log((1-p)/(N-1)) + (3*p-1/N)*np.log(p)



def ap10(p,N):
    '''
    Eq.10 from Anteneodo & Plastino (1996) for fixed n=1
    '''
    return (1-2*p+p/N)*np.log((1-p)/(N-1)) + p*(2-1/N)*np.log(p) - (p-1/N)



def Cmax(N):
    '''
    Calculates maximum complexity (Cmax) for a given N
    Numerically solves equations from Anteneodo & Plastino (1996)
    
    N: array_like, all entries expected to be integers >= 2; returns np.nan for any N < 2
    '''
    N = np.atleast_1d(N)
    
    # check than N is
    for n in N:
        if n % 1 != 0:
            raise ValueError('N must be in integer')    
    
    Cout = np.zeros_like(N, dtype='float')
    for i, n in enumerate(N):
        if n < 2:
            Cout[i] = np.nan
        else:
            if n == 2: p0 = 0.85
            else:      p0 = 2/3

            popt9  = op.fsolve(ap9, p0, args=(n))
            popt10 = op.fsolve(ap10, p0, args=(n)) 

            pall = np.zeros(n)
            pall[0] = popt9
            pall[1:] = (1-popt9)/(n-1)

            Cout[i] = -np.sum(pall*np.log(pall))*np.sum((pall-1/n)**2)

    return Cout



