import numpy as np
import warnings
import scipy.optimize as op
import warnings
import LMC
from   scipy import stats

pi = np.pi

MSME = 332948.6   # (M_sun/M_earth)
BIGG = 6.67e-11    # Newton's constant [SI units]
RSUN = 6.957e8     # solar radius [m]
MSUN = 1.988e30    # Solar mass [kg]


###

# Functions containing relevant physics

def P_to_a(P, Mstar):
    '''
    Convenience function to convert periods to semimajor axis from Kepler's Law
    
    P: orbital periods [days]
    Mstar: stellar mass [solar masses]
    '''
    Pearth = 365.24    # [days]
    aearth = 215.05    # [solar radii]
    
    return aearth * ((P/Pearth)**2 *(1/Mstar))**(1/3)


def calculate_duration(period, rho, rprs, cosi):
    '''
    Helper function to calculate transit duration predicted from a circular orbit
    
    period: orbital period [days]
    rho: stellar density [solar density]
    rprs: planet-to-star radius ratio for system
    cosi: cosine(inclination)
    '''
    G = BIGG / RSUN**3 * MSUN * (24*3600)**2    # Newton's constant [R_sun^3 * M_sun^-1 * days^-2]
    
    term3  = ((3*period)/(G*rho*pi**2))**(1/3)
    term2a = (1+rprs)**2
    term2b = ((G*rho)/(3*pi))**(2/3)
    term2c = period**(4/3)*cosi**2
    term2  = (term2a - term2b*term2c)**(1/2)
    
    return term3*term2


def residuals_for_duration_fit(x0, x1, data_dur, data_err):
    '''
    Helper function to return residuals for least squares fitting (op.leastsq)
    
    x0: vector of parameters to vary in fit (cosi)
    x1: vector of parameters to hold constant (periods, rhostar, rprs)
    data_dur: measured transit durations [days]
    data_err: corresponding errors [days]
    '''
    cosi = x0
    period, rho, rprs = x1
    
    model_dur = calculate_duration(period, rho, rprs, cosi)
    
    return (data_dur - model_dur)/data_err


def calculate_flatness(data_dur, model_dur):
    '''
    Helper function to calculate flatness
    
    data_dur: measured transit durations [days]
    model_dur: model transit durations [days] from leastsq fit
    '''
    return np.std(data_dur-model_dur)/np.sqrt(np.mean(data_dur**2))



# Functions to compute system-level complexity measures
# Quantities definied in Gilbert & Fabrycky (2019)


def mu(mp, Mstar):
    '''
    Dynamical mass
    
    mp: array of planet masses [M_earth]
    Mstar: stellar mass [M_sun]
    '''
    return np.sum(mp)/MSME/Mstar


def Q(masses):
    '''
    Mass partitioning
    
    masses: array of planet masses
    '''
    return LMC.D(masses/np.sum(masses))


def M(periods, masses):
    '''
    Monotonicity
    
    periods: array of planet periods
    masses: array of planet masses corresponding to each given period
    '''
    N = len(periods)
    rho = stats.spearmanr(periods, masses)[0]
    
    return rho*Q(masses)**(1/N)


def S(periods, mp, Mstar):
    '''
    Characteristic spacing
    
    periods: array of planet periods [days]
    mp: array of planet masses [M_earth]
    Mstar: Stellar mass [M_sun]
    '''
    a = P_to_a(periods, Mstar)
    
    radius_H = ((mp[1:]+mp[:-1])/(3*Mstar*MSME))**(1/3) * (a[1:]+a[:-1])/2
    delta_H  = (a[1:]-a[:-1])/radius_H
    
    return np.mean(delta_H)


def C(periods, warn=True):
    '''
    Gap complexity
    
    periods: array of planet periods
    warn: boolean flag to control warnings (default=True)
    '''
    if len(periods) < 3:
        if warn:
            warnings.warn('Complexity is undefined for N < 3; returning NaN')
        return np.nan
    elif len(periods) >= 3:
        order = np.argsort(periods)
  
        P = np.array(periods)[order]
        pp = np.log(P[1:]/P[:-1])/np.log(P.max()/P.min())
        
        return LMC.C(pp)
    

def f(periods, rhostar, rprs, dur, dur_err):
    '''
    periods: orbital periods [days]
    rhostar: stellar density [solar density]
    rprs: planet-to-star radius ratio
    dur: transit durations [hours]
    dur_err: corresponding errors on transit durations
    '''
    
    cosi = np.array([0.0])
    transit_params = [periods, rhostar, rprs]
        
    cosi, success = op.leastsq(residuals_for_duration_fit, cosi, args=(transit_params, dur, dur_err))
        
    #model_dur = calculate_duration(periods, rhostar, np.median(rprs), cosi)
    model_dur = calculate_duration(periods, rhostar, rprs, cosi)
    
    return calculate_flatness(dur, model_dur)