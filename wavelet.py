import numpy as np
import warnings
from scipy import signal, interpolate

def autocorr(x, sigma_x2):
    """Compute scaled autocorrelation of x"""
    res = np.correlate(x[:-1], x[1:], mode='valid') / ((len(x) - 1) * sigma_x2)
    return res[len(res)//2:]

def calc_spec_ar1(x, f):
    """Compute the theoretical spectrum of an AR1 Process"""
    sigma_x2 = np.var(x)
    # select lag1 autocorrelation
    alpha1 = autocorr(x, sigma_x2)[0]
    sigma_z2 = (1-alpha1**2) * sigma_x2
    ar1 = sigma_z2 / (1 + alpha1**2 - 2*alpha1*np.cos(2*np.pi*2*f))
    return ar1

def significance_levels(x, f):
    """
    Returns the 95 % confidence levels based on the theorectical 
    spectrum of an AR1 Process. 

    Parameters
    ----------
    x : (N,) ndarray
        Data vector.
    f : (N,) ndarray
        Frequency of the power spectrum.

    Returns
    -------
    significance : (N,) ndarray
        Significance levels
    """
    # Estimate the AR1 spectrum
    spec_ar1 = calc_spec_ar1(x, f)
    # from Chi^2 distribution table
    deg_freedom = 2
    chi2_critical = 5.9915
    # he significance levels
    significance = spec_ar1 * (chi2_critical / deg_freedom)
    return significance

def interp(t, x, missing_value=None, remove_outliers=3):
    """
    Interpolate a 1-D function using cubic splines.
    Remove infinte data points and outliers,

    Parameters
    ----------
    t : (N,) ndarray
        Independent variable
    x : (N,) ndarray
        Dependent variable. Must match the length of t.
    missing_value : float
        If not NaN; will be replaced by NaN and interpolated.
    remove_outliers : float, positive
        Outlier criterion based on z-scor. remove_outliers = 3 by default. 
        False if outliers are not to be removed.

    Returns
    -------
    - ti : (N,) ndarray
        Equally spaced t.
    - xi : (N,) ndarray
        x interpolated on ti.
    """
    # equally spaced t vector for interpolation
    ti = np.linspace(t[0], t[-1], len(t))
    
    # identify and remove infinte and missing values
    x = np.where(x==missing_value, np.nan, x)
    idx_finite = np.isfinite(x)
    t = t[idx_finite]
    x = x[idx_finite]
    n_infinite = sum(~idx_finite)
    # outliers were set to infinite
    if (n_infinite) > 0:
        warnings.warn(f"{n_infinite} infinite data points removed.")
    
    # identify and remove outlierts
    if remove_outliers:
        z_score = np.abs((x - np.mean(x)) / np.std(x))
        idx_outliers = z_score > remove_outliers
        t = t[~idx_outliers]
        x = x[~idx_outliers]
        n_outliers = sum(idx_outliers)
        if n_outliers > 0:
            warnings.warn(f"{n_outliers} outliers removed.")

    
    cs = interpolate.CubicSpline(t, x)
    xi = cs(ti)
    return ti, xi

def compute_cwt(
    x, 
    t, 
    wavelet = signal.morlet2, 
    widths = None, 
    missing_value = None,
    remove_outliers = 3,
    **kwargs
):
    """
    Performs a continuous wavelet transform on x.
    Default wavelet function is scipy.morlet2 but other functions taking at least x and a 'widths' parameter as arguments can be used. 

    Parameters
    ----------
    x : (N,) array_like
        Data on which to perform the wavelet transform.
    t : (N,) array_like
        The variable x depend on. Must have the same length as x.
    wavelet : function
        Wavelet function which takes x and widths as arguments.
    widths : (M,) sequence, optional
        Scaling parameter for the wavelet. Based on the sampling frequency by default.
    missing_value : float, optional
        If not NaN; will be replaced by NaN and interpolated.
    remove_outliers : float, positive
        Outlier criterion based on z-scor. remove_outliers = 3 by default. 
        False if outliers are not to be removed.
    kwargs
        Keyword arguments passed to wavelet function.
        
    Returns
    -------
    - cwtm : (M,N) ndarray
    - cwtm_sig : (M,N) ndarray
        Significance level. Significant if >= 1.
    - freq : (N) ndarray
    
    """
    dt = (t[-1] - t[0]) / len(t)
    fs = 1 / dt
    freq = np.linspace(0, fs/2)
    
    if "w" not in kwargs:
        w = 5
    if widths == None:
        widths = w*fs / (2*freq*np.pi)
    
    # remove infinite data and interpolate
    t, x = interp(t, x, missing_value, remove_outliers)
        
    # Compute the cwt and power
    cwtm = signal.cwt(x, wavelet, widths, **kwargs)
    cwtm = np.abs(cwtm**2)
    
    # 95 % convidence level: 'one slice' i.e. (N,)
    siglvl = significance_levels(x, freq)
    # expand to (M,N)
    siglvl = np.outer(siglvl, np.ones(len(t)))
    # where ratio > 1, power is significant
    cwtm_sig = cwtm / siglvl  
    return cwtm, cwtm_sig, freq