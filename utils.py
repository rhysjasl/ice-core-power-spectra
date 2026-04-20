# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.fft import fft, fftfreq

def discrete_avg(old_x: np.ndarray, old_y: np.ndarray, interval: int, method: str='full'):
    """
    Discretely average the data to downsample to a specified resolution
    
    Args:
        old_x (np.ndarray): Original time axis
        old_y (np.ndarray): Original "proxy" data corresponding to old_x
        interval (int): Desired sampling interval
        method (string, optional): Method for treating the last point

    Returns:
        new_x (np.ndarray): New evenly-spaced time axis at desired sampling resolution
        new_y (np.ndarray): New "proxy" data discretely averaged, corresponding to desired sampling resolution
    """
    diff = np.mean(np.diff(old_x)) # if there is missing data, new x values will be slightly shifted
    wdw = interval / 2
    x0 = old_x[0] + wdw - (diff / 2)

    # choose new x values based on method (how last point is treated)
    if method == 'full':
        new_x = np.arange(x0, old_x[-1] + (diff/2), interval)
        # this will properly allow the last point to be created only if there is enough data to cover at least half of the interval
    elif method == 'cap':
        new_x = np.arange(x0, old_x[-1] - wdw + diff, interval)
        # this will only create the last point if there is enough data to cover the full interval
    else:
        raise ValueError("method must be either 'full' or 'cap'")
    
    new_y = np.zeros_like(new_x)

    for i in range(len(new_x)):
        mask = (old_x >= new_x[i] - wdw) & (old_x < new_x[i] + wdw)
        if np.any(mask):
            new_y[i] = np.mean(old_y[mask])
        else:
            new_y[i] = np.nan # handle case where no data points fall in the interval

    return new_x, new_y

# define boxcar moving average function
def boxcar(data: np.ndarray, window_size: int):
    """
    Calculates the boxcar moving average of a 1D array.

    Args:
        data (np.ndarray): The input 1D array of numerical data.
        window_size (int): The size of the moving window.
    
    Returns:
        np.ndarray: The array containing the boxcar moving average.
    """
    wdw = np.ones(window_size) / window_size

    return np.convolve(data, wdw, mode='valid')

# define gaussian moving average function
def gaussian(data: np.ndarray, window_size: int, sigma: float=None):
    """
    Calculates the Gaussian moving average of a 1D array.

    Args:
        data (np.ndarray): The input 1D array of numerical data.
        window_size (int): The size of the moving window. Must be an odd integer.
        sigma (float, optional): The standard deviation of the Gaussian kernel.
                                 If None, it defaults to window_size / 6.

    Returns:
        np.ndarray: The array containing the Gaussian moving average.
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer.")

    if sigma is None:
        sigma = window_size / 6  # A common heuristic for sigma

    # Create the Gaussian kernel
    half_window = window_size // 2
    x = np.arange(-half_window, half_window + 1)
    gaussian_kernel = np.exp(-(x**2) / (2 * sigma**2))
    gaussian_kernel /= np.sum(gaussian_kernel)  # Normalize the kernel

    # Convolve the data with the Gaussian kernel
    # 'same' mode ensures output size is the same as input
    smoothed_data = np.convolve(data, gaussian_kernel, mode='same')

    return smoothed_data

# define function to find at which frequency a band of the moving avg PSD is 95% of its own PSD at low frequencies
def find_95_self(freqp: np.ndarray, psdp: np.ndarray, bandwidth: float=0.0001):
    """
    Finds the frequency at which the moving average PSD reaches 95% of the raw PSD.

    Args:
        freqp (np.ndarray): Frequencies of the moving average PSD.
        pspd (np.ndarray): Moving average PSD values.
        bandwidth (float): Bandwidth around each frequency to consider for averaging.
    
    Returns:
        float: Frequency at which the moving average PSD reaches 95% of the raw PSD.
    """
    threshold = np.mean(psdp[freqp < 0.001]) * 0.95  # 95% of the mean PSD at low frequencies

    # cycle from highest to lowest frequency
    for f in freqp[::-1]:
        # Find indices within the bandwidth
        mask = (freqp >= f - bandwidth / 2) & (freqp <= f + bandwidth / 2)
        if np.any(mask):
            avg_psd = np.mean(psdp[mask])
            if avg_psd >= threshold:
                return f
    
    return None  # Return None if no frequency meets the criteria

# define resampling function
def depth_to_age(depth_x: np.ndarray, depth_y: np.ndarray, resolution: float, old_depth: np.ndarray, old_age: np.ndarray):
    """
    Resample data from depth to age domain at specified resolution

    Args:
        depth_x (np.ndarray): Newly-created evenly-spaced depth axis corresponding to desired chronology
        depth_y (np.ndarray): Original "proxy" data that has been interpolated to the new evenly-spaced depth axis
        resolution (float): Desired resampling resolution to convert to time axis
        old_depth (np.ndarray): Original ice core chronology depths
        old_age (np.ndarray): Original ice core chronology ages corresponding to each depth

    Returns:
        new_age (np.ndarray): New evenly-spaced time axis for resampled synthetic data
        new_y (np.ndarray): New "proxy" data that has been re-interpolated to correspond to the new time axis
    """
    min_age = np.min(np.interp(depth_x, old_depth, old_age))
    max_age = np.max(np.interp(depth_x, old_depth, old_age))
    new_age = np.arange(min_age, max_age, resolution)
    new_depth = np.interp(new_age, old_age, old_depth)
    new_y = np.interp(new_depth, depth_x, depth_y)
    
    return new_age, new_y