# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.fft import fft, fftfreq
from scipy import stats

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
    Calculates the boxcar moving average of a 1D array

    Args:
        data (np.ndarray): The input 1D array of numerical data
        window_size (int): The size of the moving window
    
    Returns:
        np.ndarray: The array containing the boxcar moving average
    """
    wdw = np.ones(window_size) / window_size

    return np.convolve(data, wdw, mode='valid')

# define gaussian moving average function
def gaussian(data: np.ndarray, window_size: int, sigma: float=None):
    """
    Calculates the Gaussian moving average of a 1D array

    Args:
        data (np.ndarray): The input 1D array of numerical data.
        window_size (int): The size of the moving window. Must be an odd integer
        sigma (float, optional): The standard deviation of the Gaussian kernel
                                 If None, it defaults to window_size / 6

    Returns:
        np.ndarray: The array containing the Gaussian moving average
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
    Finds the frequency at which the moving average PSD reaches 95% of the raw PSD

    Args:
        freqp (np.ndarray): Frequencies of the moving average PSD
        pspd (np.ndarray): Moving average PSD values
        bandwidth (float): Bandwidth around each frequency to consider for averaging
    
    Returns:
        float: Frequency at which the moving average PSD reaches 95% of the raw PSD
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

# define function to perform all analyses
def run_analyses(x: np.ndarray, y: np.ndarray, depth: np.ndarray, age: np.ndarray, interval: int | list[int], window: int | list[int], method: str='full', sigma: float=None, segments: str='default'):
    """
    Runs all analyses for the power spectra notebook, including: 
    1) comparing FFT vs. Welch method for spectral analysis,
    2) comparing the effects of complete vs. incomplete discrete sampling on the resulting spectra, 
    3) comparing the effects of different smoothing windows on the resulting spectra, 
    4) comparing the effects of resampling from depth to age domain at different resolutions on the resulting spectra

    Args:
        x (np.ndarray): Time axis for the synthetic data
        y (np.ndarray): "Proxy" data values for the synthetic data
        depth (np.ndarray): Original ice core chronology depths
        age (np.ndarray): Original ice core chronology ages corresponding to each depth
        interval (int | list[int]): Desired sampling interval(s) for discrete sampling
        window (int | list[int]): Desired window size(s) for continuous sampling
        method (str, optional): Method for treating the last point in discrete sampling. Defaults to 'full'
        sigma (float, optional): Standard deviation for Gaussian smoothing. If None, it defaults to window_size / 6
        segments (str, optional): Method for handling segment lengths in Welch method. Defaults to 'default'
                - 'default': use default segment lengths for all analyses
                - 'number': specify a number of segments to use for all analyses (segment length will be determined by this number)
                - 'length': specify a segment length to use for all analyses (number of segments will be determined by this length)
                - 'custom': specify a list of segment lengths to use for each analysis

    Returns:
        None - will produce plots and print statements for all analyses
    """
    ### Handle segment lengths for Welch method ###
    if segments == 'default':
        seg_int = [None] * (len(interval) + 1)
        seg_wdw = [None] * (len(window) + 1)
        print('Using default segment lengths...')
    elif segments == 'number':
        num_seg = int(input('How many segments would you like?'))
        seg_int = np.zeros(len(interval) + 1)
        seg_wdw = np.zeros(len(window) + 1)
        seg_int[0] = len(x) / num_seg
        seg_wdw[0] = len(x) / num_seg
        print(f'Each spectrum produced using the Welch method will use {num_seg} segments.')
    elif segments == 'length':
        len_seg = int(input('How long should each segment be? Please enter an integer.'))
        seg_int = [len_seg] * (len(interval) + 1)
        seg_wdw = [len_seg] * (len(window) + 1)
        print(f'Each spectrum produced using the Welch method will use segment length {len_seg}.')
    elif segments == 'custom':
        while True:
                try:
                    seg_int = list(map(int, input('Please enter a list of integers containing a segment length for raw data and each interval. Enter numbers separated by space: ').split()))
                    if len(seg_int) == len(interval) + 1:
                        print('Custom segment lengths for discrete sampling successfully received.')
                        break  # Exit loop if conversion is successful
                    else: 
                        raise ValueError('Invalid list')
                except ValueError as e:
                    if str(e) == 'Invalid list':
                        print(f'Your list must contain exactly {len(interval) + 1} elements')
                    else: 
                        print('That is not a valid list. Try again.')
        while True:
                try:
                    seg_wdw = list(map(int, input('Please enter a list of integers containing a segment length for raw data and each window size. Enter numbers separated by space: ').split()))
                    if len(seg_wdw) == len(window) + 1:
                        print('Custom segment lengths for continuous sampling successfully received.')
                        break  # Exit loop if conversion is successful
                    else: 
                        raise ValueError('Invalid list')
                except ValueError as e:
                    if str(e) == 'Invalid list':
                        print(f'Your list must contain exactly {len(window) + 1} elements')
                    else: 
                        print('That is not a valid list. Try again.')
    else:
        raise ValueError('Segments method must be "default", "number", "length", or "custom".')
    
    ### Compare FFT vs. Welch ###
    print('~~~ Comparing FFT vs. Welch Method for Spectral Analysis ~~~')
    res = np.mean(np.diff(x))
    fs = 1 / res  # sampling frequency (samples/yr)

    freqy, psdy = welch(y, fs=fs, nperseg=seg_int[0]) # frequency units: cycles/yr, output value: psd
    fourier = fft(y) # output value: amplitude
    freqf = fftfreq(len(x), res)[:len(x)//2] # frequency units: cycles/yr
    psdf = 2 * np.abs(fourier)**2 / (len(x) * fs) # convert amplitude to psd

    plt.figure()
    # plot fft of initial time series
    plt.plot(freqf, psdf[:len(x)//2], label='Fast Fourier Transform', color='black')
    # plot welch method initial time series
    plt.loglog(freqy, psdy, label='Welch Method', color='red')
    plt.axvline(x=1/res/2, color='gray', linestyle='--', label=f'')
    plt.xlabel('Frequency (cycles/year)')
    plt.ylabel(r'Power Spectral Density (units$^2$/cycle/year)')
    plt.title('Power Spectral Density of Raw Synthetic Data')
    plt.grid()
    plt.legend()
    plt.show()

    ### Complete Discrete Averaging ###
    print('~~~ Comparing Effects of Complete vs. Incomplete Discrete Sampling on Resulting Spectra ~~~')
    fig1, ax1 = plt.subplots(3, 1, figsize=(8, 9)) # plots for complete discrete sampling
    fig2, ax2 = plt.subplots(3, 1, figsize=(8, 9)) # plots for incomplete discrete sampling
    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 4)) # plot for comparison

    fourier = fft(y) # output value: amplitude
    freqf = fftfreq(len(x), 1)[:len(x)//2] # frequency units: cycles/yr
    psdf = 2 * np.abs(fourier)**2 / (len(x) * 1) # convert amplitude to psd
    ax1[0].loglog(freqf, psdf[:len(x)//2], label='Raw data', color='black') # plot welch of raw data
    ax1[0].axvline(x=1/2, color='gray', linestyle='--', label=f'')
    ax2[0].loglog(freqf, psdf[:len(x)//2], label='Raw data', color='black') # plot welch of raw data
    ax2[0].axvline(x=1/2, color='gray', linestyle='--', label=f'')
    ax1[1].loglog(freqy, psdy, label='Raw data', color='black') # plot welch of raw data
    ax1[1].axvline(x=1/2, color='gray', linestyle='--', label=f'')
    ax2[1].loglog(freqy, psdy, label='Raw data', color='black') # plot welch of raw data
    ax2[1].axvline(x=1/2, color='gray', linestyle='--', label=f'')

    for i, res in enumerate(interval):
        x_ds, y_ds = discrete_avg(x, y, res, method=method)
        y_int = np.interp(x_ds, x, y) # interpolate to downsample
        fs = 1 / res  # sampling frequency (samples/yr)
        if segments == 'number':
            seg_int[i+1] = len(x_ds) / num_seg

        # perform spectral analysis for complete discrete sampling
        freqw, psdw = welch(y_ds, fs=fs, nperseg=seg_int[i+1]) # frequency units: cycles/yr, output value: psd
        fourier = fft(y_ds) # output value: amplitude
        freqf = fftfreq(len(x_ds), res)[:len(x_ds)//2] # frequency units: cycles/yr
        psdf = 2 * np.abs(fourier)**2 / (len(x_ds) * fs) # convert amplitude to psd

        # perform spectral analysis for incomplete discrete sampling
        freqwi, psdwi = welch(y_int, fs=fs, nperseg=seg_int[i+1]) # frequency units: cycles/yr, output value: psd
        fourieri = fft(y_int) # output value: amplitude
        psdfi = 2 * np.abs(fourieri)**2 / (len(x_ds) * fs) # convert amplitude to psd

        color = f'C{i}'

        # plot resulting spectra : complete discrete sampling
        ax1[0].loglog(freqf, psdf[:len(x_ds)//2], label=f'{res} yr', color=color) # plot fft
        ax1[0].axvline(x=1/res/2, color=color, linestyle='--', alpha=0.6)
        ax1[0].set_xlabel('Frequency (cycles/year)')
        ax1[0].set_ylabel(r'PSD (units$^2$/cycle/year)')
        ax1[0].set_title(f'Complete Discrete Sampling (FFT)')
        ax1[0].grid()
        ax1[0].legend()

        ax1[1].loglog(freqw, psdw, label=f'{res} yr', color=color) # plot welch
        ax1[1].axvline(x=1/res/2, color=color, linestyle='--', alpha=0.6)
        ax1[1].set_xlabel('Frequency (cycles/year)')
        ax1[1].set_ylabel(r'PSD (units$^2$/cycle/year)')
        ax1[1].set_title(f'Complete Discrete Sampling (Welch Method)')
        ax1[1].grid()
        ax1[1].legend()

        # plot resulting spectra : incomplete discrete sampling
        ax2[0].loglog(freqf, psdfi[:len(x_ds)//2], label=f'{res} yr', color=color) # plot fft
        ax2[0].axvline(x=1/res/2, color=color, linestyle='--', alpha=0.6)
        ax2[0].set_xlabel('Frequency (cycles/year)')
        ax2[0].set_ylabel(r'PSD (units$^2$/cycle/year)')
        ax2[0].set_title(f'Incomplete Discrete Sampling (FFT)')
        ax2[0].grid()
        ax2[0].legend()

        ax2[1].loglog(freqwi, psdwi, label=f'{res} yr', color=color) # plot welch
        ax2[1].axvline(x=1/res/2, color=color, linestyle='--', alpha=0.6)
        ax2[1].set_xlabel('Frequency (cycles/year)')
        ax2[1].set_ylabel(r'PSD (units$^2$/cycle/year)')
        ax2[1].set_title(f'Incomplete Discrete Sampling (Welch Method)')
        ax2[1].grid()
        ax2[1].legend()

        if res == interval[-1]:
            # handle complete discrete sampling
            ax1[2].plot(freqf, psdf[:len(x_ds)//2], label='Fast Fourier Transform', color='black') # plot fft
            ax1[2].loglog(freqw, psdw, label='Welch Method', color='red') # plot welch
            ax1[2].axvline(x=1/res/2, color='gray', linestyle='--', label=f'')
            ax1[2].set_xlabel('Frequency (cycles/year)')
            ax1[2].set_ylabel(r'PSD (units$^2$/cycle/year)')
            ax1[2].set_title(f'PSD of Downsampled Data (Sampling Resolution: {res} years)')
            ax1[2].grid()
            ax1[2].legend()

            # handle incomplete discrete sampling
            ax2[2].plot(freqf, psdfi[:len(x_ds)//2], label='Fast Fourier Transform', color='black') # plot fft
            ax2[2].loglog(freqwi, psdwi, label='Welch Method', color='red') # plot welch
            ax2[2].axvline(x=1/res/2, color='gray', linestyle='--', label=f'')
            ax2[2].set_xlabel('Frequency (cycles/year)')
            ax2[2].set_ylabel(r'PSD (units$^2$/cycle/year)')
            ax2[2].set_title(f'PSD of Downsampled Data (Sampling Resolution: {res} years)')
            ax2[2].grid()
            ax2[2].legend()

        # plot comparison
        ax3.loglog(freqw, psdwi/psdw, label=f'{res} yr', color=color) # plot welch ratio
        ax3.axvline(x=fs/2, color=color, linestyle='--', alpha=0.6)
        ax3.axhline(y=res/2, color=color, linestyle='--', alpha=0.4)  # reference line
        ax3.set_xlabel('Frequency (cycles/year)')
        ax3.set_ylabel('PSD Ratio (unitless)')
        ax3.set_title(f'PSD incomplete/complete (Welch Method)')
        ax3.grid()
        ax3.legend()
        ax3.set_ylim([1e0, 5e4]);
    
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    plt.show()

    ### Continuous Sampling ###
    print('~~~ Comparing Effects of Different Smoothing Windows on Resulting Spectra ~~~')
    # boxcar moving average
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    fig5, ax5 = plt.subplots(figsize=(8, 4))

    ax4.loglog(freqy, psdy, label='Raw data', color='black') # plot welch of raw data
    ax5.loglog(freqy, psdy, label='Raw data', color='black') # plot welch of raw data

    # produce spectra of smoothed data
    for i, wdw in enumerate(window):
        res = np.mean(np.diff(x)) # downsampling resolution (years)
        fs = 1 / res  # sampling frequency (samples/yr)
        color = f'C{i}'

        box = boxcar(y, wdw)
        gaus = gaussian(y, wdw, sigma)
        if segments == 'number':
            seg_wdw[i+1] = len(box) / num_seg

        freqwb, psdwb = welch(box, fs=fs, nperseg=seg_wdw[i+1]) # frequency units: cycles/yr, output value: psd
        freqwg, psdwg = welch(gaus, fs=fs, nperseg=seg_wdw[i+1]) # frequency units: cycles/yr, output value: psd

        ax4.loglog(freqwb, psdwb, color=color, label=f'Boxcar ({wdw})') # plot welch
        ax5.loglog(freqwg, psdwg, color=color, label=f'Gaussian ({wdw})') # plot welch

        # find frequency where smoothed PSD is 95% of its own low-freq PSD
        freq_95_selfb = find_95_self(freqwb, psdwb)
        if freq_95_selfb:
            ax4.axvline(freq_95_selfb, linestyle='--', color=color, alpha=0.6, label=f'Self 95% at {freq_95_selfb:.4f} cycles/yr')
            print(f'For Boxcar ({wdw}), 95% of own low-freq PSD reached at: {1/freq_95_selfb:.4f} yr')

        freq_95_selfg = find_95_self(freqwg, psdwg)
        if freq_95_selfg:
            ax5.axvline(freq_95_selfg, linestyle='--', color=color, alpha=0.6, label=f'Self 95% at {freq_95_selfg:.4f} cycles/yr')
            print(f'For Gaussian ({wdw}), 95% of own low-freq PSD reached at: {1/freq_95_selfg:.4f} yr')
    
    ax4.axvline(x=1/res/2, color='gray', linestyle='--', label=f'')
    ax4.set_xlabel('Frequency (cycles/year)')
    ax4.set_ylabel(r'PSD (units$^2$/cycle/year)')
    ax4.set_title(f'PSD of Downsampled Data (Boxcar Moving Average)')
    ax4.grid()
    ax4.legend()
    ax4.set_xlim(1e-4, 1e0);

    ax5.axvline(x=1/res/2, color='gray', linestyle='--', label=f'')
    ax5.set_xlabel('Frequency (cycles/year)')
    ax5.set_ylabel(r'PSD (units$^2$/cycle/year)')
    ax5.set_title(f'PSD of Downsampled Data (Gaussian Moving Average)')
    ax5.grid()
    ax5.legend()
    ax5.set_xlim(1e-4, 1e0);

    plt.show()

    ### Depth-Age Relationship ###
    print('~~~ Comparing Effects of Resampling from Depth to Age Domain at Different Resolutions ~~~')
    print('Creating new resampled time series:')
    dy = 0.5 # sampling interval (m)
    even_depth = np.arange(int(depth.min()) + dy, depth.max(), dy) # cut off the ends to ensure no extrapolation
    new_age = np.interp(even_depth, depth, age)
    y_to_depth = np.interp(new_age, x, y) # data values connected to times on even depth scale (m)

    # low_res = np.max(np.diff(new_age))  # lowest resolution in years
    low_res = np.diff(new_age)[-1]  # resolution between last two points in years
    avg_res = np.mean(np.diff(new_age))  # average resolution in years
    holocene_res = np.mean(np.diff(new_age[new_age <= 11700]))  # Holocene average resolution in years
    print(f'Lowest resolution: {low_res} years')
    print(f'Average resolution: {avg_res} years')
    print(f'Holocene average resolution: {holocene_res} years')

    # resample to specified resolutions
    low_age, low_y = depth_to_age(new_age, y_to_depth, low_res, depth, age)
    avg_age, avg_y = depth_to_age(new_age, y_to_depth, avg_res, depth, age)
    holocene_age, holocene_y = depth_to_age(new_age, y_to_depth, holocene_res, depth, age)

    fig6, ax6 = plt.subplots(figsize=(10, 4))
    ax6.plot(x, y, label='Raw Synthetic Data', color='black')
    ax6.plot(holocene_age, holocene_y, label=f'Resampled Data (Holocene Res: {holocene_res:.4f} yr)', color='C0')
    ax6.plot(avg_age, avg_y, label=f'Resampled Data (Avg Res: {avg_res:.4f} yr)', color='C1')
    ax6.plot(low_age, low_y, label=f'Resampled Data (Low Res: {low_res:.4f} yr)', color='C2')
    ax6.set_title('Resampled Time Series from Depth to Age Domain')
    ax6.set_xlabel('Time (years)')
    ax6.legend()
    plt.show()
    
    # produce spectra of resampled data
    print('Producing spectra of resampled data:')
    fig7, ax7 = plt.subplots(figsize=(8, 4))

    for i, (data, age, label) in enumerate(zip([y, holocene_y, avg_y, low_y], [x, holocene_age, avg_age, low_age], 
                               ['Raw Data: 1 yr', f'Holocene Res: {holocene_res:.4f} yr', f'Avg Res: {avg_res:.4f} yr', f'Low Res: {low_res:.4f} yr'])):
        res = np.mean(np.diff(age)) # downsampling resolution (years)
        fs = 1 / res  # sampling frequency (samples/yr)
        if i == 0:
            color = 'black'
            freqw, psdw = welch(data, fs=fs, nperseg=seg_int[0]) # frequency units: cycles/yr, output value: psd
        else:
            color = f'C{i-1}'
            freqw, psdw = welch(data, fs=fs) # frequency units: cycles/yr, output value: psd; use default segment length since resampling resolution is done dynamically for a given chronology

        ax7.loglog(freqw, psdw, color=color, alpha=0.9, label=label) # plot welch
        ax7.axvline(x=1/res/2, color=color, alpha=0.6, linestyle='--', label=f'')

    ax7.set_xlabel('Frequency (cycles/year)')
    ax7.set_ylabel(r'PSD (units$^2$/cycle/year)')
    ax7.set_title(f'PSD of Resampled Data (Depth to Age Domain)')
    ax7.grid()
    ax7.legend()
    ax7.set_xlim(1e-4, 1e0)
    plt.show()
    
# define function to create red noise spectrum
def red_noise(length: int, strength: float):
    """
    Generates a red noise spectrum of a specified length and strength.

    Args:
        length (int): The length of the time series to generate
        strength (float): The strength of the red noise (limit 0 to 1, higher value = stronger red noise)

    Returns:
        np.ndarray: A red noise time series of the specified length and strength
    """
    if not 0 <= strength <= 1:
        raise ValueError("strength must be between 0 and 1")

    red_series = np.zeros(length)
    # red_series[0] = stats.norm.rvs(size=1)
    red_series[0] = np.random.normal(0, 1)
    for i in range(1, length):
        # red_series[i] = strength * red_series[i-1] + np.sqrt(1 - strength**2) * stats.norm.rvs(size=1)
        red_series[i] = strength * red_series[i-1] + np.sqrt(1 - strength**2) * np.random.normal(0, 1)
    return red_series

# define wrapper function that also allows data selection]
def spectral_tests(depth: np.ndarray, age: np.ndarray, interval: int | list[int], window: int | list[int], data: str='white', method: str='full', sigma: float=None, segments: str='default'):
    """
    Allows user to select the type of synthetic data to create (white noise, red noise, or sine wave) and then runs all analyses for the power spectra notebook, including:
    1) comparing FFT vs. Welch method for spectral analysis,
    2) comparing the effects of complete vs. incomplete discrete sampling on the resulting spectra, 
    3) comparing the effects of different smoothing windows on the resulting spectra, 
    4) comparing the effects of resampling from depth to age domain at different resolutions on the resulting spectra

    Args:
        depth (np.ndarray): Original ice core chronology depths
        age (np.ndarray): Original ice core chronology ages corresponding to each depth
        interval (int | list[int]): Desired sampling interval(s) for discrete sampling
        window (int | list[int]): Desired window size(s) for continuous sampling
        data (str, optional): Type of synthetic data to create. Must be 'white', 'red', or 'sine'. Defaults to 'white'
        method (str, optional): Method for treating the last point in discrete sampling. Defaults to 'full'
        sigma (float, optional): Standard deviation for Gaussian smoothing. If None, it defaults to window_size / 6
        segments (str, optional): Method for handling segment lengths in Welch method. Defaults to 'default'
                - 'default': use default segment lengths for all analyses
                - 'number': specify a number of segments to use for all analyses (segment length will be determined by this number)
                - 'length': specify a segment length to use for all analyses (number of segments will be determined by this length)
                - 'custom': specify a list of segment lengths to use for each analysis

    Returns:
        x (np.ndarray): Time axis for the synthetic data
        y (np.ndarray): "Proxy" data values for the synthetic data
    """
    # specify length of time series (make sure it matches time length of age model)
    xlen = int(age.max())  # length of time series (years)
    # create even time axis
    x = np.arange(int(age.min()), xlen + 1)  # time (yr)
    
    if data == 'sine':
        print(f'You selected a single sine wave.')
        period = float(input('Please enter the period of your sine wave in years: '))
        amplitude = float(input('Please enter the amplitude of your sine wave (strong > 1, weak < 1): '))
        y = amplitude * np.sin(2 * np.pi * x / period) # data values (units)
        print(f'Your sine wave with period {period} years and amplitude {amplitude} has been created.')
    elif data == 'white' or data == 'red':
        add_signal = input(f'You selected a {data} spectrum. Would you like to add known signals? (y|n)').lower().strip()
        if add_signal in ['y', 'yes']:
            while True:
                try:
                    num_signal = int(input('How many signals would you like to add? Please enter an integer: '))
                    break  # Exit loop if conversion is successful
                except ValueError:
                    print('That is not a valid integer. Try again.')

            print(f'You entered: {num_signal} added signals')
            signals = {}
            for i in range(num_signal):
                period = float(input(f'Please enter the period of your sine wave {i+1} in years: '))
                amplitude = float(input(f'Please enter the amplitude of sine wave {i+1} (strong > 1, weak < 1): '))
                # create the sine wave signal and add it to the dictionary
                signals[f'wave_{i+1}'] = amplitude * np.sin(2 * np.pi * x / period)
                print(f'Sine wave {i+1} has a period of {period} and amplitude of {amplitude}')

            print(f'We have now collected all known signals to add to your {data}-spectrum data.')
            if data == 'white':
                y = np.random.normal(0, 1, len(x)) + sum(signals.values()) # normal distibution, mean 0, std 1, plus added signals
            if data == 'red':
                strength = float(input('Please enter the strength of the red noise between 0 and 1 (higher value = stronger red noise): '))
                if not 0 <= strength <= 1:
                    raise ValueError("strength must be between 0 and 1")
                print(f'You chose a lag-1 autocorrelation of {strength}.')
                red_spectrum = red_noise(len(x), strength)
                y = red_spectrum + sum(signals.values()) # red noise is just stronger white noise, plus added signals
            print(f'Your {data}-spectrum time series with added signals has been created.')

        elif add_signal in ['n', 'no']:
            print(f'Okay, proceeding with {data}-spectrum time series')
            if data == 'white':
                y = np.random.normal(0, 1, len(x)) # normal distibution, mean 0, std 1
            if data == 'red':
                strength = float(input('Please enter the strength of the red noise between 0 and 1 (higher value = stronger red noise): '))
                if not 0 <= strength <= 1:
                    raise ValueError("strength must be between 0 and 1")
                print(f'You chose a lag-1 autocorrelation of {strength}.')
                y = red_noise(len(x), strength) # red noise is just stronger white noise
            print(f'Your {data}-spectrum time series has been created.')
        else:
            raise ValueError("Please enter 'y' or 'n'.")

    else: 
        raise ValueError("data must be either 'white', 'red', or 'sine'")
    
    run_analyses(x=x, y=y, depth=depth, age=age, interval=interval, window=window, method=method, sigma=sigma, segments=segments)

    return x, y