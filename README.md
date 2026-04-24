# 🧊 Ice Core Power Spectra
✨ Using synthetic data to test impacts of ice core data production and handling on resulting power spectra ✨ <br/>
This analysis focuses on applications of spectral analysis techniques to synthetic data sets to assess how sampling methods, data handling, and statistical choices affect interpretability of ice core data sets in the frequency domain. The initial analyses are done with a pure white-spectrum synthetic data set with no added signals to gain a deeper understanding of the impacts of sampling and inherent resolution on the shape of the resulting spectrum. The usability of the analysis functions was expanded to explore different types of synthetic data, including customization of discrete sampling intervals and continuous sampling windows, segment length, and added known signals.
<br/>

## 🗂️ Repository contains
- `utils.py`: Script containing all analysis functions and wrapper functions to run the analyses
    - `discrete_avg`: Discretely averages the data to downsample to a specified resolution
    - `boxcar`: Calculates the boxcar moving average of a 1D array
    - `gaussian`: Calculates the Gaussian moving average of a 1D array
    - `find_95_self`: Finds the frequency at which the moving average PSD reaches 95% of the raw millennial-scale PSD
    - `depth_to_age`: Resamples data from depth to age domain at specified resolution
    - `run_analyses`: Runs all analyses for the power spectra notebook, including: 
      1. comparing FFT vs. Welch method for spectral analysis,
      2. comparing the effects of complete vs. incomplete discrete sampling on the resulting spectra, 
      3. comparing the effects of different smoothing windows on the resulting spectra, 
      4. comparing the effects of resampling from depth to age domain at different resolutions on the resulting spectra
    - `red_noise`: Generates a red noise spectrum of a specified length and strength
    - `spectral_tests`: Allows user to select the type of synthetic data to create (white noise, red noise, or sine wave) and then runs all analyses. Also allows user to save generated synthetic time series
- `power_spectra.ipynb`: Notebook that walks through all analysis steps for a pure white-spectrum data set then demonstrates the usage of the wrapper functions
- `WD2014_Chronology.tab`: Example ice core chronology used in these analyses
- `environment.yml`: Environment containing necessary libraries

## 💻 How To Use
- Everything is centralized in one Notebook, which walks you through each step of the analyses, including some reasoning and interpretation explained.
- To perform additional analyses, the `spectral_tests` function is the most straightforward and allows for a high degree of customization.
- Any ice core chronology can be used for these analyses. The user just needs to import the age model they want to use and have depth and age as separate arrays to be called in the wrapper functions. Processing an age model file is not included in the analyses due to variations in header lines, column names, delimiter, and units.
  - <strong>Note:</strong> The synthetic data uses age in years and depth in meters, so make sure to verify the units of any other age model before running the analyses.
>[!CAUTION]
>Because of how segment length is defined based off of `interval` and `window` in the wrapper function, testing just a single value must be provided in list form of length 1 rather than as an integer to avoid errors.
