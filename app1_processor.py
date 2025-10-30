import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import re
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import scipy
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import openpyxl
import os
import zipfile
from io import BytesIO
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

# ==============================================================================
# Functional Code (KEPT UNCHANGED FROM ORIGINAL FOR LOGIC INTEGRITY)
# ==============================================================================

def calculate_rms(signal):
    """Calculate the RMS value of a signal."""
    return np.sqrt(np.mean(np.square(signal)))

def scale_functions(signal, ref):
    """Scale two synchronous functions of different amplitudes using RMS."""
    rms1 = calculate_rms(signal)
    rms2 = calculate_rms(ref)
    scaling_factor = rms2 / rms1
    scaled_function1 = signal * scaling_factor
    return scaled_function1

def freq_scaling(signal, fs=48000, high=23999, low=9000):
    fft_result = np.fft.fft(signal)
    n = len(signal)
    freq = np.fft.fftfreq(n, d=1/fs)
    
    filtered_fft_low_freq = fft_result.copy()
    filtered_fft_low_freq[np.abs(freq) > low] = 0
    filtered_fft_low_freq[np.abs(freq) < 0] = 0
    ifg_low_freq_signal = np.fft.ifft(filtered_fft_low_freq).real

    filtered_fft_high_freq = fft_result.copy()
    filtered_fft_high_freq[np.abs(freq) > high] = 0
    filtered_fft_high_freq[np.abs(freq) <= low] = 0
    ifg_high_freq_signal = np.fft.ifft(filtered_fft_high_freq).real
    
    return(ifg_low_freq_signal,ifg_high_freq_signal)
    
def compensate_delay(filtered_signal, num_taps):
    """Compensate for the delay introduced by the FIR filter."""
    delay = num_taps // 2
    compensated_signal = filtered_signal[delay:]
    return compensated_signal

def find_files_with_prefix(folder_path, prefix):
    """Finds all files in the specified folder that start with the specified prefix."""
    all_arrays = []
    for filename in os.listdir(folder_path):
        if filename.startswith(prefix) and filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            array_file = np.loadtxt(file_path)
            all_arrays.extend(array_file)
            print(filename)
    return all_arrays

def read_data_from_zip(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        file_name = zip_file.namelist()[0]
        print(file_name)
        with zip_file.open(file_name) as text_file:
            content = text_file.readlines()

    channel1_data = eval(content[0]) 
    channel2_data = eval(content[1]) 
    return file_name, channel1_data, channel2_data

def dc_Offset_Removal(irsignal, fs= 48000,cutoff_frequency = 100, order = 1):
    fs = fs
    cutoff_frequency = cutoff_frequency
    nyquist = 0.5 * fs
    cutoff = cutoff_frequency / nyquist
    b, a = scipy.signal.butter(order, cutoff, btype='high')
    filtered_signal_ir = scipy.signal.lfilter(b, a, irsignal) 
    return(filtered_signal_ir)

def split_flip_cosine_similarity(array):
    mid = len(array) // 2
    first_half = array[:mid]
    second_half = np.flip(array[mid:])
    similarity = cosine_similarity([first_half], [second_half])
    return similarity[0][0]

def bandpass_Filter(laser_signal, fs = 48000, lc = 500, hc = 1800, order = 2):
    fs = fs
    lowcut= lc
    highcut=hc
    nyqs=0.5*fs
    low= lowcut/nyqs
    high=highcut/nyqs
    
    order=order
    
    b,a = scipy.signal.butter(order,[low,high], 'bandpass', analog=False)
    filtered_signal = scipy.signal.filtfilt(b,a,laser_signal,axis=0)
    return(filtered_signal)

def apply_fir_bandpass_filter(input_signal, f_sampling = 48000, f_pass = [51,500], num_taps=1001, window='hann'):
    """Design and apply a FIR bandpass filter."""
    f_nyquist = 0.5 * f_sampling
    f_low_norm = f_pass[0] / f_nyquist
    f_high_norm = f_pass[1] / f_nyquist
    taps = signal.firwin(num_taps, [f_low_norm, f_high_norm], pass_zero=False,window=window)
    filtered_signal = signal.convolve(input_signal, taps, mode='same')
    return filtered_signal

import plotly.graph_objects as go
import plotly.express as px

def create_zoomable_plotly_line(x, y, plot_title='Zoomable Plot with Plotly'):
    fig = px.line(x=x, y=y, labels={'x': 'X-axis', 'y': 'Y-axis'}, title=plot_title)
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True),
            type='linear'
        ),
        yaxis=dict(type='linear',fixedrange = False)
    )
    fig.show()

def unzip_and_create_folder(zip_file_path):
    folder_name = os.path.splitext(os.path.basename(zip_file_path))[0]
    parent_directory = os.path.dirname(zip_file_path)
    unzipped_folder_path = os.path.join(parent_directory, folder_name)
    os.makedirs(unzipped_folder_path, exist_ok=True)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(unzipped_folder_path)

    print(f"Unzipped {zip_file_path} into folder: {unzipped_folder_path}")
    
def find_Centerburst(ir_signal, distance = 1000, quantile_cutoff = 0.995):
    height = np.quantile(ir_signal,quantile_cutoff)
    peak_pos = scipy.signal.find_peaks(ir_signal, height= height, distance=distance)[0]
    return(peak_pos)

import plotly.graph_objects as go

def zero_cross_plot(laser,zero_crossings, plot_modes = 'lines',title = "Zoomable zero cross scatter"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(laser))), y=laser, mode= plot_modes, name='laser'))
    fig.add_trace(go.Scatter(x=zero_crossings, y=laser[zero_crossings],
                             mode='markers', marker=dict(symbol='cross', size=5),
                             name='Zero Crossings'))
    fig.update_layout(
        title= title,
        xaxis_title='amp',
        yaxis_title='Signal Value',
        autosize=True,
        width=1000,
        height=500,
    )
    fig.show()

def do_tri_apod(ir, window_len = 512 , sym = True):
    signal_length = len(ir)
    triangular_window = scipy.signal.windows.triang(M = window_len , sym=sym)
    apodization_window = np.zeros(signal_length)
    start_index = (signal_length - window_len) // 2
    apodization_window[start_index:start_index + window_len] = triangular_window
    apod_ir = apodization_window*ir
    
    return(apodization_window,apod_ir)
    
        
def do_mertz_phase(ir_signal, points = 512):
    half_pt = int(points/2)
    x = np.arange(0, len(ir_signal))  
    centerburst = int(len(ir_signal)/2)
    
    Ixshort = ir_signal[centerburst-int(half_pt-1):centerburst+ half_pt+1]
    xshort = x[centerburst-int(half_pt-1):centerburst+ half_pt+1]
    
    apod = 1 - np.abs(xshort - xshort[half_pt]) / (0.5 * len(xshort))
    apod[apod < 0] = 0
    
    Ixshortapod = Ixshort * apod
    
    Ixshortapodrot = np.concatenate((Ixshortapod[half_pt:], Ixshortapod[:half_pt]))
    
    bprime = np.fft.fft(Ixshortapodrot)
    bprimereal = np.real(bprime)
    bprimeimag = np.imag(bprime)
    
    thetav = np.arctan2(bprimeimag, bprimereal)

    fullthetav = np.interp(x, xshort, thetav)

    fullcosterm = np.cos(fullthetav)
    fullsinterm = np.sin(fullthetav)
    
    return(fullcosterm, fullsinterm)

def extract_and_concatenate(folder_path, file_pattern, column_name):
    excel_files = [file for file in os.listdir(folder_path) if file.startswith(file_pattern) and file.endswith('.xlsx')]
    print(excel_files)
    dfs = []

    for file in excel_files:
        df = pd.read_excel(os.path.join(folder_path, file))
        df = df[[column_name]]
        df.rename(columns={column_name: f'{file[:-5]}_{column_name}'}, inplace=True)
        dfs.append(df)

    final_dataframe = pd.concat(dfs, axis=1)
    return final_dataframe

def forman_phase_correction(ref, signal, sampling_rate=48000):
    """Perform Forman phase correction to align two signals."""
    fft_ref = np.fft.fft(ref)
    fft_signal = np.fft.fft(signal)

    phase_ref = np.angle(fft_ref)
    phase_signal = np.angle(fft_signal)

    phase_difference = phase_ref - phase_signal
    corrected_fft_signal = np.abs(fft_signal) * np.exp(1j * (phase_signal + phase_difference))
    phase_corrected = np.angle(corrected_fft_signal)

    # These prints are for debugging, kept as is
    # print("ref v/s original:", np.degrees(phase_difference).max(), np.degrees(phase_difference).mean())
    # print("ref v/s corrected:", np.degrees(phase_difference_corrected).max(), np.degrees(phase_difference_corrected).mean())

    corrected_signal = np.fft.ifft(corrected_fft_signal).real
    
    return corrected_signal

def forman_sym(input_array, nppa=64):
    """Performs Forman phase correction on a single interferogram."""
    n = len(input_array)
    burst = np.argmax(input_array)

    if nppa > burst or nppa == burst:
        if burst == 0:
            iposl = n - nppa
            workl = np.concatenate((input_array[iposl:], input_array[:nppa]))
        else:
            raise ValueError('ERROR: Specified phase array is too large')
    else:
        workl = input_array[burst - nppa + 1: burst + nppa + 1]

    workl = rotintfg(workl)
    cspec = np.fft.fft(workl)
    dphase = np.unwrap(np.angle(cspec))

    rphase = np.cos(dphase)
    iphase = np.sin(dphase)
    cphase = rphase - 1j * iphase

    nneg = (len(workl) + 1) // 2 - 1
    tl = len(workl) // 2 + 1
    t2 = 2 * nppa
    cphase[tl:t2] = np.flipud(np.conj(cphase[1:nneg + 1]))

    intphase = np.fft.ifft(cphase)
    pifg = intphase.real

    phintfg, apdfunc = triapod(np.fft.fftshift(intphase))
    
    rintfg = np.convolve(input_array, phintfg, mode='full')
    
    nout = n + nppa * 2 - 1
    rmax = np.max(rintfg)
    nburst = np.argmax(rintfg)
    iposl = nburst - burst
    ipos2 = iposl + n
    fdmat = rintfg[iposl:ipos2].real

    return fdmat, pifg

def rotintfg(input):
    """Rotate interferogram so that centerburst is at point #1 in the output array."""
    cburstpos = np.argmax(input)
    isize = len(input)
    output = np.concatenate((input[cburstpos:isize], input[0:cburstpos]))
    return output

def triapod(input, atype=1):
    """Apply triangular apodization on an interferogram."""
    cburstpos = np.argmax(input)
    npts = len(input)
    apdfunc = np.zeros(npts)
    fract = min(cburstpos, npts - cburstpos) / npts
    single_sided = 1 if fract < 0.33 else 0

    if cburstpos > 0:
        if single_sided and atype == 1:
            firsthalf = 2 * cburstpos - 1
        else:
            firsthalf = cburstpos
        apdfunc[:firsthalf] = np.arange(1, firsthalf + 1) / firsthalf
    else:
        firsthalf = 0

    secondhalf = npts - firsthalf
    apdfunc[firsthalf:] = np.arange(secondhalf, 0, -1) / secondhalf
    output = input * apdfunc

    return output, apdfunc

def do_hamming_apod(ir, window_len = 512 , sym = True):
    signal_length = len(ir)
    triangular_window = scipy.signal.windows.hamming(M = window_len , sym=sym)
    apodization_window = np.zeros(signal_length)
    start_index = (signal_length - window_len) // 2
    apodization_window[start_index:start_index + window_len] = triangular_window
    apod_ir = apodization_window*ir
    
    return(apodization_window,apod_ir)

def reduce_spacing_cubic_spline(x, y, reduction_factor=3):
    """Reduce the spacing of x-values and interpolate corresponding y-values using cubic spline."""
    new_spacing = (x[1] - x[0]) / reduction_factor
    new_x = np.arange(x[0], x[-1], new_spacing)
    spline = CubicSpline(x, y)
    new_y = spline(new_x)
    return spline, new_x, new_y

def do_hann_apod(ir, window_len = 512 , sym = True):
    signal_length = len(ir)
    triangular_window = scipy.signal.windows.hann(M = window_len , sym=sym)
    apodization_window = np.zeros(signal_length)
    start_index = (signal_length - window_len) // 2
    apodization_window[start_index:start_index + window_len] = triangular_window
    apod_ir = apodization_window*ir
    return(apodization_window,apod_ir)

def apply_signal_fft(signal):
    zpd = np.argmax(signal)
    rotir = np.concatenate([np.flip(signal[:zpd]),np.flip(signal[zpd:])])
    
    fftir = np.fft.fft(rotir)
    real = np.real(fftir)
    imag = np.imag(fftir)
    
    corrected_signal = real + imag
    return corrected_signal
    
def process_signal_fft(df):
    corrected_signals = []
    for column in df.columns:
        signal = df[column].values
        corrected_signal = apply_signal_fft(signal)
        corrected_signals.append(corrected_signal)
    return pd.DataFrame(corrected_signals, index=df.columns).T

def spec_gen(df):
    log_spec = []
    abs_spec = []
    bg = df.iloc[:,0].values
    
    for column in df.columns:
        signal = df[column].values
        
        log_signal = np.log10(np.abs(signal))
        log_bg =  np.log10(np.abs(bg))
        
        spec_signal = log_signal - log_bg
        
        log_spec.append(log_signal)
        abs_spec.append(spec_signal)
        
    return pd.DataFrame(log_spec, index=df.columns).T, pd.DataFrame(abs_spec, index=df.columns).T

from scipy.signal import savgol_filter

def savitzky_golay_smoothing(data, window_size=5, order=2):
    """Perform Savitzky-Golay smoothing on the input data."""
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")
    if window_size <= order:
        raise ValueError("Window size must be larger than order.")

    smoothed_data = savgol_filter(data, window_size, order)
    return smoothed_data

def get_tri_apod_window(signal, win_type = "tri" , sym = False):
    window_len = len(signal)*2
    print("Triangle")
    apod_window = scipy.signal.windows.triang(M = window_len , sym=sym)
    return(apod_window)

def get_hamm_apod_window(signal, sym = False):
    window_len = len(signal)*2
    print("Hamming")
    apod_window = scipy.signal.windows.hamming(M = window_len , sym=sym)
    return(apod_window)

def get_hann_apod_window(signal, sym = False):
    window_len = len(signal)*2
    print("Hanning")
    apod_window = scipy.signal.windows.hann(M = window_len , sym=sym)
    return(apod_window)

def do_apodization(signal, window_type = "tri"):
    left = signal[0:np.argmax(signal)]
    right = signal[np.argmax(signal):]
    
    if window_type == "tri":
        left_window = get_tri_apod_window(left)
        right_window = get_tri_apod_window(right)
    elif window_type == "hamm":
        left_window = get_hamm_apod_window(left)
        right_window = get_hamm_apod_window(right)
    elif window_type == "hann":
        left_window = get_hann_apod_window(left)
        right_window = get_hann_apod_window(right)
        
    left_window = left_window[0:np.argmax(left_window)]        
    right_window = right_window[np.argmax(right_window):]
    
    window = np.concatenate([left_window,right_window])
    return(window)

def get_apod_window(signal, window_type = 'tri', sym = True):
    
    len_signal_left  = len(signal[: np.argmax(signal)])
    len_signal_right = len(signal[np.argmax(signal) : ])
    
    if window_type == "tri":
        
        if len_signal_left < len_signal_right:
            apod_window = scipy.signal.windows.triang(M = len_signal_right*2 , sym=sym)
            apod_window = apod_window[(np.argmax(apod_window)-len_signal_left): (np.argmax(apod_window)+ len_signal_right)]
        else:
            apod_window = scipy.signal.windows.triang(M = len_signal_right*2 , sym=sym)
            len_left = len_signal_left-len_signal_right
            len_win = np.concatenate([np.zeros(len_left),apod_window[0:np.argmax(apod_window)]])
            apod_window = np.concatenate([len_win,apod_window[np.argmax(apod_window):]])
        
    elif window_type == "hamm":
        if len_signal_left < len_signal_right:
            apod_window = scipy.signal.windows.hamming(M = len_signal_right*2 , sym=sym)
            apod_window = apod_window[(np.argmax(apod_window)-len_signal_left): (np.argmax(apod_window)+ len_signal_right)]
        else:
            apod_window = scipy.signal.windows.hamming(M = len_signal_right*2 , sym=sym)
            len_left = len_signal_left-len_signal_right
            len_win = np.concatenate([np.zeros(len_left),apod_window[0:np.argmax(apod_window)]])
            apod_window = np.concatenate([len_win,apod_window[np.argmax(apod_window):]])
        
    elif window_type == "hann":
        if len_signal_left < len_signal_right:
            apod_window = scipy.signal.windows.hann(M = len_signal_right*2 , sym=sym)
            apod_window = apod_window[(np.argmax(apod_window)-len_signal_left): (np.argmax(apod_window)+ len_signal_right)]
        else:
            apod_window = scipy.signal.windows.hann(M = len_signal_right*2 , sym=sym)
            len_left = len_signal_left-len_signal_right
            len_win = np.concatenate([np.zeros(len_left),apod_window[0:np.argmax(apod_window)]])
            apod_window = np.concatenate([len_win,apod_window[np.argmax(apod_window):]])
    elif window_type == 'boxcar':
        apod_window = np.ones(len(signal))
    
    else:
        raise ValueError("Unknown window type")    
        
    return(apod_window) 
 

def make_ramp(signal):
    left_len = len(signal[0:np.argmax(signal)])
    right_len = len(signal[np.argmax(signal):])

    left_len, right_len

    even_ramp = np.linspace(0.5,0.5,left_len*2)
    odd_ramp = np.linspace(-0.5,0.5,left_len*2)

    final_ramp = np.round(even_ramp+odd_ramp,2) 
    
    return(final_ramp)

import numpy as np
from scipy.signal import butter, lfilter

def low_pass_filter(data, cutoff=100, fs=50000, order=2):
    """Applies a low-pass Butterworth filter to the input data."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data)
    return filtered_data


# ==============================================================================
# App1Frame Class (UPDATED UI)
# ==============================================================================

inter_points = 900

class App1Frame(ttk.Frame): # Inherit from ttk.Frame
    def __init__(self, parent):
        super().__init__(parent, padding=10) # Add padding
        self.inter_points = 900
        self.create_widgets()

    def browse_read_path(self):
        self.read_path_var.set(filedialog.askdirectory())

    def log_sample_id(self, sample_id):
        self.log_box.configure(state='normal')
        self.log_box.insert(tk.END, f"Processing Sample ID: {sample_id}\n")
        self.log_box.see(tk.END)
        self.log_box.configure(state='disabled')
        self.update_idletasks()

    def _process_single_pass(self, read_path, sampleid, ir_flip, lsr_flip, output_path, stype, pass_type=""):
        """
        Helper function to process a single pass (forward or backward) for a sample.
        (Logic kept exactly as in the original script to avoid breaking functionality)
        
        NOTE: pass_type is used for file naming but is intentionally set to "" 
        to meet the user's request to remove "_flipped" and "_unflipped" suffixes.
        This will cause files to overwrite if both passes are run for the same sample ID.
        """
        irfil = np.round((apply_fir_bandpass_filter(ir_flip,f_sampling= 48000, f_pass= [51,1000],num_taps=2501, window= "hamming")), 4)
        
        height = np.quantile(irfil, 0.997)
        print(f"\n[{pass_type}] Height (99.7th quantile):", height)
        
        peak = find_Centerburst(irfil, quantile_cutoff=0.997, distance=6000)
        print(f"\n[{pass_type}] peak :", peak)
        
        valid_peak = peak[peak > 10000]
        
        if len(valid_peak) < 2:
            print(f"[{pass_type}] Not enough valid peaks found for sample {sampleid}. Skipping.")
            return

        cut_off = np.min(np.diff(valid_peak)) + 1000
        print(f"\n[{pass_type}] Cutoff:", cut_off)
        
        if len(valid_peak) == 3 and len(peak) == 3 and np.diff(valid_peak)[0] < cut_off:
            forward_peaks = [1]
        elif len(valid_peak) == 3 and len(peak) == 3 and np.diff(valid_peak)[0] > cut_off:
            forward_peaks = [0]
        elif len(valid_peak) == 2 and len(peak) == 2 and np.min(np.diff(valid_peak)) < 40000:
            forward_peaks = [1]
        elif len(valid_peak) == 2 and len(peak) == 2 and np.min(np.diff(valid_peak)) > 40000:
            forward_peaks = [0]
        elif len(valid_peak) == 2 and len(peak) == 3 and np.where(np.diff(valid_peak) < cut_off):
            forward_peaks = [1]
        else:
            forward_peaks = np.where(np.diff(valid_peak) > cut_off)[0]
        
        print(f"[{pass_type}] Valid peak positions:", valid_peak)
        print(f"[{pass_type}] Forward peak positions:", forward_peaks)
        print(f"[{pass_type}] diff peak positions:", np.diff(valid_peak))
        
        #stat list
        stats = {
            'up_laser_zc_len': [], 'up_laser_zc_mean': [], 'up_laser_zc_std': [], 'up_laser_zc_df': [],
            'laser_zc_len': [], 'laser_zc_mean': [], 'laser_zc_std': [], 'laser_zc_df': [],
            'cb_left': [], 'cb_right': [], 'peak_index': [], 'peak_raw_maxima': [], 'peak_raw_minima': [],
            'peak_filtered_maxima': [], 'peak_left_dist': [], 'peak_right_dist': [], 'peak_left_slope': [],
            'peak_right_slope': [], 'snr_raw': [], 'snr_filtered': []
        }
        
        sampling_rate = 48000
        
        for idx, f in enumerate(forward_peaks, 1):
            if f >= len(valid_peak):
                print(f"[{pass_type}] Forward peak index {f} is out of bounds for valid_peak. Skipping.")
                continue

            print(valid_peak[f])
            
            back_peak_dist = valid_peak[f] - peak
            positive_value = int(1/2 * np.abs(min((x for x in back_peak_dist if x > 0), default=(0 - valid_peak[f]))))
            negative_value = int(1/2 * np.abs(max((x for x in back_peak_dist if x < 0), default=(len(irfil) - valid_peak[f]))))    
            
            print(f"[{pass_type}] before distance: " , positive_value)
            print(f"[{pass_type}] after distance : " , negative_value)
            
            ir_raw = ir_flip[valid_peak[f] - positive_value:valid_peak[f] + negative_value]
            max_peak_amplitude = np.max(ir_raw)
            min_peak_amplitude = np.min(ir_raw)
            noise_segment = np.concatenate((ir_raw[:100], ir_raw[-100:]))
            snr_linear = np.round(max_peak_amplitude / np.std(noise_segment),4)
            
            stats['peak_raw_maxima'].append(max_peak_amplitude)
            stats['peak_raw_minima'].append(min_peak_amplitude)
            stats['snr_raw'].append(snr_linear)
            
            ir_full = irfil[valid_peak[f] - positive_value:valid_peak[f] + negative_value]
            fil_max_peak_amplitude = np.max(ir_full)
            fil_noise_segment = np.concatenate((ir_full[:100], ir_full[-100:]))
            fil_snr_linear = np.round(fil_max_peak_amplitude / np.std(fil_noise_segment),4)
            
            stats['peak_filtered_maxima'].append(fil_max_peak_amplitude)
            stats['snr_filtered'].append(fil_snr_linear)
            
            laser_full = lsr_flip[valid_peak[f] - positive_value:valid_peak[f] + negative_value]
            
            ir_full = ir_full - ir_full.mean()  
            laser_full = laser_full-laser_full.mean()
            
            laser_fil = np.round(apply_fir_bandpass_filter(laser_full, f_sampling=48000,f_pass= [100,2000],num_taps=11, window= "hamming"),4)
            
            ir_max = np.argmax(ir_full)
            section  = laser_fil[ir_max-inter_points:ir_max+inter_points]
            laser_section_zero = np.where(np.diff(np.sign(section)))[0]
            laser_full_zero_mean = np.round(np.diff(laser_section_zero).mean(),4)
            
            zc_section = np.where(np.diff(np.sign(section)))[0]
            zc_mean = np.diff(zc_section).mean()
            zc_std = np.diff(zc_section).std()   
            zc_fq = 1/((zc_mean*2)/sampling_rate)
            
            stats['laser_zc_len'].append(len(zc_section))
            stats['laser_zc_mean'].append(zc_mean)
            stats['laser_zc_std'].append(zc_std)
            stats['laser_zc_df'].append(zc_fq)        
            
            opt = 24
            upsample_factor = np.round((opt / np.round(laser_full_zero_mean, 2)), 2)
            
            x = np.linspace(0, len(laser_fil) - 1, len(laser_fil))
            x_new = np.linspace(0, len(laser_fil) - 1, int(len(laser_fil) * upsample_factor))
            laser_upsampled = np.round(np.interp(x_new, x, laser_fil),4)                                      
            
            ir_x = np.linspace(0, len(ir_full) - 1, len(ir_full))
            ir_x_new = np.round(np.linspace(0, len(ir_full) - 1, int(len(ir_full) * upsample_factor)),4)
            ir_upsampled = np.round(np.interp(ir_x_new, ir_x, ir_full),4)                    
            
            interpolate_factor = 5
            originallen = np.linspace(0, len(ir_upsampled), len(ir_upsampled))
            newlen = np.linspace(0, len(ir_upsampled), (len(ir_upsampled)*interpolate_factor))
            
            cubic_spline_laser = CubicSpline(originallen, laser_upsampled)
            cubic_spline_ir = CubicSpline(originallen,ir_upsampled)
            
            laser_int = cubic_spline_laser(newlen)
            ir_int = cubic_spline_ir(newlen)
            
            upsampled_laser_full_zero = np.where(np.diff(np.sign(laser_int)))[0]
            ir_f = ir_int[upsampled_laser_full_zero]
            
            l_hand = ir_f[0:np.argmax(ir_f)]
            r_hand = ir_f[np.argmax(ir_f):]
            
            left_hand = -(np.flip(range(1, len(l_hand)+1)))
            right_hand= range(0,len(r_hand))
            
            pre_axis = np.concatenate([left_hand, right_hand])
            axis_nm = pre_axis*425
            axis_cm = axis_nm*1e-7
            axis_mm = axis_cm*10
            
            l_axis_mm_limit = -0.29
            r_axis_mm_limit = 0.29
            ir_f_mm = ir_f[(axis_mm >= l_axis_mm_limit) & (axis_mm <= r_axis_mm_limit)]
            
            smooth_ir_f = savitzky_golay_smoothing(ir_f_mm, window_size=5, order=2)
            
            peak_pos = find_Centerburst(smooth_ir_f, quantile_cutoff=0.995, distance=100)
            
            if len(peak_pos) > 1:
                smooth_ir_f = smooth_ir_f[peak_pos[0] + 1:]
                cb = peak_pos[1]
            elif len(peak_pos) == 1:
                cb = peak_pos[0]
            else: # No peak found
                print(f"[{pass_type}] No centerburst found in smoothed IR for sample {sampleid}, peak {f}. Skipping.")
                continue

            ir_upsampled_max = np.argmax(ir_upsampled)
            upsampled_section  = laser_upsampled[ir_upsampled_max-inter_points:ir_upsampled_max+inter_points]
            upsampled_laser_section_zero = np.where(np.diff(np.sign(upsampled_section)))[0]
            upsampled_laser_full_zero_mean = np.round(np.diff(upsampled_laser_section_zero).mean(),4)
            upsampled_laser_full_zero_std = np.diff(upsampled_laser_section_zero).std()
            upsampled_laser_full_zero_dom_freq = (1/((upsampled_laser_full_zero_mean*2)/48000))
            
            stats['up_laser_zc_len'].append(len(upsampled_laser_section_zero))
            stats['up_laser_zc_mean'].append(upsampled_laser_full_zero_mean)
            stats['up_laser_zc_std'].append(upsampled_laser_full_zero_std)
            stats['up_laser_zc_df'].append(upsampled_laser_full_zero_dom_freq) 
            
            left_join = smooth_ir_f[0:cb]
            right_join= smooth_ir_f[cb:]
            
            stats['cb_left'].append(len(left_join))
            stats['cb_right'].append(len(right_join))
            stats['peak_index'].append(f)  
            
            points = 2048
            if len(left_join) >= points:
                left = left_join[np.argmax(left_join)-points:np.argmax(left_join)]
            else:
                left =np.concatenate([np.zeros(points - len(left_join)),left_join])
            
            if len(right_join) >= points:
                right = right_join[0:points]
            else:
                right = np.concatenate([right_join,np.zeros(points - len(right_join))])
            
            max_min_left = np.argmax(left) - np.argmin(left)
            max_min_right = np.argmin(right) - np.argmax(right)
            left_slope = np.round((left[np.argmax(left)] - left[np.argmin(left)]) / (np.argmax(left) - np.argmin(left)),4)
            right_slope = np.round((right[np.argmax(right)] - right[np.argmin(right)]) / (np.argmax(right) - np.argmin(right)),4)
            
            stats['peak_left_dist'].append(max_min_left)
            stats['peak_right_dist'].append(max_min_right)
            stats['peak_left_slope'].append(left_slope)
            stats['peak_right_slope'].append(right_slope)
            
            ira = np.concatenate([left,right])
            
            try:
                sym_irf = forman_sym(ira, nppa=512)
                sym_irf_ig1 = sym_irf[0]
            except ValueError:
                print(f"[{pass_type}] Forman symmetry failed for sample {sampleid}, peak {f}. Skipping.")
                continue
            
            signal_segment = sym_irf_ig1[np.argmax(sym_irf_ig1) - len(left_join): np.argmax(sym_irf_ig1) + len(right_join)]
            
            apod_win_tri = get_apod_window(signal_segment, window_type= "tri")
            apod_win_hamm = get_apod_window(signal_segment, window_type= "hamm")
            apod_win_hann = get_apod_window(signal_segment, window_type= "hann")
            apod_win_box = get_apod_window(signal_segment, window_type= "boxcar")
            
            apod_sym_irf_ig1_tri = apod_win_tri * signal_segment
            apod_sym_irf_ig1_hamm = apod_win_hamm * signal_segment
            apod_sym_irf_ig1_hann = apod_win_hann * signal_segment
            apod_sym_irf_ig1_box = apod_win_box * signal_segment
            
            ramp_segment = make_ramp(signal_segment)
            if len(ramp_segment) < len(signal_segment):
                ramp_array = np.concatenate([ramp_segment, np.linspace(1,1,len(signal_segment) - len(ramp_segment))]) 
            else:
                ramp_array = ramp_segment[0:len(signal_segment)]
            
            ramp_apod_sym_irf_ig1_tri = apod_sym_irf_ig1_tri * ramp_array
            ramp_apod_sym_irf_ig1_hamm = apod_sym_irf_ig1_hamm * ramp_array
            ramp_apod_sym_irf_ig1_hann = apod_sym_irf_ig1_hann * ramp_array
            ramp_apod_sym_irf_ig1_box = apod_sym_irf_ig1_box * ramp_array
            
            tri_array = np.zeros(len(sym_irf_ig1))
            hamm_array = np.zeros(len(sym_irf_ig1))
            hann_array = np.zeros(len(sym_irf_ig1))
            box_array = np.zeros(len(sym_irf_ig1))
            
            tri_array[np.argmax(sym_irf_ig1) - len(left_join): np.argmax(sym_irf_ig1) + len(right_join)] = ramp_apod_sym_irf_ig1_tri
            hamm_array[np.argmax(sym_irf_ig1) - len(left_join): np.argmax(sym_irf_ig1) + len(right_join)] = ramp_apod_sym_irf_ig1_hamm
            hann_array[np.argmax(sym_irf_ig1) - len(left_join): np.argmax(sym_irf_ig1) + len(right_join)] = ramp_apod_sym_irf_ig1_hann
            box_array[np.argmax(sym_irf_ig1) - len(left_join): np.argmax(sym_irf_ig1) + len(right_join)] = ramp_apod_sym_irf_ig1_box
            
            if (int(np.round(upsampled_laser_full_zero_mean)) == 24):
                # Filename updated to remove pass_type suffix
                fn = os.path.join(output_path, f"{sampleid}_ifg_{stype}{f}.xlsx")
                ifg_df = pd.DataFrame({
                    'ifg' : ira,
                    'forman':sym_irf_ig1,            
                    'ramp_apod_tri': tri_array,
                    'ramp_apod_hamm': hamm_array,
                    'ramp_apod_hann': hann_array,
                    'ramp_apod_box': box_array
                })
                ifg_df.to_excel(fn,index=False)
                print(f"[{pass_type}] saving ifg : " + fn)
            
            # Filename updated to remove pass_type suffix
            pd.DataFrame(np.diff(zc_section)).to_excel(os.path.join(output_path, f"{sampleid}_{stype}_zc_s{f}.xlsx"), index = False)

        df = pd.DataFrame({
            'sample_id' : [str(p) for p in stats['peak_index']],
            'peak_index' : stats['peak_index'],
            'peak_raw_maxima_amp' :stats['peak_raw_maxima'],
            'peak_raw_minima_amp' :stats['peak_raw_minima'],
            'peak_filtered_maxima_amp' :stats['peak_filtered_maxima'],
            'left_max_dist':stats['peak_left_dist'],
            'right_max_dist' : stats['peak_right_dist'],
            'left_max_slope':stats['peak_left_slope'],
            'right_max_slope' : stats['peak_right_slope'],
            'snr_raw':stats['snr_raw'],
            'snr_filtered': stats['snr_filtered'],
            'cb_left' :stats['cb_left'],
            'cb_right' : stats['cb_right'],
            'laser_zc_len':stats['laser_zc_len'],
            'laser_zc_mean' : stats['laser_zc_mean'],
            'laser_zc_std' : stats['laser_zc_std'],
            'laser_zc_df' : stats['laser_zc_df'],
            'up_zc_len' : stats['up_laser_zc_len'], 
            'up_laser_zc_mean' : stats['up_laser_zc_mean'], 
            'up_laser_zc_std' : stats['up_laser_zc_std'],
            'uplaser_zc_df' : stats['up_laser_zc_df']
        })
        
        # Filename updated to remove pass_type suffix
        fn = os.path.join(output_path, f"{sampleid}{stype}_stats.xlsx")
        df.to_excel(fn,index=False)


    def submit_inputs(self):
        try:
            read_path = self.read_path_var.get().strip()
            machine_id = self.machine_id_var.get().strip()
            # Safely convert to int, providing default if empty/invalid
            try:
                start_id = int(self.start_id_var.get())
                end_id = int(self.end_id_var.get())
            except ValueError:
                messagebox.showerror("Invalid Input", "Sample ID Start/End must be integers.")
                return


            if not read_path or not machine_id:
                messagebox.showerror("Missing Input", "Please enter both Read Path and Machine ID.")
                return

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_path = os.path.join(read_path, f"{machine_id}_{timestamp}")
            os.makedirs(output_path, exist_ok=True)

            sample_ids = [str(i) for i in range(start_id, end_id + 1)]
            all_files = os.listdir(read_path)

            used_ids = set()
            selected_files = []
            for file in sorted(all_files):
                for sid in sample_ids:
                    # Note: Using a simplified check based on file naming convention
                    if f"ID_{sid}" in file and sid not in used_ids:
                        used_ids.add(sid)
                        selected_files.append((file, sid)) # Store as tuple (filename, id)
                        break

            # Process files one by one
            total_files = len(used_ids) * 2 # Forward and Backward pass
            self.progress["maximum"] = total_files
            self.progress["value"] = 0
            self.progress.start()
            self.run_button.config(state=tk.DISABLED) # Disable button during run
            
            stype = '_' # water or milk (assuming it defaults to '_')
            value_to_multiply = 5 / (2 ** 23 - 1)
            self.log_sample_id(f"Starting processing for {len(used_ids)} samples...")

            # --- FORWARD PASS (FLIPPED DATA) ---
            for i, (file, sampleid) in enumerate(selected_files):
                self.log_sample_id(f"Fwd Pass (Data Flipped): {sampleid} ")
                prefix = "ID_" + str(sampleid) 
                
                ir = np.array(find_files_with_prefix(read_path, prefix= prefix+"_sample_ir_result"))*value_to_multiply
                lsr = np.array(find_files_with_prefix(read_path, prefix = prefix+"_sample_lsr_result"))*value_to_multiply
                
                # Forward pass (flipped)
                ir_flip = np.flip(ir)[2500:-1] 
                lsr_flip = np.flip(lsr)[2500:-1]
                
                # pass_type is set to empty string "" to remove the suffix
                self._process_single_pass(read_path, sampleid, ir_flip, lsr_flip, output_path, stype, pass_type="")

                self.progress["value"] = i + 1
                self.update()
                
            # --- BACKWARD PASS (UNFLIPPED DATA) ---
            for i, (file, sampleid) in enumerate(selected_files):
                self.log_sample_id(f"Bwd Pass (Data Unflipped): {sampleid} ")
                prefix = "ID_" + str(sampleid) 

                ir = np.array(find_files_with_prefix(read_path, prefix= prefix+"_sample_ir_result"))*value_to_multiply
                lsr = np.array(find_files_with_prefix(read_path, prefix = prefix+"_sample_lsr_result"))*value_to_multiply           
                
                # Backward pass (unflipped - using ir/lsr [1:-2500] as per original code logic)
                ir_unflipped = ir[1:-2500]
                lsr_unflipped = lsr[1:-2500]
                
                # pass_type is set to empty string "" to remove the suffix
                self._process_single_pass(read_path, sampleid, ir_unflipped, lsr_unflipped, output_path, stype, pass_type="")

                self.progress["value"] = len(selected_files) + i + 1
                self.update()

            self.progress.stop()
            self.log_sample_id("✅ All done!")
            messagebox.showinfo("Success", f"Processed {len(used_ids)} samples.\nOutput saved in:\n{output_path}")

        except Exception as e:
            self.progress.stop()
            self.log_sample_id(f"❌ Error: {e}")
            messagebox.showerror("Error", str(e))
        finally:
            self.run_button.config(state=tk.NORMAL)


    def create_widgets(self):
        # Set up a main frame for padding and structure
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill='both', expand=True)

        # Configure grid weights for semi-responsiveness
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(5, weight=1) # Log box row

        row_index = 0
        
        # 1. Read Path
        ttk.Label(main_frame, text="Read Path:", bootstyle="primary").grid(row=row_index, column=0, sticky='w', pady=5, padx=5)
        self.read_path_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.read_path_var, bootstyle="primary").grid(row=row_index, column=1, sticky='ew', padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_read_path, bootstyle="info-outline").grid(row=row_index, column=2, sticky='e', padx=5)
        row_index += 1

        # 2. Machine ID
        ttk.Label(main_frame, text="Machine ID:", bootstyle="primary").grid(row=row_index, column=0, sticky='w', pady=5, padx=5)
        self.machine_id_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.machine_id_var, bootstyle="primary").grid(row=row_index, column=1, columnspan=2, sticky='ew', padx=5)
        row_index += 1

        # 3. Sample IDs (in one frame for better organization)
        id_frame = ttk.Frame(main_frame)
        id_frame.grid(row=row_index, column=0, columnspan=3, sticky='ew', pady=5, padx=5)
        id_frame.columnconfigure(1, weight=1)
        id_frame.columnconfigure(3, weight=1)

        ttk.Label(id_frame, text="Sample ID Start:").grid(row=0, column=0, sticky='e', padx=5)
        self.start_id_var = tk.StringVar(value='3722')
        ttk.Entry(id_frame, textvariable=self.start_id_var, bootstyle="secondary").grid(row=0, column=1, sticky='ew', padx=5)

        ttk.Label(id_frame, text="Sample ID End:").grid(row=0, column=2, sticky='e', padx=5)
        self.end_id_var = tk.StringVar(value='3730')
        ttk.Entry(id_frame, textvariable=self.end_id_var, bootstyle="secondary").grid(row=0, column=3, sticky='ew', padx=5)
        row_index += 1

        # 4. Progress Bar
        self.progress = ttk.Progressbar(main_frame, orient="horizontal", mode="determinate", bootstyle="success")
        self.progress.grid(row=row_index, column=0, columnspan=3, sticky='ew', padx=5, pady=10)
        row_index += 1

        # 5. Log Box (set sticky="nsew" and weight=1 for expansion)
        ttk.Label(main_frame, text="Processing Log:", bootstyle="info").grid(row=row_index, column=0, columnspan=3, sticky='w', padx=5)
        
        # Text widget needs a Scrollbar
        text_frame = ttk.Frame(main_frame)
        text_frame.grid(row=row_index + 1, column=0, columnspan=3, sticky='nsew', padx=5, pady=5)
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)
        
        # Use tk.Text as ttkbootstrap does not provide a custom text widget
        self.log_box = tk.Text(text_frame, height=10, state='disabled', wrap='word') 
        self.log_box.grid(row=0, column=0, sticky="nsew")
        
        log_scroll = ttk.Scrollbar(text_frame, command=self.log_box.yview, bootstyle="round")
        log_scroll.grid(row=0, column=1, sticky='ns')
        self.log_box['yscrollcommand'] = log_scroll.set
        row_index += 2

        # 6. Run Button
        self.run_button = ttk.Button(main_frame, text="▶ START Processing", command=self.submit_inputs, bootstyle="success", width=40)
        self.run_button.grid(row=row_index, column=0, columnspan=3, pady=15)
        row_index += 1
