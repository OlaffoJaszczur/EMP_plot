# Checking if data is beeng colected!!!!!!!!!

# import serial
# import time
#
# # Set up the serial connection
# # Update the port name and baud rate according to your Arduino configuration
# port = 'COM3'
# baud_rate = 9600
#
# try:
#     ser = serial.Serial(port, baud_rate)
#     print(f"Connected to {port} at {baud_rate} baud rate.")
#
#     while True:
#         if ser.in_waiting > 0:
#             line = ser.readline().decode('utf-8').rstrip()
#             print("Received data:", line)
#         time.sleep(0.1)
#
# except serial.SerialException as e:
#     print(f"Error: {e}")
#
# except KeyboardInterrupt:
#     print("Serial reading stopped.")
#     ser.close()

# Workiong ploting and cvs saving without FFT

import serial
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from numpy.fft import fft
from scipy.signal import butter, filtfilt
from scipy.signal import welch
from scipy.signal import spectrogram
import matplotlib
matplotlib.use('TkAgg')

# Setup serial connection
ser = serial.Serial('COM3', 9600)

# Create figure for plotting
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
# Create a new subplot for the FFT plots
fig_FFT, (ax4, ax5) = plt.subplots(2, 1, figsize=(10, 12))
# Create a new subplot for the FFT with High Pass filter plots
fig_FFT_HighPass, (ax6, ax7) = plt.subplots(2, 1, figsize=(10, 12))

data_emgRaw = []  # List to store raw EMG data
data_emg = []     # List to store processed EMG data
# Initialize lists for FFT data
fft_emgRaw = []
fft_emg = []
# Initialize lists for FFT data with High Pass fiter
fft_emgRaw_HighPass = []
fft_emg_HighPass = []

# Plot lines for combined and separate plots
plot_line_emgRaw_combined, = ax1.plot(data_emgRaw, lw=2, label='EMG Raw Value', color='blue')
plot_line_emg_combined, = ax1.plot(data_emg, lw=2, label='EMG Value', color='red')

plot_line_emgRaw_separate, = ax2.plot(data_emgRaw, lw=2, label='EMG Raw Value', color='blue')
plot_line_emg_separate, = ax3.plot(data_emg, lw=2, label='EMG Value', color='red')

# Plot lines for FFT plots
plot_line_fft_emgRaw, = ax4.plot(fft_emgRaw, lw=2, label='FFT EMG Raw', color='green')
plot_line_fft_emg, = ax5.plot(fft_emg, lw=2, label='FFT EMG', color='orange')

# Plot lines for FFT with High Pass plots
plot_line_fft_emgRaw_HighPass, = ax6.plot(fft_emgRaw_HighPass, lw=2, label='FFT High Pass EMG Raw', color='purple')
plot_line_fft_emg_HighPass, = ax7.plot(fft_emg_HighPass, lw=2, label='FFT High Pass EMG', color='black')

# Label the x and y axes
ax1.set_xlabel('Time')
ax1.set_ylabel('Amplitude')
ax2.set_xlabel('Time')
ax2.set_ylabel('Amplitude')
ax3.set_xlabel('Time')
ax3.set_ylabel('Amplitude')

ax4.set_xlabel('Frequency')
ax4.set_ylabel('Magnitude')
ax5.set_xlabel('Frequency')
ax5.set_ylabel('Magnitude')

ax6.set_xlabel('Frequency')
ax6.set_ylabel('Magnitude')
ax7.set_xlabel('Frequency')
ax7.set_ylabel('Magnitude')

# High-pass filter design and Application
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def update_line(num):
    global data_emgRaw, data_emg
    if ser.in_waiting > 0:
        serial_line = ser.readline().decode('utf-8').rstrip()
        try:
            emgRawValue, emgValue = map(float, serial_line.split())
            data_emgRaw.append(emgRawValue)
            data_emg.append(emgValue)

            plot_line_emgRaw_combined.set_xdata(np.arange(len(data_emgRaw)))
            plot_line_emgRaw_combined.set_ydata(data_emgRaw)
            plot_line_emg_combined.set_xdata(np.arange(len(data_emg)))
            plot_line_emg_combined.set_ydata(data_emg)

            plot_line_emgRaw_separate.set_xdata(np.arange(len(data_emgRaw)))
            plot_line_emgRaw_separate.set_ydata(data_emgRaw)
            plot_line_emg_separate.set_xdata(np.arange(len(data_emg)))
            plot_line_emg_separate.set_ydata(data_emg)

            ax1.relim()
            ax1.autoscale_view()
            ax2.relim()
            ax2.autoscale_view()
            ax3.relim()
            ax3.autoscale_view()
        except ValueError:
            print(f"Error in data conversion: {serial_line}")

def update_line_FFT(num):
    global fft_emgRaw, fft_emg, fft_freq  # Declare as global to modify
    if ser.in_waiting > 0:
        try:
            # Compute FFT for both sets of data and update the plots
            fft_emgRaw = np.abs(fft(data_emgRaw))[:len(data_emgRaw) // 2]  # Compute and plot only half (positive frequencies)
            fft_emg = np.abs(fft(data_emg))[:len(data_emg) // 2]
            fft_freq = np.linspace(0, 9600 / 2, len(fft_emgRaw_HighPass))  # Frequency axis

            plot_line_fft_emgRaw.set_ydata(fft_emgRaw)
            plot_line_fft_emgRaw.set_xdata(np.linspace(0, 9600, len(fft_emgRaw)))  # Replace 4800 with half your sampling rate

            plot_line_fft_emg.set_ydata(fft_emg)
            plot_line_fft_emg.set_xdata(np.linspace(0, 9600, len(fft_emg)))

            ax4.relim()
            ax4.autoscale_view()
            ax5.relim()
            ax5.autoscale_view()
        except ValueError:
            print(f"Error in data conversion FFT")

def update_line_FFT_HighPass(num):
    global fft_emgRaw_HighPass, fft_emg_HighPass, fft_freq_HighPass  # Declare as global to modify
    if ser.in_waiting > 0:
        try:
            # Apply high-pass filter
            filtered_emgRaw = highpass_filter(data_emgRaw, 100, 9600)  # sampling rate = 9600
            filtered_emg = highpass_filter(data_emg, 100, 9600)
            fft_freq_HighPass = np.linspace(0, 9600 / 2, len(fft_emgRaw_HighPass))  # Frequency axis

            # Compute FFT for the filtered data
            fft_emgRaw_HighPass = np.abs(fft(filtered_emgRaw))[:len(filtered_emgRaw) // 2]
            fft_emg_HighPass = np.abs(fft(filtered_emg))[:len(filtered_emg) // 2]

            plot_line_fft_emgRaw_HighPass.set_ydata(fft_emgRaw_HighPass)
            plot_line_fft_emgRaw_HighPass.set_xdata(np.linspace(0, 9600, len(fft_emgRaw_HighPass)))

            plot_line_fft_emg_HighPass.set_ydata(fft_emg_HighPass)
            plot_line_fft_emg_HighPass.set_xdata(np.linspace(0, 9600, len(fft_emg_HighPass)))

            ax6.relim()
            ax6.autoscale_view()
            ax7.relim()
            ax7.autoscale_view()
        except ValueError:
            print(f"Error in data conversion FFT High Pass filter")


# Register a function to save data before exiting
import atexit
def save_data_to_csv():
    df = pd.DataFrame({'Time': np.arange(len(data_emg)), 'EMG': data_emg, 'Raw EMG': data_emgRaw})
    df.to_csv('emg_data.csv', index=False)
    print("Data saved to emg_data.csv")

def save_FFT_data_to_csv():
    fft_df = pd.DataFrame({
        'Frequency': fft_freq,
        'FFT Raw EMG': fft_emgRaw,
        'FFT EMG': fft_emg
    })
    fft_df.to_csv('emg_FFT_data.csv', index=False)
    print("FFT data saved to emg_FFT_data.csv")
def save_FFT_HighPass_data_to_csv():
    fft_HighPass_df = pd.DataFrame({
        'Frequency': fft_freq_HighPass,
        'FFT Raw EMG High Pass filter': fft_emgRaw_HighPass,
        'FFT EMG High Pass filter': fft_emg_HighPass
    })
    fft_HighPass_df.to_csv('emg_FFT_High_Pass_filter_data.csv', index=False)
    print("FFT data saved to emg_FFT_High_Pass_filter_data.csv")

atexit.register(save_data_to_csv)
atexit.register(save_FFT_data_to_csv)
atexit.register(save_FFT_HighPass_data_to_csv)

ani1 = animation.FuncAnimation(fig, update_line, interval=100, cache_frame_data=False)
ani2 = animation.FuncAnimation(fig_FFT, update_line_FFT, interval=100, cache_frame_data=False)
ani3 = animation.FuncAnimation(fig_FFT_HighPass, update_line_FFT_HighPass, interval=100, cache_frame_data=False)

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax5.legend()
ax6.legend()
ax7.legend()

ax1.set_title('Combined EMG Values')
ax2.set_title('EMG Raw Value')
ax3.set_title('EMG Value')
ax4.set_title('FFT of EMG Raw Value')
ax5.set_title('FFT of EMG Value')
ax6.set_title('FFT with High Pass Filter of EMG Raw Value')
ax7.set_title('FFT with High Pass Filter of EMG Value')

plt.tight_layout()
plt.show()

# Ploting data after program stops

# Load data from CSV
df = pd.read_csv('emg_data.csv')
emg_data = df['EMG'].values
raw_emg_data = df['Raw EMG'].values

# Define the sampling rate
fs = 9600  # Replace this with the actual sampling rate


# Calculate PSD for EMG data
# Assuming data length is 141
nperseg_value = len(emg_data)  # or any smaller value for higher frequency resolution

f_emg, Pxx_emg = welch(emg_data, fs, nperseg=nperseg_value)
f_raw_emg, Pxx_raw_emg = welch(raw_emg_data, fs, nperseg=nperseg_value)


# Plot PSD
plt.figure()
plt.subplot(2, 1, 1)
plt.semilogy(f_emg, Pxx_emg)
plt.title('PSD of EMG Data')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power/Frequency [dB/Hz]')

plt.subplot(2, 1, 2)
plt.semilogy(f_raw_emg, Pxx_raw_emg)
plt.title('PSD of Raw EMG Data')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power/Frequency [dB/Hz]')

plt.tight_layout()
plt.show()


# Compute the spectrogram

# Normalize the data
emg_data_normalized = (emg_data - np.mean(emg_data)) / np.std(emg_data)
raw_emg_data_normalized = (raw_emg_data - np.mean(raw_emg_data)) / np.std(raw_emg_data)

f, t, Sxx_emg = spectrogram(emg_data, fs)
f, t, Sxx_raw_emg = spectrogram(raw_emg_data, fs)

# Plotting code
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.pcolormesh(t, f, 10 * np.log10(Sxx_emg), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Spectrogram of EMG Data')
plt.colorbar(label='Intensity [dB]')

plt.subplot(2, 1, 2)
plt.pcolormesh(t, f, 10 * np.log10(Sxx_raw_emg), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Spectrogram of Raw EMG Data')
plt.colorbar(label='Intensity [dB]')

plt.tight_layout()
plt.show()
plt.show()