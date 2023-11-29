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
import matplotlib
matplotlib.use('TkAgg')

# Setup serial connection
ser = serial.Serial('COM3', 9600)

# Create figure for plotting
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
# Create a new subplot for the FFT plots
fig_FFT, (ax4, ax5) = plt.subplots(2, 1, figsize=(10, 12))
# Create a new subplot for the FFT plots
fig_FFT_HighPass, (ax6, ax7) = plt.subplots(2, 1, figsize=(10, 12))

data_emgRaw = []  # List to store raw EMG data
data_emg = []     # List to store processed EMG data
# Initialize lists for FFT data
fft_emgRaw = []
fft_emg = []
# Initialize lists for FFT data with High Pass fiter
fft_HighPass_emgRaw = []
fft_HighPass_emg = []

# Plot lines for combined and separate plots
plot_line_emgRaw_combined, = ax1.plot(data_emgRaw, lw=2, label='EMG Raw Value', color='blue')
plot_line_emg_combined, = ax1.plot(data_emg, lw=2, label='EMG Value', color='red')

plot_line_emgRaw_separate, = ax2.plot(data_emgRaw, lw=2, label='EMG Raw Value', color='blue')
plot_line_emg_separate, = ax3.plot(data_emg, lw=2, label='EMG Value', color='red')

# Plot lines for FFT plots
plot_line_fft_emgRaw, = ax4.plot(fft_emgRaw, lw=2, label='FFT EMG Raw', color='green')
plot_line_fft_emg, = ax5.plot(fft_emg, lw=2, label='FFT EMG', color='orange')

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
            # Apply high-pass filter
            filtered_emgRaw = highpass_filter(data_emgRaw, 100, 9600)  # Replace 4800 with your actual sampling rate
            filtered_emg = highpass_filter(data_emg, 100, 9600)
            fft_freq = np.linspace(0, 9600 / 2, len(fft_emgRaw))  # Frequency axis

            # Compute FFT for the filtered data
            fft_emgRaw = np.abs(fft(filtered_emgRaw))[:len(filtered_emgRaw) // 2]
            fft_emg = np.abs(fft(filtered_emg))[:len(filtered_emg) // 2]

            plot_line_fft_emgRaw.set_ydata(fft_emgRaw)
            plot_line_fft_emgRaw.set_xdata(
                np.linspace(0, 4800, len(fft_emgRaw)))  # Replace 4800 with half your sampling rate

            plot_line_fft_emg.set_ydata(fft_emg)
            plot_line_fft_emg.set_xdata(np.linspace(0, 4800, len(fft_emg)))

            ax4.relim()
            ax4.autoscale_view()
            ax5.relim()
            ax5.autoscale_view()
        except ValueError:
            print(f"Error in data conversion:")


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

atexit.register(save_data_to_csv)
atexit.register(save_FFT_data_to_csv)

ani1 = animation.FuncAnimation(fig, update_line, interval=100, cache_frame_data=False)
ani2 = animation.FuncAnimation(fig_FFT, update_line_FFT, interval=100, cache_frame_data=False)

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax5.legend()

ax1.set_title('Combined EMG Values')
ax2.set_title('EMG Raw Value')
ax3.set_title('EMG Value')
ax4.set_title('FFT of EMG Raw Value')
ax5.set_title('FFT of EMG Value')

plt.tight_layout()
plt.show()

