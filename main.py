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

# Workiong ploting without FFT

import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

# Setup serial connection
ser = serial.Serial('COM3', 9600)

# Create figure for plotting
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))  # Three subplots

data_emgRaw = []  # List to store raw EMG data
data_emg = []     # List to store processed EMG data

# Plot lines for combined and separate plots
plot_line_emgRaw_combined, = ax1.plot(data_emgRaw, lw=2, label='EMG Raw Value', color='blue')
plot_line_emg_combined, = ax1.plot(data_emg, lw=2, label='EMG Value', color='red')

plot_line_emgRaw_separate, = ax2.plot(data_emgRaw, lw=2, label='EMG Raw Value', color='blue')
plot_line_emg_separate, = ax3.plot(data_emg, lw=2, label='EMG Value', color='red')

def update_line(num):
    if ser.in_waiting > 0:
        serial_line = ser.readline().decode('utf-8').rstrip()  # Read and decode data from serial
        try:
            emgRawValue, emgValue = map(float, serial_line.split())  # Split and convert to float
            data_emgRaw.append(emgRawValue)
            data_emg.append(emgValue)

            # Update combined plot
            plot_line_emgRaw_combined.set_xdata(np.arange(len(data_emgRaw)))
            plot_line_emgRaw_combined.set_ydata(data_emgRaw)
            plot_line_emg_combined.set_xdata(np.arange(len(data_emg)))
            plot_line_emg_combined.set_ydata(data_emg)

            # Update separate plots
            plot_line_emgRaw_separate.set_xdata(np.arange(len(data_emgRaw)))
            plot_line_emgRaw_separate.set_ydata(data_emgRaw)
            plot_line_emg_separate.set_xdata(np.arange(len(data_emg)))
            plot_line_emg_separate.set_ydata(data_emg)

            # Adjust plot limits
            ax1.relim()
            ax1.autoscale_view()
            ax2.relim()
            ax2.autoscale_view()
            ax3.relim()
            ax3.autoscale_view()
        except ValueError:
            print(f"Error in data conversion: {serial_line}")

# Update every 100ms
ani = animation.FuncAnimation(fig, update_line, interval=100, cache_frame_data=False)


# Add legends
ax1.legend()
ax2.legend()
ax3.legend()

# Set titles for subplots
ax1.set_title('Combined EMG Values')
ax2.set_title('EMG Raw Value')
ax3.set_title('EMG Value')

plt.tight_layout()
plt.show()
