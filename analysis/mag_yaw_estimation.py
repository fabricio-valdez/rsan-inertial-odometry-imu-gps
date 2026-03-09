import numpy as np
import matplotlib.pyplot as plt
from rosbags.highlevel import AnyReader
from pathlib import Path

# === Configure path to your bag ===
BAG_PATH = Path("/home/fabri/EECE5554/lab5/data/driving1")

# --- Read bag ---
with AnyReader([BAG_PATH]) as reader:
    connections = [c for c in reader.connections if c.topic == '/imu']
    timestamps = []
    mag_x, mag_y, mag_z = [], [], []

    for connection, timestamp, rawdata in reader.messages(connections=connections):
        msg = reader.deserialize(rawdata, connection.msgtype)

        mx = msg.mag_field.magnetic_field.x
        my = msg.mag_field.magnetic_field.y
        mz = msg.mag_field.magnetic_field.z

        mag_x.append(mx)
        mag_y.append(my)
        mag_z.append(mz)
        timestamps.append(timestamp * 1e-9)

# -- time array
t = np.array(timestamps)
t = t - t[0]

# === Calibration parameters (from your circle data) ===
b = np.array([4.8550e-06, 2.3550e-06, 4.3755e-05])

S = np.array([
    [0.73407698, 0.0,        0.0],
    [0.0,        0.73712231, 0.0],
    [0.0,        0.0,        3.55722389]
])

# Stack magnetometer vectors
M = np.column_stack([mag_x, mag_y, mag_z]).astype(float)

# Apply calibration
M_bias = M - b
M_cal  = M_bias @ S

mx_cal = M_cal[:, 0]
my_cal = M_cal[:, 1]

# === Raw and calibrated yaw ===
psi_raw = np.unwrap(np.arctan2(np.array(mag_x), np.array(mag_y)))
psi_cal = np.unwrap(np.arctan2(mx_cal,             my_cal))

# Convert to degrees
psi_raw_deg = np.degrees(psi_raw)
psi_cal_deg = np.degrees(psi_cal)

# === Plot ===
plt.figure(figsize=(10, 4))
plt.plot(t, psi_raw_deg, label='Raw mag yaw')
plt.plot(t, psi_cal_deg, label='Calibrated mag yaw')
plt.xlabel('Time [s]')
plt.ylabel('Yaw [deg]')
plt.title('Magnetometer Yaw: Raw vs Calibrated')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

