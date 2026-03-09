
import numpy as np
import matplotlib.pyplot as plt
from rosbags.highlevel import AnyReader
from pathlib import Path

# === Configure path to your bag ===
BAG_PATH = Path("/home/fabri/EECE5554/lab5/data/going_in_circles1") 

# --- Read bag ---
with AnyReader([BAG_PATH]) as reader:
    connections = [c for c in reader.connections if c.topic == '/imu']
    timestamps = []
    mag_x, mag_y, mag_z = [], [], []

    for connection, timestamp, rawdata in reader.messages(connections=connections):
        msg = reader.deserialize(rawdata, connection.msgtype)

        # Extract mag values
        mx = msg.mag_field.magnetic_field.x
        my = msg.mag_field.magnetic_field.y
        mz = msg.mag_field.magnetic_field.z

        mag_x.append(mx)
        mag_y.append(my)
        mag_z.append(mz)
        timestamps.append(timestamp * 1e-9)  # ns → seconds


# -- We calibrate using a simple min-max approach ---
def calibrate_mag_minmax(mx, my, mz):
    M = np.column_stack([mx, my, mz]).astype(float)

    # --- Hard-iron (bias) ---
    mins = np.nanmin(M, axis=0)
    maxs = np.nanmax(M, axis=0)
    b = (maxs + mins) / 2.0

    M_bias = M - b  # bias-corrected samples

    # --- Soft-iron (diagonal) scales (computed on bias-corrected data) ---
    # Half-ranges per axis
    sx, sy, sz = (np.nanmax(M_bias, axis=0) - np.nanmin(M_bias, axis=0)) / 2.0
    # Guard against tiny ranges
    eps = 1e-9
    sx = sx if sx > eps else eps
    sy = sy if sy > eps else eps
    sz = sz if sz > eps else eps

    s_avg = (sx + sy + sz) / 3.0
    S = np.diag([s_avg / sx, s_avg / sy, s_avg / sz])  # 3x3

    # --- Apply once to every sample ---
    M_cal = M_bias @ S  # same as @ S.T for diagonal

    return M_cal, b, S


M_cal, b, S = calibrate_mag_minmax(np.array(mag_x), np.array(mag_y), np.array(mag_z))


print('bias:', b, '\nsoft-iron: ', S )

# --- We plot Calibrated vs Raw---

# --- Extract components ---
E_raw = np.array(mag_x)
N_raw = np.array(mag_y)
E_cal = M_cal[:, 0]
N_cal = M_cal[:, 1]

# --- Plot both on the same axes ---
plt.figure(figsize=(7, 7))
plt.scatter(E_raw, N_raw, s=5, alpha=0.6, label='Uncalibrated', color='steelblue')
plt.scatter(E_cal, N_cal, s=5, alpha=0.6, label='Calibrated', color='orange')

plt.title("Fig 1: Magnetometer N–E Components (Before & After Calibration)")
plt.xlabel("Easting (µT)")
plt.ylabel("Northing (µT)")
plt.axis('equal')          # same scale for N and E
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
