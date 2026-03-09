import numpy as np
import matplotlib.pyplot as plt
from rosbags.highlevel import AnyReader
from pathlib import Path

# === Config ===
BAG_PATH   = Path("/home/fabri/EECE5554/lab5/data/driving1")
TOPIC      = "/imu"
BIAS_SECS  = 2.0   # seconds at start assumed "still" for gyro bias estimate
# ==============

def estimate_bias(x, t, bias_secs=2.0):
    x = np.asarray(x, float)
    t = np.asarray(t, float)
    mask = (t - t[0]) <= bias_secs
    if not np.any(mask):
        mask = slice(0, min(len(x), 200))
    return float(np.median(x[mask]))

def trapz_integrate(y, t):
    """Cumulative trapezoid integration with true dt."""
    y = np.asarray(y, float)
    t = np.asarray(t, float)
    out = np.zeros_like(y, dtype=float)
    if len(y) >= 2:
        dt = np.diff(t)
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dt)
    return out

# --- Read bag ---
timestamps = []
mag_x, mag_y, mag_z = [], [], []
gyro_z = []

with AnyReader([BAG_PATH]) as reader:
    connections = [c for c in reader.connections if c.topic == TOPIC]
    if not connections:
        raise RuntimeError(f"Topic {TOPIC} not found in bag.")
    for connection, timestamp, rawdata in reader.messages(connections=connections):
        msg = reader.deserialize(rawdata, connection.msgtype)

        # Mag
        mx = msg.mag_field.magnetic_field.x
        my = msg.mag_field.magnetic_field.y
        mz = msg.mag_field.magnetic_field.z

        # Gyro (rad/s)
        gz = msg.imu.angular_velocity.z

        mag_x.append(mx)
        mag_y.append(my)
        mag_z.append(mz)
        gyro_z.append(gz)
        timestamps.append(timestamp * 1e-9)  # ns → s

# Time vector
t = np.array(timestamps, float)
t = t - t[0]  # start at zero

mag_x = np.array(mag_x, float)
mag_y = np.array(mag_y, float)
mag_z = np.array(mag_z, float)
gyro_z = np.array(gyro_z, float)

# === Magnetometer calibration parameters (from your circle run) ===
b = np.array([4.8550e-06, 2.3550e-06, 4.3755e-05])

S = np.array([
    [0.73407698, 0.0,        0.0],
    [0.0,        0.73712231, 0.0],
    [0.0,        0.0,        3.55722389]
])

# --- Apply mag calibration ---
M     = np.column_stack([mag_x, mag_y, mag_z])
M_bias = M - b
M_cal  = M_bias @ S

mx_cal = M_cal[:, 0]
my_cal = M_cal[:, 1]

# --- Magnetometer yaw (raw vs calibrated) ---
psi_raw = np.unwrap(np.arctan2(mag_x, mag_y))      # radians
psi_cal = np.unwrap(np.arctan2(mx_cal,  my_cal))   # radians

psi_raw_deg = np.degrees(psi_raw)
psi_cal_deg = np.degrees(psi_cal)

# --- Gyro yaw: detrend + integrate ---
bias_gz = estimate_bias(gyro_z, t, BIAS_SECS)
gyro_z_detr = gyro_z - bias_gz

psi_gyro = trapz_integrate(gyro_z_detr, t)   # radians
psi_gyro = np.unwrap(psi_gyro)
psi_gyro_deg = np.degrees(psi_gyro)

print(f"Estimated gyro Z bias: {bias_gz:.6f} rad/s")

# === Plot comparison ===

# 1) Yaw vs time
plt.figure(figsize=(10, 5))
plt.plot(t, psi_cal_deg,  label='Mag yaw (calibrated)', linewidth=1.2)
plt.plot(t, psi_gyro_deg, label='Gyro yaw (integrated)', linewidth=1.0, alpha=0.8)
plt.xlabel('Time [s]')
plt.ylabel('Yaw [deg]')
plt.title('Yaw angle: Calibrated Magnetometer vs Integrated Gyro')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
