import numpy as np
import matplotlib.pyplot as plt
from rosbags.highlevel import AnyReader
from pathlib import Path

# === Config ===
BAG_PATH   = Path("/home/fabri/EECE5554/lab5/data/driving1")
TOPIC      = "/imu"
BIAS_SECS  = 20.0   # seconds at start assumed "still" for gyro bias estimate
alpha = 0.80       # complementary filter parameter
# ==============

def wrap_to_pi(angle):
    """Wrap angle to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

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

def quat_to_yaw(qx, qy, qz, qw):
    """
    Convert quaternion to yaw (heading) in radians.
    Assuming ENU / standard robotics convention:
    yaw about Z.
    """
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
    return np.arctan2(siny_cosp, cosy_cosp)

# --- Read bag ---
timestamps = []
mag_x, mag_y, mag_z = [], [], []
gyro_z = []
imu_yaw_list = []

with AnyReader([BAG_PATH]) as reader:
    connections = [c for c in reader.connections if c.topic == TOPIC]
    if not connections:
        raise RuntimeError(f"Topic {TOPIC} not found in bag.")
    for connection, timestamp, rawdata in reader.messages(connections=connections):
        msg = reader.deserialize(rawdata, connection.msgtype)

        # Magnetometer
        mx = msg.mag_field.magnetic_field.x
        my = msg.mag_field.magnetic_field.y
        mz = msg.mag_field.magnetic_field.z

        # Gyro (rad/s)
        gz = msg.imu.angular_velocity.z

        # IMU orientation (quaternion -> heading)
        qx = msg.imu.orientation.x
        qy = msg.imu.orientation.y
        qz = msg.imu.orientation.z
        qw = msg.imu.orientation.w
        yaw_imu = quat_to_yaw(qx, qy, qz, qw)

        mag_x.append(mx)
        mag_y.append(my)
        mag_z.append(mz)
        gyro_z.append(gz)
        imu_yaw_list.append(yaw_imu)
        timestamps.append(timestamp * 1e-9)  # ns → s

# Time vector
t = np.array(timestamps, float)
t = t - t[0]  # start at zero

mag_x = np.array(mag_x, float)
mag_y = np.array(mag_y, float)
mag_z = np.array(mag_z, float)
gyro_z = np.array(gyro_z, float)
imu_yaw = np.array(imu_yaw_list, float)

# === Magnetometer calibration parameters (from your circle run) ===
b = np.array([4.8550e-06, 2.3550e-06, 4.3755e-05])

S = np.array([
    [0.73407698, 0.0,        0.0],
    [0.0,        0.73712231, 0.0],
    [0.0,        0.0,        3.55722389]
])

# --- Apply mag calibration ---
M      = np.column_stack([mag_x, mag_y, mag_z])
M_bias = M - b
M_cal  = M_bias @ S

mx_cal = M_cal[:, 0]
my_cal = M_cal[:, 1]

# --- Magnetometer yaw (calibrated) ---
psi_cal = np.unwrap(np.arctan2(mx_cal, my_cal))   # radians

# --- Gyro yaw: detrend + integrate ---
bias_gz = estimate_bias(gyro_z, t, BIAS_SECS)
gyro_z_detr = gyro_z - bias_gz          # rad/s

psi_gyro = trapz_integrate(gyro_z_detr, t)  # radians (integrated yaw)
psi_gyro = np.unwrap(psi_gyro)

# --- Complementary filter (gyro rate + mag yaw) ---
psi_fused = np.zeros_like(psi_cal)
psi_fused[0] = psi_cal[0]  # start from mag heading

for k in range(1, len(t)):
    dt = t[k] - t[k-1]

    # 1) Predict heading from gyro (integrate rate)
    psi_pred = psi_fused[k-1] + gyro_z_detr[k] * dt
    psi_pred = wrap_to_pi(psi_pred)

    # 2) Innovation (mag - predicted), shortest angle
    err = wrap_to_pi(psi_cal[k] - psi_pred)

    # 3) Complementary update
    psi_fused[k] = wrap_to_pi(psi_pred + (1.0 - alpha) * err)

# --- Low-pass mag component (explicit LP on mag yaw) ---
psi_lp = np.zeros_like(psi_cal)
psi_lp[0] = psi_cal[0]
for k in range(1, len(t)):
    # basic 1st-order IIR low-pass with same (1-alpha)
    err_lp = wrap_to_pi(psi_cal[k] - psi_lp[k-1])
    psi_lp[k] = wrap_to_pi(psi_lp[k-1] + (1.0 - alpha) * err_lp)

# --- High-pass gyro component (fused - LP(mag)) ---
psi_hp = wrap_to_pi(psi_fused - psi_lp)

# --- IMU heading estimate (from quaternion) ---
imu_yaw = np.unwrap(imu_yaw)
imu_yaw_deg = np.degrees(imu_yaw)

# --- Convert all to degrees for plotting ---
psi_lp_deg    = np.degrees(psi_lp)       # low-pass mag
psi_hp_deg    = np.degrees(psi_hp)       # high-pass gyro contribution
psi_fused_deg = np.degrees(psi_fused)    # complementary filter output
# imu_yaw_deg already computed

# === 4 Subplots ===
fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

# 1) Low-pass magnetometer yaw
axes[0].plot(t, psi_lp_deg, label='Low-pass Mag yaw')
axes[0].set_ylabel('Yaw [deg]')
axes[0].set_title('Low-pass filtered Magnetometer Yaw')
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend()

# 2) High-pass gyro contribution
axes[1].plot(t, psi_hp_deg, label='High-pass Gyro contribution')
axes[1].set_ylabel('Yaw [deg]')
axes[1].set_title('High-pass filtered Gyro component')
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].legend()

# 3) Complementary filter fused yaw
axes[2].plot(t, psi_fused_deg, label='Fused yaw (complementary)')
axes[2].set_ylabel('Yaw [deg]')
axes[2].set_title('Complementary Filter Output')
axes[2].grid(True, linestyle='--', alpha=0.6)
axes[2].legend()

# 4) IMU heading estimate (from quaternion)
axes[3].plot(t, imu_yaw_deg, label='IMU heading (VN-100)', color='tab:orange')
axes[3].set_xlabel('Time [s]')
axes[3].set_ylabel('Yaw [deg]')
axes[3].set_title('IMU Heading Estimate')
axes[3].grid(True, linestyle='--', alpha=0.6)
axes[3].legend()

plt.tight_layout()
plt.show()
