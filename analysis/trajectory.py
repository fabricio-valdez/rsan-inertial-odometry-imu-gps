import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rosbags.highlevel import AnyReader

# ================== CONFIG ==================
BAG_PATH   = Path("/home/fabri/EECE5554/lab5/data/driving1")
TOPIC_IMU  = "/imu"
TOPIC_GPS  = "/gps"

AXIS_FWD   = "x"      # IMU forward axis
BIAS_SECS  = 20.0     # first seconds assumed ~stationary for accel bias
REST_SECS  = 20.0     # also used to zero velocity baseline

HEADING_T0   = 30.0   # start time (s) to begin heading estimate (skip parked time)
HEADING_WIN  = 10.0   # duration (s) of window used for initial heading

# Magnetometer calibration (from circle run)
b = np.array([4.8550e-06, 2.3550e-06, 4.3755e-05])
S = np.array([
    [0.73407698, 0.0,        0.0],
    [0.0,        0.73712231, 0.0],
    [0.0,        0.0,        3.55722389],
])
# ============================================

def integrate_trapz(y, t):
    """Cumulative trapezoid integration: y(t) -> integral(t)."""
    y = np.asarray(y, float)
    t = np.asarray(t, float)
    out = np.zeros_like(y)
    if len(y) >= 2:
        dt = np.diff(t)
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dt)
    return out

def estimate_bias(x, t, bias_secs):
    """Estimate constant bias from first bias_secs seconds."""
    x = np.asarray(x, float)
    t = np.asarray(t, float)
    mask = (t - t[0]) <= bias_secs
    if not np.any(mask):
        mask = slice(0, min(len(x), 200))
    return float(np.median(x[mask]))

# ---------- 1) READ IMU + GPS FROM BAG ----------
t_imu, ax_list, ay_list, az_list = [], [], [], []
mx_list, my_list, mz_list = [], [], []

t_gps, gps_E, gps_N = [], [], []

with AnyReader([BAG_PATH]) as reader:
    conns_imu = [c for c in reader.connections if c.topic == TOPIC_IMU]
    conns_gps = [c for c in reader.connections if c.topic == TOPIC_GPS]
    if not conns_imu:
        raise RuntimeError(f"IMU topic {TOPIC_IMU} not found.")
    if not conns_gps:
        raise RuntimeError(f"GPS topic {TOPIC_GPS} not found.")

    # Read everything in one pass
    for conn, t_ns, raw in reader.messages():
        if conn.topic == TOPIC_IMU:
            msg = reader.deserialize(raw, conn.msgtype)
            ax_list.append(float(msg.imu.linear_acceleration.x))
            ay_list.append(float(msg.imu.linear_acceleration.y))
            az_list.append(float(msg.imu.linear_acceleration.z))

            mx_list.append(float(msg.mag_field.magnetic_field.x))
            my_list.append(float(msg.mag_field.magnetic_field.y))
            mz_list.append(float(msg.mag_field.magnetic_field.z))

            t_imu.append(t_ns * 1e-9)

        elif conn.topic == TOPIC_GPS:
            msg = reader.deserialize(raw, conn.msgtype)
            t_gps.append(t_ns * 1e-9)
            gps_E.append(float(msg.utm_easting))
            gps_N.append(float(msg.utm_northing))

# Convert to arrays & normalize time
t_imu = np.asarray(t_imu, float)
t_imu -= t_imu[0]
ax = np.asarray(ax_list, float)
ay = np.asarray(ay_list, float)
az = np.asarray(az_list, float)
mx = np.asarray(mx_list, float)
my = np.asarray(my_list, float)
mz = np.asarray(mz_list, float)

t_gps = np.asarray(t_gps, float)
t_gps -= t_gps[0]
gps_E = np.asarray(gps_E, float)
gps_N = np.asarray(gps_N, float)

# ---------- 2) FORWARD VELOCITY FROM ACCEL ----------
if AXIS_FWD == "x":
    a_fwd_raw = ax
elif AXIS_FWD == "y":
    a_fwd_raw = ay
elif AXIS_FWD == "z":
    a_fwd_raw = az
else:
    raise ValueError("AXIS_FWD must be 'x', 'y', or 'z'.")

# Bias removal (assume first BIAS_SECS stationary)
bias_a = estimate_bias(a_fwd_raw, t_imu, BIAS_SECS)
a_fwd = a_fwd_raw - bias_a
print(f"[INFO] Forward accel bias ({AXIS_FWD.upper()}): {bias_a:.6f} m/s^2")

# Integrate to velocity
v_fwd = integrate_trapz(a_fwd, t_imu)

# Optional velocity baseline correction using first REST_SECS seconds
mask_rest = t_imu <= REST_SECS
if np.any(mask_rest):
    v_offset = float(np.mean(v_fwd[mask_rest]))
else:
    v_offset = float(np.mean(v_fwd))
v_fwd -= v_offset
print(f"[INFO] Velocity offset removed: {v_offset:.6f} m/s")

# ---------- 3) HEADING FROM MAGNETOMETER ----------
M = np.column_stack([mx, my, mz])
M_bias = M - b
M_cal = M_bias @ S
mx_cal = M_cal[:, 0]
my_cal = M_cal[:, 1]

# yaw from calibrated mag, unwrapped
psi_mag = np.unwrap(np.arctan2(mx_cal, my_cal))  # rad

# ---------- 4) ROTATE FORWARD VELOCITY INTO N/E & INTEGRATE ----------
# v_N = v_fwd * cos(psi), v_E = v_fwd * sin(psi)
v_N_imu = v_fwd * np.cos(psi_mag)
v_E_imu = v_fwd * np.sin(psi_mag)

p_N_imu = integrate_trapz(v_N_imu, t_imu)
p_E_imu = integrate_trapz(v_E_imu, t_imu)

# ---------- 5) ALIGN WITH GPS TRACK (TRANSLATION + ROTATION) ----------
# Make both trajectories relative to their own starting point
p_N_imu_rel = p_N_imu - p_N_imu[0]
p_E_imu_rel = p_E_imu - p_E_imu[0]

gps_N_rel = gps_N - gps_N[0]
gps_E_rel = gps_E - gps_E[0]

# Estimate initial heading (first straight segment) for GPS & IMU
# Use first ~100 samples (tuned for your data length)
def initial_heading_time(E, N, t, t0, dt):
    """
    Estimate heading from displacement between t0 and t0+dt.
    t0   : start time of window
    dt   : duration of window
    """
    mask = (t >= t0) & (t <= t0 + dt)
    if np.sum(mask) < 2:
        raise RuntimeError("Not enough points in heading window")

    E_seg = E[mask]
    N_seg = N[mask]

    dE = E_seg[-1] - E_seg[0]
    dN = N_seg[-1] - N_seg[0]
    return np.arctan2(dN, dE)



theta_gps0 = initial_heading_time(gps_E_rel, gps_N_rel, t_gps, HEADING_T0, HEADING_WIN)
theta_imu0 = initial_heading_time(p_E_imu_rel, p_N_imu_rel, t_imu, HEADING_T0, HEADING_WIN)
dtheta = theta_gps0 - theta_imu0

# Rotate IMU trajectory by dtheta so first straight line matches GPS
# Aplicando Ratation Matrix :)
R = np.array([[np.cos(dtheta), -np.sin(dtheta)],
              [np.sin(dtheta),  np.cos(dtheta)]])
imu_NE = np.vstack((p_E_imu_rel, p_N_imu_rel))
imu_NE_rot = R @ imu_NE
p_E_imu_aligned = imu_NE_rot[0, :]
p_N_imu_aligned = imu_NE_rot[1, :]

# Finally, translate IMU track so both start at (0,0) in the same frame
# (already true because both are relative, but this keeps intention clear)
p_E_imu_aligned -= p_E_imu_aligned[0]
p_N_imu_aligned -= p_N_imu_aligned[0]

# --- Compute scaling factor to match IMU scale to GPS scale ---
gps_dist = np.sum(np.sqrt(np.diff(gps_E)**2 + np.diff(gps_N)**2))
imu_dist = np.sum(np.sqrt(np.diff(p_E_imu_aligned)**2 + np.diff(p_N_imu_aligned)**2))

scale = gps_dist / imu_dist
#print("IMU scale factor:", scale)

# Apply scaling
pE_imu_scaled = p_E_imu_aligned * scale
pN_imu_scaled = p_N_imu_aligned * scale

# ---------- 6) PLOT BOTH TRAJECTORIES ----------
plt.figure(figsize=(8, 8))
plt.plot(gps_E_rel,        gps_N_rel,        label="GPS trajectory", linewidth=2)
plt.plot(pE_imu_scaled,  pN_imu_scaled,  label="IMU-based trajectory", linewidth=1.5)

plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("Easting [m] (relative)")
plt.ylabel("Northing [m] (relative)")
plt.title("Vehicle Trajectory: IMU (mag + accel) vs GPS")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
