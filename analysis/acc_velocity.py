import numpy as np
import matplotlib.pyplot as plt
from rosbags.highlevel import AnyReader
from pathlib import Path

# ============ Config ============
BAG_PATH   = Path("/home/fabri/EECE5554/lab5/data/driving1")
TOPIC      = "/imu"
BIAS_SECS  = 2.0    # seconds at start assumed "still" for accel bias
REST_SECS  = 2.0    # seconds we treat as 'car at rest' for velocity offset
AXIS       = "x"     # "x", "y", or "z" = forward axis in IMU frame
G_MAG      = 9.81    # m/s^2 gravity magnitude
# ================================

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
    """Estimate a constant bias from the first bias_secs seconds."""
    x = np.asarray(x, float)
    t = np.asarray(t, float)
    mask = (t - t[0]) <= bias_secs
    if not np.any(mask):
        mask = slice(0, min(len(x), 200))
    return float(np.median(x[mask]))

def quat_to_rotmat(qx, qy, qz, qw):
    """
    Convert quaternion to 3x3 rotation matrix.
    Assumes quaternion gives rotation from body -> world frame.
    """
    # normalize just in case
    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if norm == 0:
        return np.eye(3)
    qx /= norm; qy /= norm; qz /= norm; qw /= norm

    # standard DCM for body->world
    R = np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),         2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),         1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),         2*(qy*qz + qx*qw),         1 - 2*(qx*qx + qy*qy)]
    ])
    return R

# --- Read accel & orientation from bag ---
timestamps = []
a_body_x, a_body_y, a_body_z = [], [], []
g_body_x, g_body_y, g_body_z = [], [], []

with AnyReader([BAG_PATH]) as reader:
    conns = [c for c in reader.connections if c.topic == TOPIC]
    if not conns:
        raise RuntimeError(f"Topic {TOPIC} not found in bag. Available: {[c.topic for c in reader.connections]}")
    for conn, t_ns, rawdata in reader.messages(connections=conns):
        msg = reader.deserialize(rawdata, conn.msgtype)

        # Raw accel in body frame (likely includes gravity)
        ax = float(msg.imu.linear_acceleration.x)
        ay = float(msg.imu.linear_acceleration.y)
        az = float(msg.imu.linear_acceleration.z)

        # Orientation quaternion
        qx = float(msg.imu.orientation.x)
        qy = float(msg.imu.orientation.y)
        qz = float(msg.imu.orientation.z)
        qw = float(msg.imu.orientation.w)

        # Rotation from body -> world
        R_bw = quat_to_rotmat(qx, qy, qz, qw)
        # Gravity in world frame (assuming +Z up, so gravity is +G_MAG in +Z)
        g_world = np.array([0.0, 0.0, G_MAG])
        # Convert gravity into body frame: world -> body
        R_wb = R_bw.T
        g_body = R_wb @ g_world

        a_body_x.append(ax)
        a_body_y.append(ay)
        a_body_z.append(az)
        g_body_x.append(g_body[0])
        g_body_y.append(g_body[1])
        g_body_z.append(g_body[2])

        timestamps.append(t_ns * 1e-9)  # ns -> s

t = np.asarray(timestamps, float)
t -= t[0]

a_body_x = np.asarray(a_body_x, float)
a_body_y = np.asarray(a_body_y, float)
a_body_z = np.asarray(a_body_z, float)
g_body_x = np.asarray(g_body_x, float)
g_body_y = np.asarray(g_body_y, float)
g_body_z = np.asarray(g_body_z, float)

# --- Choose forward axis ---
if AXIS == "x":
    a_raw_axis   = a_body_x
    g_axis       = g_body_x
elif AXIS == "y":
    a_raw_axis   = a_body_y
    g_axis       = g_body_y
elif AXIS == "z":
    a_raw_axis   = a_body_z
    g_axis       = g_body_z
else:
    raise ValueError("AXIS must be 'x', 'y', or 'z'.")

print(f"[INFO] Using axis '{AXIS.upper()}' as forward acceleration.")



# --- Step 0: gravity removal in body frame ---
#   a_fwd_nograv = measured accel - gravity projection on that axis
a_fwd_nograv = a_raw_axis - g_axis

# --- Step 1: velocity from *raw* acceleration (no gravity, no bias removal) ---
v_fwd_raw = integrate_trapz(a_raw_axis, t)

# --- Step 2: estimate bias *after gravity removal* and subtract it ---
bias_a = estimate_bias(a_fwd_nograv, t, BIAS_SECS)
a_fwd_corr = a_fwd_nograv - bias_a
print(f"[INFO] Estimated accel bias on {AXIS.upper()} after gravity removal: {bias_a:.6f} m/s^2")

# Velocity from gravity-removed & bias-corrected acceleration
v_fwd_detr = integrate_trapz(a_fwd_corr, t)

# --- Step 3: optional velocity offset adjustment ---
mask_rest = t <= REST_SECS
if np.any(mask_rest):
    offset_v = float(np.mean(v_fwd_detr[mask_rest]))
else:
    offset_v = float(np.mean(v_fwd_detr))  # fallback

v_fwd_adjusted = v_fwd_detr - offset_v
print(f"[INFO] Velocity offset removed: {offset_v:.6f} m/s")

# --- Plots: before vs after adjustment ---
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t, v_fwd_raw, label='Forward velocity (from raw accel)')
plt.ylabel('Velocity [m/s]')
plt.title(f'Forward Velocity from Accelerometer ({AXIS.upper()} axis, raw)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, v_fwd_adjusted, label='Forward velocity (corrected)')
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.title('Forward Velocity Corrected')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()
