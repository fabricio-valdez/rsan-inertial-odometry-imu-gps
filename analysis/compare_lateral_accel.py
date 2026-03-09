#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rosbags.highlevel import AnyReader

# ==================== CONFIG ====================
BAG_PATH  = Path("/home/fabri/EECE5554/lab5/data/going_in_circles1")
IMU_TOPIC = "/imu"
GPS_TOPIC = "/gps"
BIAS_SECS = 5.0     # seconds to assume car is not moving for accel bias
# =================================================

def estimate_bias(x, t, bias_secs):
    """Estimate constant bias from first few seconds."""
    x = np.asarray(x)
    mask = (t - t[0]) <= bias_secs
    if not np.any(mask):
        mask = slice(0, min(len(x), 200))
    return float(np.median(x[mask]))

def main():

    # ------------- READ IMU (yaw rate + accel Y) -------------
    t_imu = []
    ay_raw = []
    wz_raw = []

    with AnyReader([BAG_PATH]) as reader:
        conns = [c for c in reader.connections if c.topic == IMU_TOPIC]
        if not conns:
            raise RuntimeError(f"IMU topic {IMU_TOPIC} not found in bag.")

        for conn, t_ns, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            imu = msg.imu

            t_imu.append(t_ns * 1e-9)
            ay_raw.append(float(imu.linear_acceleration.y))   # lateral accel
            wz_raw.append(float(imu.angular_velocity.z))      # yaw rate

    t_imu = np.asarray(t_imu, float)
    ay_raw = np.asarray(ay_raw, float)
    wz_raw = np.asarray(wz_raw, float)

    idx = np.argsort(t_imu)
    t_imu = t_imu[idx] - t_imu[idx][0]
    ay_raw = ay_raw[idx]
    wz_raw = wz_raw[idx]

    # ----------- Remove yaw-rate bias -----------
    bias_wz = estimate_bias(wz_raw, t_imu, BIAS_SECS)
    wz = wz_raw - bias_wz

    # ----------- Remove accel Y bias -----------
    bias_ay = estimate_bias(ay_raw, t_imu, BIAS_SECS)
    ay_obs = ay_raw - bias_ay

    print(f"[INFO] accel Y bias removed: {bias_ay:.6f} m/s²")
    print(f"[INFO] yaw-rate bias removed: {bias_wz:.6f} rad/s")

    # ---------------- READ GPS (UTM) ----------------
    t_gps = []
    easts = []
    norths = []

    with AnyReader([BAG_PATH]) as reader:
        gps_conns = [c for c in reader.connections if c.topic == GPS_TOPIC]
        if not gps_conns:
            raise RuntimeError(f"GPS topic {GPS_TOPIC} not found in bag.")

        for conn, t_ns, raw in reader.messages(connections=gps_conns):
            msg = reader.deserialize(raw, conn.msgtype)

            t_gps.append(t_ns * 1e-9)
            easts.append(float(msg.utm_easting))
            norths.append(float(msg.utm_northing))

    t_gps = np.asarray(t_gps, float)
    easts = np.asarray(easts, float)
    norths = np.asarray(norths, float)

    idx_gps = np.argsort(t_gps)
    t_gps = t_gps[idx_gps] - t_gps[idx_gps][0]
    easts = easts[idx_gps]
    norths = norths[idx_gps]

    # ------------- GPS SPEED COMPUTATION -------------
    dt = np.diff(t_gps)
    dx = np.diff(easts)
    dy = np.diff(norths)

    dt[dt <= 0] = np.nan
    speed = np.sqrt(dx**2 + dy**2) / dt   # m/s
    t_speed = 0.5 * (t_gps[1:] + t_gps[:-1])

    # Remove NANs
    speed_valid = speed[~np.isnan(speed)]
    t_speed_valid = t_speed[~np.isnan(speed)]

    # Interpolate GPS speed to IMU timestamps
    v_imu = np.interp(t_imu, t_speed_valid, speed_valid, left=np.nan, right=np.nan)

    # ----------- Modeled lateral acceleration -----------
    #  For circular motion:  a_y_model = v * omega_z
    a_y_model = v_imu * wz

    # Remove NAN regions (GPS not available)
    valid = ~np.isnan(a_y_model)
    t_valid = t_imu[valid]
    ay_obs_valid = ay_obs[valid]
    a_y_model_valid = a_y_model[valid]

    # ====================== PLOT ======================
    plt.figure(figsize=(10, 5))
    plt.plot(t_valid, ay_obs_valid, label="Observed lateral accel (IMU)", linewidth=1.2)
    plt.plot(t_valid, a_y_model_valid, label="Modeled accel (v · ω)", linewidth=1.2)
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [m/s²]")
    plt.title("Observed vs Modeled Lateral Acceleration (Circle Run)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
