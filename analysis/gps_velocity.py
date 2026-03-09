#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rosbags.highlevel import AnyReader

# ================== USER CONFIG ==================
BAG_PATH = Path("/home/fabri/EECE5554/lab5/data/driving1")
GPS_TOPIC = "/gps"   # custom_msg/msg/Customgps
# =================================================


def main():
    timestamps = []
    eastings = []
    northings = []
    hdops = []

    # --- Read bag and extract GPS data ---
    with AnyReader([BAG_PATH]) as reader:
        conns = [c for c in reader.connections if c.topic == GPS_TOPIC]
        if not conns:
            raise RuntimeError(
                f"Topic {GPS_TOPIC} not found in bag. "
                f"Available: {[c.topic for c in reader.connections]}"
            )

        for conn, t_ns, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)

            # Customgps msg layout:
            # float64 utm_easting
            # float64 utm_northing
            # float64 hdop
            e = float(msg.utm_easting)
            n = float(msg.utm_northing)
            h = float(msg.hdop)

            eastings.append(e)
            northings.append(n)
            hdops.append(h)
            timestamps.append(t_ns * 1e-9)  # ns -> seconds

    # --- Convert to numpy arrays & sort by time (safety) ---
    t = np.asarray(timestamps, float)
    e = np.asarray(eastings, float)
    n = np.asarray(northings, float)
    hdops = np.asarray(hdops, float)

    idx = np.argsort(t)
    t = t[idx]
    e = e[idx]
    n = n[idx]
    hdops = hdops[idx]

    # Start time at zero
    t0 = t[0]
    t = t - t0

    print(f"[INFO] Loaded {len(t)} GPS samples, duration {t[-1]:.1f} s")


    # --- Compute velocity from UTM positions ---
    # Differences between consecutive samples:
    dt = np.diff(t)            # Δt [s]
    de = np.diff(e)            # ΔE [m]
    dn = np.diff(n)            # ΔN [m]

    # Avoid division by zero:
    dt[dt == 0] = np.nan

    # Distance in the horizontal plane:
    dist = np.sqrt(de**2 + dn**2)   # [m]

    # Speed = distance / time
    speed_ms = dist / dt            # [m/s]
    speed_kmh = speed_ms * 3.6      # [km/h]

    # Time axis for speed: mid-point between samples
    t_mid = 0.5 * (t[1:] + t[:-1])

    # --- Print simple stats ---
    mean_speed_ms = np.nanmean(speed_ms)
    max_speed_ms = np.nanmax(speed_ms)

    print(f"[INFO] Mean speed: {mean_speed_ms:.2f} m/s ({mean_speed_ms*3.6:.2f} km/h)")
    print(f"[INFO] Max  speed: {max_speed_ms:.2f} m/s ({max_speed_ms*3.6:.2f} km/h)")

    # --- Plot speed vs time ---
    plt.figure(figsize=(10, 5))
    plt.plot(t_mid, speed_kmh, label="GPS speed", linewidth=1.2)
    plt.xlabel("Time [s]")
    plt.ylabel("Speed [km/h]")
    plt.title("Vehicle Speed from GPS UTM (E,N)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
