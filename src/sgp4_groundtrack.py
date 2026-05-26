#!/usr/bin/env python3
"""
data/ 内の OMM (Space-Track CSV) を読み込み、指定日時から指定期間だけ
SGP4 で軌道伝播し、地上トラックを描画するスクリプト。

Usage:
    python src/sgp4_groundtrack.py \
        --tle data/TLE-3026-05-25.csv \
        --start 2026-05-25T22:05:55 \
        --duration-hours 24 \
        --step-sec 60 \
        --out ground_track.png \
        --csv-out track.csv
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sgp4 import omm
from sgp4.api import Satrec, jday


# WGS84 ellipsoid constants
WGS84_A = 6378.137                       # equatorial radius [km]
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)     # first eccentricity squared


def parse_datetime(s: str) -> datetime:
    """ISO-8601 を datetime に。タイムゾーン未指定なら UTC とみなす。"""
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_satellite(path: Path) -> tuple[Satrec, dict]:
    """OMM CSV の先頭レコードを読み、初期化済み Satrec とフィールド辞書を返す。"""
    with path.open() as f:
        records = list(omm.parse_csv(f))
    if not records:
        raise ValueError(f"no OMM records found in {path}")
    fields = records[0]
    sat = Satrec()
    omm.initialize(sat, fields)
    return sat, fields


def gmst_rad(jd_ut1: float) -> float:
    """
    Greenwich Mean Sidereal Time (IAU 1982) を [rad] で返す。
    地上トラックの可視化用途では極運動・章動補正は無視。
    """
    T = (jd_ut1 - 2451545.0) / 36525.0
    gmst_s = (67310.54841
              + (876600.0 * 3600.0 + 8640184.812866) * T
              + 0.093104 * T * T
              - 6.2e-6 * T * T * T)
    gmst = (gmst_s % 86400.0) * (2.0 * np.pi / 86400.0)
    return gmst % (2.0 * np.pi)


def teme_to_ecef(r_teme: np.ndarray, jd_ut1: float) -> np.ndarray:
    """TEME → ECEF (PEF近似)。GMST まわりに Z 軸回転。"""
    theta = gmst_rad(jd_ut1)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[ c,  s, 0.0],
                  [-s,  c, 0.0],
                  [0.0, 0.0, 1.0]])
    return R @ r_teme


def ecef_to_geodetic(r_ecef: np.ndarray) -> tuple[float, float, float]:
    """ECEF [km] → (lat[deg], lon[deg], alt[km])。Bowring 反復で十分。"""
    x, y, z = r_ecef
    lon = np.degrees(np.arctan2(y, x))
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(5):
        sin_lat = np.sin(lat)
        N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        alt = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * N / (N + alt)))
    sin_lat = np.sin(lat)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    alt = p / np.cos(lat) - N
    return float(np.degrees(lat)), float(lon), float(alt)


def propagate(sat: Satrec,
              start: datetime,
              duration: timedelta,
              step: timedelta):
    """[start, start+duration] を step 刻みで SGP4 伝播し、配列群を返す。"""
    n = int(duration.total_seconds() // step.total_seconds()) + 1
    times = np.empty(n, dtype=object)
    lats = np.empty(n)
    lons = np.empty(n)
    alts = np.empty(n)
    r_teme_arr = np.empty((n, 3))
    v_teme_arr = np.empty((n, 3))

    for i in range(n):
        t = start + i * step
        jd, fr = jday(t.year, t.month, t.day,
                      t.hour, t.minute, t.second + t.microsecond * 1e-6)
        err, r_teme, v_teme = sat.sgp4(jd, fr)
        if err != 0:
            raise RuntimeError(
                f"SGP4 error at {t.isoformat()} (code {err}); "
                "satellite may have decayed or elements diverged.")
        r_ecef = teme_to_ecef(np.array(r_teme), jd + fr)
        lat, lon, alt = ecef_to_geodetic(r_ecef)
        times[i] = t
        lats[i] = lat
        lons[i] = lon
        alts[i] = alt
        r_teme_arr[i] = r_teme
        v_teme_arr[i] = v_teme
    return times, lats, lons, alts, r_teme_arr, v_teme_arr


def _split_at_antimeridian(lons, lats):
    """経度が ±180 をまたぐ箇所で線分を分割（地図上で水平線が走るのを防ぐ）。"""
    segs = []
    cur_x, cur_y = [lons[0]], [lats[0]]
    for i in range(1, len(lons)):
        if abs(lons[i] - lons[i - 1]) > 180.0:
            segs.append((cur_x, cur_y))
            cur_x, cur_y = [], []
        cur_x.append(lons[i])
        cur_y.append(lats[i])
    segs.append((cur_x, cur_y))
    return segs


def plot_ground_track(lats, lons, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for xs, ys in _split_at_antimeridian(lons, lats):
        ax.plot(xs, ys, lw=0.8, color='tab:blue')
    ax.scatter(lons[0],  lats[0],  c='green', s=40, zorder=3, label='start')
    ax.scatter(lons[-1], lats[-1], c='red',   s=40, zorder=3, label='end')

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 30))
    ax.set_yticks(range(-90, 91, 30))
    ax.grid(alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    ax.set_title(title)
    ax.legend(loc='lower left')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_csv(path: Path, times, lats, lons, alts,
              r_teme, v_teme) -> None:
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time_utc',
                    'lat_deg', 'lon_deg', 'alt_km',
                    'x_teme_km', 'y_teme_km', 'z_teme_km',
                    'vx_teme_kms', 'vy_teme_kms', 'vz_teme_kms'])
        for t, la, lo, al, r, v in zip(times, lats, lons, alts, r_teme, v_teme):
            w.writerow([t.isoformat(),
                        f"{la:.6f}", f"{lo:.6f}", f"{al:.3f}",
                        f"{r[0]:.3f}", f"{r[1]:.3f}", f"{r[2]:.3f}",
                        f"{v[0]:.6f}", f"{v[1]:.6f}", f"{v[2]:.6f}"])


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    default_tle = project_root / 'data' / 'TLE-3026-05-25.csv'

    ap = argparse.ArgumentParser(
        description="SGP4 で OMM/TLE から軌道伝播 → 地上トラック描画")
    ap.add_argument('--tle', type=Path, default=default_tle,
                    help=f'OMM/TLE CSV ファイル (default: {default_tle})')
    ap.add_argument('--start', type=parse_datetime, required=True,
                    help='開始日時 UTC (ISO-8601, 例: 2026-05-25T22:05:55)')
    ap.add_argument('--duration-hours', type=float, default=24.0,
                    help='伝播時間 [h] (default: 24)')
    ap.add_argument('--step-sec', type=float, default=60.0,
                    help='サンプル刻み [s] (default: 60)')
    ap.add_argument('--out', type=Path, default=project_root / 'ground_track.png',
                    help='地上トラック PNG 出力先')
    ap.add_argument('--csv-out', type=Path, default=None,
                    help='伝播結果 CSV 出力先 (省略可)')
    args = ap.parse_args()

    sat, fields = load_satellite(args.tle)

    # エポックから極端に離れた propagation は精度が落ちるので注意喚起
    epoch_jd = sat.jdsatepoch + sat.jdsatepochF
    start_jd, start_fr = jday(args.start.year, args.start.month, args.start.day,
                              args.start.hour, args.start.minute,
                              args.start.second + args.start.microsecond * 1e-6)
    dt_from_epoch_days = (start_jd + start_fr) - epoch_jd
    if abs(dt_from_epoch_days) > 14:
        print(f"[warn] start is {dt_from_epoch_days:+.1f} days from TLE epoch; "
              "SGP4 accuracy degrades beyond ~2 weeks.")

    times, lats, lons, alts, r_teme, v_teme = propagate(
        sat, args.start,
        timedelta(hours=args.duration_hours),
        timedelta(seconds=args.step_sec))

    title = (f"{fields.get('OBJECT_NAME', '?')} "
             f"(NORAD {fields.get('NORAD_CAT_ID', '?')}) ground track\n"
             f"{args.start.isoformat()} + {args.duration_hours} h "
             f"@ {args.step_sec} s step")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot_ground_track(lats, lons, title, args.out)
    print(f"saved ground track  -> {args.out}")
    print(f"  samples: {len(times)}  alt range: "
          f"{alts.min():.1f} - {alts.max():.1f} km")

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.csv_out, times, lats, lons, alts, r_teme, v_teme)
        print(f"saved trajectory    -> {args.csv_out}")


if __name__ == '__main__':
    main()
