#!/usr/bin/env python3
"""
Feasibility spike: validate dial detection on the kitchen stove camera.

Fetches a 2560x1440 frame from go2rtc HTTP API, applies CLAHE + median blur,
then runs a radial edge scan on each dial ROI to detect the indicator angle.

Usage (inside Docker container):
  python3 validate_dials.py [--save-debug]

The --save-debug flag saves annotated images to /output/ for visual inspection.
"""

import sys
import math
import argparse
import urllib.request

import cv2
import numpy as np


# go2rtc HTTP frame API (use host network from container)
FRAME_URL = "http://192.168.2.52:1984/api/frame.jpeg?src=kitchen"

# Approximate dial ROI centers and radii at 2560x1440.
# These are rough estimates from visual inspection of the cropped frames.
# Format: (label, center_x, center_y, radius)
# Dials are in a row from left to right: clock, D1-D10
# We skip the clock (leftmost).
DIAL_ROIS = [
    ("burner_1",       1430, 1040, 28),
    ("burner_2",       1490, 1040, 28),
    ("burner_3",       1555, 1040, 28),
    ("burner_4",       1620, 1040, 28),
    ("burner_5",       1685, 1040, 28),
    ("burner_6",       1750, 1040, 28),
    ("oven_left_temp", 1830, 1040, 28),
    ("oven_left_mode", 1895, 1040, 28),
    ("oven_right_temp",1965, 1040, 28),
    ("oven_right_mode",2030, 1040, 28),
]

# LED ROI positions (small rectangles near the oven dials)
LED_ROIS = [
    ("oven_left_led",  1860, 1010, 20, 15),
    ("oven_right_led", 1995, 1010, 20, 15),
]


def fetch_frame():
    """Fetch a JPEG frame from go2rtc HTTP API."""
    resp = urllib.request.urlopen(FRAME_URL, timeout=10)
    data = np.frombuffer(resp.read(), dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Failed to decode JPEG frame")
    print(f"Frame: {frame.shape[1]}x{frame.shape[0]}")
    return frame


def preprocess(frame):
    """Convert to grayscale, apply CLAHE and median blur."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.medianBlur(enhanced, 5)
    return blurred


def radial_edge_scan(gray_roi, cx, cy, radius, num_angles=360):
    """
    Scan radially outward from center at the rim of the dial.
    Returns the angle (degrees, 0=right, clockwise) with strongest edge response.
    Also returns the max edge strength for confidence assessment.
    """
    # Run Canny on the ROI
    edges = cv2.Canny(gray_roi, 50, 150)

    angles = np.linspace(0, 2 * math.pi, num_angles, endpoint=False)
    edge_strengths = []

    for angle in angles:
        strengths = []
        # Sample at 70-100% of radius (rim area where the notch indicator lives)
        for frac in np.linspace(0.5, 1.0, 15):
            px = int(cx + frac * radius * math.cos(angle))
            py = int(cy + frac * radius * math.sin(angle))
            if 0 <= px < edges.shape[1] and 0 <= py < edges.shape[0]:
                strengths.append(float(edges[py, px]))
        edge_strengths.append(np.mean(strengths) if strengths else 0.0)

    edge_strengths = np.array(edge_strengths)

    # Apply a small Gaussian smoothing to the angular profile to reduce noise
    kernel_size = 7
    kernel = cv2.getGaussianKernel(kernel_size, 2.0).flatten()
    # Circular convolution
    padded = np.concatenate([edge_strengths[-kernel_size//2:], edge_strengths, edge_strengths[:kernel_size//2]])
    smoothed = np.convolve(padded, kernel, mode='valid')[:num_angles]

    best_idx = np.argmax(smoothed)
    best_angle_deg = best_idx * 360.0 / num_angles
    max_strength = smoothed[best_idx]

    return best_angle_deg, max_strength


def detect_led_color(frame_bgr, x, y, w, h):
    """Classify LED color in the given ROI. Returns 'off', 'orange', or 'green'."""
    roi = frame_bgr[y:y+h, x:x+w]
    if roi.size == 0:
        return "off"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mean_h = np.mean(hsv[:, :, 0])
    mean_s = np.mean(hsv[:, :, 1])
    mean_v = np.mean(hsv[:, :, 2])

    if mean_v < 50 or mean_s < 50:
        return "off"
    elif 55 <= mean_h <= 75:
        return "green"
    elif 5 <= mean_h <= 20:
        return "orange"
    else:
        return f"unknown(H={mean_h:.0f},S={mean_s:.0f},V={mean_v:.0f})"


def main():
    parser = argparse.ArgumentParser(description="Validate dial detection on stove camera")
    parser.add_argument("--save-debug", action="store_true", help="Save annotated debug images to /output/")
    parser.add_argument("--frames", type=int, default=3, help="Number of frames to analyze")
    args = parser.parse_args()

    print("=== Oven Vision Feasibility Spike ===\n")

    for frame_idx in range(args.frames):
        print(f"--- Frame {frame_idx + 1}/{args.frames} ---")
        frame = fetch_frame()
        preprocessed = preprocess(frame)

        debug_frame = frame.copy() if args.save_debug else None

        print(f"\n{'Dial':<20} {'Angle':>6} {'Strength':>10} {'Status'}")
        print("-" * 55)

        for label, cx, cy, r in DIAL_ROIS:
            # Extract ROI (with some padding)
            pad = 10
            x1 = max(0, cx - r - pad)
            y1 = max(0, cy - r - pad)
            x2 = min(preprocessed.shape[1], cx + r + pad)
            y2 = min(preprocessed.shape[0], cy + r + pad)

            roi = preprocessed[y1:y2, x1:x2]
            local_cx = cx - x1
            local_cy = cy - y1

            if roi.size == 0:
                print(f"{label:<20} {'N/A':>6} {'N/A':>10} ROI out of bounds")
                continue

            angle, strength = radial_edge_scan(roi, local_cx, local_cy, r)

            # Confidence: strength > 30 is typically a good indicator
            confidence = "HIGH" if strength > 30 else "LOW" if strength > 15 else "NONE"

            print(f"{label:<20} {angle:>6.1f} {strength:>10.1f} {confidence}")

            if debug_frame is not None:
                # Draw ROI circle
                cv2.circle(debug_frame, (cx, cy), r, (0, 255, 0), 1)
                # Draw detected angle line
                end_x = int(cx + r * math.cos(math.radians(angle)))
                end_y = int(cy + r * math.sin(math.radians(angle)))
                cv2.line(debug_frame, (cx, cy), (end_x, end_y), (0, 0, 255), 2)
                # Label
                cv2.putText(debug_frame, f"{label}: {angle:.0f}", (cx - r, cy - r - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # LED detection
        print(f"\n{'LED':<20} {'Color':>10} {'HSV Mean'}")
        print("-" * 45)
        for label, x, y, w, h in LED_ROIS:
            color = detect_led_color(frame, x, y, w, h)
            roi = frame[y:y+h, x:x+w]
            if roi.size > 0:
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mean_hsv = np.mean(hsv, axis=(0, 1))
                print(f"{label:<20} {color:>10} H={mean_hsv[0]:.0f} S={mean_hsv[1]:.0f} V={mean_hsv[2]:.0f}")
            else:
                print(f"{label:<20} {'N/A':>10} ROI out of bounds")

            if debug_frame is not None:
                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (255, 0, 255), 1)

        if debug_frame is not None:
            path = f"/output/spike_frame_{frame_idx}.jpg"
            cv2.imwrite(path, debug_frame)
            print(f"\nDebug image saved to {path}")

        print()

    print("=== Spike Complete ===")
    print("\nNext steps:")
    print("  - Review debug images for ROI alignment")
    print("  - Adjust ROI coordinates if dials are misaligned")
    print("  - Check angle consistency across frames")
    print("  - Verify confidence levels (HIGH = good, LOW/NONE = needs tuning)")


if __name__ == "__main__":
    main()
