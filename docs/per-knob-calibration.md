# Per-Knob Template Calibration

## Problem

Template matching works excellently for angle detection (0.82-0.92 match scores)
but only when templates are matched against the same knob position they were
captured from. Cross-position matching fails because each knob slot has unique
lighting, shadow, and perspective characteristics that vary across the panel.

The ceiling-mounted camera creates position-dependent foreshortening and specular
reflection patterns that we haven't been able to correct with:
- 2D vertical stretch (degraded detection)
- 3D solvePnP protrusion correction (visible artifacts, no detection improvement)

## What Works

- HLS lightness → Otsu threshold → morphological open → binary mask
- Connected components for knob detection with spatial prior slot assignment
- Template matching (TM_CCOEFF_NORMED) on the binary mask at 135x135 crops
- Per-knob calibrated X positions to center crops correctly

## Current State

6 shared templates at 0°, 30°, 60°, 90°, 120°, 150° CCW captured from knobs 1-6.
These match their source knobs at HIGH confidence but fail on other positions.

## What's Needed

### Per-knob template sets

Each of the 10 knob slots needs its own set of angle templates. The calibration
procedure would be:

1. For each knob (1-10), one at a time:
   a. Set the knob to the "off" position (0°)
   b. Capture and save as `templates/knobs/slot_N/0deg.jpg`
   c. Rotate 30° CCW, capture → `30deg.jpg`
   d. Continue in 30° steps through 330°
   e. Total: 12 templates per knob × 10 knobs = 120 templates

2. The angle stage loads templates per-slot and only matches knob N against
   slot N's template set.

### Alternative: fewer angles with interpolation

Instead of 12 templates per knob, capture 6 (every 60°) and interpolate between
the two best matches to estimate finer angles. This halves calibration effort
(60 templates) at the cost of ~15° angular resolution.

### Semi-automated calibration

A calibration mode could be added to the binary:
```
CALIBRATION_MODE=1 CALIBRATION_SLOT=3 oven-vision
```
This would capture one frame per keypress/trigger, auto-saving the binary mask
crop to the correct template directory with incrementing angles.

## Knob Crop Positions

Final calibrated X centers (Y=125, crop size 135x135):

| Slot | X center | Notes |
|------|----------|-------|
| 1    | 50       | Clipped at left edge (118px wide) |
| 2    | 172      | |
| 3    | 294      | |
| 4    | 422      | +6px from evenly spaced |
| 5    | 544      | +6px |
| 6    | 667      | +6px |
| 7    | 789      | +6px |
| 8    | 924      | +19px |
| 9    | 1040     | +13px |
| 10   | 1169     | +19px, clipped at right edge (98px wide) |

Knobs 1 and 10 are partially clipped. Templates for these slots need padding
or the crop positions adjusted to avoid edge clipping.

## Pipeline Summary

```
reproject(solvePnP, fx=2200) → detect(HLS L, Otsu, morph open, connected components)
  → angle(per-slot template matching) → sanity → annotate
```
