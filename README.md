## Dataset

This project uses a synthetic thermal dataset with three classes:

- Normal: moderate hotspot amplitudes and reasonable spatial spread.
- Overheating: higher amplitudes and/or wider hotspot spread than normal.
- Fault: abnormal localized patterns (tight intense hotspot, secondary hotspot, or asymmetric hotspot).

Each 256x256 grayscale image includes an ambient background gradient, sensor-like noise, 1–3 component hotspots modeled as 2D Gaussians, and a slight blur to mimic IR camera texture. Metadata for each image is logged in data/labels.csv.

## How to run

Generate the dataset:

python -m src.generate_dataset

Create a montage for quick visual inspection:

python -m src.make_montage
