## Dataset

This project uses a synthetic thermal dataset with three classes:

- Normal: moderate hotspot amplitudes and reasonable spatial spread.
- Overheating: higher amplitudes and/or wider hotspot spread than normal.
- Fault: abnormal localized patterns (tight intense hotspot, secondary hotspot, or asymmetric hotspot).

Each 256x256 grayscale image includes an ambient background gradient, sensor-like noise, 1–3 component hotspots modeled as 2D Gaussians, and a slight blur to mimic IR camera texture. Metadata for each image is logged in data/labels.csv.

## How to run

Generate the dataset:

python3 -m src.generate_dataset

Create a montage for quick visual inspection:

python3 -m src.make_montage

## Baseline Model

Train the baseline transfer-learning model:

python3 -m src.train --data_dir data --epochs_head 10 --epochs_ft 5 --batch_size 32 --lr 1e-3 --lr_ft 1e-4 --seed 42

Evaluate on the test split:

python3 -m src.evaluate --data_dir data --model_path models/baseline_resnet18.pt --batch_size 64
