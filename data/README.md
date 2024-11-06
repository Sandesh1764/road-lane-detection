# Dataset layout

Expected structure (not committed — too large):

```text
dataset/
├── train/
│   ├── imgs/<sequence>/00000.jpg
│   └── label/<sequence>/00000.png
└── test/
    └── imgs/...
```

- Images: JPG/PNG ego-camera frames extracted from ROS bags (Oct–Nov 2024).
- Labels: CULane-style integer masks (`0` = background, `>0` = lane ids).
- Training uses **binary** lane vs background (`mask > 0`).

Point `--data-root dataset` when calling `train.py`.
