# Tabletop object dataset v1

This is the refreshed dataset used by `yolo26m_tabletop_v1`.

## Classes

The YOLO class IDs are:

| ID | Class |
|---:|---|
| 0 | book |
| 1 | bottle |
| 2 | earphone |
| 3 | glass |
| 4 | headphone |
| 5 | keyboard |
| 6 | laptop |
| 7 | mobile |
| 8 | mouse |
| 9 | pen |
| 10 | penstand |

The upstream export spelled class 1 as `bottole`; the project configuration
corrects the display name to `bottle`. Labels use numeric IDs, so no label files
need to change.

This dataset intentionally replaces the previous three-class contract. It does
not contain a `monitor` class.

## Splits and audit

| Split | Images | Boxes |
|---|---:|---:|
| train | 1,142 | 1,926 |
| val (`valid/`) | 148 | 234 |
| test | 151 | 259 |

Run the pre-training audit from the repository root:

```powershell
.\.venv\Scripts\python.exe tools\audit_yolo_dataset.py datasets\tabletop_v1\data.yaml
```

The 2026-09-03 audit found no missing image/label pairs, invalid YOLO boxes,
unreadable images, empty classes, or exact image duplicates across splits.

## Source and license

The source is Roboflow Universe dataset `table-03wsy`, version 1, provided by
workspace `celebalworkspace-bqx5k` under CC BY 4.0:

<https://universe.roboflow.com/celebalworkspace-bqx5k/table-03wsy/dataset/1>

The export applies EXIF auto-orientation and stretches images to 640 x 640. It
does not include generated augmentations.
