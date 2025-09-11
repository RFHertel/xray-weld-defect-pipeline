# Model Evaluation Report
============================================================

## TRAIN SET RESULTS
----------------------------------------
Total instances: 25236
Class imbalance ratio: 1.0:1

### Standard YOLOv8 Metrics:
  - mAP50: 0.578
  - mAP50-95: 0.314
  - Precision: 0.673
  - Recall: 0.488

### Enhanced Metrics:
**Macro-averaged (equal weight per class):**
  - mAP50: 0.578
  - mAP50-95: 0.314

**Micro-averaged (weighted by frequency):**
  - mAP50: 0.578
  - mAP50-95: 0.314

### Per-Class Performance:
| Class | AP50 | Precision | Recall | Support | Note |
|-------|------|-----------|--------|---------|------|
| porosity | 0.697 | 0.755 | 0.596 | 4206 |  |
| inclusion | 0.494 | 0.591 | 0.428 | 4206 |  |
| crack | 0.624 | 0.731 | 0.526 | 4206 |  |
| undercut | 0.797 | 0.677 | 0.884 | 4206 | [OK] Good performance |
| lack_of_fusion | 0.473 | 0.628 | 0.356 | 4206 |  |
| lack_of_penetration | 0.381 | 0.655 | 0.137 | 4206 |  |

## VAL SET RESULTS
----------------------------------------
Total instances: 5436
Class imbalance ratio: 109.3:1

### Standard YOLOv8 Metrics:
  - mAP50: 0.248
  - mAP50-95: 0.122
  - Precision: 0.300
  - Recall: 0.336

### Enhanced Metrics:
**Macro-averaged (equal weight per class):**
  - mAP50: 0.248
  - mAP50-95: 0.122

**Micro-averaged (weighted by frequency):**
  - mAP50: 0.579
  - mAP50-95: 0.315

### Per-Class Performance:
| Class | AP50 | Precision | Recall | Support | Note |
|-------|------|-----------|--------|---------|------|
| porosity | 0.708 | 0.802 | 0.581 | 4044 | [OK] Good performance |
| inclusion | 0.147 | 0.177 | 0.289 | 462 | [X] Poor performance |
| crack | 0.364 | 0.436 | 0.445 | 461 |  |
| undercut | 0.041 | 0.043 | 0.324 | 37 | [!] Low support |
| lack_of_fusion | 0.126 | 0.160 | 0.297 | 190 | [X] Poor performance |
| lack_of_penetration | 0.105 | 0.181 | 0.078 | 242 | [X] Poor performance |

## TEST SET RESULTS
----------------------------------------
Total instances: 5326
Class imbalance ratio: 117.9:1

### Standard YOLOv8 Metrics:
  - mAP50: 0.267
  - mAP50-95: 0.134
  - Precision: 0.311
  - Recall: 0.388

### Enhanced Metrics:
**Macro-averaged (equal weight per class):**
  - mAP50: 0.267
  - mAP50-95: 0.134

**Micro-averaged (weighted by frequency):**
  - mAP50: 0.568
  - mAP50-95: 0.309

### Per-Class Performance:
| Class | AP50 | Precision | Recall | Support | Note |
|-------|------|-----------|--------|---------|------|
| porosity | 0.708 | 0.803 | 0.589 | 3772 | [OK] Good performance |
| inclusion | 0.183 | 0.212 | 0.320 | 505 | [X] Poor performance |
| crack | 0.403 | 0.448 | 0.485 | 485 |  |
| undercut | 0.072 | 0.066 | 0.529 | 32 | [!] Low support |
| lack_of_fusion | 0.153 | 0.192 | 0.344 | 283 | [X] Poor performance |
| lack_of_penetration | 0.084 | 0.143 | 0.060 | 249 | [X] Poor performance |
