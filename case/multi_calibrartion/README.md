# Multi Calibration Synthetic Case

This case is copied from `C:\02Project\Study\Multi-DIC\case\CylinderDIC` as the
initial multiview calibration dataset. The directory name follows the project
request spelling: `multi_calibrartion`.

Contents:

- `simulate_cylinder.py`: Multi-DIC cylinder/speckle simulation script.
- `simulate_chessboard.py`: Multi-DIC multiview chessboard simulation script.
- `speckle_2048x512.bmp`: source speckle texture.
- `images/cam_*/001.bmp`: reference multiview speckle images.
- `images/cam_*/002.bmp`: deformed multiview speckle images.
- `calibrate_images/cam_*/001.bmp`: multiview chessboard calibration images.
- `calibrate_images/chessboard_meta.json`: chessboard metadata.
- `visualize_colmap_like_results.py`: creates COLMAP-style sparse scene and
  camera-observation visualizations from simulator camera metadata.

Current copied sample:

- Cameras: 12
- Speckle images: 24
- Chessboard calibration images: 12

Regenerate from the copied Multi-DIC scripts after checking their CLI options:

```powershell
C:\Users\lbd\miniconda3\python.exe case\multi_calibrartion\simulate_cylinder.py --help
C:\Users\lbd\miniconda3\python.exe case\multi_calibrartion\simulate_chessboard.py --help
C:\Users\lbd\miniconda3\python.exe case\multi_calibrartion\visualize_colmap_like_results.py
```
