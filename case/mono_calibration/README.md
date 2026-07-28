# Mono Calibration Synthetic Case

This case contains synthetic chessboard images for mono Zhang calibration tests.

Contents:

- `generate_mono_chessboard.py`: deterministic image generator.
- `images/*.bmp`: generated chessboard views with different in-plane and perspective poses.
- `meta.json`: board metadata and projected board quadrilaterals.

Board:

- Type: chessboard
- Inner corners: 9 x 6
- Square size: 25 mm
- Image size: 1280 x 960 px

Regenerate:

```powershell
C:\Users\lbd\miniconda3\python.exe case\mono_calibration\generate_mono_chessboard.py
```
