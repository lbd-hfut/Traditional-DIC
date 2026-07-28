# Stereo Calibration Synthetic Case

This case contains synthetic paired chessboard images for stereo Zhang
calibration tests. The directory name follows the project request spelling:
`stereo_calibartion`.

Contents:

- `generate_stereo_chessboard.py`: deterministic stereo pair generator.
- `left/*.bmp`: left camera chessboard views.
- `right/*.bmp`: right camera chessboard views.
- `meta.json`: board metadata, per-pair image paths, and projected board quadrilaterals.

Board:

- Type: chessboard
- Inner corners: 9 x 6
- Square size: 25 mm
- Image size: 1280 x 960 px

Regenerate:

```powershell
C:\Users\lbd\miniconda3\python.exe case\stereo_calibartion\generate_stereo_chessboard.py
```
