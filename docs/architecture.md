# Architecture

Image
  ↓
Interpolation / Correlation / Initialization
  ↓
┌──────────────┬──────────────┐
│ Subset-DIC   │ Mesh-DIC     │
└──────────────┴──────────────┘
          ↓
   2D Correspondence
          ↓
Calibration + Geometry
          ↓
Stereo / Multi-view
          ↓
3D Shape / 3D Displacement
          ↓
Postprocess

Subset-DIC and Mesh-DIC are responsible for 2D image correspondence.

3D geometry is handled by independent Calibration / Geometry / Reconstruction modules.
