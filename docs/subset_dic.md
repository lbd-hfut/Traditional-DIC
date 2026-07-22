# Subset Dic

Current Status: Project Skeleton / Interfaces Only.

TODO: Document algorithms, inputs, outputs, validation datasets, and development checkpoints.

## ROI / Mask Correlation

Subset sampling must pass a correlation weight for each sampled pixel. Samples
outside the user-supplied ROI or mask use weight `0`, so they are ignored by
SSD/ZNSSD/ZNCC. This keeps boundary subsets consistent with Ncorr-style valid
sample handling and avoids treating non-ROI pixels as real texture.
