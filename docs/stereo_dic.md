# Stereo DIC

## 双目 3D-DIC 流程

当前实现采用左参考图 `L0` 作为主坐标系，不在右参考图的亚像素点上再做时序相关。对每个左参考点 `pL0=(x,y)`，需要先由 subset 或 mesh 得到三组 2D 场：

1. `reference_disparity.csv`: 初始视差 `L0 -> R0`
2. `left_temporal.csv`: 左图时序位移 `L0 -> Llast`
3. `deformed_disparity.csv`: 最终变形时刻双目场 `L0 -> Rlast`

随后按下面关系重建：

```text
pR0 = pL0 + d(L0 -> R0)
pL1 = pL0 + d(L0 -> L1)
pR1 = pL0 + d(L0 -> Rlast)

X0 = triangulate(pL0, pR0)
X1 = triangulate(pL1, pR1)
U3D = X1 - X0
```

底层 C++ `StereoDIC` 仍接收左右相机各自的 `PointObservation`。Python 装配层会把 `L0 -> R1` 自动转换成等价的右视角位移 `R0 -> R1 = pR1 - pR0`，因此无需改变 subset/mesh/core/calibration 模块。

## 输入目录

测试算例默认目录：

```text
case/stereo_DIC/plate_center_load
```

2D 位移/视差场结果约定放在：

```text
case/stereo_DIC/plate_center_load/result/disp/subset/
case/stereo_DIC/plate_center_load/result/disp/mesh/T3/
case/stereo_DIC/plate_center_load/result/disp/mesh/Q4/
case/stereo_DIC/plate_center_load/result/disp/mesh/Q8/
```

每个目录包含三张场表，列格式为：

```text
id,x,y,u,v,correlation,valid
```

mesh 场也可使用 `node_id` 替代 `id`。其中 `x,y` 始终是左参考图 `L0` 上的主坐标，`u,v` 是相对该坐标的 2D 增量。

subset 验证 fixture 默认 `--spacing 3`，不再对 subset 场做后续稠密化。

mesh 目录同时写出节点场和由单元形函数插值得到的整场稠密位移/视差结果。三张稠密场同样都在左参考图 `L0` 坐标域上：

```text
reference_disparity_dense.csv
left_temporal_dense.csv
deformed_disparity_dense.csv
reference_disparity_dense_mag.png
left_temporal_dense_mag.png
deformed_disparity_dense_mag.png
```

mesh 的稠密 3D 重建结果由三张 `*_dense.csv` 三角化得到，输出在：

```text
result/reconstruct/mesh/T3/dense/
result/reconstruct/mesh/Q4/dense/
result/reconstruct/mesh/Q8/dense/
```

## 输出文件

3D 重建输出放在：

```text
case/stereo_DIC/plate_center_load/result/reconstruct/subset/
case/stereo_DIC/plate_center_load/result/reconstruct/mesh/T3/
case/stereo_DIC/plate_center_load/result/reconstruct/mesh/Q4/
case/stereo_DIC/plate_center_load/result/reconstruct/mesh/Q8/
```

每个重建目录包含：

```text
stereo_3d_points.csv
stereo_3d_summary.json
shape_ref_z.png
shape_def_z.png
```

mesh 目录额外输出：

```text
stereo_3d_strain_faces.csv
```

其中 `stereo_3d_points.csv` 包含 `X0,Y0,Z0`、`X1,Y1,Z1`、`Ux,Uy,Uz,Umag` 以及参考/变形重投影误差。

## 快速验证

下面命令只生成合成的初始时刻和最后变形时刻 2D 场，并验证 3D 链路，不运行 subset/mesh：

```powershell
python examples/stereo_3d.py --solver subset --skip-calibration
python examples/stereo_3d.py --solver mesh --skip-calibration
```
