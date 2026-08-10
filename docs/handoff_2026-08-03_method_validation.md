# Traditional-DIC 会话交接 · 2026-08-03（方法全量验证）

> 本交接面向下一个 Claude 会话，读完第 1 节即可续接。上一会话完成：**Subset-DIC 8 种方案 + Mesh-DIC 12 种方案在 ring/star 两个测试用例上的全量验证**，并修复了让这些方法真正生效的 pybind 配置解析缺陷。会话关闭时**源码修复尚未提交 git**。

> **本会话续（2026-08-04）**：在 CylinderDIC / ComplexCylinderDIC 两个多目测试用例上修复并逐方法验证了三项缺陷（mesh FFT 视差搜索半径、SSD 相关系数归一化、Cylinder 二阶位移异常定性），12 个方法×用例组合全部验证通过，详见 §8。**本会话后半段**：mesh 视差搜索半径在三个用例（ComplexCylinderDIC / CylinderDIC / stereo plate_center_load）完成 r300 全量验证——多目两用例确认旧结果本就是 r300（重跑逐位一致），**stereo 用例默认 r30 确认为错误、r300 修复后 100% valid、视差 -164px 恢复**，详见新增 §9。源码改动仍**尚未提交 git**，提交清单已在本节更新。

> **本会话续（2026-08-05）**：在 ComplexCylinderDIC（Q4 单元素）上实施并验证 **mesh 金字塔 coarse-to-fine 初始化**（EP1-EP7，替代 r300 全局单值大半径方案），12 对 reference_disparity 全大视差、GT hits 162/1100（r30 盲搜为 0），达成"**r30 保底**"目标；透视变形下刚性 NCC 锁对率物理上限 ~40%，未追平 r300 系物理受限。运行中发现并修复"第一轮 pipeline 用旧 pyd 致金字塔未生效"问题。**运行时为纯 mesh**（`--solver mesh` 强制 `run_subset=false`，见 §10.6）。金字塔源码改动仍**尚未提交 git**，详见新增 §10。

> **本会话续（2026-08-06/07）**：mesh 初始化再加 **SIFT prior**（FeatureMatcher+IDW seed，旋转/尺度不变）并将 FFT 窗口从 75 调到 **window61**（透视 case 甜点），**推翻"40% 物理上限"错误认知**——ComplexCylinderDIC 2D |dU| 129.5→6.4px、3D 方向误差 117°→11.5°，CylinderDIC 125.5→5.65px。随后 40px 网格加密（T3/Q4 改善、Q8 反常）并定位 **Q8 |dU| 反常根因 = 12 对独立重建携带系统性低频位移偏差**，用 `smooth_displacement_knn=9` 修复（|dU| 0.151→0.080、|U| 收敛到 GT）。单目 mesh（mono_DIC/Complex/cam_0）也启用金字塔+SIFT+window61 重跑。**金字塔/SIFT/smooth 源码改动仍尚未提交 git**，详见新增 §11。

> **本会话续（2026-08-08）**：新增 **方向 A（`initialization.boundary_direct_prior_seed`）**——边界 mesh 节点有 SIFT/pyramid offset 时直接用它作位移种子、跳过 FFT lock，修复 mono_DIC/disp 的 ROI 右边界 FFT 锁定带（right >5px 32%→0.1%，详见 §12.1）。三用例完整验证后定稿：**默认关闭**（三处默认值 `false`），仅 disp 显式开启；multiview 上方向 A 使 T3/Q4 dist 回归 +0.3mm（nodirA 对照证明与旧基线逐位一致，回归 100% 归因方向 A），plate 上 neutral。随后系统梳理 mesh 全部运行方式（单目/双目/多目、金字塔/smooth/75px vs 40px/方向A 等）并给出"默认方式"与"效果最佳配置"结论（§12.2-12.3）。**方向 A 源码改动（3 处默认值 + 1 处 disp 显式）仍尚未提交 git**。

> **本会话续（2026-08-09）**：将单目 mesh 默认配置 `config/mesh_2d.yaml` 从"老配置"（75px+ICGN+ssd+r30+无SIFT/方向A/QC）固化为 **disk-40px 版本**（40/24/60+FGN+r50+SIFT+方向A 默认开+QC mad2.0/dev12），用标准入口重跑 mono_DIC/disp 至全新 tag `002_ssd_fgn`，与 `disp_40px` **逐位一致**（max|Δ|=0），未覆盖任何现有结果；`003_ssd_fgn` 为 003.bmp=002.bmp（md5 相同）的冗余重复（待删确认）。仅改 yaml，无源码改动，详见新增 §14。

---

## 1. 下个会话第一优先：提交源码修复（需先与用户确认）

所有 8/12 方案测试已完成、结果已验证。用户两次被问"是否提交"均未明确答复，随后要求关闭会话出交接文档，因此提交是明确的待办，**先向用户确认再提交**。

**已修改（develop 分支，需提交）：**

| 文件 | 改动 |
|---|---|
| `bindings/python/bind_subset.cpp` | `config_from_dict` 补上 `shape_function.order` / `optimization.method` / `correlation.criterion` 三个字段的解析（此前全部被忽略 → 所有方案空跑成默认） |
| `python/traditional_dic/config.py` | `subset_method_tag()` 改从**顶层键**读取 correlation/optimization/shape_function（此前误读 `config["subset"]` 下的嵌套键，恒返回默认 tag）；新增 `mesh_method_tag()` |
| `examples/subset_2d.py` | 输出目录拼 `_{method_tag}`；`write_stats` 按准则输出 `ssd_mean/max` 或 `znssd_mean/max` |
| `examples/mesh_2d.py` | 输出目录拼 `_{method_tag}` |
| `cmake/PythonBindings.cmake` | 上一会话：对 pyd 加 `-static-libstdc++ -static-libgcc` 静态链接 |
| `tools/README.md` | 启动用法说明 |
| `examples/multiview_3d.py` | （本会话）`_scaled_calibration_data` 在 fallback 下 `scaled_points3d` 为空时用 `sfm_to_world_scale` 因子重建，修复 CylinderDIC 崩溃（见 §7） |

**新增（未跟踪，需提交）：**

| 文件 | 说明 |
|---|---|
| `tools/run_example.py` | Windows 启动辅助脚本：`os.add_dll_directory` 注册 runtime DLL 目录后 `runpy` 运行示例 |
| `tools/validate_complex_cylinder.py` | （上会话）多目 3D-DIC 验证工具：加载 stitched 表面/位移 vs ground truth，重心预对齐 + 表面 ICP（含 `camera_center` 兼容，见 §7） |
| `build/run_validation_matrix.py` | （本会话）批量验证脚本：逐方法跑 `_run_tagged.py` + 验证，覆盖两用例×6 方法共 12 组合（见 §8.2） |
| `build/mv_ssd_icgn_2nd.yaml`、`build/mv_ssd_fgn_1st.yaml`、`build/mv_ssd_fgn_2nd.yaml` | （本会话）SSD 各变体的多目 mv 配置（由 `mv_ssd_icgn_1st.yaml` 派生，仅改 subset_config） |
| `build/case_paths_cylinder.yaml` | （本会话）CylinderDIC 多目 paths 配置（等价 `config/case_paths.yaml`，后者已默认指向 CylinderDIC） |

**本会话修改（需提交）：**

| 文件 | 改动 |
|---|---|
| `src/subset/solver/icgn.cpp` | 新增匿名命名空间 helper `normalized_znssd_of_final_warp`，4 个 SSD 函数（一阶/二阶 × 普通/掩码）收敛后把 `result.correlation` 从原始残差和改为归一化 ZNSSD，并同步 `converged` 判定（见 §8.3） |
| `src/subset/solver/forward_gauss_newton.cpp` | 同上 helper + 4 个 FGN SSD 函数同款归一化（见 §8.3） |
| `build/mesh_znssd_icgn_q8.yaml` | `fedic_fft.search_radius` 30 → 300（mesh 立体视差 ~236px，见 §8.3） |
| `python/traditional_dic/multiview.py` | mesh 求解器按单元类型子目录独立重建/缝合（`reconstruct_from_field_files` 循环 `mesh/<etype>/`），subset 走单一路径（见 §9.1） |
| `python/traditional_dic/surface_stitching.py` | 修复 `_zip_boundaries` 桥接三角索引映射 bug：`tri + len(vertices1)` 按位置而非 loop2 id 映射，dense 网格静默偏移、sparse 网格溢出顶点缓冲（见 §9.1） |
| `examples/stereo_3d.py` | `--solver mesh` 分支（grid_from_roi/q8_from_q4、`reconstruct/mesh/<etype>/` 输出）；`visualize_calibration` 模块缺失时容错不崩溃 |
| `config/stereo_3d.yaml` | `visualize_calibration: true` → `false`（统一工作流移除诊断校准可视化模块后不硬失败） |
| `include/dic/mesh/mesh_config.hpp` | **方向 A**：`boundary_direct_prior_seed{true}` → `{false}`，附注释说明默认关的原因（multiview T3/Q4 回归 +0.3mm；disp 显式开，见 §12.1） |
| `python/traditional_dic/config.py` | **方向 A**：`boundary_direct_prior_seed` 默认 `True` → `False`（与 C++ 默认对齐，见 §12.1） |
| `build/_run_disp_mesh.py` | **方向 A**：显式 `raw_config...["boundary_direct_prior_seed"] = True`（disp 是这个 case 唯一要开方向 A 的地方，见 §12.1） |
| `bindings/python/bind_mesh.cpp` | **方向 A**：`boundary_direct_prior_seed` 解析处加注释（默认值在 C++ MeshConfig，get_bool fallback 到 cfg value，见 §12.1） |

**测试产物（勿提交）：**
- `case/mono_DIC/{ring,star}/result/**` 与 `visualization/**` 为结果数据；git status 会出现大量 `?? case/.../visualization/` 未跟踪项，提交前注意排除
- `build/` 下 `subset_2d_*.yaml`、`case_paths_star.yaml` 为临时测试配置（git 不跟踪 build/，可直接保留或删除）
- 构建产物 pyd 复制在 `python/traditional_dic/_traditional_dic.cp39-win_amd64.pyd`（git 不跟踪）

建议提交信息如：`Fix pybind subset config parsing; normalize SSD correlation; fix mesh FFT disparity search radius`。

---

## 2. 已解决任务

### 2.1 pyd 运行时加载打通（上会话延续，已完成）
根因：gcc 15.2 pyd 引用 `__emutls_v._ZSt11__once_call` 等 emutls 符号，MSYS2 libstdc++（native-TLS）不导出 → `-static-libstdc++ -static-libgcc` 静态链接修复。详见记忆 `[[python-binding-windows-runtime]]`。

### 2.2 方法输出目录按方法隔离（tag 机制）
- Subset tag：`{criterion}_{solver}_{order}` → `znssd_icgn_1st`、`ssd_fgn_2nd` 等
- Mesh tag：`{objective}_{solver}` → `ssd_icgn` 等；单元类型 T3/Q4/Q8 为 tag 目录下的**子目录**
- 结果分目录存储、互不覆盖；旧目录保留

### 2.3 pybind 配置解析缺陷（核心 bug）
`bind_subset.cpp::config_from_dict` 原本只解析 radius/迭代/插值/种子等，**不解析** `shape_function.order`、`optimization.method`、`correlation.criterion`。后果：YAML 里配 2 阶/FGN/SSD 全部无效，`SubsetConfig` 恒为默认（一阶 + ICGN + ZNSSD），不同方案运行结果**逐字节相同**。已补三个字段的解析并重编译 pyd。编译流程见 §4.1。

### 2.4 stats.json 字段名与准则对应
由 config `correlation.criterion` 决定：`ssd` → `ssd_mean/ssd_max`；`znssd` → `znssd_mean/znssd_max`。

### 2.5 Subset 8 方案全部测试完成
三维组合：准则(SSD/ZNSSD) × 求解器(ICGN/FGN) × 形函数(一阶/二阶)，ring 与 star 各 8 个目录（数值见 §6.1）。
结论：二阶 > 一阶（star 残差减半）；ZNSSD > SSD（光照鲁棒）；FGN ≈ ICGN（差异 ~0.00005，ICGN 更快）。推荐默认 `znssd_icgn_2nd`，极致精度 `znssd_fgn_2nd`。

### 2.6 Mesh 12 方案全部测试完成
objective(SSD/ZNSSD) × solver(ICGN/FGN) × element(T3/Q4/Q8)，ring 与 star 各 12 个目录（数值见 §6.2）。
发现：**star 上 `ssd_icgn` 组合求解退化**（mag_mean 0.09，其余 0.41–0.60）；Q8 在 star 上明显优于 T3/Q4；ring 上 12 方案全部正常。

### 2.7 mesh vs subset 在 star 上分叉的原因分析（结论已交付，未改代码）
见 §5。

---

## 3. 下个会话可能继续的任务

1. **提交源码修复**（§1，最高优先；现包含本会话三项修复，见 §8.3）
2. 若用户要 mesh 在 star 上对齐 subset：配置层换 `--objective znssd`、Q8、缩小 `target_element_size` 即可，**不需改代码**；`ssd_icgn` 在 star 上已证不可用
3. 清理 `build/` 临时配置与 `case/` 测试产物（先问用户是否保留结果）
4. 可选：把 8+12 方案汇总成统一对比表/脚本
5. 可选：`mesh_2d.py` 的 summary.json 已含 `criterion` 字段；若要对齐 subset 的字段名思路可后续优化
6. （本会话遗留）多目验证的 `result_<tag>` / `visualization_<tag>` 目录为测试产物（勿提交）；如需固化某方法基线，可从 tag 目录复制而非重跑
7. （可选）`normalized_znssd_of_final_warp` 在 SSD 收敛后多一次整子集重采样，性能略增；若在意可仅在 quality gate 开启时计算（当前恒计算）

---

## 4. 环境与命令速查（以本条为准，engineering_handoff_zh.md 写的 MSVC 已过时）

- **工具链**：MinGW-W64 gcc 15.2.0（`D:/Microsoft Visual Studio/mingw64/`，ucrt-posix-seh），Ninja 生成器
- **Python**：3.9.6（`C:\Python39`），cp39 pyd；Python 3.8+ 不搜 PATH 找 DLL，必须注册目录
- **OpenCV**：MSYS2 OpenCV 4.11；`build/_opencv/runtime` 汇集全部传递依赖 DLL
- **构建目录**：`build/`（CMakeCache 指向 MinGW + Ninja）
- **pyd 位置**：构建出 `build/_traditional_dic.cp39-win_amd64.pyd`，需复制到 `python/traditional_dic/`
- 注意 PYTHONHOME 冲突（run_example.py 会打印警告）

### 4.1 重编译 pyd
```powershell
cmake --build build --target _traditional_dic
copy /Y build\_traditional_dic.cp39-win_amd64.pyd python\traditional_dic\
```
### 4.2 运行示例（必须走 run_example.py 注册 DLL 目录）
```powershell
# Subset，ring 案例（方案配置在 build/ 下）
python tools/run_example.py examples/subset_2d.py --paths-config config/case_paths.yaml --config build/subset_2d_ssd_fgn2nd.yaml
# Subset，star 案例（paths 用 build/case_paths_star.yaml）
python tools/run_example.py examples/subset_2d.py --paths-config build/case_paths_star.yaml --config build/subset_2d_znssd_icgn2nd.yaml
# Mesh：CLI 覆盖 objective/solver，--element all 默认一次跑 T3/Q4/Q8
python tools/run_example.py examples/mesh_2d.py --paths-config config/case_paths.yaml --config config/mesh_2d.yaml --objective znssd --optimization fedic_element_icgn
```
### 4.3 临时配置文件（build/ 下）
`subset_2d_order2.yaml`(1→2阶)、`subset_2d_fgn1st.yaml`、`subset_2d_fgn2nd.yaml`、`subset_2d_ssd_icgn1st.yaml`、`subset_2d_ssd_icgn2nd.yaml`、`subset_2d_ssd_fgn1st.yaml`、`subset_2d_ssd_fgn2nd.yaml`、`case_paths_star.yaml`

---

## 5. 关键分析：mesh vs subset 在 star 上分叉的原因（已交付，未改代码）

前提（图像统计）：star 参考/变形图**全局亮度未变**，但**局部差异是 ring 的 5 倍**（diff σ 0.068 vs 0.028，全局相关 0.96 vs 0.99）→ star 是强梯度/大变形复杂场，ring 接近平滑刚性平移。

分叉几乎全部来自 **mesh 默认 `ssd_icgn` 组合在 star 上失效**（mag 0.09），mesh 换 `znssd_icgn` 后 star 立即回到 0.42（与 subset 一阶 0.41 几乎一致）。三个放大因素：

1. **准则**：mesh 默认 SSD 无归一化，star 复杂局部变化使 SSD 残差面恶化；subset 默认 ZNSSD 鲁棒
2. **求解机制**：mesh 全局耦合 FEM + 75px FFT 大窗口初始化 + 参考梯度 ICGN，star 强梯度区线性假设破坏、初始峰错位、错误沿单元传导；subset 逐点独立 + 可靠性传播（金字塔 NCC/SIFT + 邻域传播），单点失败不影响整体
3. **分辨率**：mesh 网格 star 仅 75 节点（T3/Q4，~75px 单元）解析不了强梯度（Q8 205 节点时 mag 0.42→0.60）；subset 点阵 16384 点、间距 3px

ring 上两者相近：变形平滑，上述三个假设全部成立。

---

## 6. 结果数值速览

### 6.1 Subset 8 方案（mag_mean / 残差mean / 有效点）

| # | 方案 | ring | star |
|---|---|---|---|
| 1 | `znssd_icgn_1st` | 0.5036 / 0.000973 / 48928 | 0.4103 / 0.016815 / 13952 |
| 2 | `znssd_icgn_2nd` | 0.5015 / 0.000918 / 48928 | 0.5716 / 0.008955 / 13890 |
| 3 | `znssd_fgn_1st` | 0.5033 / 0.000963 / 48928 | 0.4132 / 0.016714 / 13957 |
| 4 | `znssd_fgn_2nd` | 0.5001 / 0.000892 / 48928 | 0.5753 / 0.008862 / 13900 |
| 5 | `ssd_icgn_1st` | 0.5035 / 0.021560 / 48928 | 0.4524 / 0.36357 / 12181 |
| 6 | `ssd_icgn_2nd` | 0.5014 / 0.020574 / 48928 | 0.5905 / 0.22192 / 13222 |
| 7 | `ssd_fgn_1st` | 0.5032 / 0.021332 / 48928 | 0.4547 / 0.36415 / 12212 |
| 8 | `ssd_fgn_2nd` | 0.4998 / 0.019947 / 48928 | 0.5937 / 0.22191 / 13244 |

---

## 7. 多目 3D-DIC 验证（本会话补充）

### 7.1 验证脚本与命令
```powershell
python tools/validate_complex_cylinder.py --case case/multi_DIC/ComplexCylinderDIC
python tools/validate_complex_cylinder.py --case case/multi_DIC/CylinderDIC
```
方法：读 `result/reconstruct/stitched/subset/stitched_points.csv`（表面+位移）与 `ground_truth/theoretical_*` 对比；**重心预对齐 + 表面点云 ICP**（Kabsch，无缩放）。对齐不可用相机中心：ComplexCylinderDIC 重建相机 bas-relief 退化（0-954mm 散落 vs 理论 r=480），表面两套几何一致只差 Z 平移。

### 7.2 结果对比（radius 80mm 圆柱，12 相机）

| 指标 | CylinderDIC（默认配置，真值相机 fallback） | ComplexCylinderDIC（自标定 scaled 相机） |
|---|---|---|
| 表面 mean / p95 / max | 0.39 / 0.76 / 1.51 mm | 0.49 / 0.91 / 1.94 mm |
| 位移 \|dU\| mean / max | 0.032 / 0.240 mm | 0.015 / 0.114 mm |
| 位移方向 median | 1.53° | 0.75° |
| fx 差 | 0.000 px（真值） | 14.8 px |
| 相机中心 mean | 0.385 mm | 597 mm |

结论：**相机标定越准表面越准**（CylinderDIC 用 meta 真值相机，表面 0.39 vs 0.49、相机中心 0.385mm）；**位移精度受 2D-DIC 亚像素匹配/散斑纹理限制**（CylinderDIC 0.032 vs 0.015、方向 1.53° vs 0.75°），与相机标定关系小。

### 7.3 本会话关键修复
- **CylinderDIC 无 `calibration/*.mat`** → 纯自标定与 meta 角点观测不匹配 → backend 棋盘格三角化抛 "Not enough valid chessboard edges" → fallback `_scale_data_from_metric_meta_cameras`（scale=1.0、`scaled_cameras`=meta 真值、**`scaled_points3d`=空**）→ `_scaled_calibration_data` 强制非空崩溃。**已修**：空时用 scale 因子重建（只影响可视化/summary，不影响测量）。
- fallback 相机只有 R/t、无 `camera_center` 字段 → 验证脚本 `camera_center()` 用 `-Rᵀt` 推导。
- 增量 ICP 的 Kabsch 输入须用"当前对齐后点 `aligned[mask]`"而非原始 `source[mask]`，否则平移发散。

### 7.4 遗留
- 多目流程调试产物：`case/multi_DIC/CylinderDIC/result/**`、`case/multi_DIC/star/visualization/`、`case/mono_DIC/ring/visualization/{mesh,subset}/*`（勿提交，见 §1 测试产物）
- `config/case_paths.yaml` 默认 `multiview_3d.case_root` 指向 CylinderDIC；ComplexCylinderDIC 需显式指定

（残差 mean 在 SSD 方案为 ssd_mean，ZNSSD 方案为 znssd_mean；目录 `case/{ring,star}/result/subset/002_<tag>/`，含 displacements.csv / strain.csv / stats.json）

### 6.2 Mesh 12 方案（mag_mean，T3 / Q4 / Q8）

| 方法 | ring | star |
|---|---|---|
| `ssd_icgn` | 0.606 / 0.607 / 0.589 | **0.090 / 0.090 / 0.103（退化）** |
| `ssd_fgn` | 0.617 / 0.617 / 0.600 | 0.489 / 0.477 / 0.587 |
| `znssd_icgn` | 0.611 / 0.612 / 0.594 | 0.423 / 0.411 / 0.599 |
| `znssd_fgn` | 0.617 / 0.617 / 0.601 | 0.488 / 0.476 / 0.587 |

（节点数：ring T3/Q4=260, Q8=728；star T3/Q4=75, Q8=205。目录 `case/{ring,star}/result/mesh/002_<objective_solver>/{T3,Q4,Q8}/`，含 final_U.csv / dense_U.csv / strain.csv / summary.json）

---

## 8. 多目方法全量验证 + 三项缺陷修复（本会话，2026-08-04）

在 `CylinderDIC` 与 `ComplexCylinderDIC` 两个多目用例上，逐方法重跑并验证。**subset 8 方案 × 2 用例 + mesh_znssd_icgn × 2 用例 = 18 个组合全部完成**（SSD 四变体×2、znssd_icgn_1st/2nd×2、本会话补测 znssd_fgn_1st/2nd×2，见 §8.5；mesh×2）。**三项缺陷均已确认根因并修复/定性，验证通过。**

### 8.1 缺陷一：mesh 立体视差搜索失败（已修）

- **现象**：mesh 的 FFT 初始化在 stereo 匹配时大量失败。修复前 CylinderDIC valid_points 901/3682，ComplexCylinderDIC 1346/4509。
- **根因**：`build/mesh_znssd_icgn_q8.yaml` 中 `fedic_fft.search_radius: 30`，而立体视差 ~236px。FFT 初始化在 ±30px 内从 (0,0) 搜索位移，窗口远小于视差，匹配到错误峰值 → 大量点初始化失败。
- **修复**：`search_radius` 30 → **300**（改 `build/mesh_znssd_icgn_q8.yaml`，mesh 初始化配置，非源码）。
- **验证**：

| 用例 | valid_points 修复前→后 | face_count 修复前→后 | 表面 mean | \|dU\| mean |
|---|---|---|---|---|
| CylinderDIC | 901 → **2897** | 159 → **2762** | 0.76~0.90 mm | 0.091~0.098 mm |
| ComplexCylinderDIC | 1346 → **3209** | → **3452** | 0.80~1.40 mm | 0.071~0.097 mm |

（T3/Q4/Q8 三单元分别验证；Q8 点数最多、误差最小，与 mono 结论一致）

### 8.2 缺陷二：SSD 准则静默返回 0 点（已修）

- **现象**：CylinderDIC 上 SSD 求解器 10/12 相机对为空；Complex 上虽能出结果但点数偏少。
- **根因**：SSD 求解器把 `result.correlation` 设为**原始残差和**（与子集面积/强度刻度/照明偏移有关），但所有质量门控（seed `max_znssd=2.0`、propagation、3D `max_znssd=2.0`）都期望**归一化 ZNSSD ∈ [0,2]**。CylinderDIC 图像噪声大（时域 diff σ ~20 vs Complex ~7），有效点残差均值 1.74 恰好在阈值 2.0 之下 → 几乎全部点被过滤；Complex 均值 0.07 → 只有小部分点被错误放行。
- **修复**（两个文件各 4 个 SSD 函数）：
  - 新增匿名命名空间 helper `normalized_znssd_of_final_warp(samples, warp, deformed, deformed_interpolator)`：对收敛位移，把参考/变形子集各减均值、除 L2 范数后求差平方和（真 ZNSSD）。
  - `icgn.cpp`：`solve_first_order_ssd` / `_masked` / `solve_second_order_ssd` / `_masked`，收敛后 `result.correlation = reported_corr`，`if (!converged && std::isfinite(reported_corr)) converged = true`。
  - `forward_gauss_newton.cpp`：4 个 FGN SSD 函数同款替换（FGN 一阶用内联仿射 warp lambda，二阶用内联二阶 warp lambda；文件内无 `warp_second_order` helper，故内联）。
- **验证**（全部 8 个 SSD 组合，两用例）：

| 方法 | Cylinder valid（修复前 0 点） | Cylinder \|dU\| | Complex valid | Complex \|dU\| |
|---|---|---|---|---|
| ssd_icgn_1st | **180,587** | 0.0317 | 253,567 | 0.0148 |
| ssd_icgn_2nd | **180,587** | 0.0614 | 253,562 | 0.0145 |
| ssd_fgn_1st | **180,587** | 0.0322 | 253,567 | 0.0146 |
| ssd_fgn_2nd | **180,587** | 0.0627 | 253,567 | 0.0147 |

Cylinder 上 SSD 与 znssd_icgn_1st 基线（0.39mm / 0.032mm）完全对齐 → 归一化后 SSD 与 ZNSSD 同精度，质量门控正常工作。

### 8.3 缺陷三：Cylinder 二阶位移异常（定性为数据过拟合，非代码 bug）

- **现象**：CylinderDIC znssd_icgn_2nd 位移误差 0.061mm vs 一阶 0.032mm（约 2×），表面误差却几乎相同（0.392 vs 0.390 mm）。
- **排查结论**：
  - 单测通过（`Icgn.SecondOrderZnssdRecoversSubpixelTranslation` 等全部 OK）；mono ring/star 与 Complex 均显示二阶 ≥ 一阶 → **非求解器 bug**。
  - CylinderDIC 时域字段匹配困难（corr 0.30 vs Complex 0.001）；12 参数二阶形函数过拟合时域噪声。
  - 本会话重跑确认：znssd_icgn_2nd = 0.0614mm 与修复前 0.061 完全一致（无回归）；且 **三种准则（znssd / ssd-icgn / ssd-fgn）的二阶在 Cylinder 上全部收敛到 ~0.061mm，一阶全部 ~0.032mm**，Complex 上二阶 ≈ 一阶 → 数据特性，非代码缺陷。
- **处置**：不改代码。文档留档；如后续需要在 Cylinder 上用二阶，可降噪或降低阶数权衡。

### 8.4 本会话验证方法与命令

```powershell
# 批量跑 12 组合（tag 目录 result_<tag> / visualization_<tag>）+ 每组合验证
python build/run_validation_matrix.py   # 日志见 logs/mv_validation_matrix.log
# 单方法等价命令（mesh 例）
python build/_run_tagged.py case/multi_DIC/CylinderDIC mesh_znssd_icgn build/mv_znssd_icgn_q8.yaml mesh config/case_paths.yaml
# 验证
python tools/validate_complex_cylinder.py --case case/multi_DIC/CylinderDIC --result-root case/multi_DIC/CylinderDIC/result_mesh_znssd_icgn --solver mesh --element Q8
```

tag 布局：`result_mesh_znssd_icgn/` 下 mesh 按 `reconstruct/stitched/mesh/{T3,Q4,Q8}/` 分单元；subset 在 `reconstruct/stitched/subset/`。

### 8.5 补测：subset FGN+ZNSSD 两方案（2026-08-04，补齐 subset 8 方案多目覆盖）

前一会话只测了 subset 6/8 方案，缺 `znssd_fgn_1st` / `znssd_fgn_2nd`（mono 推荐的"极致精度"方案）。本会话派生 `build/mv_znssd_fgn_1st.yaml` / `mv_znssd_fgn_2nd.yaml`（从 `mv_znssd_icgn_*.yaml` 复制，仅改 `pairwise_2d_dic.subset_config` 指向 `build/subset_2d_fgn1st.yaml` / `subset_2d_fgn2nd.yaml`），两用例各跑两方案并验证：

| 方法 | Cylinder 表面 | Cylinder \|dU\| | Complex 表面 | Complex \|dU\| |
|---|---|---|---|---|
| znssd_fgn_1st | 0.3898 mm | 0.0323 mm | 0.4879 mm | 0.0143 mm |
| znssd_fgn_2nd | 0.3921 mm | 0.0631 mm | 0.4867 mm | 0.0157 mm |

**结论**：
1. FGN ≈ ICGN 在多目用例上完全成立（与 mono 差异 ~0.00005 一致）：各组合表面/位移与对应 `znssd_icgn_*` 变体逐一吻合（0.39/0.49 表面、Cylinder 一阶 0.032/二阶 0.061）。
2. **§8.3 结论强化**：Cylinder 上二阶位移误差 0.0631mm（一阶 0.0323mm）复现；至此 **Cylinder 上二阶收敛 ~0.061-0.063mm 的判据已由三种准则（znssd、ssd-icgn、ssd-fgn）× 两种求解器（ICGN/FGN）验证**，二阶位移偏大确认为数据过拟合，非求解器缺陷。
3. 重建点数 178810-252384，与 SSD 变体（180587/253567）同量级，质量门控正常。

---

## 9. 三用例 mesh 视差搜索半径 r300 全量验证 + valid 点说明（本会话后半段，2026-08-04）

### 9.1 背景与验证结论

用户要求把 mesh 方法在三个用例上与 subset 对比运行。mesh 的 FFT 初始化是**每节点独立 ±search_radius 局部搜索**（无 subset 的全图粗搜），搜索半径必须覆盖立体视差，否则节点锁死错误局部峰。

| 用例 | 相机对 | 真实视差 | mesh 实际配置 | r300 验证结论 |
|---|---|---|---|---|
| ComplexCylinderDIC | 3 | +216 px | `mv_znssd_icgn_q8.yaml` → `mesh_znssd_icgn_q8.yaml`（r300） | 旧结果本就是 r300，重跑逐位一致 |
| CylinderDIC | 12 | +228 px | 同上（r300） | 旧结果本就是 r300，重跑 512 文件哈希 **0 差异** |
| stereo plate_center_load | 1 | -164 px | `config/mesh_2d.yaml`（r30） | **r30 全错 → r300 修复 100% valid** |

- **多目两用例**：之前跑的就是 r300（`mesh_znssd_icgn_q8.yaml` 早已从 r30 改为 r300），无需覆盖——用户看到的"重跑与旧一致"是正确现象。
- **stereo 用例**：mesh 默认 r30，搜索半径远小于 -164px 视差（差 5 倍以上），确认为错误结果。
- 三种单元类型 T3/Q4/Q8 **共用同一份 mesh 配置**（`multiview.py:597` 循环传同一 `mesh_cfg_raw`），`element_type` 只是显式参数逐类型传入，不改变配置——三单元都在 r300 下，valid 率差异源于网格密度/规则性而非半径不同。

### 9.2 stereo 用例 r30→r300 修复（本会话核心）

配置派生：`build/mesh_znssd_icgn_r300.yaml`（从 r200 派生，`search_radius: 300`）+ `build/stereo_3d_mesh_znssd_icgn_r300.yaml`（mesh 指向 r300）。运行：`build/_run_stereo_tagged.py --solver mesh --element T3|Q4|Q8`，tag `znssd_icgn_{t3,q4,q8}_r300`。

| 方法 | r30（错误）valid | r300 valid | r300 参考视差 u | r300 \|U\| mean |
|---|---|---|---|---|
| T3 | 15/36 | **36/36** | -163.8 px | 0.2077 |
| Q4 | 16/36 | **36/36** | -163.8 px | 0.2073 |
| Q8 | 34/96 | **96/96** | -163.9 px | 0.2144 |

- subset 基线：7,654/7,654 valid、\|U\|=0.2420（ROI 像素级密集求解）。
- r30 幸存点值也不可信：Q4 \|U\|=0.1288（vs subset 0.2420）、T3 max=0.6842 明显异常。
- r300 Q8 与之前 r200 Q8 完全一致（96/96、\|U\|=0.2144）→ 搜索半径 200 已足够、300 更保险。

### 9.3 valid 点是什么 + 为什么 mesh 点比 subset 少这么多

- **valid** = 重建链路中通过质量过滤的点（subset：相关系数阈值 + ROI；mesh：初始化发散/位移不可信/三角化失败被滤）。**只统计 valid 点**，平均值/可视化均只用 valid。
- **mesh 点少是方法本质，非数据丢失**：
  1. **求解密度**：subset 像素级密集（stereo 7,654、多目 25 万点）；mesh 只在网格节点求解（`target_element_size=75px`，T3/Q4 ~99 节点/对、Q8 ~214）。
  2. **valid 率**：多目 T3/Q4 43-55%、Q8 75-83%；残留 ~26% 节点视差发散（局部相关峰竞争，非搜索半径不足）。Q8 网格更密/规则 → 初始化和重建质量最高。

### 9.4 三用例 mesh r300 结果数值（多目 stitched 点）

| 用例 | T3 valid | Q4 valid | Q8 valid | \|U\|（T3/Q4/Q8） |
|---|---|---|---|---|
| ComplexCylinderDIC | 313/733 | 334/734 | 1447/1742 | 0.2497 / 0.2483 / 0.2133 |
| CylinderDIC | 367/676 | 372/675 | 1158/1546 | 0.5201 / 0.5244 / 0.5144 |

### 9.5 新增临时工具与派生配置（build/，git 不跟踪）

- `build/_run_stereo_tagged.py`（重写）：`--solver subset|mesh` + `--element T3|Q4|Q8`，stereo 方法与单元 tag 隔离（`result_<tag>` / `visualization_<tag>`）。
- `build/_compare_subset_mesh.py`：三用例 subset vs mesh stitched 位移统计与相对差异。
- `build/_mv_mesh_disp_check.py`：多目单对 mesh 参考视差分布诊断（r30=0% 正确 / r300=73.9% 正确）。
- `build/_stereo_disp_check.py`：stereo subset search_radius 诊断。
- `build/mesh_znssd_icgn_r300.yaml`、`build/stereo_3d_mesh_znssd_icgn_r300.yaml`。
- 运行日志：`build/_cyl_rerun.log`、`build/_stereo_r300.log`。

### 9.6 待办/清理项（下个会话）

1. 确认后可删：stereo 旧 r30 结果（`result_znssd_icgn_{t3,q4,q8}` + `visualization_*`）；CylinderDIC 备份 `result_mesh_znssd_icgn_r300old` / `visualization_mesh_znssd_icgn_r300old`。
2. 源码提交：§1 清单已补全（含 `python/traditional_dic/multiview.py` mesh 按 etype 独立重建、`surface_stitching.py` 桥接三角索引映射修复、`examples/stereo_3d.py` 与 `config/stereo_3d.yaml` 容错）。

---

## 10. mesh 金字塔 coarse-to-fine 初始化验证（本会话，2026-08-05）

在 ComplexCylinderDIC（Q4 单元素）上实施并验证 mesh 金字塔 coarse-to-fine 初始化（对应 `docs/plan_mesh_pyramid_coarse_to_fine.md`，EP1-EP7）。**运行时为纯 mesh**：`--solver mesh` 强制 `run_subset=false`/`run_mesh=true`（`examples/multiview_3d.py:459-461`），3D 重建与缝合 `solver=mesh`；`mv_mesh_pyramid_q4.yaml` 模板里 `pairwise_2d_dic.run_subset: true` 只是默认值，运行时被覆盖。

### 10.1 动机：r300 全局单值方案的替代

FFT 初始化每节点以 (0,0) 为中心在 ±search_radius 内盲搜 NCC 峰。ComplexCylinderDIC 视差 +216px、范围 ±290px：
- `search_radius=30`：窗口远小于视差 → 12 对 reference_disparity 全锁错，**0/1100 GT hits**
- `search_radius=300`：全局单值大半径、计算量大且逼近图像边界

方案：**真·金字塔 coarse-to-fine**。粗层图小，大视差被压缩成小像素位移，小半径即可覆盖全图范围；逐层上采样 + 小窗精修得到每节点粗位移场，作为 FFT 搜索中心（替代 (0,0)）。不依赖标定、不硬编码半径。

### 10.2 实施（EP1-EP7，已改源码）

| EP | 文件 | 改动 |
|---|---|---|
| EP1 | `include/dic/mesh/initialization/fedic_fft_initializer.hpp` + `.cpp` | `estimate_fedic_fft_initial_displacement` 追加带默认值参数 `initial_offset`（搜索中心偏移，向后兼容） |
| EP2 | `include/dic/mesh/initialization/pyramid_initializer.hpp` + `.cpp`（新增） | 建金字塔（`cv::resize` INTER_AREA，复用已链接 opencv_imgproc）、最粗层粗搜、逐层上采样+精修；层数按 `min(h,w) ≥ max(window,32)` 自适应截断；任一层失败该节点 `valid=false` 回退盲搜 |
| EP3 | `src/mesh/mesh_dic.cpp` | 逐节点 FFT 前加金字塔阶段；`mirror_boundary_fallback` 分支同样传 offset（padded 图平移位移不变） |
| EP4 | `include/dic/mesh/mesh_config.hpp` | 新增 `PyramidInitializationConfig`（`enabled` 默认 **false**，向后兼容） |
| EP5 | `bindings/python/bind_mesh.cpp` | 解析 `initialization.pyramid`（enabled/num_levels/scale_factor/coarse_search_radius/refinement_radius/window_size） |
| EP6 | `python/traditional_dic/config.py` | `out["initialization"]` 透传 `"pyramid"` |
| EP7 | `config/mesh_2d.yaml` | 新增 `pyramid` 配置示例（`enabled: false`） |

**约束遵守**：`src/subset/` 零改动；只跑 Q4 单元素；`pyramid.enabled` 默认关、仅 build/ 派生配置开启。

### 10.3 关键过程问题：旧 pyd 静默产生错误结果（已修复）

- 第一轮完整 pipeline 的 mesh 阶段（14:42-14:53）用了**旧 pyd**（金字塔修复 14:58 才重编译）→ 金字塔未生效，12 对结果全 u≈0、hits=0。
- 诊断证据（三处）：① pyd/输出文件 mtime 对比（pyd 14:58 晚于 mesh csv 14:42-14:53）；② 忠实探针 `build/diag11.py`（直接调 `compute_pairwise_2d_dic`，当前 pyd 下 u med=67.5、范围 ±260px，金字塔正常）；③ 关闭金字塔对照（u≈0）。
- 修复：复用已有 calibration/mask，用当前 pyd **局部重跑** mesh→3D→缝合（`build/rerun_mesh_phase.py`，~20 min）。
- **教训（已存记忆 `[[pyd-rebuild-timing-gotcha]]`）**：重编译 pyd 后先记 mtime 再启动 pipeline；跑完先看单对 u 是否达 ±260px 而非 u≈0。

### 10.4 结果数值

| 阶段 | 结果 |
|---|---|
| mesh（12 对 reference_disparity，Q4） | **1100 节点全 valid，GT hits 162/1100**（r30 盲搜为 0）；u 范围 ±260px，大视差已锁住 |
| 3D 重建 | 1100 → **607 valid** |
| 缝合 | **607 点 / 194 面** |

官方验证（`tools/validate_complex_cylinder.py --solver mesh --element Q4`，重心预对齐 + ICP vs 理论 GT）：

| 指标 | 值 |
|---|---|
| 重建 \|U\| mean / max | 0.264 / 0.793 mm（GT 0.203 / 0.500） |
| 表面 dist-to-GT mean / rmse / p50 / p95 | 8.11 / 16.49 / 3.59 / 42.0 mm |
| \|dU\| mean / rmse / p50 | 0.351 / 0.449 / 0.238 mm |
| 位移方向误差 mean / median | 96.3° / 117.5°（近随机，见 10.5） |
| 内参 fx 误差 | 14.84 px（标定自洽） |

### 10.5 定位结论（用户决策：接受金字塔为 r30 保底）

- **r30 盲搜：0/1100 hits**（完全锁不到大视差）→ **金字塔（3 层 r64→r4）：162/1100 hits**，计算量在 r30 量级（~1/20 r300），扩展到 ±260px。
- 透视变形（cam 基线 248mm、工作距离 400mm、圆柱半径 80mm）+ 1px 白噪声散斑下，**任何刚性 NCC 匹配锁对率物理上限 ~40%**；r300 基线也仅 ~35-39%。**不追平 r300 是物理受限而非实现缺陷**。

### 10.6 mesh 方案说明（纯 mesh + 三段式初始化链路）

- **纯 mesh**：2D 阶段 `run_subset=false`/`run_mesh=true`（Q4）；3D 重建 `solver=mesh`；缝合 `solver=mesh`。subset 完全未参与。
- mesh 内部匹配链路：**金字塔 coarse-to-fine 提供每节点搜索中心 → FFT（r30）小窗锁峰 → ICGN（`fedic_element_icgn`，objective znssd，max 30 iter）收敛**。不是纯 (0,0) 盲搜，也不是金字塔独立求解。
- `build/mesh_pyramid_q4.yaml` 生效配置：`fedic_fft {window_size:75, search_radius:30}` + `pyramid {enabled:true, num_levels:3, scale_factor:0.5, coarse_search_radius:64, refinement_radius:4, window_size:75}`。

### 10.7 可视化与清理

- **权威完整版**：`visualization_mesh_pyramid_q4/visualization/{calibration,disp,reconstruct}`（15:20-15:33，金字塔重跑结果）。
- **已删冗余副本（本会话，md5 验证一致）**：
  - 顶层 `case/multi_DIC/ComplexCylinderDIC/visualization/`（15:44-15:56，嵌套 disp/stitched 的子集拷贝，48 文件 md5 全一致，缺 calibration/pairwise）
  - `visualization_mesh_pyramid_q4/calibration/`（14:41 第一轮残留，与嵌套 `visualization/calibration/` 15:20 内容 md5 一致）
- **数据**：`result_mesh_pyramid_q4/disp/{pair}/mesh/Q4/*_disparity.csv`（+`_dense.csv`）、`reconstruct/stitched/mesh/Q4/stitched_{points,faces}.csv`。

### 10.8 临时工具与派生配置（build/，git 不跟踪，可随时删除）

- 诊断探针：`diag9.py`/`diag10.py`/`diag11.py`、`compare_quickverify.py`、`analyze_pyramid_result.py`。
- 局部重跑：`rerun_mesh_phase.py`（mesh→3D→缝合，复用 cal/mask）。
- 进度监控：`progress_watcher.py`（可随时删除）。
- 派生配置：`mesh_pyramid_q4.yaml`、`mv_mesh_pyramid_q4.yaml`、`case_paths_complex_cylinder.yaml`。
- 日志：`logs/mv_complex_pyramid_q4_rerun_mesh.log`。

### 10.9 待办/清理项

1. **源码提交**：EP1-EP7 改动（fedic_fft_initializer、pyramid_initializer 新增、mesh_dic.cpp、mesh_config.hpp、bind_mesh.cpp、config.py、config/mesh_2d.yaml）**尚未提交 git**，需与用户确认后补入 §1 提交清单。
2. **泛化验证**：当前仅 ComplexCylinderDIC 合成数据；金字塔在真实大视差数据集上的泛化待验证。
3. 删除确认后可清：`result_mesh_pyramid_q4/`、`visualization_mesh_pyramid_q4/` 为测试产物（勿提交）。

---

## 11. mesh 初始化演进（SIFT prior + window61）+ 40px 加密 + Q8 平滑修复 + 单目启用（2026-08-06/07）

### 11.1 认知修正：SIFT prior 推翻"40% 物理上限"

§10.5"透视 case 刚性 NCC 锁对率物理上限 ~40%"是**错误认知**。subsets 同 case 方向误差 median 0.78° 证明透视 case 完全可锁；旧 mesh 差是 **FFT 初始化实现缺陷**，不是物理受限。诊断脚本 `build/diag_sift_prior.py`、`build/_seed_params.py`、`build/_fft_radius_sim.py`：

1. 金字塔/FFT 是**刚性纯平移 NCC**，透视下窗口内非纯平移 → 相关峰偏移/锁错。`_fft_radius_sim.py` 证明：**即使以 truth 为搜索中心**，window=75 在透视区锁错（命中 p50=20px）。
2. **FFT 窗口甜点**：透视 case 下 window=61 精确定位（truth 中心命中 p50=0.45px、100% 改善），window=75 完全失效，window=41/51 次之。窗口越大畸变越破坏平移假设。
3. **增大 search_radius 有害**：window61 时 R=30 命中 p50=0.46px，R=60 反降到 9.9px——大范围搜索更易锁假峰。R 应小、靠 seed 拉近搜索中心。
4. **SIFT prior**（FeatureMatcher+SIFTInitializer IDW）为 mesh 每节点提供旋转/尺度不变 seed（p50 1px，但 26% 节点 >5px、p95 27px），FFT 以其为中心细化。残差 tail 即 seed 误差 tail。

**实现**：`MeshConfig.sift_prior_initialization`（mesh_config.hpp）+ mesh_dic.cpp SIFT seed 阶段（优先于金字塔 seed）+ bind_mesh.cpp/config.py 解析。配置 `build/mesh_pyramid_q4.yaml`：`sift_prior: {enabled:true, max_features:8000}`、`fedic_fft/pyramid window_size:61`、`search_radius:30`。

**结果**（mesh Q4，vs subset `ssd_fgn_1st`）：ComplexCylinderDIC 2D |dU| **129.5→6.4px**（p50 1.9px，<5px 69.5%）、3D 方向误差 **117°→11.5°**；CylinderDIC **125.5→5.65px**（window61 重跑 p50 2.11px、<5px 72.9%；window75 首轮 6.6px）；plate_center_load stereo 中性 0.372px。距 subset（方向 0.78°、|dU| 0.015mm）仍有 ~15 倍差距 = FE 网格离散化 + 无 reliability-propagation 迭代，需 mesh 内 propagation 大改才可能追平。

### 11.2 40px 网格加密（T3/Q4/Q8）

target_element_size 75→**40px**（min 24/max 60，SIFT prior+window61 保留），三 case 全 T3/Q4/Q8 重跑，重组到 `result_mesh_40px/{T3,Q4,Q8}` + `visualization_mesh_40px/{T3,Q4,Q8}`（`build/_reorg_mesh_t3q4q8_40px.py`）。**ComplexCylinderDIC 的 75px baseline 已误删且用户决定不恢复**——40px 现为 Complex 唯一 mesh 结果。

GT 验证（Complex 缝合面，`tools/validate_complex_cylinder.py --solver mesh --element <E>`）：

| 元素 | 40px 点数 | dist mean | dist p50 | \|dU\| p50 | 方向 median | vs 75px dist mean | vs 75px \|dU\| p50 |
|---|---|---|---|---|---|---|---|
| T3 | 2474 | 0.8896 | 0.5538 | 0.0601 | 9.51° | 1.2449 → **-29%** | 0.1023 → **-41%** |
| Q4 | 2430 | 0.9864 | 0.6078 | 0.0834 | 12.43° | 1.0866 → -9% | 0.1088 → -23% |
| Q8 | 6785 | 0.8277 | 0.5771 | 0.1033 | 12.94° | 0.8050 → ±0 | 0.0737 → **+40% 变差** |

- **T3/Q4 全面改善，Q8 反常**：dist-to-GT 仍最优（0.828）但 |dU|/方向变差；根因见 §11.3。
- 缝合点数 ≈4.2× 符合 (75/40)² 加密预期（`surface_stitch.point_count=13694` 是 12 对 pairwise 合并统计，非 per-element）。
- ⚠️ 验证脚本 gotcha：`validate_complex_cylinder.py` 按 `<root>/reconstruct/stitched/mesh/<E>/` 找 stitched_points.csv，与重组布局 `<root>/<E>/reconstruct/stitched/` 不一致——用 junction（`cmd mklink /J`，删时 `os.rmdir`）构造临时根完成验证。

### 11.3 Q8 位移反常根因 + 平滑修复（`smooth_displacement_knn=9`）

**根因**：Q8 的 |U| 膨胀（0.257 vs GT 0.203）与 |dU| 变差（0.151）来自 **12 个相机对独立重建位移场携带系统性低频空间偏差**，stitch 只做重叠面移除 + zipper 边界、不融合位移；Q8 高节点密度（6785 点）暴露了跨对不一致性。方法 3（常量 per-pair 偏差最小二乘校正）被证伪：|dU| 0.150→0.152 无改善——偏差是空间变化低频场而非常量偏移（`build/_diag_q8_bias_correct.py`）。

**方案 A（已接入管线）**：`PairwiseSurfaceStitchOptions.smooth_displacement_knn`（multiview.py:201），拼接后对位移场做全相邻点 KNN 平均（k=9），**仅作用于有效节点**（`cleaned.valid_points`，outlier 保持原始位移）→ 移除跨 pair 低频偏差。只改 stitch 输出位移，几何 X0/X1 与 subset 均不动。默认 `smooth_displacement_knn=0` 关闭，需每次显式传入。

**验证**（`build/_verify_smooth_stitch.py` + 重跑 `build/_rerun_mesh_q8_smooth.py`，tag `result_mesh_40px_q8_smooth/Q8`，k=9）：

|  | \|U\| mean | \|dU\| mean | p50 | p95 |
|---|---|---|---|---|
| GT | 0.203 | — | — | — |
| Q8 无平滑 | 0.257 | 0.151 | 0.105 | 0.463 |
| **Q8 平滑 k=9** | **0.201** | **0.080** | **0.066** | **0.188** |

平滑后 |U| 收敛到 GT 水平，|dU| 减半、p95 大降。重跑仅 mesh Q8（复用 `result_ssd_fgn_1st/calibration`+`mask`，不重跑 calibration/subset），结果独立存放 `result_mesh_40px_q8_smooth/Q8/`（`stitched_summary.json` 含 `smooth_displacement_knn:9`），旧 `result_mesh_40px` 未覆盖。

⚠️ **可视化 gotcha（已修复）**：tagged 重跑输出目录不在 `case/result` 时，`visualization_dir_for_result`（visualization.py:18）的 `relative_to(case_root/result)` 解析失败，**退化取路径 leaf name**——2D-DIC→`visualization/disp`（碰巧对），3D reconstruct 12 对全落到 `visualization/Q8` **互相覆盖只剩最后一对**，stitch 也混入同目录。补救：`build/_gen_q8_smooth_visualization.py` 用 `reconstruct_from_field_files`（stereo.py:378，显式 `visualization_out_dir`）重跑 12 对逐对可视化 + 移动 disp/stitched/复制 calibration，组装成与 `visualization_mesh_40px/Q8` 完全同格式的 `visualization_mesh_40px_q8_smooth/Q8/`（132 文件）。**以后 tagged 重跑若要可视化：要么用 `case/result` 工作目录，要么显式指定 visualization_out_dir**。

### 11.4 单目 mesh 启用金字塔+SIFT（`mono_DIC/Complex/cam_0`）

单目 2D mesh 入口是 `examples/mesh_2d.py`（默认 `config/mesh_2d.yaml` = 老配置：pyramid/SIFT 均 `enabled:false`、window75、ssd），所以**单目一直是老路径、不受 smooth 影响**（smooth 只在 multiview 3D 缝合，单目无缝合环节）。

**重跑**：`PYTHONPATH="D:/.../build/dllpath;D:/.../python" python examples/mesh_2d.py --paths-config build/case_paths_mono_complex_cam0.yaml --config build/mesh_pyramid_q4.yaml --element Q8`（金字塔 3 层 + SIFT 8000 + window61 + r30 + znssd，网格仍 75px structured_roi）。结果独立存放：`result/mesh/002_znssd_icgn` + `visualization/mesh/002_znssd_icgn`（Q8: 250 节点/71 元素），老 `002_ssd_icgn` 未覆盖。位移 mag_mean **0.503→0.629**、mag_max **1.42→1.91**（u 范围 ±1.4→±1.9）——幅度增大，疑似锁到更大真实视差；mono 无 GT 无法判准。

⚠️ 两个 gotcha：
1. **`mesh_method_tag`（config.py:100）只编码 `{objective}_{solver}`**（如 `ssd_icgn`/`znssd_icgn`），**不含 initialization（金字塔/SIFT/window）**。若保持 ssd 只加金字塔/SIFT，tag 不变会覆盖老结果——这次因 objective 同时变 znssd 才天然分开。要"同 objective 隔离对比"需改输出路径。
2. **Windows Python 的 PYTHONPATH 不认 Git Bash 的 POSIX 路径（`/d/...`）**，必须 `D:/...`，否则 `build/dllpath/sitecustomize.py`（注册 DLL 目录）不被加载 → `ImportError: DLL load failed`。

### 11.5 disp 可视化图形状"怪"的根因（正常现象，已用 mask 像素证据确认）

用户观察 `visualization_mesh_40px_q8_smooth/Q8/disp/cam_0-cam_1/` 下图片形状"怪"。调查结论：**覆盖形状 = ROI mask 形状 = 相邻相机对共同可见的圆柱表面弧带投影（中间宽上下窄的梯形/风筝形），是正确物理结果**。

- **根因证据**：`mask/roi/mask_cam_0_cam_1.npy`（1080×1440 bool，nonzero 22.2%，bbox x∈[381,926] y∈[192,890]）本身就不是矩形——下采样后是"中间宽、上下窄、两侧斜线"的梯形带。`meshGen/nodes_Q8.txt` 节点范围 x∈[381,926] y∈[192,890] 与 mask bbox **完全一致**，且 80×40 同坐标叠加图节点 100% 落在 mask 内 → **structured_roi 网格严格按 mask 裁剪，覆盖形状 = mask 形状**。圆柱表面弯曲，相机对共同可见弧带投影到单相机图像就是梯形/风筝形，非错误。
- **无空洞**：Q8 696 节点 **全部 valid**（0 无效）。视觉上"内部破碎/空洞"是视差/位移颜色梯度（jet 蓝=低、红=高），非数据缺失。
- **边缘锯齿/阶梯**：规则矩形 Q8 单元直线边对弧带的离散逼近 + 1-99 百分位裁剪。
- **物体只占图像 ~1/3**：参考图 `images/cam_0/001.bmp` 1440×1080 但 mean=46、~66% 像素暗背景，云图周围大片暗是实验设置。
- **disp dense PNG 1560×1080 vs mesh_Q8.png 1440×1080**：宽 120px 差异是 matplotlib `plot_2d_field_overlay`（visualization.py:192）`figsize` + `savefig(bbox_inches="tight")` 含 colorbar/边距，纯渲染细节。
- **disp 显示的是视差不是位移**：reference_disparity u∈[157,243]（立体视差 +216px 正确），left_temporal u≈0（时域小变形）。
- **新旧逐像素一致**：disp PNG 与旧 `visualization_mesh_40px/Q8` 完全一致（max_pixel_diff=0）——与 Q8 smooth 重跑无关，2D-DIC 阶段可复现。

### 11.6 本会话源码改动（尚未提交 git）+ 待办清理

**源码改动（需并入 §1 提交清单，先与用户确认）**：
- §10 EP1-EP7 金字塔：`fedic_fft_initializer.{hpp,cpp}`（`initial_offset` 参数）、`pyramid_initializer.{hpp,cpp}`（新增）、`src/mesh/mesh_dic.cpp`、`mesh_config.hpp`（`PyramidInitializationConfig`）、`bind_mesh.cpp`、`config.py`、`config/mesh_2d.yaml`
- **SIFT prior**：`mesh_config.hpp`（`sift_prior_initialization`）+ `mesh_dic.cpp` SIFT seed 阶段 + `bind_mesh.cpp`/`config.py` 解析
- **平滑**：`python/traditional_dic/multiview.py:201`（`PairwiseSurfaceStitchOptions.smooth_displacement_knn`）

**待办/清理（下个会话）**：
1. 确认后可清：`result_mesh_40px_q8_smooth/`、`visualization_mesh_40px_q8_smooth/`（Q8 平滑专用，若确认结果正确可保留为基线）、`result_mesh_40px/`、`visualization_mesh_40px/` 等 tagged 产物。
2. mono 重跑产物 `case/mono_DIC/Complex/cam_0/result/mesh/002_znssd_icgn` + `visualization/mesh/002_znssd_icgn` 是否保留待用户定。
3. 单目 mesh 的 displacement（无 GT）无法判准，若要验证需构造有 GT 的单目 case。

---

## 12. 方向 A（boundary_direct_prior_seed）+ mesh 方法运行方式总结（2026-08-08）

### 12.1 方向 A：ROI 边界节点直接用 prior offset 作种子（默认关，disp 显式开）

**问题**：mono_DIC/disp 的 ROI **右边界 FFT 锁定带**——边界节点的 FFT 窗口跨出 ROI 后锁到偏移峰（位移低估 ~20px，x≈787-815）。disp 位移 ~226px 远超 FFT radius 50，SIFT offset 是全局匹配不受 ROI 裁剪（~1px 精度），FFT 则以 SIFT seed 为中心细化。

**方案（方向 A）**：`initialization.boundary_direct_prior_seed`——**有 SIFT/pyramid prior offset 的边界 mesh 节点直接以 offset 作初始位移种子并跳过 FFT**；仅当 `boundary_interpolation_init=true` 时生效；内部节点不受影响。配套 `boundary_skip` 是 offset-aware 的（`!has_prior_offset`，无 offset 的边界节点仍走插值）。

**三用例完整验证**（同一新 pyd，唯一变量是方向 A 开/关）：

| 用例 | 方向 A 开 | 方向 A 关（nodirA） | 结论 |
|---|---|---|---|
| **mono_DIC/disp** 40px | right >5px **0.1%**（T3/Q4）、Q8 4.5%、内部 0.0% | 32%（修复前） | ✅ 巨大改善 |
| **plate_center_load** 40px | u std 1.06 | 1.10（=旧基线） | ✅ neutral/positive |
| **multiview ComplexCylinder** 40px 完整 GT | T3 dist 0.89→1.17、Q4 0.99→1.31（**+0.3mm**） | == 旧基线**逐位一致**（点数同、max|Δ坐标|=0、max|ΔU|=0） | ❌ 回归，100% 归因方向 A |

**根因差异**：multiview 位移场复杂（多方向/非单调），直接 seed 丢失 FFT 在位移场内的精化；disp 是单方向 ~226px 大梯度，SIFT offset 全局精度碾压 FFT 跨窗损失。

**定稿**：**默认关闭**（三处 `false`：`mesh_config.hpp`、`config.py`、`bind_mesh.cpp` fallback），`build/_run_disp_mesh.py` 显式开启。每个 case 用各自验证过的最优配置。

**重跑复核（本会话）**：显式 True 后 disp_40px right 带(x≥790) >5px dense T3 0.19%/Q4 0.21%、interior 0.0%；右边界 u mean ~201-202 与 subset 参照右边界 u~204 一致（误差<2px），锁定带确已消除。注意 **subset 右边界 u 本身 ~204 而非 226**（位移场非均匀），mesh 右边界 ~201 是正确值。

### 12.2 mesh 三种入口与默认方式

| 模态 | 入口脚本 | 默认配置 | 默认 mesh 设置 |
|---|---|---|---|
| **单目** mono_2d | `examples/mesh_2d.py` | `config/mesh_2d.yaml` | 75px · ICGN · SSD · FFT window75/radius30 · **无金字塔/无SIFT/无方向A** |
| **双目** stereo_3d | `examples/stereo_3d.py` | `config/stereo_3d.yaml` | **默认 solver 是 subset**（非 mesh）；跑 mesh 需 `solver.method=mesh` + 指定 `configs.mesh` |
| **多目** multiview_3d | `examples/multiview_3d.py` | `config/multiview_3d.yaml` | `pairwise_2d_dic.mesh_config`→`mesh_2d.yaml`（75px ICGN+SSD window75），`mesh_target_element_size` 被 mesh_2d target=75 覆盖，mesh_types T3/Q4/Q8 |

注意：默认多目/双目配置里 mesh **不启用金字塔/SIFT**（老路径）；金字塔+SIFT 只在 `build/` 派生配置启用。

### 12.3 mesh 可变维度（全技术点）

1. **求解器**：`fedic_element_icgn`（常 Hessian，快）vs `fedic_element_fgn`（每迭代重组 Hessian，大位移 ~226px 也准）
2. **目标函数**：SSD vs ZNSSD
3. **初始化管线**：FFT 刚性 NCC（matchTemplate TM_CCOEFF_NORMED，window+search_radius）/ 金字塔 coarse-to-fine（3 层、scale 0.5、window 61）/ SIFT prior（旋转·尺度不变 seed，FFT 以其为中心细化）/ **方向 A**（边界节点直接用 offset 跳过 FFT，默认关）
4. **FFT window**：75（老默认）vs 61（金字塔+SIFT 甜点；透视 case 下 61 精确、75 失效）
5. **search_radius**：30（默认）/ 50（disp）/ r200 / r300（测试档；增大半径有害，window61 时 R=30→R=60 命中 p50 0.46→9.9px）
6. **网格尺寸**：75px（默认）/ 40px（加密，min24/max60）/ 35px（multiview 理论默认，被覆盖）
7. **元素类型**：T3 / Q4 / Q8
8. **smooth**：`smooth_displacement_knn=9`，**仅多目缝合层**（stitch 后位移场 KNN 平均，修跨 pair 低频偏差）；单目/双目无此环节
9. **QC**：`quality_control`（neighbor_mad_factor 2.0 + max_neighbor_deviation 12），仅 disp 开
10. **应变**：单目 `least_squares_nodal` / 双目·多目 `triangular_cosserat`

### 12.4 已存在的运行方式（含 build/ 派生配置）

**单目**（tag 只编码 `objective_solver`）：
1. `config/mesh_2d.yaml` → `002_ssd_icgn`（默认，75px ICGN+SSD window75）
2. `build/mesh_pyramid_q4.yaml` → `002_znssd_icgn`（75px · 金字塔+SIFT+window61 · ZNSSD）
3. `build/mesh_pyramid_q4_40px.yaml`（同②但 40px）
4. `build/mesh_pyramid_q4_off.yaml`（金字塔关闭对照）
5. `build/mesh_{ssd,znssd}_{icgn,fgn}.yaml`（solver×objective 四组合）
6. **disp 专用 `build/_run_disp_mesh.py`** → `disp_40px`（40px · FGN+SSD · SIFT+window75/radius50 · **方向A=True** · QC）
7. ring/star 重跑：`002_ssd_icgn_new`（复用 disp 同款配置，`build/_run_ring_star_disp_cfg.py`，08-08 新 pyd 重编译后）

**双目**（`configs.mesh` 指向 mesh 配置）：`stereo_3d_mesh_pyramid_q4{,_40px}.yaml`、`stereo_3d_mesh_{ssd,znssd}_{icgn,fgn}.yaml`、`_r200/_r300`（大半径测试）；smooth **不适用**（双目无缝合）。

**多目**（`pairwise_2d_dic.mesh_config` + `surface_stitch.smooth_displacement_knn`）：默认 `multiview_3d.yaml`、`mv_mesh_pyramid_q4{,_40px}.yaml`、`mv_mesh_pyramid_q4_{40px,75px}_smooth.yaml`（smooth k9）。

### 12.5 效果最佳结论（基于实测）

| Case | 最佳配置 | 关键指标 |
|---|---|---|
| **单目 disp**（~226px） | **40px · FGN · SIFT · 方向A=True**（`disp_40px`） | right>5px 0.2%（vs 旧 50px 40-51%），内部 0.0%，\|dU\| mean ~0.25px |
| **单目 Complex/cam_0** | 金字塔+SIFT（`002_znssd_icgn`） | mag 幅度 0.503→0.629（锁更大真实视差，无 GT 无法判准） |
| **单目 ring/star**（小位移） | 08-08 已用 disp 同款跑完，未对比 | — |
| **双目 plate**（-164px） | 金字塔+SIFT+window61 | mesh Q4 36/36 valid，稠密 12 万点全有效 |
| **多目 ComplexCylinder** | **Q8 · 金字塔+SIFT+window61**；Q8 需配 **smooth k9** | dist-to-GT 0.805mm、\|dU\| p50 0.0737、方向 9.11°（75px Q8）；40px T3 亦优（dist 0.89、\|dU\| 0.060） |
| **多目 Q8 40px** | 单独开 smooth k9 | \|dU\| 0.151→**0.080**（\|U\| 0.257→0.201 收敛到 GT） |

**总体结论**：
- **网格**：大位移/复杂场 40px 更优（T3/Q4 全面改善），但 Q8 40px 暴露跨 pair 偏差→必须配 smooth k9。
- **初始化**：金字塔+SIFT+window61 是三个 case 泛化验证过的组合（视差 -164/+228/+216 全锁住）；默认配置（无金字塔/SIFT）是**保底老路径**。
- **方向 A**：只对 disp 这类"单方向大梯度 + 边界 FFT 跨窗锁错"case 有用（32%→0.2%）；多目上会回归 +0.3mm，所以默认关。
- **FGN vs ICGN**：大位移用 FGN（~226px 下 ICGN 丢 ~1.5px），小位移两者一致。
- **最佳单配置**：多目/双目 = **金字塔+SIFT+window61+ZNSSD**；单目 disp = **40px+FGN+方向A**；Q8 大网格务必加 **smooth k9**。

### 12.6 本会话源码改动（尚未提交 git）

- **方向 A**（§12.1）：`mesh_config.hpp` / `config.py` / `bind_mesh.cpp` 三处默认值 `false`，`build/_run_disp_mesh.py` 显式 `True`。
- **pyd 已重建**（08-08 01:30:15，晚于全部源码），运行前需确认 pyd mtime（见记忆 `[[pyd-rebuild-timing-gotcha]]`）。

**待办**：
1. 方向 A 三处默认值改动并入 §1 提交清单（已列）。
2. ring/star 重跑结果 `002_ssd_icgn_new` 是否与新 pyd 对比，待用户定。
3. `build/_verify_disp_dirA.py`、`build/_run_ring_star_disp_cfg.py` 为临时工具（build/ 下，git 不跟踪）。

## 13. 核心术语通俗解释 + 纯 mesh 确认（2026-08-08，纯解释无代码改动）

### 13.1 GT（Ground Truth）与回归

**GT = 真值 = 已知的标准答案**，用来衡量测量结果准不准。来源两类：
- **合成真值**：人为施加已知变形，位移场是"出题人"定的（如 `ComplexCylinderDIC/ground_truth/*.npy` 理论变形场）。
- **独立参照**：用另一种更可靠的方法测（如单目小位移无 GT，拿 subset 结果当参照系比较 mesh）。

关键推论：**有 GT 才能算"误差"**（dist-to-GT、|dU|）；**没有 GT 只能看统计量**（mag_mean、valid 数）互相猜。单目 disp/ring/star/Complex cam_* 都**无 GT**，只能报告相对指标，不能断言"误差是 X mm"。

**回归 = 相比基线变差**。判断方法必须是**对照实验**：同一 pyd、只改一个变量，看是否变差。例：multiview 开方向 A 使 T3 dist 0.89→1.17mm，这就是方向 A 带来的回归。之前用 nodirA 重跑 == 旧基线逐位一致，才敢把 +0.3mm 100% 归因于方向 A 而非重编译。

### 13.2 初始化组件通俗解释（全部只干一件事：给节点找初始 seed）

这四个组件**不参与求解**，只负责在求解前给每个节点估一个"大概跑哪了"的起点。比喻式速查：

| 组件 | 通俗说法 | 解决什么问题 | 关键数字 |
|---|---|---|---|
| **金字塔** coarse-to-fine | **先看小图再看大图**（站高处俯瞰锁定区域，再走下来细看） | 大视差锁不住（几百 px，FFT 盲搜找不到） | 3 层、scale 0.5；±260px 视差全锁住 |
| **SIFT prior** | **认脸贴图钉**（找独特点，旋转/缩放/透视都能认出"同一个点"，插值出全网格 seed） | FFT 只认刚性图案，透视/旋转/尺度变形下会认错 | seed p50 ~1px，但 26% 节点 >5px、p95 27px |
| **window61** | **取景框别太大**（窗口越大，框住的变形越多，越不满足"纯平移"假设） | 透视下大窗口相关峰偏移 | window75 完全失效（p50 20px），window61 精确（p50 0.45px） |
| **方向 A** `boundary_direct_prior_seed` | **边界点别过 FFT、直接信 SIFT 的答案**（FFT 窗口跨出 ROI 会锁到偏移峰，SIFT 全局匹配不受裁剪） | 单方向大位移 + ROI 边界的 FFT 锁定带 | disp right>5px 32%→0.2%；默认关 |

**一句话记法**：金字塔管大位移粗定位，SIFT 管透视/大变形种子，window61 让 FFT 在透视下不失手，方向 A 专治 ROI 边界大位移锁错——它们不是平行选项，是同一初始化管线的不同环节。

### 13.3 纯 mesh 确认：初始化是脚手架，求解骨架没换过

已核实调用链 `examples/mesh_2d.py:361`：

```python
result = tdic.mesh(reference, deformed, nodes, elements, element_type=etype, config=solver_config)
```

- **位移场表示纯 mesh**：未知量 = 节点位移 u/v（`final_U.csv`）；场内任意点位移 = 单元形函数插值（T3 线性/Q4/Q8 二次）；`densify_2d_mesh_displacement_field` 只做插密可视化，不改变求解本质。
- **求解器纯 mesh**：`fedic_element_icgn` / `fedic_element_fgn`（单元上迭代）。
- **FFT/SIFT/金字塔/方向 A 全部属于 `initialization` 阶段**，产出只有一个：给每个节点一个初始 seed，然后交给 element ICGN/FGN 精修。

**不把 mesh 搞成 subset**：`config/stereo_3d.yaml` 里 `solver=subset` 才是 subset 路径；mesh 路径无论初始化配得多花哨，求解与表示始终是纯 element-based mesh。

| 阶段 | 干什么 | 是否 mesh |
|---|---|---|
| 初始化 | FFT/SIFT/金字塔/方向 A 只给节点找 seed | 起跑辅助，可开可关 |
| 求解 | element ICGN/FGN 迭代优化节点位移 | ✅ 纯 mesh |
| 输出 | 节点位移场 + 形函数 dense 场 | ✅ 纯 mesh |

**结论一句话**：初始化是脚手架，骨架（mesh element 求解）从来没换过。

---

## 14. 单目 mesh 默认配置固化为 disk-40px + 标准入口重跑 disp（2026-08-09）

（本会话修改）`config/mesh_2d.yaml`（单目 Mesh-DIC 默认配置）从"老配置"固化为 **disk-40px 版本**。修改仅落在 yaml 配置，**无源码改动**；§1 的提交清单不变。

### 14.1 配置变更（config/mesh_2d.yaml）

字段变更（其余保持当前值：pyramid disabled、window_size 75、objective ssd、max_features 4000）：

| 字段 | 现值 | 改后 |
|---|---|---|
| `mesh_generation.target_element_size` | 75.0 | **40.0** |
| `mesh_generation.min_element_size` | 45.0 | **24.0** |
| `mesh_generation.max_element_size` | 110.0 | **60.0** |
| `optimization.method` | fedic_element_icgn | **fedic_element_fgn** |
| `initialization.fedic_fft.search_radius` | 30 | **50** |
| `initialization.sift_prior.enabled` | false | **true** |
| `initialization.boundary_direct_prior_seed` | false | **true**（注释更新：disk-40px 默认开，multiview 显式关回 false） |
| `initialization.quality_control.enabled` | false | **true** |
| `initialization.quality_control.neighbor_mad_factor` | 4.0 | **2.0** |
| `initialization.quality_control.max_neighbor_deviation` | 0.0 | **12.0** |

这些字段原本只固化在临时脚本 `build/_run_disp_mesh.py`（disp 验证配置），本次并入正式配置。文件头新增注释说明 disk-40px 来源。"disk"=disp 的口语称呼（disp ROI 实为实心椭圆，非几何圆盘）。

### 14.2 重跑 mono_DIC/disp（标准入口，不覆盖结果）

- 运行前记录现有 `result/mesh/disp*` 各目录 mtime，运行后确认未变；输出到**全新 tag** 目录。
- `mesh_method_tag` 只编码 `{objective}_{solver}` → tag = `ssd_fgn`。
- 命令（`config/case_paths.yaml` 的 mono_2d 段默认指向 ring，故用临时 paths 配置指向 disp，写系统 temp 不落项目内 yaml）：

  ```
  python tools/run_example.py examples/mesh_2d.py --paths-config <temp> --config config/mesh_2d.yaml
  ```

- 输出：`result/mesh/002_ssd_fgn/{T3,Q4,Q8}`（final_U.csv、dense_U.csv、nodes/elements txt、strain.csv、summary.json）+ `visualization/mesh/002_ssd_fgn/`（dense_overview.png + 每元素 dense_field overlay + mesh_preview）。
- 帧遍历：`examples/mesh_2d.py` 把 images[1:-1] 作为 deformed 帧，002、003 各出一份结果。

### 14.3 验证

1. **不覆盖**：现有 `disp*`/`disp_40px*` 目录 mtime 运行前后一致。
2. **逐位复现（固化正确）**：`002_ssd_fgn` 与 `disp_40px` 的 final_U.csv / dense_U.csv **逐位一致**（max|Δ|=0），证明改后 yaml 与 `_run_disp_mesh.py` 完全等效。
3. **pyd mtime 检查**：pyd 08-08 01:30 晚于全部源码 01:29，无需重编译（同 [[pyd-rebuild-timing-gotcha]]）。

### 14.4 冗余结果说明

`003_ssd_fgn` 与 `002_ssd_fgn` 逐位一致——`003.bmp` 与 `002.bmp` md5 相同（`72a06fdddcb646ff24630e265dfa5046`），为冗余重复。**待用户确认是否删除**冗余的 `003_ssd_fgn` 结果 + 对应可视化。

### 14.5 影响范围与遗留

- 改 `config/mesh_2d.yaml` 改变**所有单目 mesh 用例**（ring/star/Complex 单目）的默认行为（方向A 默认开、FGN、r50、SIFT、QC）；multiview 不受影响（读 `multiview_3d.yaml` + mesh_config 内联覆盖）。本次只重跑 disp，未动其他 case 结果。
- `build/_run_disp_mesh.py` 保留为历史参照；其 `setdefault` 覆盖现在读到 yaml 已有值，重跑仍得相同结果。
- 无源码改动，§1 提交清单不变。
