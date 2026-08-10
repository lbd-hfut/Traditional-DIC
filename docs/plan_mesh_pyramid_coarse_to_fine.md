# 方案 C 实施点：mesh 真·金字塔 coarse-to-fine 初始化

> 状态：**方案设计，未实施**。本文只描述实施点，不涉及代码改动。
> 目的：解决 mesh FFT 初始化"每节点从 (0,0) 盲搜大半径"的问题，使 mesh 在
> stereo / CylinderDIC / ComplexCylinderDIC（乃至未来新用例）上无需手工调大
> `search_radius` 即可覆盖大视差（-164 / +228 / +216 px，变化范围到 ±290）。

---

## 1. 背景与现状

当前 mesh 初始化链路（只读 `search_radius` + `window_size` 两个旋钮）：

- `src/mesh/mesh_dic.cpp:492-529` "Displacement init"：**逐节点独立**调用
  `estimate_fedic_fft_initial_displacement(reference, deformed, pt, search_radius, window_size)`，
  每节点以 **(0,0) 为中心**在 ±search_radius 内做局部 NCC（FFT）盲搜；
  之后 `apply_mesh_initialization_quality_control` + `fill_missing_nodal_initialization` 做质控与邻域填充。
- `src/mesh/initialization/fedic_fft_initializer.cpp:58-150`：核心实现，
  第 98-101 行搜索窗中心**硬编码**为节点坐标 `(x0, y0)`，即位移零点。
- `include/dic/mesh/initialization/fedic_fft_initializer.hpp:19-24`：接口
  `estimate_fedic_fft_initial_displacement(Image, Image, point, search_radius, window_size)`。

**问题**：搜索半径必须预先硬编码覆盖最大视差。`r30` 在 stereo 全错、`r300` 才活；
但 `r300` 是全局单值、计算量大、且 ComplexCylinder 实测视差到 ±290 已逼近边界。

**方案 C 思想**：金字塔 coarse-to-fine。粗层图小，大视差被压缩成小像素位移，
在粗层用**小半径**即可完成覆盖全图范围的粗搜；逐层上采样 + 小窗精修，
最终得到每节点粗位移场，作为 FFT 初始化的**搜索中心**（而非 (0,0)）。
不依赖标定（ComplexCylinder 的病根）、不依赖硬编码半径（自动适配任意视差范围）。

---

## 2. 实施点总览

| 编号 | 文件 | 内容 | 核心性 |
|---|---|---|---|
| EP1 | `fedic_fft_initializer.hpp/.cpp` | 给 FFT 初始化加"搜索中心偏移"参数 | ★ 核心，全部依赖它 |
| EP2 | 新增 `pyramid_initializer.{hpp,cpp}` | 金字塔位移场估计模块 | ★ 核心 |
| EP3 | `src/mesh/mesh_dic.cpp` | 接入金字塔阶段，初始化用 seed | 接线 |
| EP4 | `include/dic/mesh/mesh_config.hpp` | `MeshConfig` 新增 `pyramid_initialization` 结构 | 数据 |
| EP5 | `bindings/python/bind_mesh.cpp` | 配置解析新增 pyramid 字段 | 数据 |
| EP6 | `python/traditional_dic/config.py` | `initialization.pyramid` 透传 | 数据 |
| EP7 | `config/*.yaml` | 配置示例 | 数据 |

---

## 3. 各实施点详情

### EP1 ★  FFT 初始化支持搜索中心偏移

**接口**（`include/dic/mesh/initialization/fedic_fft_initializer.hpp:19-24`）：
给现有函数追加一个**带默认值**的参数，向后兼容：

```cpp
FEDICFFTInitialDisplacement estimate_fedic_fft_initial_displacement(
    const Image& reference,
    const Image& deformed,
    const Eigen::Vector2d& point,
    int search_radius,
    int window_size,
    const Eigen::Vector2d& initial_offset = Eigen::Vector2d::Zero());  // 新增
```

**实现**（`src/mesh/initialization/fedic_fft_initializer.cpp:58-150`）：
- 第 98-101 行搜索窗的取窗原点从 `(x0, y0)` 改为 `(x0 + u0, y0 + v0)`（`u0,v0 = initial_offset`）；
  `reference_patch` 仍取节点附近 `window_size` 区域不变。
- 第 80-83 行的边界检查相应改为：搜索窗 `(x0+u0) ± search_radius` 必须在图内。
- 峰值解释不变：`matchTemplate` 找到的峰值仍是"相对参考 patch 的位移"，
  最终初始位移 = `initial_offset + 峰位置`。

**注意**：`mesh_dic.cpp:508-515` 的 `mirror_boundary_fallback` 分支同样要传 `initial_offset`。

---

### EP2 ★  新增金字塔位移场估计模块

**新文件**：`src/mesh/initialization/pyramid_initializer.{hpp,cpp}`
（也可并入 `fedic_fft_initializer.cpp`，建议独立便于单测）。

**配置结构**（供 EP4 复用）：

```cpp
struct PyramidInitializationConfig {
    bool enabled{false};              // 默认关，验证后再开
    int num_levels{4};                // 层数（会按图像尺寸自适应截断）
    double scale_factor{0.5};         // 每层降采样比例
    int coarse_search_radius{16};     // 最粗层（粗层坐标）搜索半径
    int refinement_radius{4};         // 每层细化半径
    int window_size{75};              // 复用 fedic window_size
};
```

**接口**：

```cpp
// 返回逐节点粗位移（全分辨率坐标）；valid=false 表示该节点金字塔失败（回退盲搜）。
std::vector<InitialDisplacement> estimate_pyramid_initial_displacements(
    const Image& reference, const Image& deformed,
    const std::vector<Eigen::Vector2d>& points,
    const PyramidInitializationConfig& cfg);
```

**实现逻辑**：
1. **建金字塔**：对 reference/deformed 逐级 `cv::resize`（或 Image 降采样）生成高斯金字塔；
   层 k 尺寸 = 原图 × `scale_factor^k`。
2. **坐标映射**：节点坐标缩放到各层 `pt_k = pt × scale_factor^k`。
3. **最粗层**：每节点在粗层用 EP1 函数（`initial_offset=Zero`）做 ±`coarse_search_radius` 粗搜。
4. **上采样 + 细化**：粗位移 × `scale_factor`（放大回细层）作为下一层 `initial_offset`，
   再在 ±`refinement_radius` 小窗精修；逐层到全分辨率。
5. **输出**：全分辨率粗位移场 + valid 掩码。

**覆盖保证（关键设计）**：
- 最粗层可覆盖的真实位移范围 ≈ `coarse_search_radius / scale_factor^num_levels`。
  例：scale=0.5、4 层 → 最粗层缩 1/16；`coarse_search_radius=16` 覆盖 ±16×16=±256px，
  stereo（-164）、Cylinder（+228）、ComplexCylinder（+216）全部落入。
- **层数自适应**：`num_levels` 截断使最粗层短边 ≥ `max(window_size, 32)`，防过粗失去纹理。
- **失败回退**：某节点粗层无有效峰 → 该节点 `valid=false`，由 EP3 回退到全分辨率盲搜。

---

### EP3  `mesh_dic.cpp` 接入

**位置**：`src/mesh/mesh_dic.cpp` 第 492 行 "Displacement init" 处。

伪流程（替换现第 499-526 行的逐节点盲搜为"seed 引导的小窗搜索"）：

```cpp
// 7a. 可选：金字塔粗配准 → 逐节点 seed
std::vector<InitialDisplacement> pyramid_seed;
if (config_.pyramid_initialization.enabled) {
    pyramid_seed = estimate_pyramid_initial_displacements(
        reference, deformed, node_points, config_.pyramid_initialization);
}
// 7b. 逐节点：以 seed 为中心做 FFT 小窗细化；seed 无效则 (0,0) 盲搜兜底
for (int i = 0; i < n_nodes; ++i) {
    Eigen::Vector2d offset = (pyramid_seed 有效) ? (u_i, v_i) : Zero;
    estimate_fedic_fft_initial_displacement(
        reference, deformed, pt,
        config_.fedic_fft_initialization.search_radius,   // 保持默认 30 即可
        config_.fedic_fft_initialization.window_size,
        /*initial_offset=*/offset);
}
// 7c. 现有 quality_control + fill_missing_nodal_initialization 原样保留（527-529 行）
```

要点：
- `search_radius` 可回退到默认 30（金字塔负责大位移，FFT 只做小窗精修）。
- 现有 `apply_mesh_initialization_quality_control` / `fill_missing_nodal_initialization`
  继续兜底，保证金字塔失败节点不拖累全局。
- 求解器（ICGN/FGN）、准则（SSD/ZNSSD）、mesh 生成**一律不动**。

---

### EP4  `MeshConfig` 新增字段

**文件**：`include/dic/mesh/mesh_config.hpp:49-77`。

在 `fedic_fft_initialization{}`（61-65 行）附近新增：

```cpp
struct PyramidInitializationConfig {
    bool enabled{false};
    int num_levels{4};
    double scale_factor{0.5};
    int coarse_search_radius{16};
    int refinement_radius{4};
    int window_size{75};
} pyramid_initialization{};
```

可选：`MeshNodalInitializationMethod`（35-37 行）增加枚举 `PYRAMID_FFT`，
便于将来 `initialization.method` 显式选择。

---

### EP5  `bind_mesh.cpp` 配置解析

**文件**：`bindings/python/bind_mesh.cpp:279-316`。

在 `init.contains("fedic_fft")` 块（287-296 行）之后新增：

```cpp
if (init.contains("pyramid")) {
    py::dict pyr = py::cast<py::dict>(init["pyramid"]);
    cfg.pyramid_initialization.enabled =
        get_bool(pyr, "enabled", cfg.pyramid_initialization.enabled);
    cfg.pyramid_initialization.num_levels =
        get_int(pyr, "num_levels", cfg.pyramid_initialization.num_levels);
    cfg.pyramid_initialization.scale_factor =
        get_double(pyr, "scale_factor", cfg.pyramid_initialization.scale_factor);
    cfg.pyramid_initialization.coarse_search_radius =
        get_int(pyr, "coarse_search_radius", cfg.pyramid_initialization.coarse_search_radius);
    cfg.pyramid_initialization.refinement_radius =
        get_int(pyr, "refinement_radius", cfg.pyramid_initialization.refinement_radius);
    cfg.pyramid_initialization.window_size =
        get_int(pyr, "window_size", cfg.pyramid_initialization.window_size);
}
```

---

### EP6  `config.py` 透传

**文件**：`python/traditional_dic/config.py:59-63`。

`out["initialization"]` 下补一行：

```python
out["initialization"] = {
    "method": initialization.get("method", "fedic_fft"),
    "fedic_fft": initialization.get("fedic_fft", {}),
    "pyramid": initialization.get("pyramid", {}),        # 新增
    "quality_control": initialization.get("quality_control", {}),
}
```

（`out["mesh"]["search_radius"]` 第 56 行默认 30 保留作兜底，不改。）

---

### EP7  YAML 配置示例

在 `config/mesh_2d.yaml`（及 stereo/多目派生配置）的 `initialization` 下加：

```yaml
initialization:
  method: fedic_fft
  fedic_fft:
    window_size: 75
    search_radius: 30          # 可保持默认 30，大位移交给金字塔
  pyramid:
    enabled: true
    num_levels: 4
    scale_factor: 0.5
    coarse_search_radius: 16
    refinement_radius: 4
    window_size: 75
  quality_control:
    enabled: true
    # ... 现有质控参数原样
```

---

## 4. 验证与验收标准

**三用例 + 回归**（均以 `search_radius=30` 默认，不回退 300）：

| 用例 | 期望 |
|---|---|
| stereo plate_center_load | T3/Q4/Q8 100% valid、参考视差 ~-164 px |
| CylinderDIC | 参考视差 ~+228 px、valid 率 ≥ r300 基线 |
| ComplexCylinderDIC | 参考视差 ~+216 px、valid 率 ≥ r300 基线（3209 点、Q8 75-83%） |
| mono ring/star（回归） | mesh 结果不退化（mag_mean 与现有记录一致） |

**数值对比**：与 r300 结果比 `|U| mean`、表面/位移误差（`tools/validate_complex_cylinder.py`）应一致或更好。

**性能**：总耗时 ≤ r300（粗层窗口小，搜得便宜）。

**单测**（`tests/`）：合成已知平移/视差图 → 金字塔输出位移场 ≈ 真值；
含无纹理、重复纹理、边界节点用例。

---

## 5. 风险与回退

| 风险 | 缓解 |
|---|---|
| 无纹理/纯色 → 粗层无唯一峰 | 节点 `valid=false` → 回退全分辨率盲搜 + 现有质控/邻域填充兜底 |
| 重复/周期纹理 → 粗层锁周期性错峰 | 复用 `peak_to_entropy` / `peak_to_correlation_energy` 指标过滤 |
| 强非刚性/大旋转 → 粗层错配 | `refinement_radius` 逐层修正 + 质控 `max_neighbor_deviation` |
| 过粗丢纹理 | 层数按 `min(h,w) ≥ max(window,32)` 自适应截断 |
| 边界/mirror padding | 复用现有 `mirror_boundary_fallback`（EP1 需把 offset 传入该分支） |
| 默认行为回归 | `pyramid.enabled=false` 默认关，逐用例开启验证 |

---

## 6. 为什么这个方案能覆盖三个例子（依据）

1. **不依赖标定**：金字塔是纯图像数据驱动，ComplexCylinder 的自标定 bas-relief 退化对
   粗配准无影响（方案 D 在此失效、C 不失效）。
2. **不依赖硬编码半径**：粗层把 ±300px 视差压缩为 ±18px 量级，小半径即可全范围覆盖，
   逐层细化自动适配任何视差范围，无需按用例手调。
3. **与 subset 对齐思路**：subset 之所以不用手调半径，是靠金字塔 + 邻域传播 +
   全图粗搜——方案 C 就是把这一层补到 mesh 的 FFT 初始化上。

---

## 7. 遗留 / 待确认

- `pyramid.enabled` 默认开还是关（建议先关，逐用例验证后开）。
- 金字塔层是否参与**时域（temporal）匹配**（位移小，收益小但无害），还是只做立体（disparity）主链路。
- 是否需要在 Python 侧暴露层数/半径的按用例覆盖（类似 `build/*.yaml` 派生配置机制）。
