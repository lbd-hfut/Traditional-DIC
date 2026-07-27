# MeshDIC 融合指导文档

> 将此文档复制给新对话，配合访问本地项目路径即可开始融合工作。

---

## 1. 背景

- **Traditional-DIC** (`C:\02Project\Research\Traditional-DIC`)：我们刚刚完成 Subset-DIC 模块的完整实现（ICGN + 整数搜索 + 种子选择 + RG-DIC 可靠性传播 + Python 绑定）。
- **MeshDIC** (`C:\02Project\Research\MeshDIC`)：独立的全局 FE-DIC 项目，包含 B样条插值、全局-局部映射(G2L)、形函数、刚度组装、Global ICGN / Forward-GN 求解器、应变计算。

目标：**将 MeshDIC 的核心能力融合进 Traditional-DIC 的 mesh 模块框架下**，使其与现有 Subset-DIC 共享基础模块（图像、插值、相关准则、初始化），但不修改 Subset 模块代码。

---

## 2. 关键原则

1. **不修改 Subset 模块**：`include/dic/subset/`、`src/subset/`、`tests/subset/` 保持不动。
2. **不修改基础模块**：`include/dic/core/`、`include/dic/interpolation/`、`include/dic/correlation/`、`include/dic/initialization/` 保持不动（可新增方法，不改变现有接口）。
3. **Mesh-DIC 作为独立子模块**：代码放在 `src/mesh/` 和 `include/dic/mesh/` 下。
4. **Python 绑定通过 pybind11**：`bindings/python/bind_mesh.cpp`。

---

## 3. MeshDIC 核心架构概览

```
MeshDIC/src/cpp/
  bspline.h/cpp       -- Quintic B-spline 引擎（与 dic/interpolation/bspline 重叠）
  local_icgn.h/cpp    -- 局部 ICGN 求解器（用于节点位移初始化）
  shape_func.h/cpp    -- Q4/Q8/T3 形函数
  global2local.h/cpp  -- 全局像素 -> 单元自然坐标映射 (G2L)
  stiffness.h/cpp     -- FE 刚度矩阵组装 + 残差 + Global ICGN/Forward-GN
  strain.h/cpp        -- 应变计算
  mesh_dic.h/cpp      -- 顶层入口（占位）
  bindings.cpp        -- pybind11 绑定（依赖 OpenCV）

MeshDIC/src/mesh_dic/
  solver.py           -- Python 侧求解流程编排
  config.py           -- 配置 dataclass
  mesh_gen.py         -- Gmsh 网格生成 / 文件 I/O
  field.py            -- 像素位移场插值
```

### MeshDIC 求解流程

```
ROI Mask -> Gmsh 生成网格 -> 节点/单元/Inform 文件
                                       |
Ref/Def 图像 -> B样条系数 -> B样条梯度(fx,fy) -> 全局->局部映射(G2L)
                                       |
                      每节点整数搜索+局部ICGN -> U_init
                                       |
                      组装 FE 刚度矩阵(Hessian)
                                       |
                      Global ICGN / Forward-GN 迭代求解 U
                                       |
                      应变计算 + 像素位移场插值 -> 结果
```

---

## 4. Traditional-DIC 现有 mesh 模块状态

`include/dic/mesh/` 目录下已有框架头文件，`src/mesh/` 下大部分是占位实现：

| 文件 | 状态 | 对应 MeshDIC |
|------|------|-------------|
| `element/q4.hpp/.cpp` | **占位** | shape_func (Q4) |
| `element/q8.hpp/.cpp` | **占位** | shape_func (Q8) |
| `element/t3.hpp/.cpp` | **占位** | shape_func (T3) |
| `coordinate/global_to_natural.hpp/.cpp` | **占位** | global2local |
| `solver/assembler.hpp/.cpp` | **占位** | stiffness (assemble) |
| `solver/global_gauss_newton.hpp/.cpp` | **占位** | stiffness (global_forward_gn) |
| `solver/global_icgn.hpp/.cpp` | **占位** | stiffness (global_icgn) |

`src/mesh/` 下的 .cpp 几乎全是 `(void)param; return {};`，`tests/mesh/` 下全是 `TEST(XXX, Placeholder) { SUCCEED(); }`。

---

## 5. 融合映射方案

### 5.1 B样条引擎 -> 复用现有模块

MeshDIC 的 `bspline.h/cpp` 与 Traditional-DIC 的 `dic/interpolation/bspline.hpp/cpp` 功能重叠。**直接使用 `BSplineInterpolator`** 做变形图像子像素插值，不搬 `BsplineEngine`。

### 5.2 形函数 -> 直接替换占位

将 `shape_func.h/cpp` 的 Q4/Q8/T3 实现复制到：
- `src/mesh/element/q4.cpp`
- `src/mesh/element/q8.cpp`
- `src/mesh/element/t3.cpp`

### 5.3 全局->局部映射 -> 替换占位

将 `global2local.h/cpp` 复制到：
- `src/mesh/coordinate/global_to_natural.cpp`

### 5.4 刚度组装+全局求解器 -> 替换占位

将 `stiffness.h/cpp` 的组装逻辑和求解器复制到：
- `src/mesh/solver/assembler.cpp`（刚度矩阵+残差组装）
- `src/mesh/solver/global_icgn.cpp`（Global ICGN）
- `src/mesh/solver/global_gauss_newton.cpp`（Global Forward-GN）

新增：
- `src/mesh/solver/local_icgn.cpp`（节点位移初始化用）

### 5.5 应变计算 -> 新增

- `src/mesh/postprocess/strain.cpp`（或整合到 `src/postprocess/strain_2d.cpp`）

### 5.6 网格生成 -> Python 侧保留

`mesh_gen.py` 依赖 `gmsh`/`pygmsh`/`meshio`，保留在 Python 侧，不移植 C++。

### 5.7 Python 绑定

`bindings/python/bind_mesh.cpp` 重写，暴露：ElementType枚举、形函数、G2L映射、刚度组装、Global ICGN/Forward-GN、应变计算。

Python API `python/traditional_dic/mesh.py` 重写，参考 `subset.py` 的模式：

```python
from traditional_dic import mesh

result = mesh.solve(
    ref_img, def_img,
    roi_mask=roi_path,
    mesh_size=30.0, element_type="Q8",
    method="forward_gn", alpha=1.0,
    max_iter=10, tol=1e-3,
)
```

---

## 6. 实施步骤

| 步骤 | 内容 | 优先级 |
|------|------|--------|
| 1 | 形函数 + G2L 映射（其他模块依赖） | 最高 |
| 2 | B样条对接：确保 BSplineInterpolator 可用 | 高 |
| 3 | 局部 ICGN + 刚度组装 | 高 |
| 4 | Global 求解器 (ICGN + Forward-GN) | 高 |
| 5 | 应变计算 | 中 |
| 6 | Python 绑定 + Python API (mesh.py) | 中 |
| 7 | 集成测试（ring 算例） | 最后 |

---

## 7. 重要注意事项

1. **命名空间统一为 `dic`**（MeshDIC 原来用 `meshdic`）。
2. **Eigen 依赖已满足**（SparseMatrix 需要 `<Eigen/Sparse>`）。
3. **OpenCV 可选**：如果保留 MeshDIC 的 `cv::Mat` 用法，需在 `#ifdef TRADITIONAL_DIC_HAS_OPENCV` 内。优先用 Traditional-DIC 的 `Image` 类替代。
4. **不要改 Subset 模块**：`src/subset/`、`include/dic/subset/`、`tests/subset/` 不动。
5. **不要改基础模块**：`Image`、`Mask`、`Interpolator` 接口保持稳定。
6. **网格生成可选**：gmsh/pygmsh/meshio 是 Python 侧的可选依赖，用户可以预生成网格文件。
