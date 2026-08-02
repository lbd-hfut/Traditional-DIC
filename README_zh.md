# Traditional-DIC 中文说明

Traditional-DIC 是一个基于 C++17 与 Python 的传统数字图像相关项目，支持单目二维 Subset-DIC、二维 Mesh-DIC、双目 3D-DIC 与多目 3D-DIC。完整英文技术说明见 [README.md](README.md)；本文件提供中文使用、编译和引用入口。

## 功能与结构

- Subset-DIC：圆形/ROI 截断子区、镜像填充、整数搜索与 SIFT 初始化、种子选择、ncorr 风格可靠性传播、一二阶形函数、SSD/ZNSSD、ICGN/FGN。
- Mesh-DIC：T3/Q4/Q8 单元、ROI 自动网格或手工网格、FE-DIC FFT 节点初始化、稀疏全局求解、正则化、节点场与稠密场。
- 三维流程：棋盘格标定、双目三角化和三维位移；多目自标定、尺度恢复、相机对选择、两两重建和曲面拼接。

```text
输入图像/ROI -> 预处理、B-spline 插值、初始化
             -> Subset-DIC 或 Mesh-DIC -> 相机几何
             -> 双目/多目重建 -> 三维位移与可视化
```

`include/dic/` 为 C++ 公共接口，`src/` 为实现，`bindings/` 与 `python/` 为 Python 绑定/流程，`examples/` 为主程序，`config/` 为 YAML，`case/` 为案例输入和生成结果。

## 环境与编译

必需依赖：CMake 3.16+、C++17 编译器、Eigen3、yaml-cpp、Python 3、NumPy、Pillow、PyYAML。OpenCV（图像/SIFT/标定）、Ceres、pybind11、SciPy、Matplotlib 视所用功能安装。

示例依赖 Python 绑定。先激活依赖所在 Conda 环境，再在 MSVC x64 开发环境中配置并编译：

```powershell
conda activate <environment>
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && cmake -S . -B build -DTRADITIONAL_DIC_BUILD_PYTHON=ON'
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && cmake --build build --config Release'
python -c "import traditional_dic; print(\"traditional_dic import succeeded\")"
```

后续始终复用 `build/`。若 CMake 找不到依赖，先激活包含依赖的环境，或设置 `CMAKE_PREFIX_PATH`/`OpenCV_DIR`；Python、OpenCV 与编译器架构必须一致。

## 配置规则

`subset_2d.yaml`、`mesh_2d.yaml`、`calibration.yaml`、`stereo_3d.yaml`、`multiview_3d.yaml` 只包含算法参数。案例根目录、图像、ROI、标定板目录和输出根目录全部写在 [config/case_paths.yaml](config/case_paths.yaml)，且除 `case_root` 外均相对案例根目录。

- 单目：`images_dir` 排序后，首图是参考图、末图是 ROI，所有中间图按变形序列逐张求解。
- 双目：左右目录首图是参考图、末图是变形图；左右标定板目录单独配置。
- 多目 `roi.mode: auto`：每个相机目录首图参考、末图变形，程序自动生成相机对 ROI。
- 多目 `roi.mode: last_image`：首图参考、倒数第二张变形、末图为该相机 ROI；其作为主相机时，ROI 用于时序和视差 DIC。

`output.result_root` 与 `output.visualization_root` 默认分别是 `result` 和 `visualization`。数值 CSV、JSON、网格和重建数据写入 `result/`；图像写入 `visualization/`。

## 运行主程序

所有主程序支持 `--paths-config`，以替换案例 YAML 而不是修改算法 YAML。

```powershell
python examples\subset_2d.py --paths-config config\case_paths.yaml --config config\subset_2d.yaml
python examples\mesh_2d.py --paths-config config\case_paths.yaml --config config\mesh_2d.yaml --element all
python examples\stereo_3d.py --paths-config config\case_paths.yaml --stereo-config config\stereo_3d.yaml --calibration-config config\calibration.yaml --solver subset --compute-fields
python examples\multiview_3d.py --paths-config config\case_paths.yaml --config config\multiview_3d.yaml --solver subset
```

Mesh 单元选择为 `T3`、`Q4`、`Q8` 或 `all`。双目可用 `--skip-calibration` 复用相机参数。多目可用 `--solver subset|mesh|both` 选择分支，`--resume` 复用已完成两两二维场。

## 数值与输出约定

SSD/ZNSSD 越小越好，ZNCC 越大越好；ZNCC 用于初始化/质量评分而不是当前 Gauss-Newton 目标。B-spline 支持 1、3、5 阶。Subset 的 `reliability_propagation.spacing` 是 ncorr 间隔数，实际全分辨率步长为 `spacing + 1`，不是下采样比例。

二维结果包含 `x,y,u,v,correlation,valid`；Mesh 额外包含节点和稠密插值场。三维结果包含参考/变形三维坐标、`Ux,Uy,Uz,Umag`、重投影误差和有效性。双目用 `L0->R0`、`L0->L1`、`L0->R1` 三个二维场重建 `X0`、`X1` 和 `U3D=X1-X0`。

## 致谢

本项目在算法思想、数值约定、验证参考和流程设计上受益于以下开源工作；使用相关方法发表研究时，请引用其原始论文和软件：

- [justinblaber/ncorr_2D_matlab](https://github.com/justinblaber/ncorr_2D_matlab)
- [YangMechanicsGroupUTAustin/2D_FE_Global_DIC](https://github.com/YangMechanicsGroupUTAustin/2D_FE_Global_DIC)
- [SolavLab/DuoDIC](https://github.com/SolavLab/DuoDIC)
- [MultiDIC/MultiDIC](https://github.com/MultiDIC/MultiDIC)
- [colmap/colmap](https://github.com/colmap/colmap)

## 引用本项目

发表使用 Traditional-DIC 的成果时，请引用实际使用版本或 commit：

```bibtex
@software{leebda_traditional_dic_2026,
  author = {LeeBDa},
  title = {Traditional-DIC: Traditional Digital Image Correlation Workflows},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/lbd-hfut/Traditional-DIC},
  note = {Version <used-version>}
}
```

请将 `<used-version>` 替换为 release tag 或 commit hash；发布归档 DOI 后应优先引用 DOI。

## 开发与许可

Subset-DIC 与 Mesh-DIC 的求解器修改应保持独立，除非确有共享数值模块需要变动。不要把案例路径写回算法 YAML。`build/`、`result/`、`visualization/`、Python 缓存和编译产物均为生成内容。项目采用 [MIT License](LICENSE)。
