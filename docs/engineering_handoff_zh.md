# Traditional-DIC Windows 交接与工程约定

本文档用于把 `develop` 分支交给同事进行测试、调试和升级。目标是让工程代码、测试脚本、临时工具、编译产物和日志彼此隔离，避免测试函数或调试脚本混入核心代码。

## 1. Windows 编译环境

推荐平台：

- Windows 10/11 x64
- Visual Studio 2022 或 Visual Studio Build Tools 2022，安装 `Desktop development with C++`
- MSVC x64 编译器和 Windows SDK
- CMake 3.16 或更高版本
- Git
- Python 3.9 或更高版本，推荐 Conda 环境
- Ninja 可选；也可以直接使用 Visual Studio CMake Generator

主要 C++ 依赖：

- Eigen3，项目会在找不到本地包时通过 CMake `FetchContent` 获取
- yaml-cpp，项目会在找不到本地包时通过 CMake `FetchContent` 获取
- OpenCV，可选但图像读取、SIFT、标定等功能需要
- Ceres Solver，可选但部分优化求解能力需要
- pybind11，构建 Python 绑定时需要

主要 Python 依赖：

- numpy
- Pillow
- PyYAML
- scipy
- matplotlib
- scikit-build-core
- pybind11

推荐在 Conda 环境中准备 Python 侧依赖，并确保 Python、OpenCV、Ceres、MSVC 都是 x64 架构。

## 2. Windows 编译命令

在仓库根目录执行。先进入 Conda 环境，再进入 MSVC x64 开发环境。

```powershell
conda activate <environment>
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && cmake -S . -B build -DTRADITIONAL_DIC_BUILD_PYTHON=ON'
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && cmake --build build --config Release'
python -c "import traditional_dic; print('traditional_dic import succeeded')"
```

如果使用 Visual Studio Build Tools，`VsDevCmd.bat` 可能位于：

```text
C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat
```

如果 CMake 找不到 OpenCV 或 Ceres，可以设置：

```powershell
$env:CMAKE_PREFIX_PATH="<dependency-prefix>"
$env:OpenCV_DIR="<opencv-build-or-install>\x64\vc17\lib"
```

## 3. 编译文件和生成文件位置

所有编译产物必须放在根目录下的 build 类目录中：

- `build/`：默认构建目录
- `build_debug/`：Debug 构建目录
- `build_relwithdeb/`：RelWithDebInfo 构建目录
- `build_asan/`：带 sanitizer 或专项诊断的构建目录

不要把 CMake 生成的 `.sln`、`.vcxproj`、`CMakeFiles/`、`CMakeCache.txt`、`.dll`、`.lib`、`.pdb` 等文件提交到源码目录。根目录 `.gitignore` 已经忽略这些产物。

## 4. 测试、工具、日志目录约定

- `tests/`：只放测试代码、测试脚本、测试数据说明和测试夹具。新增单元测试、回归测试、临时验证脚本都放这里。
- `tools/`：只放独立工具脚本，例如数据转换、批处理、结果检查、可视化辅助、环境检查脚本。
- `logs/`：只放运行日志、调试日志、测试日志和临时诊断输出。

禁止事项：

- 不要把测试函数写进 `src/`、`include/` 或 `bindings/`。
- 不要让生产代码依赖 `tests/` 或 `tools/`。
- 不要让 `examples/` 反向依赖 `tests/` 中的脚本。
- 不要把日志写到仓库根目录或源码目录。
- 不要把临时调试脚本放到 `src/`、`include/`、`python/traditional_dic/`、`bindings/`。

允许事项：

- `tests/` 可以调用已安装或已构建的包和公开 API。
- `tools/` 可以读取 `config/`、`case/`、`result/`、`visualization/` 等数据，但不应成为生产运行链路的一部分。
- 需要长期保留的工具应有简短 README 或脚本头部说明。

## 5. 源码目录边界

- `include/`：C++ 公共头文件和稳定接口。
- `src/`：C++ 核心实现。
- `bindings/`：Python 绑定实现。
- `python/`：Python 包代码。
- `examples/`：面向用户的示例入口，不放临时测试逻辑。
- `benchmarks/`：性能基准测试，不替代功能测试。
- `config/`：算法参数和案例路径配置。
- `case/`：案例输入、案例配置和案例生成结果。
- `docs/`：用户文档、交接文档和工程规范。
- `tests/`：测试专用内容。
- `tools/`：独立辅助工具。
- `logs/`：运行日志和调试输出。

核心原则：生产代码只能依赖生产代码和正式第三方依赖；测试和工具可以依赖生产代码，但生产代码不能反向依赖测试或工具。

## 6. 给智能体的协作规则

每次让 Claude、DeepCode、Codex、KimiCode 或其他智能体修改本项目时，先让它读取根目录的 `AGENTS.md`，再读取本文档。

智能体必须遵守：

- 改动前先确认当前分支和工作区状态。
- 不要删除或回滚用户已有改动。
- 新增测试放 `tests/`。
- 新增临时或独立工具放 `tools/`。
- 运行产生的日志放 `logs/`。
- 编译产物只放 `build*` 目录。
- 修改核心算法时保持 `include/`、`src/`、`bindings/`、`python/` 的职责边界。
- 完成后说明修改了哪些文件、如何编译、如何验证。

