# DrakeUni

[English](README.md)

DrakeUni 是 UniLab Drake 后端使用的实验性批量仿真运行时。它在 Python
层维护 UniLab 的 MJCF contract，并通过可选的 C++/pybind11 扩展调用本地
Drake 安装执行批量物理仿真。

当前范围：

- 解析并物化 UniLab MJCF 模型 contract；
- 控制批量环境数和 worker 线程数；
- 提供与 UniLab Drake 适配器对接的 reset、step、状态和传感器 API。

这不是完整的 Drake backend。任务语义（基体定义、观测、奖励、训练和
rollout 调度）由 UniLab backend/task 层负责。

## 目录结构

```text
src/drakeuni/runtime/       # Python contract、materializer 和运行时
src/drakeuni/compiled/      # C++ Drake 批量执行器源码和本地扩展
scripts/build_drake_batch.py # 针对本地 Drake 前缀编译扩展
tests/                      # parser/materialization 测试
```

## 安装与开发

项目使用 uv 管理开发环境：

```bash
make sync
make install
make check
```

原生扩展需要本地 Drake C++ 安装前缀（包含 `include/drake`、
`include/pybind11` 和共享库）：

```bash
make build DRAKE_HOME=/path/to/drake/install
make test-no-sync
```

扩展会生成在 `src/drakeuni/compiled/_drake_env_pool*`，属于本地构建产物，
不会提交到 Git。CI 默认执行跨平台 Python lint/test 和 sdist 构建，不假定
runner 上存在机器相关的 Drake 前缀。

## 运行时 API

```python
from drakeuni.runtime import DrakeBatchConfig, create_runtime

runtime = create_runtime(
    DrakeBatchConfig(
        model_file="/path/to/scene_flat.xml",
        num_envs=32,
        sim_dt=0.002,
        nthread=8,
    )
)
```

推荐从 `drakeuni.runtime` 进入；`DrakeEnvPool` 和编译扩展属于底层实现。
