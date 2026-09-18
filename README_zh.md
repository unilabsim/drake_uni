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
src/drake_uni/runtime/       # Python contract、materializer 和运行时
src/drake_uni/compiled/      # C++ Drake 批量执行器源码和本地扩展
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

扩展会生成在 `src/drake_uni/compiled/_drake_env_pool*`，属于本地构建产物，
不会提交到 Git。CI 默认执行跨平台 Python lint/test 和 sdist 构建，不假定
runner 上存在机器相关的 Drake 前缀。

## 运行时 API

```python
from drake_uni.runtime import DrakeBatchConfig, create_runtime

runtime = create_runtime(
    DrakeBatchConfig(
        model_file="/path/to/scene_flat.xml",
        num_envs=32,
        sim_dt=0.002,
        nthread=8,
    )
)
```

推荐从 `drake_uni.runtime` 进入；`DrakeEnvPool` 和编译扩展属于底层实现。

## 原生模型属性回读

`DrakeBatchRuntime.native_model_properties()` 返回物化后 Drake 模型的冷路径
版本化快照，不会把 Drake 对象暴露给调用方。当前属性 contract 版本为 1。

```python
properties = runtime.native_model_properties()
assert properties.contract_version == 1
```

快照包含：

- Drake body 顺序、名称、默认质量、COM 和完整 3×3 转动惯量。惯量以 body
  原点为参考点、在 body 坐标系中表示；焊接的 world body 报告为质量、COM
  和惯量均为零的静态 body。
- SceneGraph geometry 按 body 索引和名称排序，包含规范化名称、所属 body
  索引、primitive 类型、原生参数以及 proximity/collision 角色标记。数值
  数组为脱离来源的只读数组。

Primitive 参数每个 geometry 固定打包为三个原生值：sphere 为
`(radius, 0, 0)`；box 和 ellipsoid 使用三个完整长度/轴长；capsule 和
cylinder 使用 `(radius, length, 0)`；half-space 全为零。mesh、convex mesh
以及其他无法用标量精确表达内容身份的形状会 fail closed，不会返回误导性
参数。
