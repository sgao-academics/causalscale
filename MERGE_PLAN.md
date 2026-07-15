# causalscale V3 融合方案

## 目标
融合 causalscale V1 (公开面) + CausalDiscoveryEngine V2 (企业核) = causalscale V3.0.0

## 融合架构

```
causalscale V3/
├── core/                          # V2 核心引擎（原 causal_engine_v2/）
│   ├── engine.py                  # CausalDiscoveryEngine (原 causal_engine.py)
│   ├── adaptive_rank.py           # 自适应秩选择
│   ├── multi_scale.py             # 多尺度分解
│   ├── uncertainty.py             # 不确定性量化
│   ├── optimization.py            # 混合精度+余弦退火
│   ├── dag_utils.py               # DAG约束+反事实+Granger
│   ├── theory.py                  # 收敛诊断+显著性检验
│   ├── lowrank.py                 # 保留 V1 的轻量 LowRankGNN (向后兼容)
│   ├── dag_constraint.py          # 保留 V1 的 NOTEARS (向后兼容)
│   └── cluster_gate.py            # 保留 V1 的 CAGate (向后兼容)
│
├── api.py                         # 统一 API: CausalDiscovery(data, method="auto")
│                                  # method: lowrank/multi_scale/cluster_aware/full/auto
│
├── cli.py                         # 保留 causalscale 6条命令
├── web/app.py                     # 保留 Streamlit 界面
├── pretrained/                    # 保留预训练模型+HF自动下载
├── apps/                          # 保留 3个应用模块
├── examples/                      # 保留 4个 Jupyter 教程
└── utils/                         # 保留数据工具
```

## 分步操作

### Step 1: 拷贝 V2 引擎到 causalscale/core/
- `causal_engine_v2/causal_engine.py` → `core/engine.py`
- `causal_engine_v2/adaptive_rank.py` → `core/adaptive_rank.py`
- `causal_engine_v2/multi_scale.py` → `core/multi_scale.py`
- `causal_engine_v2/uncertainty.py` → `core/uncertainty.py`
- 修复所有 import 路径（`from .adaptive_rank` 加 `.core.` 前缀）

### Step 2: 重写 api.py
- `CausalDiscovery(data, method="auto")`
- `method="auto"` → 根据 d 自动选择：
  - d <= 100 → `cluster_aware` (小样本)
  - 100 < d <= 500 → `multi_scale` (中层)
  - d > 500 → `lowrank` (大规模)
- 保留 `get_network()`, `plot()`, `summary()`
- 新增 `get_edges(confidence=0.8)`, `counterfactual()`, `generate_report()`

### Step 3: 更新 __init__.py
- 导出 V2 所有模块
- 版本升级到 3.0.0
- 新增 `__all__` 包含: CausalDiscovery, CausalDiscoveryEngine (alias), BootstrapEnsemble, 等

### Step 4: 更新 setup.py
- version='3.0.0'
- 新增可选依赖: huggingface_hub, streamlit
- entry_points 保留 CLI

### Step 5: 回归测试
- 合成 DAG d=30/100/200, 4种模式全跑
- `pip install -e .` 验证
- CLI 6条命令验证
- Streamlit 启动验证

## 免费 vs 授权

| 模式 | causalscale (免费) | V3 完整版 (授权) |
|:--|:--|:--|
| `lowrank` | ✅ | ✅ |
| `multi_scale` | ❌ | ✅ |
| `cluster_aware` | ❌ | ✅ |
| `full` | ❌ | ✅ |
| Bootstrap | ❌ | ✅ |
| 反事实 | ❌ | ✅ |
| 混合精度 | ❌ | ✅ |
| 预训练模型 | ✅ | ✅ |
| CLI | ✅ | ✅ |
| Streamlit | ✅ | ✅ |
| Jupyter教程 | ✅ | ✅ |

## 预计工时
- 文件拷贝+路径修复: 30分钟
- api.py 重写: 1小时
- 测试+调试: 30分钟
- 总计: 2小时
