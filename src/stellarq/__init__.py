"""stellarq —— 星衡量化引擎

把「因子 → 检验 → 组合 → 预测 → 未来星图」做成一条可复现、带时点纪律的流水线。

设计原则（与未来星图的「因子星座 + 事件脉冲 + 情景引力井」一脉相承）：

1. **每个节点都带证据**：因子、情景、预测都附样本量、区间与失效条件，
   不允许出现无法追溯到数值的「亮度」。
2. **时点纪律**：全部计算在 `panel`（date × symbol）上按 as-of 规则推进，
   前向收益一律 shift(-h)，标签与特征之间不允许共享未来信息。
3. **不确定性显式**：点预测必须配区间（block bootstrap），情景必须配先验
   与证据似然，事件必须带「观察点 / 阈值」。
"""

__version__ = "0.1.0"

import os as _os
import sys as _sys

# 源码目录直接运行时（未 pip install）自动把 src 挂到 sys.path，保证包内相对导入可用。
_SRC = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)

DISCLAIMER = "研究工具输出，不构成投资建议；所有统计量只描述给定样本，不代表未来。"
