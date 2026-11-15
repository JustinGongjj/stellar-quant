"""冲击成本校准：Almgren-Chriss 的平方根模型里 k 到底该取多少。

模型 impact = k × σ_daily × sqrt(Q/ADV)。
业界常用 k≈0.3 是「σ 用年化」或「Q/ADV 用 1% 口径」时的数值，直接套用到
「σ_daily + Q/ADV 实际比例」会把成本放大 1~2 个数量级。

校准锚点（A 股公开研究的通行量级）：
  * 单笔占 ADV 1% 时，冲击成本约 10~15 bp
  * 占 ADV 5% 时约 25~35 bp
  * 占 ADV 10% 时约 40~60 bp
反解：impact/1e4 = k*sqrt(q/adv)  →  k = impact_bp/1e4/sqrt(q/adv)
"""
import numpy as np
anchors=[(0.01,0.0012),(0.05,0.0030),(0.10,0.0050)]
for q,imp in anchors:
    print(f'ADV{q:.0%}: 目标 {imp*1e4:.0f}bp → k = {imp/np.sqrt(q):.4f}')
k=np.mean([imp/np.sqrt(q) for q,imp in anchors])
print('推荐 k =', round(k,4))
for q in (0.001,0.005,0.01,0.03,0.05,0.10,0.20):
    print(f'  q/ADV {q:6.1%} → {k*np.sqrt(q)*1e4:6.1f} bp')
