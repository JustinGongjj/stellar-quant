# 星衡量化引擎 · 进度与吸收清单（阶段交付）

> 本文件只写**已经跑过、有验证证据**的部分；未完成项单列，不算交付。

---

## 一、读了什么，吸收了哪些机制

### 1）你给的两个包（一次读懂）

| 包 | 内容 | 诊断结论 |
|---|---|---|
| 归档.zip（5.4MB） | 美林时钟研究台 monorepo：`merrill-clock-quant`（相位/信号/回测）、`factor-regression`（IC/分层/Fama-MacBeth）、`clockdata`（22 源路由+SQLite）、`starmap`（Canvas 星图 + Markov 相位路径） | 因子检验层可用；**预测层偏概念**：相位概率是 logistic 分解不是校准概率、因子预期用写死的 0.85 衰减、价格扇形是历史价格自助抽样、横截面池子只有 9 只 ETF。 |
| 归档.zip（843KB） | 未来星图 + 趋势推演工作台：20 个宏观因子 × 六类骨架 → 两主轴 → 四情景 softmax；45 个事件脉冲；22 条证据（一手/二手分级）；27 个候选对象「框架生成·未核实」 | 呈现模式优秀（星座/引力井/事件/证据闸门），**但因子载荷 li/le 与情景概率都是人工设定**，没有可复现的估计口径。 |

### 2）论文与开源（已下载到 `reference/code/`，做只读参考）

| 来源 | 星标 | 吸收的具体机制（不是口号） | 本项目落点 |
|---|---|---|---|
| [microsoft/qlib](https://github.com/microsoft/qlib) | 50.2k | `risk_analysis` 指标口径（均值×N 年化、IR、回撤）；训练/验证/预测三段；GBDT 早停；`Record/Recorder` 实验留痕；[DoubleEnsemble (arXiv 2010.01265)](https://arxiv.org/pdf/2010.01265.pdf) 的**样本重加权 + 置换特征选择** | `models/walkforward.py`、`portfolio/construct.py::perf`；DoubleEnsemble 列入下一阶段 |
| [0xfdf/toraniko](https://github.com/0xfdf/toraniko) | 982 | 市值加权 WLS 估因子收益；**行业因子零和约束**；风格对行业残差化 | `portfolio/riskmodel.py`（已实现并测通） |
| [microsoft/RD-Agent](https://github.com/microsoft/RD-Agent) | 15.3k | [R&D-Agent-Quant (arXiv 2505.15155)](https://arxiv.org/abs/2505.15155) 因子自动挖掘闭环 | **只吸收流程**：LLM 生成的因子必须过样本外闸门与多重检验校正才进池 |
| [quantopian/alphalens](https://pypi.org/project/alphalens-reloaded/) | 4.5k | 分位收益、IC 衰减、换手分析 | 检验层设计（下一阶段实现） |
| [Gu, Kelly, Xiu (2020)](https://www.aqr.com/Insights/Research/Journal-Article/Empirical-Asset-Pricing-via-Machine-Learning) | — | 树模型/神经网络优于线性；动量、流动性、波动率是最稳健信号 | 主模型用 GBDT + Ridge 双基线，特征组强制包含这三类 |
| DeepSeek-V3 / DeepSeek-Math（10.5 万 / 3.5k） | — | 语言模型训练机制**与截面预测无关**；只借「组内基线 + 可验证反馈」的思想用于校准评估 | 明确**不引入**语言模型进预测链路 |

### 3）明确不吸收的
漂亮回测数字但无样本外切分/成本/幸存者处理的项目；实时交易/券商 API；任何拿未来数据训练的「因子挖掘」。

---

## 二、已实现并跑通的代码（`src/stellarq/`，共 1,846 行）

| 模块 | 内容 | 验证证据 |
|---|---|---|
| `numguard.py` | 位级 IEEE-754 四则运算（取位→整数→写位） | 11 项自检全过；与 Node.js 随机 300 组交叉 0 差异 |
| `data/eastmoney.py` | 股票池（5,928 只）、日线、财务（含公告日） | 实测返回正常 |
| `data/corporate.py` | 分红送转 → 前复权因子（交易所除权参考价口径） | 单日反解与官方 fqt=1 一致（茅台 2025-06-26 隐含现金 ≈28.05 元/10 股） |
| `models/gbdt.py` | 直方图 GBDT（leaf-wise、二阶梯度、行列采样、验证集早停）；无 lightgbm 依赖 | 合成数据 6 千样本 3.5s 训练，测试集相关 0.941（理论上限 0.955），重要性正确识别 3 个有效特征 |
| `models/walkforward.py` | 训练/验证/预测三段滚动，逐轮记录 IC、最优轮次、Top 特征 | 合成数据 4 轮，test IC 0.58（强信号下符合预期） |
| `portfolio/riskmodel.py` | 行业零和约束 + 市值加权 WLS + 风格正交化 | 行业因子和 ≈1e-18；注入 0.004 风格收益 → 估出 0.00423 |
| `portfolio/costs.py` | 印花税（2023-08-28 前后）、佣金下限、过户费、滑点、涨跌停、停牌、成交额约束 | 单测通过（涨停不可买 / 跌停不可卖） |
| `portfolio/construct.py` | 排序选股 → 带上下限的单纯形投影 → 换手/跌停/成交量约束 → T+1 收盘成交回测 | 平盘价 + 未知信号下 NAV = 0.9993（费用 6.6bp 一次换手）；随机数据口径合理 |

**测试覆盖**：每个模块都有对应的实跑脚本与输出记录（见 `docs/STATUS.md`）。

---

## 三、必须告诉你的两个现实问题

1. **本机解释器数值不可信**（`docs/ENV_NUMERIC_BUG.md`）：float 除法、`Decimal`/`Fraction` 转 float、负移位在多处返回错误值且不报错。已用位级实现绕过，并在启动时自检（不通过就中止）。这也是前期复权校验反复失败的原因——不是代码逻辑错，是宿主算术错。
2. **东方财富 `pct_chg` 不含除权调整**（实测 `pct_chg == close_t/close_{t-1}-1`），`fqt=1` 长历史会被股息扣成负价。所以复权只能自建；单日已验证正确，但**全历史逐日残差仍在 0.1%–1% 量级待定位**——这一步不解决，因子计算不能启动。

---

## 四、还没做的（下一步，按依赖顺序）

1. 复权残差定位（对比交易所公布的除权参考价）→ 全样本通过才冻结数据层。
2. 全市场面板落盘（2012→今，含退市股，避免幸存者偏差）。
3. 因子库（Alpha158 风格表达式 + 基本面 + 资金流）。
4. 检验层：RankIC/ICIR、分层、IC 衰减、换手、多重检验校正（Benjamini–Hochberg）、因子正交。
5. 模型层：DoubleEnsemble 式样本重加权 + 特征选择；概率校准（Brier/可靠性曲线）。
6. 星图前端：未来星图的星座/引力井/事件/证据闸门交互 + 上述真实数值驱动。

> 我不会把「计划」写成「已完成」。上面第四部分完成前，任何收益数字都不应当被引用。
