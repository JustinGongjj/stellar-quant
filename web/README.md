# 未来星图 · 前端与服务

**线上地址**：https://dsh-kxwqmbvzrptjdsfn.apps.gai.cn/ （公开访问返回 401，需登录平台）
**本地**：http://127.0.0.1:8791/

## 页面结构（按原「未来星图」框架重做，数据全为真实回测）

| 区块 | 内容 | 数据来源 |
|---|---|---|
| **因子星座** | 8 个星簇环形布局；亮度=\|ICIR\|；红=正向、绿=避免型；点击看成员 | `factors.clusters` |
| **情景引力井** | 复苏/过热/滞胀/衰退四井；井的大小=历史相位占比；当前相位高亮 | `scenarios` |
| **事件脉冲** | 8 类公告事件；波形振幅=\|t\|/8；过校正的标「已过校正」 | `events` |
| **证据闸门** | A 通过 / B 待验 / D 否定 / F 证伪，含失效条件 | `evidence` |
| **模型预测** | RankIC、ECE、样本外期数、**数据滞后** | `prediction` |
| **相位轨迹** | 近 48 个月相位色带 | `timeline` |

## 服务

零依赖 Node（内置 `http`），端口由 `PORT` 环境变量给定：

```bash
PORT=8791 node web/server/app.js
```

- `GET /` → `starmap_v2.html`
- `GET /api/<name>.json` → 映射 `reports/data/<name>.json`（文件名白名单，防路径穿越）
- `GET /healthz` → `{"ok":true}`

已注册为托管服务 `stellar-starmap`（端口 8791），Pod 重启后自动拉起。

## 数据更新

```bash
PYTHONPATH=src python3 scripts/prediction_framework.py   # 重算四层框架
PYTHONPATH=src python3 scripts/build_starmap_v2.py        # 补充引力井/事件/时间轴
python3 scripts/build_starmap_data.py                     # 生成 web/forecast_data.js
```

## 移动端

< 640px 自动切换为「每簇一行」的线性星座带，成员不显示文字，点击簇查看明细。
