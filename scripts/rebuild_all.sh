#!/bin/bash
# 完整重建流水线：顺序不能乱，后一步依赖前一步的输出
set -e
cd /home/node/default/stellar-quant
export PYTHONPATH=src
echo "1/8 构建基础域"
python3 scripts/build_atlas_data.py > /dev/null
echo "2/8 商业航天域"
python3 scripts/build_space_domain.py > /dev/null
echo "3/8 AI 与具身智能域"
python3 scripts/build_ai_embodied.py > /dev/null
echo "4/8 低空经济（自定义域）"
python3 scripts/custom_domain.py --id lowalt --name 低空经济 --src reports/research/lowalt.json > /dev/null
echo "5/9 中国域（替换瓯海位置，瓯海移到末尾）"
python3 scripts/build_china_domain.py > /dev/null
echo "6/9 重叠冲击场景坐标"
python3 scripts/fix_scenario_coords.py
echo "6.5 用最新宏观数据刷新中国域"
python3 scripts/refresh_china_data.py > /dev/null
echo "7/9 挂载知识库与产业链"
python3 scripts/attach_insight.py > /dev/null
python3 scripts/attach_research.py > /dev/null
python3 scripts/attach_doc_analysis.py > /dev/null
python3 scripts/attach_entity_graph.py > /dev/null
echo "8/9 预测与风险"
python3 scripts/predict_impact_v3.py > /dev/null 2>&1 || true
python3 scripts/apply_shares.py > /dev/null
python3 scripts/predict_impact_v3.py > /dev/null
python3 scripts/build_industry_risk.py > /dev/null
echo "9/9 生成页面"
python3 scripts/build_atlas_html.py
python3 scripts/patch_workbench.py
echo "10/10 打包成果（可在线下载）"
python3 scripts/make_package.py
