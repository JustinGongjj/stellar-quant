"""检查可用的历史产业事件样本量。"""
import sys; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'src'))
import pandas as pd, requests, re, collections
from stellarq.bridge.insight import InsightHub
h=InsightHub(); docs=h.research_docs()
# 从知识库找带日期的产业事件
pat=re.compile(r'(20\d\d)-(\d{2})-(\d{2})[^\n]{0,80}?(发射|首飞|回收|获证|量产|发布|涨价|禁运|制裁)')
evs=[]
for d in docs:
    for m in pat.finditer(d['body']):
        evs.append((f'{m.group(1)}-{m.group(2)}-{m.group(3)}', m.group(4), d['title'][:30]))
print('知识库事件:', len(evs))
for e in evs[:15]: print(' ', e)
