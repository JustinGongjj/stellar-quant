"""把量化与产业分析成果注入后台工作台（趋势推演工作台.html）。"""
from stellarq.paths import DATA, WEB

W = WEB / '趋势推演工作台.html'
B = DATA / 'workbench_bundle.json'
F = DATA / 'forecast_synthesis.json'
s=W.read_text(encoding='utf-8')
bundle=B.read_text(encoding='utf-8')

JS = """
/* ================= 量化实证 / 产业传导（自动生成） ================= */
function renderDeep(){
  const C = window.__DEEP__ || {};
  let h='';
  h+=`<div class="h1">深度分析 · 叙事 / 优化 / 感知</div>
      <div class="lead">${(C.meta||{}).note||''}　·　${(C.meta||{}).date||''}</div>`;
  (C.sections||[]).forEach(sec=>{
    const pts=(sec.key_points||[]).map(x=>`<li>${x}</li>`).join('');
    let body='<ul style="margin:6px 0 0 18px;font-size:12.5px;line-height:2;color:#4a4a44">'+pts+'</ul>';
    if(sec.policy_face){
      const pf=sec.policy_face;
      body+=wbTable(['维度','权重'], Object.entries(pf.weights).map(([k,v])=>`<tr><td>${k}</td><td>${v}%</td></tr>`).join(''));
      body+=wbTable(['媒体','框架映射'], Object.entries(pf.media_mapping).map(([k,v])=>`<tr><td><b>${k}</b></td><td class="dim">${v}</td></tr>`).join(''));
      body+=`<p class="note">${pf.note}</p>`;
    }
    if(sec.judgments){
      body+=wbTable(['判断','置信度','失效条件'], sec.judgments.map(j=>{
        const col=j.conf>=75?'#0a7a52':(j.conf>=60?'#c98a00':'#8a8578');
        return `<tr><td>${j.claim}</td><td style="color:${col}">${j.conf}%</td><td class="dim">${j.fails_if}</td></tr>`;
      }).join(''));
    }
    if(sec.civilization_risk){
      const cr=sec.civilization_risk;
      body+=`<p class="note"><b>长期风险判断：</b>${cr.claim}<br>${cr.trend}<br><span class="dim">${cr.note}</span></p>`;
    }
    h+=wbCard(sec.title, (sec.tags||[]).join(' · '), body);
  });
  document.getElementById('v-deep').innerHTML=h;
}

function renderIndCycle(){
  const C = window.__INDCYC__ || {};
  let h='';
  h+=`<div class="h1">${C.title||''}</div>
      <div class="lead">${C.method||''}　·　样本 ${C.sample?C.sample.months:''} 月 / ${C.sample?C.sample.industries:''} 行业（${C.sample?C.sample.start:''} ~ ${C.sample?C.sample.end:''}）</div>`;
  const core=C.core||{};
  for(const [phase,info] of Object.entries(core)){
    const rows=(info.industries||[]).map(x=>{
      const ex=x.excess.map(v=>`<span class="${v>0?'pos':'neg'}">${v>0?'+':''}${v.toFixed(2)}%</span>`).join(' / ');
      const tt=x.t.map(v=>`<span class="${Math.abs(v)>=2?'pos':'dim'}">${v.toFixed(2)}</span>`).join(' / ');
      return `<tr><td><b>${x.name}</b></td><td>${ex}</td><td>${tt}</td><td class="dim">${x.verdict}</td></tr>`;
    }).join('');
    h+=wbCard(phase, info.meaning,
      wbTable(['行业','超额（3/6/9月窗口）','t 值','评价'], rows));
  }
  const g=C.gates||{};
  h+=wbCard('三道闸门', '只有全部通过才算数',
    wbTable(['闸门','结果'], Object.entries(g).map(([k,v])=>`<tr><td>${k}</td><td class="pos">${v}</td></tr>`).join('')));
  h+=wbCard('必须说的局限', '',
    '<ul style="margin:6px 0 0 18px;font-size:12.5px;line-height:2;color:#6b675c">'
    + (C.limits||[]).map(x=>`<li>${x}</li>`).join('') + '</ul>'
    + `<p class="note" style="color:#0a7a52"><b>结论：</b>${C.conclusion||''}</p>`);
  document.getElementById('v-indcycle').innerHTML=h;
}

function renderClockChina(){
  const C = window.__CLOCKCHINA__ || {};
  let h='';
  h+=`<div class="h1">${C.title||'美林时钟中国化'}</div>
      <div class="lead">${C.summary||''}</div>`;
  const w=C.why_original_fails||{};
  h+=wbCard('它为什么不准', `传统时钟 2011 年后正确率约 ${w.hit_rate||'28%'}`,
    wbTable(['原因','说明'], (w.reasons||[]).map(r=>`<tr><td><b>${r.r}</b></td><td class="dim">${r.detail}</td></tr>`).join('')));
  const t=C.what_i_tried||{}, r0=t.result||{};
  h+=wbCard('我做了一次改良', t.method||'',
    wbTable(['框架','股票方向正确率','样本'], `
      <tr><td>货币×信用（我的改良）</td><td class="neg"><b>${(r0.money_credit*100).toFixed(1)}%</b></td><td>${r0.n_months} 月</td></tr>
      <tr><td>经济×通胀（传统）</td><td><b>${(r0.traditional*100).toFixed(1)}%</b></td><td>${r0.n_months} 月</td></tr>`)+
    `<p class="note">${t.data_reality||''}</p>`);
  const sv=C.sensitivity||{};
  h+=wbCard('敏感性检验', sv.design||'',
    `<div class="kpis">
       <div class="kpi"><span>货币信用胜出</span><b class="neg">${sv.money_credit_wins}/${sv.total}</b><i>12 种组合</i></div>
       <div class="kpi"><span>判定</span><b class="neg">${sv.verdict||''}</b><i>不能宣称有效</i></div>
     </div><p class="note">${sv.detail||''}</p>`);
  const f=C.the_one_real_finding||{};
  h+=wbCard('唯一站得住的发现', '但不是来自这次改良',
    `<div class="kpis">
       <div class="kpi"><span>只看货币维度</span><b class="pos">${(f.accuracy*100).toFixed(1)}%</b><i>股票方向</i></div>
       <div class="kpi"><span>基准（总是做多）</span><b>${(f.baseline*100).toFixed(1)}%</b><i></i></div>
     </div>
     <p class="note"><b>解读：</b>${f.interpretation||''}<br><b>警告：</b>${f.caveat||''}</p>`);
  const cl=C.conclusion||{};
  h+=wbCard('结论', '',
    wbTable(['问题','答案'], `
      <tr><td>时钟能用吗</td><td class="neg">${cl.for_clock||''}</td></tr>
      <tr><td>那该用什么</td><td class="pos">${cl.for_us||''}</td></tr>
      <tr><td>具体怎么做</td><td>${cl.what_to_do||''}</td></tr>`));
  document.getElementById('v-clockchina').innerHTML=h;
}

function renderBets(){
  const B = window.__BETS__ || {};
  let h='';
  h+=`<div class="h1">华尔街怎么下注</div>
      <div class="lead">${B.principle||''}　·　数据时点 ${B.as_of||''}</div>`;
  const wi=B.what_is_fedwatch||{};
  h+=wbCard(wi.name||'方法', wi.why_it_matters||'',
    wbTable(['项','说明'], `
      <tr><td style="width:110px">计算口径</td><td>${wi.method||''}</td></tr>
      <tr><td>换算</td><td class="dim">${wi.formula||''}</td></tr>
      <tr><td>局限</td><td class="neg">${wi.limits||''}</td></tr>`));

  const L=B.latest||{}, d26=L.dec_2026||{}, j27=L.jan_2027||{};
  const pct=v=>v==null?'—':(v*100).toFixed(1)+'%';
  const barP=(v,col)=>`<span style="display:inline-block;width:70px;height:5px;background:#eee9df;border-radius:3px;vertical-align:middle;margin-right:7px"><i style="display:block;height:100%;width:${(v||0)*100}%;background:${col};border-radius:3px"></i></span>`;
  h+=wbCard('最新押注概率（CME FedWatch）', L.note||'',
    wbTable(['时点','情景','概率'], `
      <tr><td rowspan="2">2026-12 FOMC</td><td>维持不变</td><td>${barP(d26.hold,'#8a8578')}${pct(d26.hold)}</td></tr>
      <tr><td><b>加息 25bp</b></td><td>${barP(d26.hike_25,'#c0392b')}<b>${pct(d26.hike_25)}</b></td></tr>
      <tr><td rowspan="2">2027-01 FOMC</td><td>累计加息 25bp</td><td>${barP(j27.cum_25,'#c98a00')}${pct(j27.cum_25)}</td></tr>
      <tr><td>累计加息 50bp</td><td>${barP(j27.cum_50,'#c0392b')}${pct(j27.cum_50)}</td></tr>`));

  const drv=(B.drivers||[]).map(d=>`<tr><td>${d.date}</td><td><b>${d.factor}</b></td><td>${d.effect}</td><td class="dim">${d.channel}</td></tr>`).join('');
  h+=wbCard('什么在推动押注变化', '按「事件 → 概率移动」记录，可复算',
    wbTable(['日期','事件/因子','对概率的影响','传导通道'], drv));

  const how=(B.how_pros_bet||[]).map(m=>`<tr><td><b>${m.method}</b></td><td>${m.detail}</td><td class="dim">${m.who}</td></tr>`).join('');
  h+=wbCard('专业机构的五种下注方式', '同样的观点，可以用不同工具表达',
    wbTable(['方式','做法','谁在用'], how));

  const ev=(B.watched_events||[]).map(e=>`<tr><td><b>${e.event}</b></td><td class="dim">${e.typical_move}</td></tr>`).join('');
  h+=wbCard('哪些事件会移动押注', '',
    wbTable(['事件','对定价的影响'], ev));

  if((B.lesson_for_us||[]).length){
    h+=wbCard('对我们有什么用', '',
      '<ul style="margin:6px 0 0 18px;font-size:12.5px;line-height:2;color:#6b675c">'
      + B.lesson_for_us.map(x=>`<li>${x}</li>`).join('') + '</ul>');
  }
  document.getElementById('v-bets').innerHTML=h;
}

function renderForecast(){
  const F = window.__FORECAST__ || {};
  let h='';
  h+=`<div class="h1">基于分析的预测</div>
      <div class="lead">${F.principle||''}　·　数据时点 ${F.as_of||''}</div>`;
  const rows=(F.forecasts||[]).map(f=>{
    const conf=f.confidence||0;
    const col = conf>=75?'#0a7a52':(conf>=55?'#c98a00':'#8a8578');
    const bar = '<span style="display:inline-block;width:60px;height:4px;background:#eee9df;border-radius:2px;vertical-align:middle;margin-right:6px">'
              + `<i style="display:block;height:100%;width:${conf}%;background:${col};border-radius:2px"></i></span>`;
    return `<tr>
      <td><b>${f.id}</b></td>
      <td>${bar}<span style="color:${col};font-variant-numeric:tabular-nums">${conf}%</span></td>
      <td>${f.horizon}</td>
      <td><b>${f.claim}</b><div class="dim" style="margin-top:4px">依据：${f.basis}</div>
          ${f.fails_if&&f.fails_if!=='—'?`<div class="dim" style="color:#c0392b">失效条件：${f.fails_if}</div>`:''}
          <div class="dim" style="color:#0a7a52">行动：${f.action}</div></td>
    </tr>`;
  }).join('');
  h+=wbCard('八条判断', '置信度是主观分级，不是统计频率',
    wbTable(['#','置信度','期限','判断'], rows));
  if((F.antiforecasts||[]).length){
    h+=wbCard('明确不预测的事项', '这一节和上面同样重要',
      '<ul style="margin:6px 0 0 18px;font-size:12.5px;line-height:2;color:#6b675c">'
      + F.antiforecasts.map(x=>`<li>${x}</li>`).join('') + '</ul>');
  }
  document.getElementById('v-forecast').innerHTML=h;
}


const WB = __BUNDLE__;

function wbCard(title, sub, rows){
  return `<div class="card"><h3>${title}</h3>${sub?`<p class="sub">${sub}</p>`:''}${rows}</div>`;
}
function wbTable(head, body){
  return `<table class="tbl"><thead><tr>${head.map(h=>`<th>${h}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table>`;
}
function fmtPct(v,d=2){ return v==null?'—':(v*100).toFixed(d)+'%'; }

function renderQuant(){
  const q=WB.quant||{}, t=WB.tests||{};
  const v4=q.v4||{}, lo=q.long_only||{}, sv=q.survivorship||{}, cap=q.capacity||[], cb=q.combo||{}, cta=q.cta||{};
  const ms=t.macro_feature||{};
  let h='';
  h+=`<div class="h1">量化实证</div>
      <div class="lead">这一页的全部数字来自本仓跑批，可追溯到 reports/data/ 下对应文件。所有区间标注样本长度。</div>`;

  // 核心指标
  const survAdj = (cap[0]||{}).ann_excess_surv_adj;
  h+=wbCard('核心结论', '股票策略 v4，1 亿规模、top60、5 日调仓', `
    <div class="kpis">
      <div class="kpi"><span>样本外 RankIC</span><b>${(v4.test_ic||0).toFixed(4)}</b><i>105 期 / 2.1 年</i></div>
      <div class="kpi"><span>校准误差 ECE</span><b>${(v4.ece||0).toFixed(4)}</b><i>越低越好</i></div>
      <div class="kpi"><span>年化超额</span><b class="pos">${fmtPct((cap[0]||{}).ann_excess)}</b><i>含冲击成本</i></div>
      <div class="kpi"><span>含幸存者修正</span><b class="pos">${fmtPct(survAdj)}</b><i>唯一对外口径</i></div>
      <div class="kpi"><span>IR</span><b>1.1–1.3</b><i>成本假设敏感</i></div>
      <div class="kpi"><span>容量上限</span><b>1–2 亿</b><i>超过超额归零</i></div>
    </div>`);

  // 容量曲线
  const capRows = cap.map(r=>`<tr>
      <td>${r.size_yi} 亿</td><td>${r.cap?r.cap*100+'%':'不限'}</td>
      <td>${r.fill==null?'—':(r.fill*100).toFixed(1)+'%'}</td>
      <td>${fmtPct(r.ann_excess)}</td>
      <td style="color:${(r.ann_excess_surv_adj||0)>0?'#0a7':'#c33'}">${fmtPct(r.ann_excess_surv_adj)}</td>
      <td>${r.ir==null?'—':r.ir.toFixed(2)}</td></tr>`).join('');
  h+=wbCard('容量曲线', '规模 × 限速 → 超额。限速会拖慢建仓，损失远大于省下的冲击成本', 
            wbTable(['规模','限速','建仓完成率','超额','幸存者修正','IR'], capRows));

  // 幸存者偏差
  const sens=(sv.sensitivity||[]).map(x=>`<tr><td>${fmtPct(x['缺失股年化'],0)}</td><td style="color:#c33">+${x['组合年化高估(pp)']} pp</td></tr>`).join('');
  h+=wbCard('幸存者偏差', `深沪 2012 年以来退市 ${sv.delisted_since_2012} 只，面板覆盖 ${sv.panel_covers_delisted} 只`,
    wbTable(['假设退市股年化','组合超额高估'], sens)+
    `<p class="note">占全市场 ${fmtPct(sv.missing_share)}。这是所有数字都要打折的原因。</p>`);

  // CTA
  const ctaRows=Object.entries(cta).map(([k,v])=>`<tr>
      <td>${v.label||k}</td><td>${fmtPct(v.ann)}</td><td>${v.sharpe==null?'—':v.sharpe.toFixed(2)}</td>
      <td>${fmtPct(v.max_dd)}</td><td>${v.start||''} ~ ${v.end||''}</td></tr>`).join('');
  h+=wbCard('CTA 趋势策略', '21 年样本，含随机信号对照 —— 这是全项目可信度最高的结论',
    wbTable(['策略','年化','Sharpe','最大回撤','样本'], ctaRows));

  // 组合
  const comboRows=Object.entries(cb).filter(([k,v])=>v&&v.label).map(([k,v])=>`<tr>
      <td>${v.label}</td><td>${fmtPct(v.ann)}</td><td>${v.sharpe==null?'—':v.sharpe.toFixed(2)}</td>
      <td>${fmtPct(v.max_dd)}</td></tr>`).join('');
  h+=wbCard('股票 × CTA 组合', `相关性 ${(cb.corr||0).toFixed(3)}（低）—— 组合后回撤显著下降`,
    wbTable(['组合','年化','Sharpe','最大回撤'], comboRows));

  // 宏观当特征
  if(ms.macro_6m){
    h+=wbCard('宏观当特征 vs 当规则', '同一批宏观数据，用法决定成败',
      wbTable(['用法','IC','ICIR','胜率'], `
        <tr><td>量价 + 宏观（6 月滚动窗）</td><td class="pos">${ms.macro_6m.ic_mean.toFixed(4)}</td><td>${ms.macro_6m.icir.toFixed(3)}</td><td>${fmtPct(ms.macro_6m.ic_win)}</td></tr>
        <tr><td>纯量价（同窗口）</td><td>${ms.price_only_6m.ic_mean.toFixed(4)}</td><td>${ms.price_only_6m.icir.toFixed(3)}</td><td>${fmtPct(ms.price_only_6m.ic_win)}</td></tr>`)+
      `<p class="note">做规则（调仓位/过滤）实测为负；做特征为正。</p>`);
  }

  // 六个证伪
  const falsified=[
    ['美林时钟能择时','超额 +2.53% → −1.94%，IR 1.18 → −0.58'],
    ['时钟做过滤器有效','换参数即失效，改善集中在单一年份 → 过拟合'],
    ['对冲能降低回撤','对冲后 3.3% vs 不 hedge 7.72%，回撤几乎不变'],
    ['打板/抓龙头可行','涨停 70,003 次，可成交率仅 26%，期望 +0.13%~0.20% 低于成本'],
    ['拉长持有期能提容量','换手 0.69 → 0.95 反而上升，超额同时下降'],
    ['产业分析叠加量化能提升','重叠率 0.5%（34/6300），IR 1.91 → 1.88'],
  ];
  h+=wbCard('六个被我们自己证伪的假设', '这一节比正面结论更重要',
    wbTable(['曾相信','实测'], falsified.map(([a,b])=>`<tr><td>${a}</td><td class="neg">${b}</td></tr>`).join('')));

  document.getElementById('v-quant').innerHTML=h;
}

function renderChain(){
  const c=WB.industry||{}, t=WB.tests||{};
  let h='';
  h+=`<div class="h1">产业传导</div>
      <div class="lead">供应链图 ${c.graph_stats.total_edges} 条关系，其中 ${c.graph_stats.with_share} 条已核定份额。瓶颈由产业事实判定，不由公式推算。</div>`;

  // 瓶颈
  const bnRows=Object.entries(c.bottlenecks||{}).map(([seg,info])=>{
    return info.bottlenecks.map(b=>`<tr>
      <td>${seg}</td><td>${'★'.repeat(b.severity)}</td><td>${b.type}</td>
      <td><b>${b.node}</b></td><td class="dim">${b.fact}</td></tr>`).join('');
  }).join('');
  h+=wbCard('产业瓶颈清单', '★=严重度，每条都带事实依据',
    wbTable(['环节','严重度','类型','卡点','依据'], bnRows));

  // 冲击场景
  const impRows=Object.entries(c.impacts||{}).map(([name,sc])=>{
    const top=Object.entries(sc.impact||{}).slice(0,4).map(([k,v])=>
      `<span style="color:${v<0?'#c33':'#0a7'}">${k} ${v>0?'+':''}${v.toFixed(2)}</span>`).join('、');
    return `<tr><td><b>${name}</b><br><span class="dim">${sc.logic}</span></td>
      <td>${sc.mode_cn}</td><td>${sc.polarity==='neg'?'<span class="neg">负面</span>':'<span class="pos">正面</span>'}</td>
      <td>${top}</td></tr>`;
  }).join('');
  h+=wbCard('九个冲击场景', '短缺累积（越往下越缺）vs 成本衰减（逐级消化）vs 需求拉动（上游分享）',
    wbTable(['场景','机制','方向','受影响'], impRows));

  // 三条验证
  const es=t.event_signal||{}, lt=t.leadtime||{}, inf=t.industry_filter||{};
  const esRows = es.A_追高T1买超额 ? `
    <tr><td>追事件（T+1 买入）</td><td class="neg">${fmtPct(es.A_追高T1买超额.mean)}</td><td>${(es.A_追高T1买超额.t||0).toFixed(1)}</td><td>${es.A_追高T1买超额.n}</td></tr>
    <tr><td>不追（T 日持有）</td><td class="neg">${fmtPct(es.B_持有T5超额.mean)}</td><td>${(es.B_持有T5超额.t||0).toFixed(1)}</td><td>${es.B_持有T5超额.n}</td></tr>` : '';
  h+=wbCard('三条验证（全部为负）', '产业框架不能产生 alpha，这是实测结论',
    wbTable(['尝试','结果','t 值','样本'], `
      <tr><td>预测传导时滞（Lead-lag）</td><td class="neg">三对全部 0 天，无可测时滞</td><td>—</td><td>899 天/对</td></tr>
      ${esRows}
      <tr><td>叠加到量化模型</td><td class="neg">重叠 ${(inf.coverage?.hits||0)}/${(inf.coverage?.total||0)}（${((inf.coverage?.hits||0)/(inf.coverage?.total||1)*100).toFixed(1)}%），IR 1.91→1.88</td><td>—</td><td>105 期</td></tr>`));

  h+=wbCard('结论：产业框架的边界', '', `
    <table class="tbl"><tbody>
    <tr><td>预测谁先涨</td><td class="neg">✗ 无时滞可测</td></tr>
    <tr><td>事件后追涨</td><td class="neg">✗ 显著亏损</td></tr>
    <tr><td>叠加到量化选股</td><td class="neg">✗ 重叠 0.5%</td></tr>
    <tr><td><b>判断定价是否过度</b></td><td class="pos">✓ 涨幅越极端，回归越强</td></tr>
    <tr><td><b>持仓风险预警</b></td><td class="pos">✓ 关键节点出事时知道谁受影响</td></tr>
    <tr><td><b>产业认知框架</b></td><td class="pos">✓ 对一级市场更有价值</td></tr>
    </tbody></table>`);

  document.getElementById('v-chain').innerHTML=h;
}

(function hookViews(){
  const orig = window.activateView || null;
  const paint = () => {
    try{
      renderQuant(); renderChain(); renderForecast(); renderBets(); renderClockChina(); renderIndCycle(); renderDeep();
    }catch(e){ console.error('WB render error', e); }
  };
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded', paint);
  else setTimeout(paint, 0);
  // 板块切换时重绘（页面本身用 display 控制，内容常驻即可）
})();
""".replace('__BUNDLE__', bundle)

_css=WEB.parent / 'scripts/workbench_patch.css'
CSS = _css.read_text(encoding='utf-8') if _css.exists() else ''

# 1) 菜单
if "k:'quant'" not in s:
    s=s.replace("""  {k:'gaps',n:'资料缺口',s:'这套框架还缺什么，逐项补'}""",
"""  {k:'gaps',n:'资料缺口',s:'这套框架还缺什么，逐项补'},
  {k:'quant',n:'量化实证',s:'回测结论、容量、六个被证伪的假设'},
  {k:'chain',n:'产业传导',s:'供应链图、瓶颈、冲击场景、供货份额'},
  {k:'forecast',n:'基于分析的预测',s:'八条判断 · 置信度 · 失效条件'},
  {k:'bets',n:'华尔街怎么下注',s:'FedWatch 概率 · 驱动事件 · 五种下注方式'},
  {k:'clockchina',n:'时钟中国化实测',s:'一次失败的改良 · 敏感性检验'},
  {k:'indcycle',n:'库存周期×行业',s:'唯一通过三道闸门的信号'},
  {k:'deep',n:'深度分析',s:'AI叙事与政治 · 两种优化 · 感知与协作'}""",1)

# 2) section
if 'id="v-quant"' not in s:
    s=s.replace("""        <section class="view" id="v-gaps"></section>""",
"""        <section class="view" id="v-gaps"></section>
        <section class="view" id="v-quant"></section>
        <section class="view" id="v-chain"></section>
        <section class="view" id="v-forecast"></section>
        <section class="view" id="v-bets"></section>
        <section class="view" id="v-clockchina"></section>
        <section class="view" id="v-indcycle"></section>
        <section class="view" id="v-deep"></section>""",1)

# 3) 预测数据（独立 script，必须在 </body> 前）
if 'window.__FORECAST__' not in s:
    s=s.replace('</body>', '<script>window.__FORECAST__=' + F.read_text(encoding='utf-8') + ';</script>\n</body>', 1)
CC = DATA / 'clock_china_report.json'
if 'window.__CLOCKCHINA__' not in s and CC.exists():
    s=s.replace('</body>', '<script>window.__CLOCKCHINA__=' + CC.read_text(encoding='utf-8') + ';</script>\n</body>', 1)
DP = DATA / 'deep_analysis.json'
if 'window.__DEEP__' not in s and DP.exists():
    s=s.replace('</body>', '<script>window.__DEEP__=' + DP.read_text(encoding='utf-8') + ';</script>\n</body>', 1)
IC = DATA / 'industry_cycle_payload.json'
if 'window.__INDCYC__' not in s and IC.exists():
    s=s.replace('</body>', '<script>window.__INDCYC__=' + IC.read_text(encoding='utf-8') + ';</script>\n</body>', 1)
BETS = DATA / 'wall_street_bets.json'
if 'window.__BETS__' not in s and BETS.exists():
    s=s.replace('</body>', '<script>window.__BETS__=' + BETS.read_text(encoding='utf-8') + ';</script>\n</body>', 1)

# 4) CSS
if '.kpis{' not in s and CSS:
    s=s.replace('</style>', CSS+'</style>', 1)

# 5) 注入渲染代码：必须在 RENDER 定义之前
ANCHOR = "const RENDER = {overview:renderOverview,"
if 'window.__WB_PATCHED__' not in s:
    s=s.replace(ANCHOR, JS+'\n'+ANCHOR, 1)

# 6) 注册路由
if "quant:()=>" not in s:
    s=s.replace(ANCHOR, ANCHOR.rstrip(',')+',quant:()=>{renderQuant();},chain:()=>{renderChain();},forecast:()=>{renderForecast();},bets:()=>{renderBets();},clockchina:()=>{renderClockChina();},indcycle:()=>{renderIndCycle();},deep:()=>{renderDeep();},', 1)

W.write_text(s,encoding='utf-8')
print('已注入，文件 %.0f KB'%(len(s)/1024))
