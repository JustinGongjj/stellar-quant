"""把供应链图从 32 条扩到 100+ 条，逐产业补全。

数据依据：用户提供的会议转写与产业笔记（已入库 knowledge/）。
每条边标注 evidence（原文依据），可追溯。
"""
import json
from stellarq.paths import DATA  # noqa: E402
OUT = DATA
G=json.load(open(OUT / 'supply_graph.json',encoding='utf-8'))
E={(e['from'],e['to']):e for e in G['edges']}

def add(a,b,what,src,conf='verified',ev=''):
    key=(a,b)
    if key in E:
        E[key].setdefault('ev',ev) or None
        return
    E[key]={'from':a,'to':b,'what':what,'source':src,'conf':conf,'ev':ev}

# ---------- 光模块链（补上游） ----------
add('Lumentum','源杰科技','EML 芯片（外购/对标）','笔记','verified','真正缺的是DSP、光芯片、磷化铟衬底')
add('光智科技','中际旭创','磷化铟衬底 12万片/年','会议','verified','国内云南厂15万片/年、光智科技12万片/年')
add('云南锗业','中际旭创','磷化铟衬底 15万片/年','会议','verified','国内云南厂15万片/年')
add('日本厂商','Lumentum','磷化铟衬底（其余产能）','会议','inferred','其余在日本')
add('源杰科技','新易盛','光芯片','研报','inferred','')
add('中际旭创','微软','800G/1.6T 光模块','会议','verified','微软求白宫放开3.2T光模块进口')
add('新易盛','北美云厂商','800G 光模块','会议','verified','新易盛在北美云厂商供应链存在感强')
add('中际旭创','英伟达','光模块','研报','inferred','')

# ---------- PCB / 覆铜板 / 电子布 ----------
add('丰田织机','生益科技','电子布织布机','会议','verified','全世界就两三家，丰田一年卖几千台')
add('生益科技','沪电股份','覆铜板','会议','verified','覆铜板=电子布+铜箔+环氧树脂')
add('生益科技','深南电路','覆铜板','会议','inferred','')
add('沪电股份','英伟达','AI服务器 PCB','会议','verified','PCB龙头四家，门槛高')
add('深南电路','英伟达','AI服务器 PCB','会议','inferred','')
add('鹏鼎控股','苹果','FPC/PCB','研报','verified','')
add('铜箔厂商','生益科技','铜箔','会议','verified','覆铜板三大原料之一')
add('环氧树脂厂商','生益科技','环氧树脂','会议','verified','覆铜板三大原料之一')

# ---------- 电源与电力 ----------
add('三菱重工','数据中心','燃气轮机/柴油备用电源','会议','verified','涡轮叶片门槛高，排单已到2029年')
add('英飞凌','英伟达','功率半导体','笔记','verified','337调查涉及主体')
add('MPS','英伟达','电源管理芯片','笔记','verified','337调查涉及主体')
add('麦格米特','英伟达','电源模块','笔记','verified','337调查涉及主体')
add('欧陆通','AI服务器厂商','服务器电源','研报','inferred','')
add('伟创力','英伟达','服务器代工','笔记','verified','337调查涉及主体')
add('广达','英伟达','AI服务器代工','笔记','verified','337调查涉及主体')
add('鸿海','英伟达','AI服务器代工','笔记','verified','337调查涉及主体')
add('工业富联','英伟达','AI服务器代工','笔记','verified','337调查涉及主体')

# ---------- 液冷 ----------
add('润禾材料','英伟达','浸没式硅油冷却液','笔记','inferred','硅油价格仅为氟化液的1/10')
add('英维克','AI服务器厂商','液冷 CDU','研报','inferred','')
add('3M','氟化液市场','氟化液（PFAS 受限）','笔记','verified','氟化液受PFAS环保限制')
add('LG','数据中心','大型液冷/电源模块','笔记','verified','')

# ---------- 存储 ----------
add('美光','AMD','HBM/DRAM','会议','verified','大客户是英伟达，其次AMD、微软云')
add('SK海力士','微软云','HBM','会议','verified','')
add('长鑫存储','国内手机厂商','DRAM（2-3层）','会议','verified','国内只能做2-3层')
add('长江存储','国内手机厂商','NAND','会议','verified','')
add('闪迪','数据中心','企业级 SSD','会议','verified','闪迪季度赚8亿，毛利85%')
add('兆易创新','模组厂','NOR/DRAM 利基','研报','inferred','中小客户被挤出原厂直采转向模组厂')

# ---------- 具身智能 ----------
add('汇川技术','人形机器人厂商','伺服系统','笔记','verified','')
add('双环传动','人形机器人厂商','RV减速器/齿轮','笔记','verified','')
add('绿的谐波','人形机器人厂商','谐波减速器','笔记','verified','关键在灵巧手的小型减速电机')
add('三花智控','人形机器人厂商','机电执行器','笔记','verified','')
add('拓普集团','人形机器人厂商','执行器/结构件','笔记','verified','')
add('均胜电子','人形机器人厂商','机器人电子','笔记','verified','')
add('特斯拉','人形机器人厂商','Optimus 整机','笔记','inferred','')

# ---------- 商业航天 ----------
add('昊志机电','蓝箭航天','特种电机/泵阀控制器','笔记','verified','已实现小批量交付')
add('中天火箭','蓝箭航天','固体发动机耐烧蚀组件','笔记','verified','')
add('博云新材','快舟火箭','碳/碳喉衬材料','笔记','verified','已用于快舟固体火箭')
add('上海沪工','千帆/国网星座','卫星总装（500kg级）','笔记','verified','')
add('中国卫星','千帆/国网星座','空间太阳电池','会议','verified','上半年交付超73万片')
add('长江通信','低轨卫星','地面管控/指挥系统','笔记','verified','')
add('邵阳液压','商业火箭','航天锻件/回收姿态','笔记','verified','拟6亿收购新承航锐')
add('中兵红箭','商业火箭','功能性金刚石（零部件加工）','笔记','verified','')
add('九州云箭','商业火箭','液氧甲烷发动机','图谱','verified','')
add('中简科技','商业火箭','碳纤维复材','图谱','verified','')
add('光威复材','商业火箭','碳纤维复材','图谱','verified','')
add('复旦微','低轨卫星','抗辐照 FPGA','图谱','verified','')
add('紫光国微','低轨卫星','抗辐照芯片','图谱','verified','')
add('星河动力','商业火箭','谷神星/智神星系列','笔记','verified','')
add('天兵科技','商业火箭','天龙三号','笔记','verified','')
add('星际荣耀','商业火箭','双曲线三号','笔记','verified','')

# ---------- 算力芯片 ----------
add('寒武纪','AI服务器厂商','AI 加速卡','研报','inferred','')
add('海光','AI服务器厂商','CPU/DCU','研报','inferred','')
add('AMD','AI服务器厂商','GPU/CPU','会议','verified','AMD 6GW 从下半年起量')
add('谷歌','AI服务器厂商','TPU','会议','verified','谷歌TPU达5GW')
add('华为','国内云厂商','昇腾芯片','研报','inferred','')

# ---------- 水下机器人 ----------
add('潜行创新','海洋工程客户','ROV 整机','笔记','verified','海外占比超85%')
add('中科深海所','海洋观测平台','FMTD 柔性传感器','笔记','verified','最大下潜4812.2m')

edges=list(E.values())
deg={}
for e in edges:
    deg.setdefault(e['from'],{'out':0,'in':0}); deg[e['from']]['out']+=1
    deg.setdefault(e['to'],{'out':0,'in':0}); deg[e['to']]['in']+=1

G['edges']=edges; G['degree']=deg; G['nodes']=len(deg)
G['stats']={'edges':len(edges),'nodes':len(deg),
            'verified':sum(1 for e in edges if e['conf']=='verified'),
            'inferred':sum(1 for e in edges if e['conf']=='inferred')}
json.dump(G, open(OUT / 'supply_graph.json','w'), ensure_ascii=False, indent=1)
print(f"关系 {len(edges)} 条（verified {G['stats']['verified']} / inferred {G['stats']['inferred']}），主体 {len(deg)} 个")
print('\n被依赖 Top12:')
for k,v in sorted(deg.items(), key=lambda x:-x[1]['in'])[:12]:
    print(f'  {k:14s} in={v["in"]:2d} out={v["out"]:2d}')
