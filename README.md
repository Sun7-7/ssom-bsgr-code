# GR-SSOM：雷达脉冲分选主方法

这是投稿期的最小公开实现，包含固定随机脉冲编码器、30×30 SSOM、未知K初始化和NSPR样本分区细化，以及独立仿真PDW。沿用原仓库地址和历史，替换旧协议与入口。

可从PDW运行主方法并输出指标和预测。真实采集数据、八种基线、消融、参数搜索、内部预测、稿件与审计材料暂不公开，因此仿真结果不能替代论文真实记录的数值。

## 安装与运行

建议Python 3.11；CPU可运行，安装支持CUDA的PyTorch后自动使用GPU。

```bash
python -m venv .venv
# Windows: .venv/Scripts/activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python scripts/smoke_test.py
python scripts/run_simulation.py
```

默认3个场景(K=4/6/8)、每场景2个10000脉冲窗口、种子42/123/456，共18次。输出 outputs/simulation：逐次ACC/ARI/NMI/Macro-F1/K绝对误差、场景均值、NPZ预测、运行环境和来源哈希。再次运行须使用新输出目录。

```bash
python scripts/run_simulation.py --seeds 42 --device cpu --out outputs/cpu_check
python scripts/run_main.py --pdw data/simulation/synthetic_k6.csv --window 10000 --count 2 --seed 123 --out outputs/single_record
python scripts/generate_simulation.py --out outputs/regenerated_data
```

run_main默认步长等于窗口长度；--start指定起点，--start-step指定其他步长。--seed实际控制SSOM初始化和细化随机性；GMM种子按窗口起点和K确定。输入要求TOA/RF/PW/DOA/PA/Label，标签只用于评分。见 [数据格式](docs/data_format.md)。

## 方法与协议

1. RF/PW/PA和DOA的sin/cos构建标准化特征，不使用TOA选择分区。
2. K=4..20的tied-GMM，增益为相邻模型总对数似然之差。阈值为全部16个增益最大值的10%，从第三个增益起选首次低于阈值的左端K，否则取20。这不是BIC最小化或旧8%早期中位数规则。
3. RF/PW/PA/DOA驱动固定随机脉冲编码器；SSOM地图训练600步。固定编码器不代表地图零训练。
4. NSPR结合连续距离和BMU支持分布选择合并或拆分；操作对象是样本分区，不合并、删除或重训地图神经元。

[配置说明](configs/main.json)描述源码固定协议；改JSON本身不改变计算。依赖版本和CPU/GPU可能影响浮点结果，每次批量运行记录版本。

## 目录

| 目录 | 内容 |
|---|---|
| src | 主方法，pipeline.py为入口，refinement.py为NSPR |
| scripts | 单记录运行、批量仿真、仿真生成、冒烟核验 |
| configs | 冻结主方法协议说明 |
| data/simulation | 独立合成CSV和SHA清单 |
| docs | 数据格式与投稿期公开范围 |

src/energy.py仅因模型依赖保留计数接口，本公开运行不输出能耗结论，运算代理不能当成硬件实测。仿真使用固定独立范围和高斯扰动，不读取私有数据或经验分布，可用公开生成器和种子重新生成核对SHA。公开边界见 [发布范围](docs/release_scope.md)。许可MIT，见 [LICENSE](LICENSE)。
