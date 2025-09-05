# awesome-computer-using-agents

1. [Benchmark](#1-benchmark)
2. [Architecture](#2-architecture)
3. [Data Synth](#3-data-synth)
4. [Reward Model](#4-reward-model)
5. [Training Algorithm](#5-training-algorithm)
6. [Others](#6-others)

## 1. Benchmark
[2508] [OSWorld-Verified](https://xlang.ai/blog/osworld-verified)
> 修正了OSWorld的一些bug：问题模糊、网站打不开；  
> 指出社区下一步重点是构造轨迹数据、验证器、更有效的测评  
> **开源了[现有的CUA在osworld测试集上的测评轨迹](https://huggingface.co/datasets/xlangai/ubuntu_osworld_verified_trajs)，挺有参考价值的**

🌟 [2506] [OSWorld-Human: Benchmarking the Efficiency of Computer-Use Agents](https://arxiv.org/pdf/2506.16042)
> 人类30秒完成的任务（如调整文档行距），代理需耗时12分钟  
> 人工标注OSWorld的369个真实任务，构建人类操作轨迹，发现人类操作步骤比AI代理少1.4–2.7倍  
> 提出新评估指标：WES（加权效率得分）

[2504] [Computer Agent Arena: Compare & Test AI Agents on Crowdsourced Real-World Computer Use Tasks](https://arena.xlang.ai)
> 针对CUA的竞技场  
> 博客里提到会通过这个arena**收集用户偏好的指令数据**，并在未来开源

[2404] [OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments](https://os-world.github.io)
> 369个task，每个task对应一个脚本作为验证器 

## 2. Architecture
🌟 [2508] [CoAct-1: Computer-using Agents with Coding as Actions](https://arxiv.org/pdf/2508.03923)
> 三大智能体分工​: 1) Orchestrator​：任务分解与决策中枢; 2) Programmer​：通过代码与系统直接交互; 3) GUI Operator​：基于视觉模型执行图形操作  
> CoAct-1 vs GTA-1: 成功率60.76 vs 45.2，平均步数10.15 vs 15.22

[2507] [GTA1: GUI Test-time Scaling Agent](https://arxiv.org/pdf/2507.05791)
> planner + judge + grouder  
> planner输出多个candidate action，judge从中选择action，grouder转换为带坐标的动作
> rl训练grouder，o3作为planner+judge，osworld能达到45%

🌟 [2505][Scaling Computer-Use Grounding via User Interface Decomposition and Synthesis](https://osworld-grounding.github.io)
> 用JEDI-7B作为grouding模型，配合GPT-4o规划器，在OSWorld基准上的成功率高达27%  
> jedi-7b-o3在OSWorld上则是高达50
> 训练后的模型（如JEDI-7B）​不直接输出PyAutoGUI代码，而是预测结构化动作（38页附录A.4）

[2504][Agent S2: A Compositional Generalist-Specialist Framework for Computer Use Agents](https://arxiv.org/pdf/2504.00906)


## 3. Data Synth
### 3.1 Computer
🌟 [2509][UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning](https://arxiv.org/pdf/2509.02544)


🌟 [2508][Mobile-Agent-v3: Foundamental Agents for GUI Automation](https://arxiv.org/pdf/2508.15144)
> 预训练+sft得到gui-owl-7b在osworld上29.4；在osworld上专门rl后是34.9  


[2508][ComputerRL: Scaling End-to-End Online Reinforcement Learning for Computer Use Agents](https://arxiv.org/pdf/2508.14040)
> We manually collect extensive, well-defined tasks and corresponding evaluation functions. 通过人工构建任务、验证函数  
> If the trajectory successfully solves the task, we assign a reward of 1 to every action that is both correctly formatted and substantially contributes to the solution. 验证函数应该是复杂到每个action是否对任务完成有益。


[2508][SEA: Self-Evolution Agent with Step-wise Reward for Computer Use](https://arxiv.org/pdf/2508.04037)
> 7b模型在osworld上30.1，比SE-Agent、OpenCUA都更强  
> 基于QWen2.5-VL-72B训练出来的step model可以判断action的正确性，但没展示prompt  
> 通过few-shot in-context Learning去生产task

🌟 [2508] [SEAgent: Self-Evolving Computer Use Agent with
Autonomous Learning from Experience](https://arxiv.org/pdf/2508.04700)
> Curriculum Generator: 生成任务，搞了4245个任务  
> World State Model: 也就是reward model, 用的是基于osworld chome的43个task得到的860条轨迹训练qwen2.5vl-7b得到的

[2508] [VeriGUI: Verifiable Long-Chain GUI Dataset](https://arxiv.org/pdf/2508.04026)
> VeriGUI 构建了首个支持 ​长链复杂任务​（数百步操作）和 ​子任务级验证​ 的 GUI 数据集，覆盖桌面和网页环境。  
> 数据集持续更新中，目前只有130网页任务，桌面任务即将发布。  
> 桌面验证机制基于截图和系统属性的状态检查，由人类专家在标注过程中实现函数化

🌟 [2508] [OpenCUA: Open Foundations for Computer-Use Agents](https://opencua.xlang.ai)
> 634名标注员针对200+应用/网站，标记了20k+任务的轨迹  
> 开源了标注软件、数据集  
> 纯sft训练qwen-vl-32b，osworld上分数34.8

[2506] [AgentSynth: Scalable Task Generation for Generalist Computer-Use Agents](https://arxiv.org/pdf/2506.14205)
> 利用信息不对称，链式生成子任务​，组合成复杂任务。任务生成成功率98%，但评估时SOTA模型在Level 6任务成功率仅4%​​  
> 可控任务难度: 通过子任务数量精确调控复杂度，Level 6任务平均需40–60步操作  
> ​高真实性任务: 人工评估显示：任务可行性87%​、子任务连贯性91%​、角色相关性94%​  
> 揭示LLM代理三大缺陷：鼠标点击不精准（59.1%动作）、状态跟踪错误、错误恢复能力缺失（第5.3节）

[2505] [Scaling Computer-Use Grounding via User Interface Decomposition and Synthesis](https://osworld-grounding.github.io)
> 大规模合成数据集JEDI​ (400w) 和精标的基准测试OSWORLD-G​ (564)  
> 通过UI构造grouding数据，并处理成(图像+指令→动作)的格式  
> 通过指令-截图错配生成不可行动作样本，增强模型拒绝能力
> 用JEDI-7B作为grouding模型，配合GPT-4o规划器，在OSWorld基准上的成功率高达27%

[2505][ZeroGUI: Automating Online GUI Learning at Zero Human Cost](https://arxiv.org/pdf/2505.23762)
> 复用用osworld的task config，用LLM生成更多指令
> task verifier用qwen2.5vl-32b


🌟 [2501] [UI-TARS: Pioneering Automated GUI Interaction with Native Agents](https://arxiv.org/pdf/2501.12326)
> grounding数据从公开数据集中收集  
> 轨迹数据两部分：145k移动端轨迹，团队标注的PC轨迹  
> 用VLM给轨迹数据打上思维链  
> 迭代【训练-推轨迹-校正轨迹】这个过程  
> 校正轨迹通过多级过滤实现：1）规则启发式：移除无效动作；2）VLM评分：过滤低分轨迹；3）人工审核：截断错误步骤，保留有效前缀  
> 两种dpo轨迹对：1）公式6，错误步骤-正确步骤；2）公式7，错误发生后，不采取补救-采取补救

### 3.2 Browser
[2507] [WebShaper: Agentically Data Synthesizing via
Information-Seeking Formalization](https://arxiv.org/pdf/2507.15061)
> formulation-driven data synth，将question用集合交并的语言形式化  
> 基于形式化描述，利用expander agent拓展seed question

[2504] [Breaking the Data Barrier – Building GUI Agents Through Task Generalization](https://arxiv.org/pdf/2504.10127)
> 针对Qwen2-VL-7B-Instruct，提出在预训练和微调之间引入一个中间训练阶段，使用11种不同任务的数据集（包括多模态和纯文本任务），以提升模型的基础能力（如推理、感知和规划），对GUI任务帮助很大  
> 纯文本数学数据（MathInstruct）在WebArena基准上提升5.6%，在AndroidWorld上提升5.4%  
> 多模态数学数据提升AndroidWorld性能6.3%

[2502] [InSTA: Towards Internet-Scale Training For Agents](https://arxiv.org/abs/2502.06776)
> 自动化三级数据流水线: LLM任务生成器、LLM轨迹生成器​、LLM轨迹过滤器  
> 基于浏览器沙盒实例​生产真实的数据


[2412] [Aguvis: Unified Pure Vision Agents for Autonomous GUI Interaction](https://aguvis-project.github.io)
> stage-1: 基础定位数据（103.6万条）  
> stage-2: 规划推理轨迹数据（3.5万条），浏览器6,253，移动端27,647  
> 轨迹数据由GPT-4o生成思考链三元组(obs, reason, action)

[2412] [ICLR2025] [AgentTrek: Agent Trajectory Synthesis via Guiding Replay with Web Tutorials](https://agenttrek.github.io)
> 爬了1880w教程，LLM筛选出23w，教程喂给VLM取执行任务，获得1w轨迹
> hf数据集应该是可用的，虽然hf的可视化展示是单轮的，但论文提到轨迹是多轮的，但以单轮的格式存储

### 3.3 Mobile


## 4. Reward Model
[2508][ComputerRL: Scaling End-to-End Online Reinforcement Learning for Computer Use Agents](https://arxiv.org/pdf/2508.14040)
> We manually collect extensive, well-defined tasks and corresponding evaluation functions. 通过人工构建任务、验证函数  
> If the trajectory successfully solves the task, we assign a reward of 1 to every action that is both correctly formatted and substantially contributes to the solution. 验证函数应该是复杂到每个action是否对任务完成有益。

[2506][Agent-RewardBench: Towards a Unified Benchmark for Reward Modeling across Perception, Planning, and Safety in Real-World Multimodal Agents](https://arxiv.org/pdf/2506.21252)
> 覆盖agent任务太多样：网页导航（417）​、具身智能（317）、旅行规划（180）

[2505][ZeroGUI: Automating Online GUI Learning at Zero Human Cost](https://arxiv.org/pdf/2505.23762)
> task verifier用qwen2.5vl-32b

🌟 [2505] [UI-Genie: A Self-Improving Approach for Iteratively Boosting MLLM-based Mobile GUI Agents](https://arxiv.org/pdf/2505.21496)
> 多粒度奖励模型​：同时支持step-level和outcome-level评估
> 核心要点在于如何搞到数据来训练reward model

[2504] [AgentRewardBench: Evaluating Automatic Evaluations of Web Agent Trajectories](https://arxiv.org/abs/2504.08942)
> 在不同浏览器benchmarkWebArena/VisualWebArena/AssistantBench/WorkArena/WorkArena++上推了GPT-4o/Claude 3.7S/Qwen2.5-VL/Llama 3.3的轨迹1392条，人工精心标注任务成功、副作用、重复行为  
> 基于这些GT，分析现有benchmark的评估的准确率，揭示规则评估的严重缺陷，和LLM评估的瓶颈  
> 提出新型简化评估框架，就是改了prompt

## 5. Training Algorithm

| Model | Score | CT | SFT | DPO | RFT | RLVR | Max Steps |
|-------|-------|----|----|-----|-----|------|-----------|
| uitars-2 | 47.5 | ✅ | ✅ | | ✅ | ✅ | ∞ |
| autoglm-os-9b | 47.3 | | | | ✅ | | |
| seagent-7b | 34.5 | | ✅ | ✅ | | | |
| opencua-32b/7b | 34.1/28.2 | | ✅ | | | | 50 |
| sea-7b | 30.1 | | | | ✅ | | |
| gui-owl-7b | 29.4 | | ✅ | | | | 50 |
| uitars-72b-dpo | 25.8 | | ✅ | ✅ | | | 50 |

🌟 [2509][UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning](https://arxiv.org/pdf/2509.02544)
**训练流程：**
- **CT (Curriculum Training)**: 任务教程、教学视频、in-situ annotation
- **SFT (Supervised Fine-Tuning)**: interactive annotation
- **RLVR (Reinforcement Learning with Verifier Reward)**: 基于验证器奖励的强化学习

**三类训练任务：**
- **GUI-Browsing**: Deep research任务，但不能调用搜索API，只能操作网页
- **GUI-General**: 多样化的网站操作任务
- **Gameplay**: 浏览器中的小游戏，合成游戏

**训练算法：** PPO (Proximal Policy Optimization)

**奖励设计：**
- **Gameplay**: 使用function验证
- **GUI-Browsing**: 使用LLM根据GT匹配
- **GUI-General**: 使用UI-TARS-2作为ORM (Outcome Reward Model)

**模型融合策略：** 同一个SFT模型分别用三类任务进行RL训练，然后进行参数平均 (params avg)

🌟 [2508][Mobile-Agent-v3: Foundamental Agents for GUI Automation](https://arxiv.org/pdf/2508.15144)
> 预训练+sft得到gui-owl-7b在osworld上29.4；在osworld上专门rl后是34.9  

- **Pre-training Phase（预训练阶段）**：
  - 涵盖基础 UI 理解、交互轨迹、通用推理的大规模预训练语料
  - 持续预训练 Qwen2.5-VL，强化 GUI 元素识别、动作预测、通用推理等基础能力
- **Iterative Tuning Phase（迭代调优阶段）**：
  - 在桌面、移动等真实环境中部署模型执行大规模任务
  - 将轨迹清洗、评分后转换为多样化推理数据集


[2508][ComputerRL: Scaling End-to-End Online Reinforcement Learning for Computer Use Agents](https://arxiv.org/pdf/2508.14040)
> We manually collect extensive, well-defined tasks and corresponding evaluation functions.  
> 通过人工构建任务、验证函数  

> If the trajectory successfully solves the task, we assign a reward of 1 to every action that is both correctly formatted and substantially contributes to the solution.  
> 验证函数应该是复杂到每个action是否对任务完成有益。 

**SFT**：
  - Model pool提供轨迹数据

**RLVR**：
  - **Step-Level Group Relative Policy Optimization**：
    - 一个query的所有rollout的所有step作为一个group
    - 每个step计算reward，进而计算advantage
  - **奖励设计**：
    - 若轨迹成功完成任务，则为所有格式正确且对解决方案有实质贡献的动作分配1分奖励
    - 否则，未通过验证的轨迹或格式不当的动作均得0分
  - **Entropulse策略**：
    - 收集rollout，筛选成功的
    - RL之间穿插SFT训练这些，可以提高entropy、突破性能瓶颈


[2508][SEA: Self-Evolution Agent with Step-wise Reward for Computer Use](https://arxiv.org/pdf/2508.04037)
> 7b模型在osworld上30.1，比SE-Agent、OpenCUA都更强  
> 基于QWen2.5-VL-72B训练出来的step model可以判断action的正确性，但没展示prompt  
> 通过few-shot in-context Learning去生产task

🌟 [2508] [SEAgent: Self-Evolving Computer Use Agent with
Autonomous Learning from Experience](https://arxiv.org/pdf/2508.04700)
> Curriculum Generator: 生成任务，搞了4245个任务  
> World State Model: 也就是reward model, 用的是基于osworld chome的43个task得到的860条轨迹训练qwen2.5vl-7b得到的

**SEAgent框架：**
1. **Task Initialization（任务初始化）**
2. **Autonomous Evaluation（自主评估）**
3. **RFT（Rejection Fine-Tuning）**
4. **Task Update（任务更新）**

**RFT技术细节：**
- **错误动作惩罚**：对识别出的错误动作进行惩罚性训练
- **正确动作模仿**：学习和模仿正确的操作序列



🌟 [2508] [OpenCUA: Open Foundations for Computer-Use Agents](https://opencua.xlang.ai)
> 634名标注员针对200+应用/网站，标记了20k+任务的轨迹  
> 开源了标注软件、数据集  
> 纯sft训练qwen-vl-32b，osworld上分数34.8

**轨迹数据特性：**
- **数据规模**：包含22,625条人工标注的计算机使用任务
- **平台分布**：Windows平台12K条、macOS平台5K条、Ubuntu平台5K条
- **技术规格**：屏幕分辨率涵盖720p至4K
- **复杂度**：每条轨迹平均18.6步，体现了任务的复杂性
- **覆盖范围**：数据覆盖140多个应用程序和190多个网站，常涉及多应用工作流、专业工具和不常用功能
- **数据集特色**：与现有GUI数据集相比，AGENTNET是首个兼具真实性、复杂性、多样性和多模态特性的桌面轨迹级数据集

**训练数据混合：**
- **多层级轨迹COT**：混合不同层级的轨迹思维链
  - L1（动作）
  - L2（思考 + 动作）
  - L3（观察 + 思考 + 动作）
- **数据组成**：轨迹数据、grounding数据、通用SFT数据

**训练流程：**
- **三种训练策略**：
  - **仅阶段2**：opencua-qwen2-7b和opencua-a3b
  - **阶段1+阶段2**：opencua-32b
  - **联合策略**：opencua-7b

- **具体配比**：
  - **仅阶段2**：56%轨迹数据、14%grounding数据、30%通用SFT数据
  - **阶段1+阶段2**：
    - 阶段1：grounding数据、教程式演示、状态转换描述数据、通用VL和通用text SFT数据
    - 阶段2：45%轨迹数据、20%grounding数据、35%通用数据
  - **联合策略**：20%轨迹数据、20%grounding数据、60%通用数据


[2505][ZeroGUI: Automating Online GUI Learning at Zero Human Cost](https://arxiv.org/pdf/2505.23762)
> 复用用osworld的task config，用LLM生成更多指令
> task verifier用qwen2.5vl-32b

[2504] [UI-TARS-1.5](https://seed-tars.com/1.5/)
> 没有披露太多细节，只提到了“UI-TARS-1.5 integrates advanced reasoning enabled by reinforcement learning”  
> osworld上分数42.5，7b的则是28

🌟 [2501] [UI-TARS: Pioneering Automated GUI Interaction with Native Agents](https://arxiv.org/pdf/2501.12326)
> grounding数据从公开数据集中收集  
> 轨迹数据两部分：145k移动端轨迹，团队标注的PC轨迹  
> 用VLM给轨迹数据打上思维链  
> 迭代【训练-推轨迹-校正轨迹】这个过程  
> 校正轨迹通过多级过滤实现：1）规则启发式：移除无效动作；2）VLM评分：过滤低分轨迹；3）人工审核：截断错误步骤，保留有效前缀  
> 两种dpo轨迹对：1）公式6，错误步骤-正确步骤；2）公式7，错误发生后，不采取补救-采取补救

## 6. Others
### x.1 General Agent
[2507] [Kimi K2: Open Agentic Intelligence](https://moonshotai.github.io/Kimi-K2/)

[2507] [GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning](https://arxiv.org/pdf/2507.01006)
> osworld上分数不太好只有14.7，但可能是没有专项微调的原因，潜力应该挺大

### x.2 Coding
[Qwen3-Coder: Agentic Coding in the World](https://qwenlm.github.io/blog/qwen3-coder/)

### x.3 Math


### x.4 Benchmark
🌟 [2507] [Establishing Best Practices for Building Rigorous
Agentic Benchmarks](https://arxiv.org/pdf/2507.02825)
> 指出了现有agent benchmark的问题，例如webarena就有问题  
> 提出了ABC检查表，作为检查、构建agent benchmark的原则  
> 作为一种meta-research，比较有启发性

[2507] [Deep Research Comparator: A Platform For Fine-grained Human Annotations of Deep Research Agents](https://arxiv.org/pdf/2507.05495)
> 针对deep research的arena，亮点是除了最终结果，还可以对中间步骤点赞

[2501] [ACEBench: Who Wins the Match Point in Tool Usage?](https://arxiv.org/pdf/2501.12851)