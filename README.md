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
[2507] [GTA1: GUI Test-time Scaling Agent](https://arxiv.org/pdf/2507.05791)
> planner + judge + grouder  
> planner输出多个candidate action，judge从中选择action，grouder转换为带坐标的动作
> rl训练grouder，o3作为planner+judge，osworld能达到45%

🌟 [2505][Scaling Computer-Use Grounding via User Interface Decomposition and Synthesis](https://osworld-grounding.github.io)
> 用JEDI-7B作为grouding模型，配合GPT-4o规划器，在OSWorld基准上的成功率高达27%  
> 训练后的模型（如JEDI-7B）​不直接输出PyAutoGUI代码，而是预测结构化动作（38页附录A.4）

## 3. Data Synth
### 3.1 Computer
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

[2412] [Aguvis: Unified Pure Vision Agents for Autonomous GUI Interaction](https://aguvis-project.github.io)
> stage-1: 基础定位数据（103.6万条）  
> stage-2: 规划推理轨迹数据（3.5万条），浏览器6,253，移动端27,647  
> 轨迹数据由GPT-4o生成思考链三元组(obs, reason, action)

[2412] [ICLR2025] [AgentTrek: Agent Trajectory Synthesis via Guiding Replay with Web Tutorials](https://agenttrek.github.io)
> 爬了1880w教程，LLM筛选出23w，教程喂给VLM取执行任务，获得1w轨迹
> hf数据集应该是可用的，虽然hf的可视化展示是单轮的，但论文提到轨迹是多轮的，但以单轮的格式存储

## 4. Reward Model
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
[2504] [UI-TARS-1.5](https://seed-tars.com/1.5/)
> 纯秀肌肉，没有披露更多细节，只提到了“UI-TARS-1.5 integrates advanced reasoning enabled by reinforcement learning”  
> osworld上分数42.5，7b的则是28


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