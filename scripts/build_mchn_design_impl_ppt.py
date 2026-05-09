import os
import zipfile
from pathlib import Path

from build_mchn_report_ppt import (
    Slide,
    add_footer,
    app_xml,
    content_types,
    core_xml,
    package_rels,
    presentation_rels,
    presentation_xml,
    slide_layout_rels,
    slide_layout_xml,
    slide_master_rels,
    slide_master_xml,
    theme_xml,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "现代Hopfield网络污染车牌字符识别_实验设计与实现_学术版.pptx"


def section_slide(title, subtitle, idx):
    s = Slide(dark=True)
    s.text(0.85, 2.15, 1.5, 0.5, f"{idx:02d}", size=26, color="60A5FA", bold=True)
    s.text(0.85, 2.88, 11.3, 0.7, title, size=34, color="FFFFFF", bold=True)
    s.rect(0.88, 3.85, 2.0, 0.04, fill="3B82F6", line=None)
    s.text(0.88, 4.25, 10.8, 0.65, subtitle, size=21, color="BFDBFE")
    return s


def formula_box(slide, x, y, w, h, title, formula, note=None):
    slide.rect(x, y, w, h, fill="F8FAFC", line="CBD5E1", radius=True)
    slide.text(x + 0.18, y + 0.16, w - 0.36, 0.32, title, size=15, color="2563EB", bold=True)
    slide.text(x + 0.20, y + 0.58, w - 0.40, 0.75, formula, size=22, color="111827", bold=True)
    if note:
        slide.text(x + 0.20, y + h - 0.58, w - 0.40, 0.38, note, size=12.5, color="64748B")


def build_slides():
    slides = []

    s = Slide(dark=True)
    s.text(0.75, 1.05, 11.8, 0.86, "现代 Hopfield 网络污染车牌字符识别", size=33, color="FFFFFF", bold=True)
    s.text(0.78, 2.02, 11.0, 0.55, "实验设计与实现专题汇报", size=25, color="BFDBFE", bold=True)
    s.rect(0.78, 2.9, 11.7, 0.02, fill="3B82F6", line=None)
    s.text(0.78, 3.48, 11.3, 1.55, "本汇报只讨论实验系统如何设计、模型如何实现、污染如何建模、评价协议如何构造；不展开端到端整车识别和实验结果分析。", size=23, color="E5E7EB")
    s.text(0.8, 6.55, 8.8, 0.34, "关键词：Associative Memory · Modern Hopfield Network · Polluted Character Recognition", size=12.5, color="9CA3AF")
    slides.append(s)

    s = Slide("研究任务形式化", "1. 问题定义")
    s.text(0.75, 1.55, 5.75, 4.55, "目标是研究在字符图像受到污染时，现代 Hopfield 网络是否能够通过模板记忆完成稳定识别。\n\n任务不依赖整车检测，而是聚焦归一化字符层面的鲁棒识别：给定一个受到污染的车牌字符图像，预测其所属类别。", size=18, color="1F2937")
    formula_box(
        s,
        6.9,
        1.55,
        5.25,
        1.55,
        "输入与输出",
        "x ∈ [0,1]^D,   y ∈ {1,…,C}",
        "D=64×32=2048，C 为字符类别数",
    )
    formula_box(
        s,
        6.9,
        3.45,
        5.25,
        1.65,
        "污染识别目标",
        "ŷ = fθ( Pτ,s(x) )",
        "Pτ,s 表示类型 τ、强度 s 的污染算子",
    )
    s.text(6.95, 5.45, 5.0, 0.7, "实验关注：当 τ 和 s 变化时，fθ 的识别准确率、Top-K 命中率和混淆模式如何变化。", size=15, color="475569")
    slides.append(s)

    s = Slide("符号体系与数据结构", "1. 问题定义")
    rows = [
        ["符号", "含义", "项目中的对应实现"],
        ["x", "干净字符图像向量", "TemplateLoader 加载并归一化"],
        ["x̃", "污染后的查询字符", "PollutedCharDataset / CharPolluter"],
        ["M ∈ R^{K×D}", "Hopfield 记忆矩阵", "每行是一个模板字符"],
        ["L ∈ {1,…,C}^K", "模板标签向量", "template_labels"],
        ["β", "逆温度/注意力锐化系数", "MCHN beta 参数"],
        ["φ(·)", "特征变换函数", "binary / centered / profile 等"],
    ]
    s.table(0.75, 1.45, [1.55, 4.1, 5.65], 0.62, rows)
    s.text(0.9, 6.15, 11.0, 0.48, "这一符号体系把代码中的数据加载、污染生成、模型检索和评价指标统一到同一个数学框架中。", size=16, color="334155", bold=True)
    slides.append(s)

    slides.append(section_slide("模板记忆构建", "从字符图片到现代 Hopfield 记忆矩阵", 1))

    s = Slide("字符归一化流程", "2. 模板记忆构建")
    steps = [
        ("灰度化", "I ← Gray(I)"),
        ("尺寸统一", "I ∈ R^{64×32}"),
        ("阈值二值化", "B = 1[I > μ+λσ]"),
        ("前景方向校正", "边界过亮则反色"),
        ("紧致裁剪", "依据前景 bbox"),
        ("居中重采样", "放入 64×32 canvas"),
    ]
    for i, (name, eq) in enumerate(steps):
        x = 0.58 + (i % 3) * 4.15
        y = 1.55 + (i // 3) * 1.75
        s.rect(x, y, 3.35, 1.1, fill="FFFFFF", line="CBD5E1", radius=True)
        s.text(x + 0.18, y + 0.16, 2.85, 0.32, name, size=17, color="0F172A", bold=True)
        s.text(x + 0.18, y + 0.58, 2.85, 0.28, eq, size=14.5, color="2563EB", bold=True)
        if i % 3 != 2:
            s.text(x + 3.46, y + 0.42, 0.35, 0.25, "→", size=22, color="2563EB", bold=True)
    formula_box(
        s,
        1.05,
        5.28,
        11.1,
        0.92,
        "最终模板向量",
        "mᵢ = vec( Normalize(Iᵢ) ) ∈ R²⁰⁴⁸",
        None,
    )
    slides.append(s)

    s = Slide("记忆矩阵与类别映射", "2. 模板记忆构建")
    formula_box(s, 0.9, 1.52, 5.45, 1.35, "模板堆叠", "M = [m₁; m₂; …; mK] ∈ R^{K×D}", "K 为模板数，D=2048")
    formula_box(s, 0.9, 3.15, 5.45, 1.35, "标签映射", "L = [ℓ₁,ℓ₂,…,ℓK],  ℓᵢ∈{1,…,C}", "省份简称、数字、字母统一编码")
    formula_box(s, 0.9, 4.78, 5.45, 1.35, "类别索引", "g: label ↔ class_id", "idx_to_label / label_to_idx")
    s.text(6.9, 1.62, 5.05, 3.6, "设计意图：\n\n• 现代 Hopfield 网络的“参数”主要来自模板记忆，而不是大规模梯度训练。\n• 新增字符样本时，只需将模板写入 M，并同步更新标签映射。\n• 多模板同类存储可以表达字体、笔画粗细和形态差异。", size=18, color="1F2937")
    s.text(6.95, 5.62, 5.0, 0.45, "对应实现：dataset/lp_dataset.py 中 TemplateLoader。", size=13.5, color="64748B")
    slides.append(s)

    s = Slide("记忆增强策略", "2. 模板记忆构建")
    s.text(0.78, 1.5, 5.5, 4.75, "为了缓解干净模板与污染查询之间的形态差异，我在实验中构造了标签保持的模板增强记忆。\n\n增强类型包括：\n• 上下左右 1 像素平移\n• 膨胀与腐蚀\n• 轻度平均模糊\n• 旋转、剪切、横向缩放\n\n增强后的记忆仍然用于 Hopfield 检索，不改变类别标签。", size=17.5)
    formula_box(s, 6.65, 1.52, 5.45, 1.2, "增强集合", "T = {Id, shift, dilate, erode, blur, affine}", None)
    formula_box(s, 6.65, 3.05, 5.45, 1.35, "增强记忆", "M⁺ = ⋃_{Tj∈T} Tj(M)", "标签同步复制：L⁺ = repeat(L, |T|)")
    formula_box(s, 6.65, 4.75, 5.45, 1.22, "实验目的", "提高 φ(x̃) 与 φ(M⁺) 的可匹配性", "尤其针对轻微位移和形变")
    slides.append(s)

    slides.append(section_slide("污染模型设计", "用可控参数模拟不同类型的字符退化", 2))

    s = Slide("统一污染建模框架", "3. 污染模型设计")
    s.text(0.8, 1.55, 5.7, 3.95, "污染生成不是随机随意破坏，而是被定义为一个参数化算子族。\n\n每种污染类型 τ 对应一个图像变换 Pτ,s，其中 s∈[0,1] 为污染强度。实验通过改变 s 观察模型性能随污染程度的退化曲线。", size=18)
    formula_box(s, 6.9, 1.55, 5.15, 1.3, "污染查询", "x̃ = Pτ,s(x)", "τ∈{mask, noise, salt, blur, fog, dirt, affine}")
    formula_box(s, 6.9, 3.2, 5.15, 1.3, "实验强度", "s ∈ {0, 0.1, 0.2, 0.4, 0.6, 0.8}", "覆盖无污染到严重污染")
    formula_box(s, 6.9, 4.85, 5.15, 1.2, "混合污染", "Pmix,s = Pτr,s ∘ … ∘ Pτ1,s", "随 s 增大组合 1~3 种污染")
    slides.append(s)

    s = Slide("像素级污染公式", "3. 污染模型设计")
    formula_box(s, 0.72, 1.45, 5.75, 1.18, "高斯噪声 noise", "x̃ = clip(x + ε),  ε∼N(0, σ²)", "σ = 0.03 + 0.30s")
    formula_box(s, 6.85, 1.45, 5.75, 1.18, "椒盐噪声 salt_pepper", "x̃j ∈ {0,1} with probability p", "p = 0.01 + 0.22s")
    formula_box(s, 0.72, 3.15, 5.75, 1.18, "雾化 fog", "x̃ = (1−α)x + α·1", "α = 0.15 + 0.45s")
    formula_box(s, 6.85, 3.15, 5.75, 1.18, "模糊 blur", "x̃ = Gk * x", "k ≈ 3 + 6s，取奇数核")
    s.text(0.9, 5.35, 11.2, 0.62, "这些污染主要改变像素强度、边缘清晰度和前景/背景对比度，用于测试模型对低层图像退化的鲁棒性。", size=17, color="334155", fill="EFF6FF", line="BFDBFE")
    slides.append(s)

    s = Slide("结构级污染公式", "3. 污染模型设计")
    formula_box(s, 0.72, 1.45, 5.72, 1.28, "遮挡 mask", "x̃ = x ⊙ (1−R) + cR", "R 为随机矩形块，块数与面积随 s 增大")
    formula_box(s, 6.88, 1.45, 5.72, 1.28, "污渍 dirt", "x̃ = x ⊙ (1−D) + ρD", "D 为随机圆形斑点，ρ∈[0,0.35]")
    formula_box(s, 0.72, 3.28, 5.72, 1.28, "仿射 affine", "x̃(u) = x(A_s^{-1}u)", "A_s 由旋转、平移、缩放、剪切组成")
    formula_box(s, 6.88, 3.28, 5.72, 1.28, "强度控制", "|θ|≤10s, |Δ|≤3s, scale≈1±0.12s", "剪切幅度约 ±6s")
    s.text(0.88, 5.55, 11.15, 0.55, "结构级污染更容易破坏字符的笔画拓扑，因此也是识别任务中更具有挑战性的部分。", size=17, color="334155", bold=True)
    slides.append(s)

    slides.append(section_slide("现代 Hopfield 检索模型", "从模板记忆到类别预测的数学实现", 3))

    s = Slide("现代 Hopfield 更新规则", "4. MCHN 模型实现")
    s.text(0.78, 1.5, 5.65, 4.75, "现代 Hopfield 网络在连续空间中执行单步或少步检索。本项目采用单步检索形式：查询字符与模板记忆计算相似度，经 softmax 得到注意力分布，再对记忆矩阵加权求和。\n\n从实现上看，它等价于一种以模板为 Key/Value 的注意力检索层。", size=18)
    formula_box(s, 6.78, 1.42, 5.38, 1.18, "相似度", "S(q,M) = φ(q) · φ(M)ᵀ", "默认使用归一化点积")
    formula_box(s, 6.78, 2.95, 5.38, 1.18, "注意力", "a = softmax( β S(q,M) )", "β 控制检索分布的尖锐程度")
    formula_box(s, 6.78, 4.48, 5.38, 1.18, "检索重构", "z = aM", "z 是由记忆模板加权得到的重构字符")
    slides.append(s)

    s = Slide("类别评分与预测", "4. MCHN 模型实现")
    formula_box(s, 0.78, 1.52, 5.55, 1.35, "模板级预测", "i* = argmaxᵢ aᵢ", "最相似记忆模板索引")
    formula_box(s, 0.78, 3.2, 5.55, 1.35, "类别级得分", "score_c(q)= max_{i:Lᵢ=c} βSᵢ(q)", "同类多模板取最大响应")
    formula_box(s, 0.78, 4.88, 5.55, 1.35, "最终类别", "ŷ = argmax_c score_c(q)", "输出字符类别")
    s.text(6.86, 1.55, 5.1, 3.5, "为什么采用类别级最大得分：\n\n• 同一个字符类别可能有多张模板。\n• 污染后查询只需与其中一个高质量模板匹配即可。\n• 最大池化比平均池化更适合多形态模板记忆。\n• 这种设计也方便加入模板增强。", size=17.5)
    s.text(6.9, 5.42, 5.0, 0.45, "对应实现：class_attention_projection_scores / ensemble_hopfield_scores。", size=13.2, color="64748B")
    slides.append(s)

    s = Slide("特征变换设计", "4. MCHN 模型实现")
    rows = [
        ["模式", "数学形式", "设计作用"],
        ["raw", "φ(x)=x", "保留原始灰度信息"],
        ["centered", "φ(x)=x−mean(x)", "抑制整体亮度偏移"],
        ["bipolar", "φ(x)=2x−1", "转为双极性模式"],
        ["binary", "φ(x)=2·1[x>μ+λσ]−1", "突出前景笔画"],
        ["profile", "φ(x)=[pool,row,col,edge,ink]", "提取粗粒度结构"],
        ["hybrid_shape", "φ(x)=[pooled,projection,edge]", "兼顾局部与形状特征"],
    ]
    s.table(0.65, 1.42, [1.65, 4.35, 5.3], 0.58, rows)
    s.text(0.88, 6.08, 11.0, 0.55, "特征设计思想：用结构信息补偿像素模板匹配在污染、缺笔和轻微形变下的脆弱性。", size=16, color="334155", bold=True)
    slides.append(s)

    s = Slide("多特征 Hopfield 集成", "4. MCHN 模型实现")
    s.text(0.78, 1.48, 5.5, 4.78, "单一特征模式很难同时适应所有污染类型。因此我构造了多个 Hopfield 检索器，每个检索器共享同一记忆矩阵，但使用不同特征变换、相似度度量和 β。\n\n各检索器输出类别 log-probability 后进行平均融合。", size=18)
    formula_box(s, 6.72, 1.45, 5.45, 1.35, "第 r 个检索器", "pr(c|q)=softmax(scoreᵣ(q))c", None)
    formula_box(s, 6.72, 3.1, 5.45, 1.45, "对数域融合", "log p(c|q)=log( 1/R Σr pr(c|q) )", "代码中用 logsumexp 稳定计算")
    formula_box(s, 6.72, 4.92, 5.45, 1.18, "集成预测", "ŷ = argmax_c log p(c|q)", "提升跨污染鲁棒性")
    slides.append(s)

    slides.append(section_slide("基线方法与评价协议", "保证实验可对照、可复现、可解释", 4))

    s = Slide("对照方法设计", "5. 基线与评价协议")
    formula_box(s, 0.72, 1.4, 5.75, 1.25, "传统 Hopfield", "W = (1/D) Σᵢ yyᵀ,  y^{t+1}=sign(Wyᵗ)", "离散双极性关联记忆基线")
    formula_box(s, 6.85, 1.4, 5.75, 1.25, "最近邻", "ŷ = L_{argmaxᵢ cos(q,mᵢ)}", "直接模板匹配基线")
    formula_box(s, 0.72, 3.15, 5.75, 1.25, "类别原型", "μc = mean({mᵢ | Lᵢ=c})", "每类一个平均模板")
    formula_box(s, 6.85, 3.15, 5.75, 1.25, "CNN", "minθ  E[ CE(hθ(x̃), y) ]", "监督学习分类器基线")
    s.text(0.9, 5.35, 11.05, 0.55, "基线覆盖了关联记忆、非参数模板匹配、类别原型和神经网络分类器四类思路。", size=16.5, color="334155", bold=True)
    slides.append(s)

    s = Slide("实验划分与采样协议", "5. 基线与评价协议")
    s.text(0.78, 1.5, 5.55, 4.85, "为了避免同一模板同时出现在训练和测试中，实验采用按类别分层的模板划分。\n\n训练模板用于构建记忆矩阵和训练 CNN；测试模板通过污染算子生成查询样本。\n\n每个污染类型、每个强度等级均独立采样，得到可比较的准确率曲线。", size=18)
    formula_box(s, 6.76, 1.52, 5.4, 1.25, "分层划分", "D = Dtrain ∪ Dtest,  Dtrain∩Dtest=∅", "每类保留测试模板")
    formula_box(s, 6.76, 3.14, 5.4, 1.25, "测试查询", "x̃j = Pτ,s(xj),  xj∈Dtest", "虚拟样本按需生成")
    formula_box(s, 6.76, 4.76, 5.4, 1.25, "污染网格", "τ × s = 7 × 6", "七类污染、六档强度")
    slides.append(s)

    s = Slide("评价指标", "5. 基线与评价协议")
    formula_box(s, 0.75, 1.45, 5.65, 1.25, "Top-1 Accuracy", "Acc = (1/N) Σj 1[ŷj = yj]", "主指标")
    formula_box(s, 6.82, 1.45, 5.65, 1.25, "Top-K Accuracy", "Acc@K = (1/N) Σj 1[yj∈TopK(qj)]", "观察候选排序质量")
    formula_box(s, 0.75, 3.15, 5.65, 1.25, "混淆矩阵", "Cuv = Σj 1[yj=u ∧ ŷj=v]", "分析相似字符误识别")
    formula_box(s, 6.82, 3.15, 5.65, 1.25, "鲁棒性曲线", "Aτ(s) = Acc( f, Pτ,s(Dtest) )", "刻画性能随污染强度退化")
    s.text(0.92, 5.35, 11.0, 0.58, "评价目标不是只报告单点准确率，而是形成污染类型、污染强度、方法类别之间的系统性对比。", size=16.5, color="334155", bold=True)
    slides.append(s)

    s = Slide("消融实验设计", "5. 基线与评价协议")
    rows = [
        ["消融对象", "对照设置", "验证问题"],
        ["特征模式", "raw / binary / shape / profile", "哪类表征更适合污染字符"],
        ["模型集成", "single vs ensemble", "多特征融合是否有效"],
        ["记忆增强", "NoAug vs Aug", "模板增强是否提升匹配"],
        ["β 参数", "β∈{0.1,0.5,1,2,5,10}", "注意力锐度如何影响检索"],
        ["容量实验", "不同 K / D 比例", "现代 Hopfield 的存储能力"],
    ]
    s.table(0.75, 1.5, [2.3, 4.1, 5.1], 0.72, rows)
    s.text(0.9, 5.85, 11.1, 0.55, "消融实验服务于方法论证明：不是只看最终模型，而是解释每个设计选择为什么被保留。", size=16.5, color="334155", bold=True)
    slides.append(s)

    slides.append(section_slide("工程实现结构", "将数学流程落到可复现实验代码", 5))

    s = Slide("代码模块对应关系", "6. 工程实现结构")
    rows = [
        ["模块", "主要职责"],
        ["dataset/lp_dataset.py", "字符归一化、模板加载、污染字符数据集、污染算子"],
        ["models/mchn.py", "现代 Hopfield 网络、特征变换、相似度计算、检索输出"],
        ["models/traditional_hopfield.py", "传统 Hopfield 对照模型"],
        ["main_eval.py", "训练 CNN、构建记忆、运行鲁棒性/消融/容量实验"],
        ["utils/metric_visuals.py", "曲线、热力图、混淆矩阵等可视化"],
        ["app.py", "交互式单字符污染测试界面"],
    ]
    s.table(0.72, 1.42, [3.7, 7.85], 0.68, rows)
    s.text(0.9, 6.15, 11.0, 0.45, "实现结构围绕“数据构造 → 模型检索 → 对照评估 → 可视化输出”四个环节组织。", size=16, color="334155", bold=True)
    slides.append(s)

    s = Slide("实验主流程伪代码", "6. 工程实现结构")
    s.rect(0.85, 1.45, 11.55, 4.95, fill="111827", line=None, radius=True)
    code = (
        "Input: template roots, pollution set T, severity set S\n"
        "1  loader ← TemplateLoader(chars2, charsChinese)\n"
        "2  train_idx, test_idx ← StratifiedSplit(loader.labels)\n"
        "3  M_train, L_train ← templates[train_idx]\n"
        "4  M_aug, L_aug ← AugmentMemory(M_train, L_train)\n"
        "5  H ← BuildHopfieldEnsemble(M_aug)\n"
        "6  for τ in T:\n"
        "7      for s in S:\n"
        "8          Q ← { Pτ,s(x) | x ∈ test templates }\n"
        "9          scores ← EnsembleHopfieldScores(H, Q)\n"
        "10         Acc, Acc@K, Confusion ← Evaluate(scores, labels)\n"
        "11         Save curves / csv / figures"
    )
    s.text(1.15, 1.72, 10.9, 4.35, code, size=16, color="E5E7EB")
    slides.append(s)

    s = Slide("可复现性设计", "6. 工程实现结构")
    s.text(0.8, 1.55, 5.55, 4.65, "为了保证实验可重复，项目在实现中固定了关键控制变量：\n\n• 随机种子 seed\n• 训练/测试模板划分比例\n• 污染类型与强度网格\n• 每个强度等级的采样数量\n• CNN 训练轮数、batch size\n• MCHN β 值和特征模式\n• 输出目录和 CSV/PNG 文件命名规则", size=17.5)
    formula_box(s, 6.78, 1.55, 5.35, 1.2, "统一实验配置", "E = (seed, split, τ, s, N, β, φ)", None)
    formula_box(s, 6.78, 3.08, 5.35, 1.35, "结果产物", "R = {csv, curve, heatmap, confusion}", "便于论文图表复用")
    formula_box(s, 6.78, 4.88, 5.35, 1.2, "复现实验入口", "python main_eval.py --pollution all", "按参数控制不同实验")
    slides.append(s)

    s = Slide("本部分小结", "实验设计与实现总结")
    s.rect(0.85, 1.5, 11.55, 4.65, fill="FFFFFF", line="CBD5E1", radius=True)
    s.text(
        1.15,
        1.88,
        10.95,
        3.72,
        "• 实验对象被形式化为污染字符 x̃ 的分类问题，避免把研究重心分散到整车检测。\n"
        "• 模板字符经过统一归一化后构成现代 Hopfield 记忆矩阵 M。\n"
        "• 污染由参数化算子 Pτ,s 描述，实现了像素级和结构级退化建模。\n"
        "• MCHN 通过 φ(q) 与 φ(M) 的相似度、softmax 注意力和类别级最大池化完成检索识别。\n"
        "• 对照方法、消融实验和评价指标共同构成可复现、可解释的实验设计。",
        size=19,
        color="1F2937",
    )
    slides.append(s)

    for i, slide in enumerate(slides, start=1):
        if i not in (1, 4, 8, 12, 16, 20, len(slides)):
            add_footer(slide, i)
    return slides


def write_pptx():
    slides = build_slides()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types(len(slides)))
        zf.writestr("_rels/.rels", package_rels())
        zf.writestr("ppt/presentation.xml", presentation_xml(len(slides)))
        zf.writestr("ppt/_rels/presentation.xml.rels", presentation_rels(len(slides)))
        zf.writestr("ppt/slideMasters/slideMaster1.xml", slide_master_xml())
        zf.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", slide_master_rels())
        zf.writestr("ppt/slideLayouts/slideLayout1.xml", slide_layout_xml())
        zf.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", slide_layout_rels())
        zf.writestr("ppt/theme/theme1.xml", theme_xml())
        zf.writestr("docProps/core.xml", core_xml())
        zf.writestr("docProps/app.xml", app_xml(len(slides)))
        for i, slide in enumerate(slides, start=1):
            zf.writestr(f"ppt/slides/slide{i}.xml", slide.xml())
            zf.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", slide.rels_xml())
    print(OUT)


if __name__ == "__main__":
    os.chdir(ROOT)
    write_pptx()
