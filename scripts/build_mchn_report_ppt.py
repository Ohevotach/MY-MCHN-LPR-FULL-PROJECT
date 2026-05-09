import csv
import os
import zipfile
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "现代Hopfield网络污染车牌字符识别_汇报PPT_实验设计结果分析后续工作.pptx"

SLIDE_W = 12192000
SLIDE_H = 6858000


def emu(x):
    return int(round(x * 914400))


def pct(value):
    return f"{float(value):.1f}%"


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def text_runs(text, size=22, color="1F2937", bold=False):
    parts = str(text).split("\n")
    paragraphs = []
    for part in parts:
        paragraphs.append(
            '<a:p>'
            '<a:r>'
            f'<a:rPr lang="zh-CN" sz="{int(size * 100)}" b="{1 if bold else 0}">'
            f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
            '<a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/>'
            "</a:rPr>"
            f"<a:t>{escape(part)}</a:t>"
            "</a:r>"
            "</a:p>"
        )
    return "".join(paragraphs)


class Slide:
    def __init__(self, title=None, section=None, dark=False):
        self.parts = []
        self.rels = []
        self.next_id = 2
        self.next_rid = 2
        self.dark = dark
        bg = "F8FAFC" if not dark else "111827"
        self.rect(0, 0, 13.333, 7.5, fill=bg, line=None)
        if section:
            self.text(0.58, 0.32, 4.2, 0.34, section, size=10, color="64748B" if not dark else "9CA3AF", bold=True)
        if title:
            self.text(0.58, 0.66, 12.0, 0.55, title, size=27, color="0F172A" if not dark else "FFFFFF", bold=True)
            self.rect(0.58, 1.32, 1.25, 0.035, fill="2563EB", line=None)

    def _shape_id(self):
        val = self.next_id
        self.next_id += 1
        return val

    def rect(self, x, y, w, h, fill="FFFFFF", line="CBD5E1", radius=False):
        sid = self._shape_id()
        geom = "roundRect" if radius else "rect"
        fill_xml = '<a:noFill/>' if fill is None else f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>'
        line_xml = '<a:ln><a:noFill/></a:ln>' if line is None else f'<a:ln w="9525"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>'
        self.parts.append(
            f"""
<p:sp>
  <p:nvSpPr><p:cNvPr id="{sid}" name="Shape {sid}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
  <p:spPr>
    <a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
    <a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom>
    {fill_xml}{line_xml}
  </p:spPr>
</p:sp>"""
        )

    def text(self, x, y, w, h, text, size=20, color="1F2937", bold=False, fill=None, line=None, margin=0.08):
        sid = self._shape_id()
        fill_xml = '<a:noFill/>' if fill is None else f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>'
        line_xml = '<a:ln><a:noFill/></a:ln>' if line is None else f'<a:ln w="9525"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>'
        self.parts.append(
            f"""
<p:sp>
  <p:nvSpPr><p:cNvPr id="{sid}" name="TextBox {sid}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
  <p:spPr>
    <a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
    <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
    {fill_xml}{line_xml}
  </p:spPr>
  <p:txBody>
    <a:bodyPr wrap="square" lIns="{emu(margin)}" tIns="{emu(0.03)}" rIns="{emu(margin)}" bIns="{emu(0.03)}"><a:spAutoFit/></a:bodyPr>
    <a:lstStyle/>
    {text_runs(text, size=size, color=color, bold=bold)}
  </p:txBody>
</p:sp>"""
        )

    def pill(self, x, y, w, h, text, fill="DBEAFE", color="1D4ED8"):
        self.rect(x, y, w, h, fill=fill, line=None, radius=True)
        self.text(x + 0.05, y + 0.06, w - 0.10, h - 0.08, text, size=13, color=color, bold=True)

    def image(self, path, x, y, w, h):
        path = Path(path)
        if not path.exists():
            self.text(x, y, w, h, f"图片缺失：{path.name}", size=16, color="B91C1C", fill="FEE2E2", line="FCA5A5")
            return
        rid = f"rId{self.next_rid}"
        self.next_rid += 1
        target = f"../media/{path.name}"
        self.rels.append((rid, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image", target))
        sid = self._shape_id()
        self.parts.append(
            f"""
<p:pic>
  <p:nvPicPr><p:cNvPr id="{sid}" name="{escape(path.name)}"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>
  <p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
  <p:spPr>
    <a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
    <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
  </p:spPr>
</p:pic>"""
        )

    def table(self, x, y, col_widths, row_h, rows, header_fill="1D4ED8"):
        for r, row in enumerate(rows):
            cx = x
            for c, cell in enumerate(row):
                fill = header_fill if r == 0 else ("FFFFFF" if r % 2 else "F1F5F9")
                color = "FFFFFF" if r == 0 else "111827"
                self.rect(cx, y + r * row_h, col_widths[c], row_h, fill=fill, line="CBD5E1")
                self.text(cx + 0.03, y + r * row_h + 0.05, col_widths[c] - 0.06, row_h - 0.07, cell, size=11.5, color=color, bold=(r == 0), margin=0.02)
                cx += col_widths[c]

    def xml(self):
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      {''.join(self.parts)}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""

    def rels_xml(self):
        items = [('rId1', 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout', '../slideLayouts/slideLayout1.xml')] + self.rels
        rels = "\n".join(
            f'<Relationship Id="{rid}" Type="{typ}" Target="{escape(target)}"/>' for rid, typ, target in items
        )
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
{rels}
</Relationships>"""


def add_footer(slide, n):
    slide.text(11.95, 7.08, 0.8, 0.24, f"{n:02d}", size=9, color="94A3B8")


def build_slides():
    ranking = read_csv(ROOT / "results" / "summary_method_ranking.csv")
    mchn_rows = [r for r in ranking if r["method"] == "Modern Hopfield"]
    order = ["noise", "salt_pepper", "fog", "blur", "mask", "affine", "dirt"]
    by_pollution = {r["pollution"]: r for r in mchn_rows}
    final_avg = sum(float(by_pollution[p]["final_accuracy"]) for p in order) / len(order)
    mean_avg = sum(float(by_pollution[p]["mean_accuracy"]) for p in order) / len(order)

    slides = []

    s = Slide(dark=True)
    s.text(0.75, 1.3, 11.8, 0.8, "现代 Hopfield 网络污染车牌字符识别", size=34, color="FFFFFF", bold=True)
    s.text(0.78, 2.28, 10.8, 0.65, "实验设计与实现 · 实验结果与分析 · 后续工作", size=21, color="BFDBFE")
    s.rect(0.78, 3.25, 11.7, 0.02, fill="3B82F6", line=None)
    s.text(0.78, 3.72, 11.1, 1.25, "汇报重点：我完成了从字符模板记忆构建、污染建模、现代 Hopfield 检索实现，到多基线鲁棒性评估与可视化分析的完整实验闭环。", size=22, color="E5E7EB")
    s.text(0.78, 6.65, 6.8, 0.3, f"生成时间：{datetime.now().strftime('%Y-%m-%d')}", size=11, color="9CA3AF")
    slides.append(s)

    s = Slide("汇报结构", "目录")
    cards = [
        ("01", "实验设计与实现", "研究目标、技术路线、MCHN 模型、污染生成、评价流程"),
        ("02", "实验结果与分析", "鲁棒性曲线、对比实验、消融、容量与混淆分析"),
        ("03", "后续工作", "真实域适配、模板记忆优化、可解释性和工程化整理"),
    ]
    for i, (num, title, desc) in enumerate(cards):
        y = 1.65 + i * 1.55
        s.rect(0.85, y, 11.65, 1.1, fill="FFFFFF", line="D8DEE9", radius=True)
        s.text(1.05, y + 0.16, 0.82, 0.45, num, size=24, color="2563EB", bold=True)
        s.text(2.0, y + 0.16, 3.5, 0.36, title, size=21, color="0F172A", bold=True)
        s.text(2.0, y + 0.58, 9.6, 0.25, desc, size=15, color="475569")
    slides.append(s)

    s = Slide("我做了什么", "实验设计与实现")
    s.text(0.78, 1.55, 5.5, 4.8, "• 搭建现代 Hopfield 字符识别核心：把模板字符作为关联记忆，输入污染字符后做单步检索与类别判定。\n• 构建车牌字符模板记忆库：统一归一化为 32×64 灰度字符向量，覆盖数字、字母和省份简称。\n• 设计污染模拟器：遮挡、噪声、椒盐、模糊、雾化、污渍、仿射变形，以及混合污染。\n• 建立完整对照实验：CNN、传统 Hopfield、近邻、欧氏近邻、类别原型。\n• 自动生成实验产物：准确率曲线、热力图、混淆矩阵、Top-K、消融和容量实验。", size=17, color="1F2937")
    s.rect(6.72, 1.65, 5.5, 4.5, fill="EFF6FF", line="BFDBFE", radius=True)
    s.text(7.02, 1.95, 4.9, 0.5, "项目边界", size=22, color="1D4ED8", bold=True)
    s.text(7.05, 2.62, 4.8, 2.4, "本 PPT 聚焦“污染车牌字符识别”。\n\n不展开端到端整车识别，也不把整车定位、车牌检测作为主要贡献。\n\n核心问题是：字符已经被裁剪或归一化后，如何在污染条件下稳定识别。", size=17, color="334155")
    slides.append(s)

    s = Slide("总体技术路线", "实验设计与实现")
    steps = [
        ("字符模板", "数字/字母/省份简称\n归一化为 32×64"),
        ("记忆矩阵", "模板向量写入 MCHN\n并构造增强记忆"),
        ("污染查询", "对干净字符施加\n遮挡/噪声/污渍等"),
        ("MCHN 检索", "计算相似度与注意力\n返回类别和重构"),
        ("多方法评估", "与 CNN/传统 Hopfield\n近邻/原型对比"),
    ]
    for i, (title, desc) in enumerate(steps):
        x = 0.55 + i * 2.52
        s.rect(x, 2.0, 2.12, 1.35, fill="FFFFFF", line="CBD5E1", radius=True)
        s.text(x + 0.15, 2.16, 1.82, 0.32, title, size=17, color="0F172A", bold=True)
        s.text(x + 0.15, 2.62, 1.82, 0.52, desc, size=11.5, color="475569")
        if i < len(steps) - 1:
            s.text(x + 2.14, 2.42, 0.4, 0.3, "→", size=24, color="2563EB", bold=True)
    s.text(0.78, 4.25, 11.7, 1.25, "技术路线的重点不是训练一个复杂分类器，而是把“模板记忆 + 污染查询 + 相似度检索”组织成可解释、可对照、可复现实验流程。", size=21, color="1E293B", fill="E0F2FE", line="BAE6FD")
    slides.append(s)

    s = Slide("MCHN 实现方式", "实验设计与实现")
    s.text(0.72, 1.52, 5.65, 4.85, "• 记忆：M ∈ R^{K×D}，每行对应一个字符模板。\n• 查询：污染字符 q 先经过特征变换，再与记忆矩阵计算相似度。\n• 更新：softmax(β·sim(q,M)) 形成注意力权重。\n• 输出：加权检索得到重构字符，同时用最大注意力/类别分数得到预测。\n• 特征模式：binary、centered、binary_centered、hybrid_shape、profile。\n• 集成策略：多种特征模式共同投票，增强对不同污染类型的鲁棒性。", size=16.5)
    s.rect(6.78, 1.65, 5.2, 3.6, fill="111827", line=None, radius=True)
    s.text(7.08, 1.95, 4.6, 0.4, "核心检索公式", size=20, color="FFFFFF", bold=True)
    s.text(7.08, 2.75, 4.65, 1.2, "z = softmax(β · qMᵀ)M\n\npred = argmax attention", size=26, color="BFDBFE", bold=True)
    s.text(7.08, 4.35, 4.55, 0.45, "实现位置：models/mchn.py", size=13, color="CBD5E1")
    slides.append(s)

    s = Slide("污染建模与评价设计", "实验设计与实现")
    rows = [
        ["污染类型", "模拟含义", "实验作用"],
        ["mask", "局部遮挡/字符缺笔", "验证遮挡恢复能力"],
        ["noise", "高斯噪声", "验证随机扰动鲁棒性"],
        ["salt_pepper", "椒盐噪声", "验证离散坏点干扰"],
        ["blur", "高斯模糊", "验证边缘退化影响"],
        ["fog", "整体变亮/雾化", "验证低对比度识别"],
        ["dirt", "污渍斑点", "模拟污染遮挡"],
        ["affine", "旋转/平移/缩放/剪切", "验证几何变化"],
    ]
    s.table(0.72, 1.52, [2.0, 4.45, 5.05], 0.52, rows)
    s.text(0.85, 6.0, 11.0, 0.45, "评价方式：在不同污染强度 0.0 / 0.1 / 0.2 / 0.4 / 0.6 / 0.8 下统计准确率，并保存曲线、热力图和混淆矩阵。", size=16, color="334155")
    slides.append(s)

    s = Slide("重构与识别示例", "实验设计与实现")
    s.image(ROOT / "results" / "mchn_reconstruction_demo.png", 0.85, 1.55, 6.0, 4.85)
    s.text(7.15, 1.75, 4.85, 3.95, "这页展示的是 MCHN 的核心直觉：\n\n• 左：干净模板字符\n• 中：污染后的查询输入\n• 右：从关联记忆中检索出的重构结果\n\n我用这类可视化证明模型不是黑盒地给出类别，而是在模板记忆空间里完成“匹配-恢复-判别”。", size=17, color="1F2937")
    slides.append(s)

    s = Slide("整体鲁棒性结果", "实验结果与分析")
    s.image(ROOT / "results" / "summary_mean_accuracy_heatmap.png", 0.7, 1.45, 5.95, 4.9)
    s.image(ROOT / "results" / "summary_final_severity_heatmap.png", 6.8, 1.45, 5.95, 4.9)
    s.text(0.85, 6.38, 11.6, 0.38, f"MCHN 在 7 类污染上的平均准确率为 {mean_avg:.1f}%，最高强度下平均仍有 {final_avg:.1f}%。", size=15.5, color="334155", bold=True)
    slides.append(s)

    s = Slide("MCHN 在不同污染下的表现", "实验结果与分析")
    s.image(ROOT / "results" / "mchn_pollution_severity_curves.png", 0.75, 1.45, 7.0, 5.05)
    rows = [["污染", "最高强度准确率", "平均准确率"]]
    for p in order:
        r = by_pollution[p]
        rows.append([p, pct(r["final_accuracy"]), pct(r["mean_accuracy"])])
    s.table(8.05, 1.55, [1.35, 1.85, 1.65], 0.46, rows)
    s.text(8.08, 5.55, 4.6, 0.8, "分析：噪声、椒盐、雾化基本保持高准确率；遮挡和仿射仍较稳定；污渍污染下降最明显，是后续优化重点。", size=14.5, color="334155")
    slides.append(s)

    s = Slide("与基线方法对比", "实验结果与分析")
    s.image(ROOT / "results" / "robustness_mask_methods_curve.png", 0.7, 1.45, 5.95, 4.65)
    s.image(ROOT / "results" / "robustness_dirt_methods_curve.png", 6.78, 1.45, 5.95, 4.65)
    s.text(0.85, 6.18, 11.5, 0.55, "在遮挡和污渍这类更接近真实污染的场景中，MCHN 显著优于传统 Hopfield；与 CNN 相比，MCHN 的优势在于不依赖大规模训练，模板记忆可解释、可直接增删。", size=15.5, color="334155")
    slides.append(s)

    s = Slide("消融实验：哪些设计有效", "实验结果与分析")
    s.image(ROOT / "results" / "ablation_final_bar.png", 0.82, 1.48, 6.1, 4.75)
    s.image(ROOT / "results" / "beta_ablation_accuracy.png", 7.08, 1.48, 5.25, 4.75)
    s.text(0.9, 6.24, 11.2, 0.5, "结果表明：单一 raw 特征不够鲁棒；二值/形状特征明显提升；多特征集成和模板增强把 mixed 污染 0.6 下的准确率提升到 94.0%。", size=15.5, color="334155", bold=True)
    slides.append(s)

    s = Slide("容量实验：现代 Hopfield 的优势", "实验结果与分析")
    s.image(ROOT / "results" / "capacity_random_pattern_retrieval_accuracy.png", 0.8, 1.55, 6.15, 4.75)
    s.image(ROOT / "results" / "capacity_real_template_retrieval_accuracy.png", 7.1, 1.55, 5.35, 4.75)
    s.text(0.9, 6.28, 11.3, 0.42, "随机模式实验中，Modern Hopfield 在 K=4096、特征维度 2048、10% 翻转噪声下仍保持 100% 检索；经典 Hopfield 超过约 0.14Nf 后明显失效。", size=15, color="334155")
    slides.append(s)

    s = Slide("混淆矩阵与错误分析", "实验结果与分析")
    s.image(ROOT / "results" / "confusion_mask_modern_hopfield.png", 0.75, 1.45, 5.6, 5.15)
    s.image(ROOT / "results" / "confusion_dirt_modern_hopfield.png", 6.75, 1.45, 5.6, 5.15)
    s.text(0.9, 6.62, 11.2, 0.3, "错误主要集中在形状相近字符、省份简称与笔画较少字符；遮挡和污渍会放大局部笔画缺失带来的歧义。", size=14.2, color="334155")
    slides.append(s)

    s = Slide("结果小结", "实验结果与分析")
    s.rect(0.85, 1.58, 11.5, 4.55, fill="FFFFFF", line="CBD5E1", radius=True)
    s.text(1.18, 1.9, 10.85, 3.5, f"• MCHN 在单字符污染识别上完成了完整闭环：模板记忆、污染生成、检索识别、重构可视化和多方法对比。\n• 在 7 类污染上，平均准确率达到 {mean_avg:.1f}%，最高污染强度下平均为 {final_avg:.1f}%。\n• 与传统 Hopfield 相比，MCHN 在容量和污染鲁棒性上优势明显。\n• CNN 在部分污染上仍然非常强，但需要训练；MCHN 的亮点是模板可解释、少训练依赖、适合字符模板记忆任务。\n• 当前主要短板是污渍和几何变形下的细粒度相似字符混淆。", size=18, color="1F2937")
    slides.append(s)

    s = Slide("后续工作", "后续工作")
    items = [
        ("真实域模板增强", "从更多真实裁剪字符中提取高质量模板，缓解模板字体与真实车牌字体差异。"),
        ("污染感知特征", "针对污渍和遮挡加入局部笔画完整性、结构骨架、连通域等特征。"),
        ("自适应记忆选择", "按字符类别、位置先验和污染类型动态筛选记忆模板，提高相似字符区分度。"),
        ("更充分的统计验证", "扩展样本量，补充置信区间、显著性检验和更多失败案例分析。"),
    ]
    for i, (title, desc) in enumerate(items):
        y = 1.55 + i * 1.2
        s.pill(0.9, y, 2.2, 0.45, title)
        s.text(3.35, y + 0.03, 8.8, 0.42, desc, size=15.5, color="334155")
    slides.append(s)

    s = Slide("结束页", None, dark=True)
    s.text(0.8, 1.8, 11.5, 0.72, "谢谢聆听", size=38, color="FFFFFF", bold=True)
    s.text(0.84, 2.75, 10.6, 1.0, "本项目的核心贡献：把现代 Hopfield 网络落到污染车牌字符识别任务中，并完成了可复现、可对照、可分析的实验体系。", size=24, color="BFDBFE")
    s.text(0.86, 5.2, 9.8, 0.45, "汇报文件：现代Hopfield网络污染车牌字符识别_汇报PPT_实验设计结果分析后续工作.pptx", size=13, color="CBD5E1")
    slides.append(s)

    for i, slide in enumerate(slides, start=1):
        if i not in (1, len(slides)):
            add_footer(slide, i)
    return slides


def content_types(slide_count):
    overrides = [
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    for i in range(1, slide_count + 1):
        overrides.append(f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>')
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Default Extension="jpg" ContentType="image/jpeg"/>
  <Default Extension="jpeg" ContentType="image/jpeg"/>
  {''.join(overrides)}
</Types>"""


def package_rels():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""


def presentation_xml(slide_count):
    sld_ids = "\n".join(f'<p:sldId id="{255+i}" r:id="rId{i+1}"/>' for i in range(1, slide_count + 1))
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
  xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst>{sld_ids}</p:sldIdLst>
  <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>"""


def presentation_rels(slide_count):
    rels = ['<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>']
    for i in range(1, slide_count + 1):
        rels.append(f'<Relationship Id="rId{i+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>')
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
{''.join(rels)}
</Relationships>"""


def slide_master_xml():
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
  xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree>
    <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
    <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
  </p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>"""


def slide_master_rels():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>"""


def slide_layout_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
  xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree>
    <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
    <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
  </p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>"""


def slide_layout_rels():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>"""


def theme_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="MCHN Theme">
  <a:themeElements>
    <a:clrScheme name="MCHN">
      <a:dk1><a:srgbClr val="111827"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
      <a:dk2><a:srgbClr val="1F2937"/></a:dk2><a:lt2><a:srgbClr val="F8FAFC"/></a:lt2>
      <a:accent1><a:srgbClr val="2563EB"/></a:accent1><a:accent2><a:srgbClr val="0EA5E9"/></a:accent2>
      <a:accent3><a:srgbClr val="10B981"/></a:accent3><a:accent4><a:srgbClr val="F59E0B"/></a:accent4>
      <a:accent5><a:srgbClr val="EF4444"/></a:accent5><a:accent6><a:srgbClr val="64748B"/></a:accent6>
      <a:hlink><a:srgbClr val="2563EB"/></a:hlink><a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="MCHN Fonts">
      <a:majorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/><a:cs typeface="Microsoft YaHei"/></a:majorFont>
      <a:minorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/><a:cs typeface="Microsoft YaHei"/></a:minorFont>
    </a:fontScheme>
    <a:fmtScheme name="MCHN Format"><a:fillStyleLst/><a:lnStyleLst/><a:effectStyleLst/><a:bgFillStyleLst/></a:fmtScheme>
  </a:themeElements>
</a:theme>"""


def core_xml():
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
  xmlns:dc="http://purl.org/dc/elements/1.1/"
  xmlns:dcterms="http://purl.org/dc/terms/"
  xmlns:dcmitype="http://purl.org/dc/dcmitype/"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>现代Hopfield网络污染车牌字符识别</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>"""


def app_xml(slide_count):
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
  xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex OpenXML Builder</Application>
  <PresentationFormat>Widescreen</PresentationFormat>
  <Slides>{slide_count}</Slides>
</Properties>"""


def write_pptx():
    slides = build_slides()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    media_paths = set()
    for slide in slides:
        for _, typ, target in slide.rels:
            if typ.endswith("/image"):
                media_paths.add(ROOT / "results" / Path(target).name)

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
        for path in sorted(media_paths):
            if path.exists():
                zf.write(path, f"ppt/media/{path.name}")

    print(OUT)


if __name__ == "__main__":
    os.chdir(ROOT)
    write_pptx()
