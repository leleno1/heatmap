🧬 LabPlot Pro: Advanced Heatmap

专为科研人员打造的下一代热图分析工具。

LabPlot Pro 是一个基于 Python 和 Streamlit 构建的高级数据可视化平台。它旨在填补“Excel 绘图太简陋”与“R 语言编程太难”之间的空白，让生物学家、医生和研究人员能够通过简单的点击，生成符合 Nature, Cell 等顶级期刊标准的复杂热图。

✨ 核心亮点 (Key Features)

1. 三大核心图表模式

矩形热图 (Standard/Cluster): 支持双向层级聚类 (Hierarchical Clustering)，内置多种距离算法 (Euclidean, Correlation) 和链接方法。

三角热图 (Correlation Triangle): 专为展示相关性矩阵设计，自动屏蔽冗余的上三角区域，支持 P 值显著性星号标记。

气泡热图 (Bubble Plot): 引入“大小”维度展示数据。支持自定义形状（方块、圆圈、菱形）和三角遮罩模式。

2. 🤖 AI 智能生物学解读 (AI Insight)

内置 AI 大脑: 集成 Google Gemini API。

一键分析: 程序自动提取高表达基因或强相关变量，发送给 AI 进行生物学通路富集分析和假设生成。

自定义问答: 支持针对当前数据向 AI 提出具体的生物学问题。

3. 严谨的统计学内核

显著性标记: 自动计算 Pearson 相关系数及 P 值，支持 * (P<0.05), ** (P<0.01), *** (P<0.001) 标注。

稳健标准化: 提供 Robust Z-Score (基于中位数/IQR)，防止离群值破坏配色。

数据清洗: 支持 Log2/Log10 变换，以及多种缺失值处理策略（保留、剔除、均值填充）。

4. 高级分组注释 (Metadata Annotation)

多组学风格: 支持上传 Metadata 文件，自动匹配样本 ID。

自动着色: 自动为分组信息生成颜色条 (Annotation Bars)，并对齐到热图上方或左侧，完美复刻多组学分析图表。

5. 出版级导出

矢量支持: 支持导出 PDF, SVG 格式，无限放大不失真。

高清位图: 支持 PNG, JPG, TIFF 导出，DPI 可调节（最高 600 DPI）。

智能配色: 内置科研常用色盘 (Viridis, RdBu_r, YlOrRd 等)，并根据数据类型智能推荐。

🛠️ 安装与运行 (Installation)

确保您的电脑上已安装 Python 3.8+。

1. 克隆或下载本项目

git clone [https://github.com/yourusername/labplot-pro.git](https://github.com/yourusername/labplot-pro.git)
cd labplot-pro


2. 安装依赖

建议使用虚拟环境 (Virtualenv/Conda)。

pip install -r requirements.txt


3. 启动应用

streamlit run app.py


应用将在浏览器中自动打开，通常地址为 http://localhost:8501。

📖 使用指南 (Quick Start)

📂 数据输入:

上传你的表达量矩阵 (.csv 或 .xlsx)。

第一列应为基因名/行名，第一行为样本名/列名。

(可选) 勾选“转置”以互换行列。

📊 图表定义:

选择图表类型（矩形、三角、气泡）。

设置聚类、标准化模式（建议表达量用 Z-Score，相关性用 Auto-Correlation）。

🎨 视觉美化:

调整宽高度、字体、配色方案。

开启“标注模式”以显示数值或星号。

🏷️ 分组注释 (高级):

在矩形热图聚类模式下，上传 Metadata.csv。

选择要展示的分组列，程序将自动添加颜色条。

🤖 AI 解读:

输入 Google Gemini API Key。

点击“开始智能分析”获取数据洞察。

📦 依赖库 (Requirements)

本项目依赖以下优秀的开源库：

Streamlit: Web 应用框架

Pandas: 数据处理

Seaborn & Matplotlib: 绘图引擎

Scipy: 统计与聚类计算

OpenPyXL: Excel 支持

Requests: API 调用

📄 License

此项目采用 MIT 许可证。欢迎 Fork 和 PR！
