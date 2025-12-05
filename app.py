import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import io
from scipy.stats import pearsonr
import matplotlib as mpl
import matplotlib.lines as mlines
import requests
import json

# ==========================================
# 🔐 商业化配置 (License Config)
# ==========================================
PRO_LICENSE_KEY = "LABPLOT2025"  # 你可以随时修改这个密码
FREE_DPI_LIMIT = 150             # 免费版最大 DPI
PRO_DPI_LIMIT = 600              # Pro 版最大 DPI

# -----------------------------------------------------------------------------
# 1. 配置与工具类 (Infrastructure)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="LabPlot Pro: Advanced Heatmap",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 基础绘图设置
sns.set_theme(style="white")
plt.rcParams['axes.unicode_minus'] = False

class DataProcessor:
    """处理数据的加载、清洗、变换、过滤的核心逻辑"""
    
    @staticmethod
    def load_file(file):
        try:
            if file.name.endswith('.csv'):
                return pd.read_csv(file, index_col=0)
            else:
                return pd.read_excel(file, index_col=0)
        except Exception as e:
            st.error(f"文件读取错误: {e}")
            return None

    @staticmethod
    def clean(df, method):
        df_num = df.apply(pd.to_numeric, errors='coerce')
        if "Keep NaN" in method: return df_num 
        if "Drop Rows" in method: return df_num.dropna()
        if "Fill with Mean" in method: return df_num.fillna(df_num.mean())
        if "Fill with Min" in method: return df_num.fillna(df_num.min().min())
        return df_num.fillna(0)

    @staticmethod
    def transform(df, method):
        if "Log2" in method: return np.log2(df.abs() + 1)
        if "Log10" in method: return np.log10(df.abs() + 1)
        return df

    @staticmethod
    def filter(df, method, top_n, selected_ids):
        if "Variance" in method:
            vars = df.var(axis=1)
            return df.loc[vars.nlargest(top_n).index], f"Top {top_n} Var"
        if "Specific IDs" in method and selected_ids:
            valid = [i for i in selected_ids if i in df.index]
            return df.loc[valid], "Manual Select"
        return df, "All Data"

    @staticmethod
    def normalize(df, mode):
        if mode == "Row (按行-Std)": return df.sub(df.mean(1), axis=0).div(df.std(1), axis=0)
        if mode == "Column (按列-Std)": return df.sub(df.mean(0), axis=1).div(df.std(0), axis=1)
        if mode == "Robust Z-Score":
            med = df.median(1)
            iqr = df.quantile(0.75, 1) - df.quantile(0.25, 1)
            return df.sub(med, axis=0).div(iqr.replace(0, 1), axis=0)
        return df

    @staticmethod
    def calc_correlation(df):
        df_clean = df.dropna()
        cols = df_clean.columns
        n = len(cols)
        corr = np.zeros((n, n))
        p_vals = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j: 
                    corr[i, j] = 1.0
                    p_vals[i, j] = 0.0
                else:
                    r, p = pearsonr(df_clean.iloc[:, i], df_clean.iloc[:, j])
                    corr[i, j] = r
                    p_vals[i, j] = p
        
        return pd.DataFrame(corr, index=cols, columns=cols), pd.DataFrame(p_vals, index=cols, columns=cols)

class MetadataManager:
    """[增强版] 管理样本/基因的元数据注释，支持多列与自动对齐"""
    def __init__(self):
        self.meta_df = None
        self.row_colors = None
        self.col_colors = None

    def upload_ui(self, main_df):
        with st.sidebar.expander("🏷️ 4. 分组注释 (Annotations)", expanded=True):
            # [Fix] 添加 unique key 防止 DuplicateElementId 错误
            file = st.file_uploader("上传分组信息 (Metadata.csv)", type=["csv", "xlsx"], help="第一列必须是样本ID，用于匹配主数据", key="metadata_uploader")
            
            if file:
                try:
                    self.meta_df = DataProcessor.load_file(file)
                    self.meta_df.index = self.meta_df.index.astype(str)
                    main_df_cols = main_df.columns.astype(str)
                    main_df_rows = main_df.index.astype(str)
                    
                    match_cols = len(main_df_cols.intersection(self.meta_df.index))
                    match_rows = len(main_df_rows.intersection(self.meta_df.index))
                    
                    target_axis = None
                    
                    if match_cols > 0 and match_cols >= match_rows:
                        target_axis = 'col'
                        st.success(f"✅ 检测到列(样本)注释: 匹配 {match_cols}/{len(main_df.columns)}")
                    elif match_rows > 0:
                        target_axis = 'row'
                        st.success(f"✅ 检测到行(基因)注释: 匹配 {match_rows}/{len(main_df.index)}")
                    else:
                        st.error("❌ Metadata 的索引与主数据的行/列名均不匹配！")
                        return

                    selected_cols = st.multiselect(
                        "选择要展示的分组条带", 
                        self.meta_df.columns,
                        default=self.meta_df.columns[:1].tolist()
                    )
                    
                    if selected_cols:
                        color_df = pd.DataFrame(index=self.meta_df.index)
                        st.caption("🎨 分组图例预览:")
                        legend_cols = st.columns(min(len(selected_cols), 4))
                        
                        for idx, col in enumerate(selected_cols):
                            series = self.meta_df[col]
                            unique_vals = series.unique()
                            pal = sns.color_palette("husl", len(unique_vals))
                            lut = dict(zip(unique_vals, pal))
                            color_df[col] = series.map(lut)
                            
                            with legend_cols[idx % 4]:
                                st.markdown(f"**{col}**")
                                for val, color in list(lut.items())[:5]:
                                    st.color_picker(f"{val}", '#%02x%02x%02x' % (int(color[0]*255), int(color[1]*255), int(color[2]*255)), disabled=True, key=f"{col}_{val}")
                        
                        if target_axis == 'col':
                            self.col_colors = color_df
                        else:
                            self.row_colors = color_df

                except Exception as e:
                    st.error(f"Metadata 处理出错: {e}")

class AIAssistant:
    """[Pro功能] AI 智能数据解读"""
    
    @staticmethod
    def analyze_data(df, chart_type, api_key, user_query=None):
        if not api_key:
            return "⚠️ 请输入 Google Gemini API Key 以启用 AI 分析。"
        
        # 1. 构建数据摘要 (防止 Token 超出)
        summary = ""
        if "矩形" in chart_type:
            # 找出均值最高的 Top 5 和最低的 Top 5
            means = df.mean(axis=1).sort_values(ascending=False)
            top5 = means.head(5).index.tolist()
            bottom5 = means.tail(5).index.tolist()
            summary = f"Data Type: Expression Matrix. \nTop 5 High Expression: {top5}. \nTop 5 Low Expression: {bottom5}."
        else:
            # 相关性矩阵，找出强相关 (r>0.8)
            # 这里的 df 已经是 correlation matrix
            corr = df.where(np.triu(np.ones(df.shape), k=1).astype(bool)).stack()
            strong_pos = corr[corr > 0.8].head(5).index.tolist()
            strong_neg = corr[corr < -0.8].head(5).index.tolist()
            summary = f"Data Type: Correlation Matrix. \nStrong Positive Pairs: {strong_pos}. \nStrong Negative Pairs: {strong_neg}."

        # 2. 构建 Prompt
        base_prompt = f"""
        Act as a senior bioinformatics scientist. Analyze the following data summary derived from a heatmap:
        {summary}
        
        Provide a concise biological insight (max 150 words) covering:
        1. Potential biological functions or pathways of the top markers (assume they are gene symbols or metabolites).
        2. A brief hypothesis about the sample condition or correlation pattern.
        3. Use professional tone.
        """
        
        # 如果用户有特定问题，将其加入 Prompt
        if user_query and user_query.strip():
            prompt = f"{base_prompt}\n\nImportant - The user has a specific question/instruction:\n{user_query}\nPlease prioritize answering the user's specific question while using the data summary as context."
        else:
            prompt = base_prompt
        
        # 3. 调用 API
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"
        headers = {'Content-Type': 'application/json'}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            if response.status_code == 200:
                result = response.json()
                return result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'No response.')
            else:
                return f"API Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Request Failed: {e}"

class Visualizer:
    """绘图引擎"""
    
    @staticmethod
    def setup_font(font_name, scale):
        sns.set_context("notebook", font_scale=scale)
        font_list = [font_name, 'SimHei', 'Arial', 'sans-serif']
        if 'Times' in font_name:
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = font_list
        else:
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = font_list

    @staticmethod
    def get_cmap(cmap_name, bad_color='lightgrey'):
        try:
            cmap = mpl.colormaps.get_cmap(cmap_name).copy()
        except:
            cmap = plt.cm.get_cmap(cmap_name).copy()
        cmap.set_bad(color=bad_color)
        return cmap

    @staticmethod
    def get_annot_matrix(df, p_df, mode):
        if mode == "None": return None
        n, m = df.shape
        annot = np.empty((n, m), dtype=object)
        for i in range(n):
            for j in range(m):
                txt = ""
                val = df.iloc[i,j]
                if pd.isna(val):
                    annot[i,j] = ""
                    continue
                if "Values" in mode: txt += f"{val:.2f}"
                if "Stars" in mode and p_df is not None:
                    p = p_df.iloc[i,j]
                    if p < 0.001: txt += "\n***" if txt else "***"
                    elif p < 0.01: txt += "\n**" if txt else "**"
                    elif p < 0.05: txt += "\n*" if txt else "*"
                annot[i, j] = txt
        return pd.DataFrame(annot, index=df.index, columns=df.columns)

    @staticmethod
    def draw_clustermap(df, meta_mgr, cmap, cbar_label, **kwargs):
        row_colors = None
        col_colors = None
        
        if meta_mgr.row_colors is not None:
            row_colors = meta_mgr.row_colors.reindex(df.index)
        
        if meta_mgr.col_colors is not None:
            col_colors = meta_mgr.col_colors.reindex(df.columns)

        g = sns.clustermap(
            df,
            row_colors=row_colors,
            col_colors=col_colors,
            cmap=cmap,
            cbar_kws={'label': cbar_label},
            **kwargs
        )
        return g

    @staticmethod
    def draw_bubble_plot(df, ax, cmap, scale_factor=100, rotation=45, vmin=None, vmax=None, triangular=False, annot_df=None, marker='o', cbar_label="Value"):
        df_plot = df.copy()
        if triangular:
            mask = np.triu(np.ones(df_plot.shape), k=1).astype(bool)
            df_plot = df_plot.mask(mask)
            if annot_df is not None: annot_df = annot_df.mask(mask)

        df_reset = df_plot.reset_index()
        index_name = df_reset.columns[0]
        
        # [修复] 强制指定 var_name 和 value_name，防止原索引名作为列名导致 KeyError
        df_melt = df_reset.melt(id_vars=index_name, var_name='variable', value_name='value')
        
        df_melt = df_melt.dropna(subset=['value'])
        
        x_labels = df.columns
        y_labels = df.index
        x_map = {label: i for i, label in enumerate(x_labels)}
        y_map = {label: i for i, label in enumerate(y_labels)}
        
        df_melt['x'] = df_melt['variable'].map(x_map)
        df_melt['index_mapped'] = df_melt[index_name].map(y_map)
        
        size_values = df_melt['value'].abs()
        max_val = size_values.max() if size_values.max() != 0 else 1
        df_melt['size'] = (size_values / max_val) * scale_factor * 5
        
        scatter = ax.scatter(
            x=df_melt['x'],
            y=df_melt['index_mapped'],
            s=df_melt['size'],
            c=df_melt['value'],
            cmap=cmap,
            marker=marker,
            vmin=vmin, vmax=vmax,
            alpha=0.9, edgecolors='grey', linewidth=0.5
        )
        
        handles = []
        labels = []
        for r in [1.0, 0.5, 0.25]:
            val = max_val * r
            s = r * scale_factor * 5
            h = mlines.Line2D([], [], color='grey', marker=marker, linestyle='None',
                            markersize=np.sqrt(s), label=f'{val:.2f}')
            handles.append(h)
            labels.append(f'{val:.2f}')
        
        # [修复] 调整图例位置到 (1.35, 1) 避免与 Colorbar 冲突
        ax.legend(handles, labels, title="|Val|", loc='upper left', bbox_to_anchor=(1.35, 1), frameon=False)

        if annot_df is not None:
            for idx, row in df_melt.iterrows():
                r_idx = int(row['index_mapped'])
                c_idx = int(row['x'])
                try:
                    txt = annot_df.loc[y_labels[r_idx], x_labels[c_idx]]
                    if pd.notna(txt) and str(txt) != "":
                        ax.text(c_idx, r_idx, txt, ha='center', va='center', fontsize=8, color='black')
                except: pass

        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=rotation, ha='right' if rotation > 0 else 'center')
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)
        ax.invert_yaxis()
        
        # [修复] 强制等比例显示，保证气泡是圆的
        ax.set_aspect('equal')
        
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.colorbar(scatter, ax=ax, label=cbar_label)
        return ax

# -----------------------------------------------------------------------------
# 3. 主程序逻辑
# -----------------------------------------------------------------------------

def main():
    st.title("🧬 LabPlot Pro: v3.0 Commercial")
    
    # === 0. 商业化激活区 (Lock) ===
    with st.sidebar.expander("🔑 Pro 版激活 (License)", expanded=True):
        license_input = st.text_input("输入解锁码", type="password", help="关注公众号'AIBio Research'回复'heatmap'免费获取")
        is_pro = (license_input == PRO_LICENSE_KEY)
        if is_pro:
            st.success("✅ Pro 版已激活！所有功能解锁。")
        else:
            st.info("🔒 当前为免费版，关注公众号'AIBio Research'回复'heatmap'免费获取解锁码")
            st.caption(f"限制：最高 {FREE_DPI_LIMIT} DPI，不支持矢量导出，无法使用 AI。")

    # --- 1. 数据输入 ---
    with st.sidebar.expander("📂 1. 数据输入 (Data)", expanded=True):
        # [Fix] 添加 unique key 防止 DuplicateElementId 错误
        file = st.file_uploader("主矩阵 (Matrix)", type=["csv", "xlsx"], key="main_matrix_uploader")
        do_transpose = st.checkbox("转置主数据 (行列互换)", value=False, help="如果你的文件是'行=样本'，请勾选此项")
        clean_method = st.selectbox("清洗", ["Drop Rows", "Keep NaN (保留缺失值)", "Fill 0", "Fill Mean"], 0)
        trans_method = st.selectbox("变换", ["None", "Log2", "Log10"], 0)
        
        use_filter = st.checkbox("启用过滤")
        filter_type, filter_n = "None", 50
        if use_filter:
            filter_type = st.radio("策略", ["Variance", "Specific IDs"])
            if filter_type == "Variance": filter_n = st.number_input("Top N", 50)
            
    # --- 2. 图表定义 ---
    with st.sidebar.expander("📊 2. 图表定义 (Chart)", expanded=True):
        chart_type = st.radio("类型", ["A. 矩形热图", "B. 三角热图", "C. 气泡热图"])
        
        norm_mode = "None"
        cluster_on = False
        is_corr = False
        triangular_bubble = False
        
        if "矩形" in chart_type:
            cluster_on = st.checkbox("聚类 (Clustering)", True)
            norm_mode = st.selectbox("处理模式", ["None (原始值)", "Row (按行-Std)", "Column (按列-Std)", "Standard Z-Score", "Robust Z-Score", "Auto-Correlation (计算相关性)"], 1)
            if "Auto-Correlation" in norm_mode: is_corr = True
            
        elif "三角" in chart_type:
            st.info("自动计算相关性")
            is_corr = True
            
        elif "气泡" in chart_type:
            triangular_bubble = st.checkbox("仅显示下三角", False)
            bubble_scale = st.slider("气泡大小", 10, 300, 100)
            if triangular_bubble:
                norm_mode = "Auto-Correlation"
                is_corr = True
            else:
                norm_mode = st.selectbox("标准化", ["None", "Row (按行-Std)", "Robust Z-Score", "Auto-Correlation"], 0)
                if "Auto-Correlation" in norm_mode: is_corr = True

    # --- 3. 视觉美化 ---
    with st.sidebar.expander("🎨 3. 视觉 (Style)", expanded=True):
        w = st.slider("宽", 4, 20, 10)
        h = st.slider("高", 4, 20, 8)
        
        st.markdown("#### 配色设置")
        seq_cmaps = ["viridis", "YlOrRd", "Blues", "Reds", "magma"] 
        div_cmaps = ["RdBu_r", "coolwarm", "vlag", "Spectral_r"]    
        
        is_diverging = is_corr or "Row" in norm_mode or "Z-Score" in norm_mode
        if is_diverging:
            st.caption("推荐: 双向配色")
            current_options = div_cmaps + seq_cmaps
        else:
            st.caption("推荐: 单向渐变")
            current_options = seq_cmaps + div_cmaps
            
        selected_cmap_name = st.selectbox("配色方案", current_options, 0)
        
        # [恢复] 色彩范围锁定功能
        use_manual_scale = st.checkbox("锁定色彩范围 (Lock Scale)", value=False, help="手动指定最小(vmin)和最大(vmax)值，用于统一多图标准")
        vmin_manual, vmax_manual = None, None
        if use_manual_scale:
            col_v1, col_v2 = st.columns(2)
            with col_v1: vmin_manual = st.number_input("Min", value=-2.0, step=0.5)
            with col_v2: vmax_manual = st.number_input("Max", value=2.0, step=0.5)
        
        default_label = "Value"
        if is_corr: default_label = "Pearson r"
        elif "Row" in norm_mode or "Z-Score" in norm_mode: default_label = "Z-Score"
        elif "Log" in trans_method: default_label = "Log Expression"
        
        cbar_label_input = st.text_input("图例标签", default_label)
        annot_mode = st.selectbox("标注", ["None", "Values", "Stars", "Values + Stars"])
        font_name = st.selectbox("字体", ["Arial", "Times New Roman", "Verdana", "SimHei"], 0)
        font_scale = st.slider("字号", 0.5, 2.5, 1.2)
        
        marker_char = 'o'
        if "气泡" in chart_type:
            st.markdown("---")
            marker_sel = st.selectbox("气泡形状", ["Circle (o)", "Square (s)", "Diamond (D)", "Triangle (^)", "Hexagon (h)"], 0)
            marker_char = marker_sel.split("(")[1][0]
        
        st.markdown("---")
        custom_title = st.text_input("自定义标题 (Title)", "", help="留空则不显示标题")
        
    # --- 4. AI 智能解读 (Locked) ---
    with st.sidebar.expander("🤖 4. AI 智能解读 (AI Insight)", expanded=False):
        if is_pro:
            gemini_key = st.text_input("Gemini API Key", type="password")
            user_query = st.text_area("自定义问题 (可选)", placeholder="例如：请分析这些基因与癌症通路的关联...", help="留空则进行自动通用解读")
            start_ai = st.button("🧠 开始智能分析")
        else:
            st.warning("🔒 AI 智能分析功能仅限 Pro 版可用。\n请在侧边栏上方输入解锁码激活。")
            start_ai = False
            gemini_key = ""
            user_query = ""

    meta_mgr = MetadataManager()
    
    # --- 执行 ---
    if not file:
        st.info("👈 请先在左侧上传主数据矩阵")
        return

    df = DataProcessor.load_file(file)
    if df is None: return
    
    if do_transpose:
        df = df.T
        st.caption(f"ℹ️ 已转置数据，当前维度: {df.shape}")

    # [Fix] 只在这里调用 upload_ui，确保只创建一个 uploader
    if "矩形" in chart_type and cluster_on:
        meta_mgr.upload_ui(df) 
    
    df = DataProcessor.clean(df, clean_method)
    df = DataProcessor.transform(df, trans_method)
    
    sel_ids = []
    if use_filter and filter_type == "Specific IDs":
        sel_ids = st.sidebar.multiselect("选择ID", df.index, df.index[:5].tolist())
    df, filter_msg = DataProcessor.filter(df, filter_type, filter_n, sel_ids)

    p_df = None
    if is_corr:
        df_plot, p_df = DataProcessor.calc_correlation(df)
    else:
        df_plot = DataProcessor.normalize(df, norm_mode)

    # --- AI Analysis Trigger ---
    if start_ai and is_pro:
        with st.status("🤖 AI 正在思考中...", expanded=True) as status:
            st.write("正在提取关键特征...")
            # 传递 user_query 到 AI 分析函数
            ai_result = AIAssistant.analyze_data(df_plot, chart_type, gemini_key, user_query)
            st.write("正在生成报告...")
            status.update(label="✅ 分析完成", state="complete", expanded=True)
            st.markdown("### 🧬 AI 生物学解读报告")
            st.info(ai_result)

    Visualizer.setup_font(font_name, font_scale)
    annot = Visualizer.get_annot_matrix(df_plot, p_df, annot_mode)
    final_cmap = Visualizer.get_cmap(selected_cmap_name, bad_color='lightgrey')
    
    st.write("---")
    with st.spinner("Rendering..."):
        try:
            fig = None
            
            # [新增] 计算色彩范围
            if use_manual_scale:
                # 用户强制锁定
                c_min, c_max = vmin_manual, vmax_manual
                c_center = 0 if is_diverging else None
            else:
                # 智能自动
                robust_min, robust_max = np.nanpercentile(df_plot.values, 2), np.nanpercentile(df_plot.values, 98)
                if is_diverging: # Z-score/Corr 强制对称
                    lim = max(abs(robust_min), abs(robust_max))
                    c_min, c_max, c_center = -lim, lim, 0
                else: # 原始值
                    c_min, c_max, c_center = robust_min, robust_max, None

            # 1. 矩形热图
            if "矩形" in chart_type:
                if cluster_on:
                    g = Visualizer.draw_clustermap(
                        df_plot, meta_mgr,
                        figsize=(w, h), cmap=final_cmap, annot=annot, fmt="",
                        cbar_label=cbar_label_input, 
                        method='average', metric='euclidean',
                        vmin=c_min, vmax=c_max, center=c_center, # 应用色彩控制
                        tree_kws={'linewidths': 1.5}
                    )
                    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
                    fig = g.fig
                else:
                    fig, ax = plt.subplots(figsize=(w, h))
                    sns.heatmap(df_plot, ax=ax, cmap=final_cmap, annot=annot, fmt="", 
                                vmin=c_min, vmax=c_max, center=c_center, # 应用色彩控制
                                cbar_kws={'label': cbar_label_input})
                    plt.xticks(rotation=45, ha='right')
            
            # 2. 三角热图
            elif "三角" in chart_type:
                fig, ax = plt.subplots(figsize=(w, h))
                mask = np.triu(np.ones_like(df_plot))
                sns.heatmap(df_plot, mask=mask, ax=ax, cmap=final_cmap, annot=annot, fmt="", square=True,
                            vmin=c_min, vmax=c_max, center=c_center, # 应用色彩控制
                            cbar_kws={'label': cbar_label_input})
                plt.xticks(rotation=45, ha='right')
                
            # 3. 气泡热图
            elif "气泡" in chart_type:
                fig, ax = plt.subplots(figsize=(w, h))
                Visualizer.draw_bubble_plot(
                    df_plot, ax, final_cmap, bubble_scale, 45, 
                    annot_df=annot, triangular=triangular_bubble,
                    vmin=c_min, vmax=c_max, # 应用色彩控制
                    marker=marker_char, cbar_label=cbar_label_input
                )
            
            if fig:
                # 只在有自定义输入时显示标题
                if custom_title:
                    fig.suptitle(custom_title, y=1.02, fontsize=16)
                st.pyplot(fig)
                
                st.markdown("### 📥 下载图表")
                c1, c2 = st.columns(2)
                
                # [Lock] 下载格式与DPI的商业化逻辑
                if is_pro:
                    # Pro: 全格式，高 DPI
                    save_fmt = c1.selectbox("格式 (Pro Unlocked)", ["PDF", "SVG", "TIFF", "PNG", "JPG"], 0)
                    max_dpi = PRO_DPI_LIMIT
                else:
                    # Free: 仅位图，低 DPI
                    save_fmt = c1.selectbox("格式 (Free Limit)", ["PNG", "JPG"], 0)
                    max_dpi = FREE_DPI_LIMIT
                    
                save_dpi = c2.number_input("DPI", 72, max_dpi, min(300, max_dpi), 50)
                
                buf = io.BytesIO()
                save_fmt_lower = save_fmt.lower()
                if save_fmt_lower == "jpg": save_fmt_lower = "jpeg"
                
                fig.savefig(buf, format=save_fmt_lower, dpi=save_dpi, bbox_inches='tight', facecolor='white')
                
                # 按钮文案区分
                dl_label = f"下载 Pro {save_fmt}" if is_pro else f"下载 Free {save_fmt}"
                st.download_button(dl_label, buf.getvalue(), f"plot.{save_fmt_lower}")
                
                if not is_pro:
                    st.caption("💡 想要 PDF 矢量图和 600 DPI？请在左侧激活 Pro 版。")

        except Exception as e:
            st.error(f"绘图失败: {e}")
            st.write("调试建议: 检查数据是否包含非数值字符。如果看到方框乱码，请在视觉设置中切换字体为 SimHei。")

if __name__ == "__main__":
    main()


