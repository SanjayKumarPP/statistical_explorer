import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Statistical Explorer", page_icon="📊", layout="wide")

st.markdown("""
<style>
.stat-card {
    background: white; border-radius: 10px; padding: 14px 18px; margin: 6px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-left: 4px solid #4f8ef7;
}
.stat-label { font-size: 12px; color: #888; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
.stat-value { font-size: 20px; color: #1a1a2e; font-weight: 700; margin-top: 2px; }
.section-header {
    font-size: 17px; font-weight: 700; color: #1a1a2e;
    border-bottom: 2px solid #4f8ef7; padding-bottom: 5px; margin: 18px 0 10px 0;
}
.hbox { background:#eef3ff; border-radius:8px; padding:12px 16px; border:1px solid #c0d0f8; margin:6px 0; }
.reject { border-left:4px solid #e74c3c!important; background:#fdf0ef!important; }
.ok     { border-left:4px solid #27ae60!important; background:#edfaf4!important; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.title("📊 Statistical Explorer")
st.markdown("Upload any CSV/Excel dataset and instantly compute descriptive statistics, visualise distributions, and run hypothesis tests.")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload Dataset (CSV or Excel)", type=["csv","xlsx","xls"])
    alpha = st.slider("Significance Level (α)", 0.01, 0.20, 0.05, 0.01)
    st.markdown("---")
    st.markdown("**Project:** Statistical Analysis Tool  \n**By:** Sanjay Kumar Pujari Pushparaj")

# ── Load / Demo Data ─────────────────────────────────────────────────────────
@st.cache_data
def load_data(f):
    return pd.read_csv(f) if f.name.lower().endswith(".csv") else pd.read_excel(f)

if uploaded_file:
    df = load_data(uploaded_file)
else:
    st.info("👆 No file uploaded — showing demo dataset. Upload your own CSV/Excel in the sidebar.")
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "Age":        np.random.normal(35, 8, n).astype(int).clip(18, 65),
        "Salary":     np.random.lognormal(10.5, 0.5, n).round(2),
        "Score":      np.random.normal(72, 15, n).round(1).clip(0, 100),
        "YearsExp":   np.random.exponential(5, n).round(1).clip(0, 30),
        "Department": np.random.choice(["Engineering","Marketing","Sales","HR"], n),
        "Grade":      np.random.choice(["A","B","C","D"], n, p=[0.25,0.35,0.25,0.15]),
    })

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols     = df.select_dtypes(include=["object","category"]).columns.tolist()

st.markdown(f"**Dataset:** {df.shape[0]} rows × {df.shape[1]} cols &nbsp;|&nbsp; {len(numeric_cols)} numeric, {len(cat_cols)} categorical")
st.dataframe(df.head(8), use_container_width=True)

# ── Column Selection ──────────────────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    sel_col  = st.selectbox("Primary column for analysis", numeric_cols or df.columns.tolist())
with c2:
    comp_col = st.selectbox("Compare with (optional)", ["None"] + [c for c in numeric_cols if c != sel_col])

series = df[sel_col].dropna()

# ── Descriptive Statistics ────────────────────────────────────────────────────
mean_val   = series.mean()
median_val = series.median()
mode_vals  = series.mode().tolist()
if len(mode_vals) > 10:
    mode_display = "No unique mode"
else:
    mode_display = ", ".join([f"{m:.2f}" if isinstance(m, (float, np.floating)) else str(m) for m in mode_vals[:3]])
std_val    = series.std()
var_val    = series.var()
range_val  = series.max() - series.min()
iqr_val    = stats.iqr(series)
skew_val   = series.skew()
kurt_val   = series.kurtosis()
cv_val     = (std_val / mean_val * 100) if mean_val != 0 else float("nan")
if len(series) > 1:
    sem_val = stats.sem(series)
    ci_low, ci_high = stats.t.interval(1 - alpha, df=len(series)-1, loc=mean_val, scale=sem_val)
else:
    sem_val = np.nan
    ci_low, ci_high = np.nan, np.nan

st.markdown('<div class="section-header">📐 Descriptive Statistics</div>', unsafe_allow_html=True)

cards = [
    ("Mean",            f"{mean_val:.4f}"),
    ("Median",          f"{median_val:.4f}"),
    ("Mode",            mode_display),
    ("Std Deviation",   f"{std_val:.4f}"),
    ("Variance",        f"{var_val:.4f}"),
    ("Range",           f"{range_val:.4f}"),
    ("IQR",             f"{iqr_val:.4f}"),
    ("Coeff. Variation",f"{cv_val:.2f}%"),
    ("Skewness",        f"{skew_val:.4f}"),
    ("Kurtosis",        f"{kurt_val:.4f}"),
    ("Std Error Mean",  f"{sem_val:.4f}"),
    (f"CI {int((1-alpha)*100)}% Low",  f"{ci_low:.4f}"),
    (f"CI {int((1-alpha)*100)}% High", f"{ci_high:.4f}"),
    ("Min",             f"{series.min():.4f}"),
    ("Max",             f"{series.max():.4f}"),
    ("Count",           str(len(series))),
]
cols4 = st.columns(4)
for i, (lbl, val) in enumerate(cards):
    with cols4[i % 4]:
        st.markdown(f'<div class="stat-card"><div class="stat-label">{lbl}</div><div class="stat-value">{val}</div></div>', unsafe_allow_html=True)

st.markdown("**Percentiles**")
pct_df = pd.DataFrame({
    "P1": [series.quantile(0.01)], "Q1 (P25)": [series.quantile(0.25)],
    "Median (P50)": [series.quantile(0.50)], "Q3 (P75)": [series.quantile(0.75)],
    "P99": [series.quantile(0.99)]
}).round(4)
st.dataframe(pct_df, use_container_width=True, hide_index=True)

# ── Distribution Plots ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Distribution Plots</div>', unsafe_allow_html=True)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.patch.set_facecolor("#f8f9fa")
PAL = "#4f8ef7"

# 1 Histogram + KDE
ax = axes[0, 0]
ax.hist(series, bins=30, color=PAL, alpha=0.6, edgecolor="white", density=True)
series.plot.kde(ax=ax, color="#e74c3c", lw=2)
ax.axvline(mean_val,   color="#27ae60", ls="--", lw=1.5, label=f"Mean={mean_val:.2f}")
ax.axvline(median_val, color="#f39c12", ls=":",  lw=1.5, label=f"Median={median_val:.2f}")
ax.set_title("Histogram + KDE"); ax.legend(fontsize=8); ax.set_facecolor("white")

# 2 Box Plot
ax = axes[0, 1]
ax.boxplot(series, vert=True, patch_artist=True, notch=True,
           boxprops=dict(facecolor=PAL, alpha=0.5),
           medianprops=dict(color="#e74c3c", lw=2))
ax.set_title("Box Plot (with notch)"); ax.set_facecolor("white"); ax.set_xticklabels([sel_col])

# 3 Q-Q Plot
ax = axes[0, 2]
(osm, osr), (slope, intercept, r) = stats.probplot(series, dist="norm")
ax.scatter(osm, osr, color=PAL, s=12, alpha=0.6)
ax.plot(osm, slope*np.array(osm)+intercept, color="#e74c3c", lw=1.5)
ax.set_title(f"Q-Q Plot  (R²={r**2:.3f})"); ax.set_xlabel("Theoretical Quantiles"); ax.set_ylabel("Sample Quantiles"); ax.set_facecolor("white")

# 4 Violin Plot
ax = axes[1, 0]
parts = ax.violinplot(series, showmedians=True, showmeans=True)
for pc in parts["bodies"]: pc.set_facecolor(PAL); pc.set_alpha(0.5)
ax.set_title("Violin Plot"); ax.set_facecolor("white"); ax.set_xticks([1]); ax.set_xticklabels([sel_col])

# 5 Empirical CDF
ax = axes[1, 1]
sorted_s = np.sort(series)
cdf = np.arange(1, len(sorted_s)+1) / len(sorted_s)
ax.plot(sorted_s, cdf, color=PAL, lw=2)
ax.fill_between(sorted_s, cdf, alpha=0.15, color=PAL)
ax.set_title("Empirical CDF"); ax.set_xlabel(sel_col); ax.set_ylabel("Cumulative Probability"); ax.set_facecolor("white")

# 6 Scatter or Run Chart
ax = axes[1, 2]
if comp_col != "None":
    tmp = df[[sel_col, comp_col]].dropna()
    ax.scatter(tmp[sel_col], tmp[comp_col], color=PAL, alpha=0.4, s=18)
    m, b, rv, pv, _ = stats.linregress(tmp[sel_col], tmp[comp_col])
    xl = np.linspace(tmp[sel_col].min(), tmp[sel_col].max(), 100)
    ax.plot(xl, m*xl+b, color="#e74c3c", lw=2, label=f"r={rv:.3f}")
    ax.set_title(f"{sel_col} vs {comp_col}"); ax.set_xlabel(sel_col); ax.set_ylabel(comp_col); ax.legend(fontsize=8)
else:
    ax.plot(series.values[:100], color=PAL, lw=1.2, alpha=0.8)
    ax.axhline(mean_val, color="#e74c3c", ls="--", lw=1.5, label=f"Mean={mean_val:.2f}")
    ax.set_title("Run Chart (first 100 obs.)"); ax.set_xlabel("Index"); ax.set_ylabel(sel_col); ax.legend(fontsize=8)
ax.set_facecolor("white")

plt.tight_layout(pad=1.5)
st.pyplot(fig)
plt.close()

# ── Hypothesis Tests ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔬 Hypothesis Tests</div>', unsafe_allow_html=True)

def hbox(title, stat_name, stat_val, p_val, note=""):
    css = "reject" if p_val < alpha else "ok"
    verdict = (f"❌ Reject H₀ &nbsp; (p = {p_val:.4f} &lt; α = {alpha})"
               if p_val < alpha else
               f"✅ Fail to Reject H₀ &nbsp; (p = {p_val:.4f} ≥ α = {alpha})")
    n_str = f"<br><small style='color:#555'>{note}</small>" if note else ""
    return (f'<div class="hbox {css}"><strong>{title}</strong><br>'
            f'{stat_name} = {stat_val:.4f} &nbsp;|&nbsp; <em>p</em> = {p_val:.4f}<br>'
            f'<strong>{verdict}</strong>{n_str}</div>')

tabs = st.tabs(["Normality", "One-Sample t", "Two-Sample", "Chi-Square", "ANOVA", "Correlation"])

# Normality
with tabs[0]:
    st.markdown("**H₀: The data follows a normal distribution.**")
    stat_sw, p_sw = stats.shapiro(series[:5000])
    st.markdown(hbox("Shapiro-Wilk Test", "W", stat_sw, p_sw, "Recommended for n < 2000"), unsafe_allow_html=True)
    ad = stats.anderson(series, dist='norm')
    st.markdown(f'<div class="hbox ok"><strong>Anderson-Darling Test</strong><br>A² = {ad.statistic:.4f}<br>Critical Values: {np.round(ad.critical_values,4)}</div>', unsafe_allow_html=True)
    stat_dag, p_dag = stats.normaltest(series)
    st.markdown(hbox("D'Agostino-Pearson (skew + kurtosis)", "K²", stat_dag, p_dag), unsafe_allow_html=True)

# One-Sample t
with tabs[1]:
    pop_mean = st.number_input("Hypothesised mean (H₀: μ = ?)", value=float(round(mean_val, 2)), step=0.1)
    tail     = st.radio("Test type", ["Two-tailed", "Greater than", "Less than"], horizontal=True)
    alt_map  = {"Two-tailed": "two-sided", "Greater than": "greater", "Less than": "less"}
    t_stat, p_t = stats.ttest_1samp(series, popmean=pop_mean, alternative=alt_map[tail])
    st.markdown(hbox(f"One-Sample t-Test  (H₀: μ = {pop_mean})", "t", t_stat, p_t,
                     f"df = {len(series)-1}  |  sample mean = {mean_val:.4f}  |  {int((1-alpha)*100)}% CI = ({ci_low:.4f}, {ci_high:.4f})"),
                unsafe_allow_html=True)

# Two-Sample
with tabs[2]:
    if comp_col != "None":
        g1 = df[sel_col].dropna(); g2 = df[comp_col].dropna()
        t2, p2   = stats.ttest_ind(g1, g2, equal_var=False)
        pooled_sd = np.sqrt((np.var(g1, ddof=1) + np.var(g2, ddof=1)) / 2)
        cohens_d = ((np.mean(g1) - np.mean(g2)) / pooled_sd) if pooled_sd > 0 else np.nan
        lw, plw  = stats.levene(g1, g2)
        mw, pmw  = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        st.markdown(hbox(f"Welch's t-Test: {sel_col} vs {comp_col}", "t", t2, p2, f"Unequal-variance assumed | Cohen\'s d = {cohens_d:.4f}"), unsafe_allow_html=True)
        st.markdown(hbox("Levene's Test for Equal Variances", "W", lw, plw), unsafe_allow_html=True)
        st.markdown(hbox("Mann-Whitney U (non-parametric)", "U", mw, pmw), unsafe_allow_html=True)
    elif cat_cols:
        gc = st.selectbox("Group by categorical column", cat_cols)
        gs = [g.dropna().values for _, g in df.groupby(gc)[sel_col] if len(g.dropna()) >= 2]
        if len(gs) >= 2:
            t2, p2 = stats.ttest_ind(gs[0], gs[1], equal_var=False)
            st.markdown(hbox(f"Welch's t-Test (first two groups of {gc})", "t", t2, p2), unsafe_allow_html=True)
    else:
        st.info("Select a comparison column or ensure a categorical column exists.")

# Chi-Square
with tabs[3]:
    if cat_cols:
        chi_col = st.selectbox("Categorical column", cat_cols)
        obs = df[chi_col].value_counts().values
        exp = np.full(len(obs), len(df[chi_col].dropna()) / len(obs))
        c2, pc2 = stats.chisquare(obs, exp)
        st.markdown(hbox(f"Goodness-of-Fit (H₀: uniform distribution of '{chi_col}')", "χ²", c2, pc2,
                         f"df = {len(obs)-1}"), unsafe_allow_html=True)
        if len(cat_cols) >= 2:
            c2b = st.selectbox("Second column (independence)", [c for c in cat_cols if c != chi_col])
            ct  = pd.crosstab(df[chi_col], df[c2b])
            ci2, pci, dof, _ = stats.chi2_contingency(ct)
            st.markdown(hbox(f"Independence Test: {chi_col} × {c2b}", "χ²", ci2, pci, f"df = {dof}"), unsafe_allow_html=True)
    else:
        st.info("No categorical columns found.")

# ANOVA
with tabs[4]:
    if cat_cols:
        ag = st.selectbox("Group by", cat_cols, key="ag")
        gs = [g.dropna().values for _, g in df.groupby(ag)[sel_col] if len(g.dropna()) >= 2]
        if len(gs) >= 2:
            fs, pf = stats.f_oneway(*gs)
            all_data = np.concatenate(gs)
            grand_mean = np.mean(all_data)
            ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in gs)
            ss_total = sum((x - grand_mean) ** 2 for x in all_data)
            eta_sq = (ss_between / ss_total) if ss_total > 0 else np.nan
            ks, pk = stats.kruskal(*gs)
            st.markdown(hbox(f"One-Way ANOVA: {sel_col} across '{ag}'", "F", fs, pf, f"{len(gs)} groups  |  H₀: all means equal | η² = {eta_sq:.4f}"), unsafe_allow_html=True)
            st.markdown(hbox(f"Kruskal-Wallis (non-parametric)", "H", ks, pk), unsafe_allow_html=True)
            grp_sum = df.groupby(ag)[sel_col].agg(["count","mean","std","median"]).round(4)
            st.dataframe(grp_sum, use_container_width=True)
    else:
        st.info("No categorical columns for ANOVA grouping.")

# Correlation
with tabs[5]:
    if len(numeric_cols) >= 2:
        method = st.radio("Method", ["pearson","spearman","kendall"], horizontal=True)
        corr_m = df[numeric_cols].corr(method=method)
        fig_c, ax_c = plt.subplots(figsize=(max(5, len(numeric_cols)), max(4, len(numeric_cols)-1)))
        sns.heatmap(corr_m, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5, ax=ax_c, square=True)
        ax_c.set_title(f"{method.capitalize()} Correlation Heatmap")
        st.pyplot(fig_c); plt.close()
        if comp_col != "None":
            tmp = df[[sel_col, comp_col]].dropna()
            if tmp[sel_col].nunique() > 1 and tmp[comp_col].nunique() > 1:
                rp, pp = stats.pearsonr(tmp[sel_col], tmp[comp_col])
                rs, ps = stats.spearmanr(tmp[sel_col], tmp[comp_col])
                st.markdown(hbox(f"Pearson r: {sel_col} vs {comp_col}", "r", rp, pp), unsafe_allow_html=True)
                st.markdown(hbox(f"Spearman ρ: {sel_col} vs {comp_col}", "ρ", rs, ps), unsafe_allow_html=True)
            else:
                st.warning("Correlation cannot be computed because one variable is constant.")
    else:
        st.info("Need at least 2 numeric columns.")

# ── Download ──────────────────────────────────────────────────────────────────
st.markdown("---")
summary_rows = {
    "Statistic": ["Count","Mean","Median","Mode","Std Dev","Variance","Range","IQR",
                  "Skewness","Kurtosis","CV (%)","Std Error Mean",
                  f"CI {int((1-alpha)*100)}% Low", f"CI {int((1-alpha)*100)}% High",
                  "Min","Max","Q1 (P25)","Q3 (P75)"],
    "Value": [len(series), mean_val, median_val, mode_vals[0] if mode_vals else float("nan"),
              std_val, var_val, range_val, iqr_val,
              skew_val, kurt_val, cv_val, sem_val,
              ci_low, ci_high,
              series.min(), series.max(), series.quantile(0.25), series.quantile(0.75)]
}
csv_out = pd.DataFrame(summary_rows).to_csv(index=False)
st.download_button("⬇️ Download Summary Statistics (CSV)", data=csv_out,
                   file_name=f"{sel_col}_statistics.csv", mime="text/csv")
