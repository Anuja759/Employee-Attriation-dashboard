"""Palo Alto Networks – Employee Attrition Risk Dashboard (self-contained).
Run:  python -m streamlit run app.py     (keep Palo_Alto_Networks.csv in the same folder)"""
import os, numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

st.set_page_config(page_title="Attrition Risk | Palo Alto Networks", page_icon="🛡️", layout="wide")
CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Palo_Alto_Networks.csv")
CAT = ["BusinessTravel","Department","EducationField","Gender","JobRole","MaritalStatus","OverTime"]
COLORS = {"Low": "#2e9e5b", "Medium": "#f0a30a", "High": "#d64545"}

def engineer(df):
    d = df.copy()
    d["IncomePerExpYear"] = d["MonthlyIncome"] / (d["TotalWorkingYears"] + 1)
    d["PromotionDelay"] = (d["YearsSinceLastPromotion"] >= 3).astype(int)
    d["PromoDelayRatio"] = d["YearsSinceLastPromotion"] / (d["YearsAtCompany"] + 1)
    d["EngagementScore"] = d[["JobSatisfaction","EnvironmentSatisfaction","RelationshipSatisfaction","JobInvolvement","WorkLifeBalance"]].mean(axis=1)
    d["WorkloadStress"] = ((d["OverTime"] == "Yes") & (d["WorkLifeBalance"] <= 2)).astype(int)
    d["HopperIndex"] = d["NumCompaniesWorked"] / (d["TotalWorkingYears"] + 1)
    return d

def encode(df, cols=None):
    d = engineer(df).drop(columns=["Attrition","EmployeeID"], errors="ignore")
    d = pd.get_dummies(d, columns=CAT, dtype=float)
    return d if cols is None else d.reindex(columns=cols, fill_value=0.0)

def band(p, lo, hi):
    return np.where(p < lo, "Low", np.where(p <= hi, "Medium", "High"))

@st.cache_resource(show_spinner="Training model on your data…")
def build(df):
    X, y = encode(df), df["Attrition"].values
    cols = list(X.columns)
    mk = lambda: LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000)
    # honest test metrics
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    m0 = make_pipeline(StandardScaler(), mk()).fit(Xtr, ytr); p0 = m0.predict_proba(Xte)[:, 1]; yh = (p0 >= .5).astype(int)
    metrics = dict(Accuracy=accuracy_score(yte, yh), Precision=precision_score(yte, yh), Recall=recall_score(yte, yh), F1=f1_score(yte, yh), ROC_AUC=roc_auc_score(yte, p0))
    # calibrate probabilities so 30%/60% bands are meaningful
    oof = cross_val_predict(make_pipeline(StandardScaler(), mk()), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), method="decision_function")
    cal = LogisticRegression().fit(oof.reshape(-1, 1), y)
    sc = StandardScaler().fit(X); model = mk().fit(sc.transform(X), y)
    return model, sc, cal, cols, metrics

# ---------- data ----------
st.sidebar.title("🛡️ Attrition Risk")
if os.path.exists(CSV): raw = pd.read_csv(CSV)
else:
    up = st.sidebar.file_uploader("Upload Palo_Alto_Networks.csv", type="csv")
    if up is None: st.title("Attrition Risk Dashboard"); st.info("Place Palo_Alto_Networks.csv next to app.py, or upload it in the sidebar."); st.stop()
    raw = pd.read_csv(up)
raw = raw.copy(); raw.insert(0, "EmployeeID", np.arange(1, len(raw) + 1))
model, sc, cal, COLS, METRICS = build(raw)

def prob_of(df):
    z = model.decision_function(sc.transform(encode(df, COLS)))
    return cal.predict_proba(z.reshape(-1, 1))[:, 1]

def reasons(row):
    x = sc.transform(encode(row, COLS))[0]; ct = pd.Series(model.coef_[0] * x, index=COLS); g = {}
    for c, v in ct.items():
        b = next((k for k in CAT if c.startswith(k + "_")), c); g[b] = g.get(b, 0) + v
    return pd.Series(g).sort_values()

data = raw.copy(); data["AttritionProb"] = prob_of(raw)

# ---------- sidebar filters ----------
page = st.sidebar.radio("View", ["Risk Dashboard", "Employee Risk Profile", "Department & Role View", "Explainability & What-If"])
st.sidebar.markdown("---")
depts = st.sidebar.multiselect("Department", sorted(data.Department.unique()), default=sorted(data.Department.unique()))
roles_all = sorted(data[data.Department.isin(depts)].JobRole.unique())
roles = st.sidebar.multiselect("Job role", roles_all, default=roles_all)
lo, hi = st.sidebar.slider("Risk thresholds: Medium starts / High starts", 0.05, 0.95, (0.30, 0.60), 0.05)
data["RiskBand"] = band(data.AttritionProb, lo, hi)
view = data[data.Department.isin(depts) & data.JobRole.isin(roles)]
st.sidebar.caption("Calibrated probabilities. Decision support only – never use for discipline, pay or promotion decisions.")
if view.empty: st.warning("No employees match the current filters."); st.stop()

# ---------- pages ----------
if page == "Risk Dashboard":
    st.title("Attrition Risk Dashboard")
    c = st.columns(5)
    c[0].metric("Employees", f"{len(view):,}"); c[1].metric("High risk", int((view.RiskBand == "High").sum()))
    c[2].metric("Medium risk", int((view.RiskBand == "Medium").sum())); c[3].metric("Low risk", int((view.RiskBand == "Low").sum()))
    c[4].metric("Avg probability", f"{view.AttritionProb.mean():.1%}")
    l, r = st.columns(2)
    cnt = view.RiskBand.value_counts().reindex(["Low", "Medium", "High"]).fillna(0).reset_index(); cnt.columns = ["RiskBand", "Employees"]
    l.plotly_chart(px.bar(cnt, x="RiskBand", y="Employees", color="RiskBand", color_discrete_map=COLORS, text="Employees", title="Risk category distribution"))
    r.plotly_chart(px.histogram(view, x="AttritionProb", nbins=30, title="Distribution of attrition probability").add_vline(x=lo, line_dash="dash").add_vline(x=hi, line_dash="dash"))
    st.subheader("Priority list: highest-risk current employees")
    top = view[view.Attrition == 0].sort_values("AttritionProb", ascending=False)
    cols = ["EmployeeID", "Department", "JobRole", "JobLevel", "OverTime", "MonthlyIncome", "YearsSinceLastPromotion", "AttritionProb", "RiskBand"]
    st.dataframe(top[cols].head(25).style.format({"AttritionProb": "{:.1%}"}), hide_index=True)
    st.download_button("⬇️ Download full scored list (CSV)", view[cols + ["Attrition"]].sort_values("AttritionProb", ascending=False).to_csv(index=False), "scored_employees.csv", "text/csv")

elif page == "Employee Risk Profile":
    st.title("Employee Risk Profile")
    eid = st.selectbox("Employee ID", view.EmployeeID.tolist())
    row = data[data.EmployeeID == eid]; s = row.iloc[0]; p = float(s.AttritionProb); b = s.RiskBand
    l, r = st.columns([1, 2])
    g = go.Figure(go.Indicator(mode="gauge+number", value=p * 100, number={"suffix": "%"}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": COLORS[b]},
        "steps": [{"range": [0, lo * 100], "color": "#e6f4ea"}, {"range": [lo * 100, hi * 100], "color": "#fdf3dc"}, {"range": [hi * 100, 100], "color": "#fbe3e3"}]}))
    g.update_layout(height=260, margin=dict(t=30, b=0)); l.plotly_chart(g); l.markdown(f"### Risk category: **{b}**")
    r.write(f"**{s.JobRole}** · {s.Department} · Job level {s.JobLevel} · Age {s.Age} · {s.YearsAtCompany} yrs at company · Overtime: {s.OverTime} · Monthly income: {s.MonthlyIncome:,}")
    ct = reasons(row); up, dn = ct[ct > 0.1].sort_values(ascending=False).head(5), ct[ct < -0.1].head(3)
    a, bb = r.columns(2)
    a.markdown("**🔺 Raising risk**"); [a.write(f"• {k} (+{v:.2f})") for k, v in up.items()] or a.write("None significant")
    bb.markdown("**🔻 Protective**"); [bb.write(f"• {k} ({v:.2f})") for k, v in dn.items()] or bb.write("None significant")
    show = pd.concat([ct.head(6), ct.tail(6)]); show = show[~show.index.duplicated()].rename("Impact").rename_axis("Feature").reset_index()
    st.plotly_chart(px.bar(show, x="Impact", y="Feature", orientation="h", title="Contribution to this employee's risk (right = higher risk)"))
    if s.Attrition == 1: st.caption("Note: this employee is recorded as having left.")

elif page == "Department & Role View":
    st.title("Department & Role Risk View")
    g = view.groupby("Department").agg(Employees=("EmployeeID", "count"), AvgRisk=("AttritionProb", "mean"), HighRisk=("RiskBand", lambda s: (s == "High").sum()), ActualAttrition=("Attrition", "mean")).reset_index()
    st.plotly_chart(px.bar(g, x="Department", y="AvgRisk", color="AvgRisk", color_continuous_scale="Reds", title="Average attrition probability by department"))
    st.dataframe(g.style.format({"AvgRisk": "{:.1%}", "ActualAttrition": "{:.1%}"}), hide_index=True)
    rl = view.groupby("JobRole").agg(Employees=("EmployeeID", "count"), AvgRisk=("AttritionProb", "mean"), HighRisk=("RiskBand", lambda s: (s == "High").sum())).reset_index().sort_values("AvgRisk")
    st.plotly_chart(px.bar(rl, x="AvgRisk", y="JobRole", orientation="h", color="HighRisk", title="Average risk by job role (colour = # high-risk)"))
    st.subheader("Risk band counts by role")
    st.dataframe(view.pivot_table(index="JobRole", columns="RiskBand", values="EmployeeID", aggfunc="count", fill_value=0))

else:
    st.title("Explainability & What-If")
    t1, t2 = st.tabs(["Global drivers", "What-if scenario"])
    with t1:
        grp = {}
        for c, v in pd.Series(model.coef_[0], index=COLS).items():
            k = next((k for k in CAT if c.startswith(k + "_")), c); grp[k] = grp.get(k, 0) + abs(v)
        imp = pd.Series(grp).sort_values().tail(15).rename("Importance").rename_axis("Feature").reset_index()
        st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h", title="Top 15 drivers (standardized |coefficient|, grouped by original feature)"))
        st.markdown("**Model performance (held-out 20% test set, Logistic Regression)**")
        st.dataframe(pd.DataFrame([METRICS]).round(3), hide_index=True)
    with t2:
        eid = st.selectbox("Employee", view.EmployeeID.tolist(), key="wi")
        base = raw[raw.EmployeeID == eid].copy(); mod = base.copy(); gv = lambda c: int(base[c].iloc[0])
        c1, c2, c3 = st.columns(3)
        mod["OverTime"] = c1.selectbox("OverTime", ["Yes", "No"], index=["Yes", "No"].index(base.OverTime.iloc[0]))
        mod["JobSatisfaction"] = c1.slider("Job satisfaction", 1, 4, gv("JobSatisfaction"))
        mod["WorkLifeBalance"] = c2.slider("Work-life balance", 1, 4, gv("WorkLifeBalance"))
        mod["EnvironmentSatisfaction"] = c2.slider("Environment satisfaction", 1, 4, gv("EnvironmentSatisfaction"))
        mod["StockOptionLevel"] = c3.slider("Stock option level", 0, 3, gv("StockOptionLevel"))
        mod["YearsSinceLastPromotion"] = c3.slider("Years since last promotion", 0, 15, gv("YearsSinceLastPromotion"))
        mod["MonthlyIncome"] = c3.number_input("Monthly income", 1000, 25000, gv("MonthlyIncome"), 500)
        p0, p1 = float(prob_of(base)[0]), float(prob_of(mod)[0]); m = st.columns(3)
        m[0].metric("Current risk", f"{p0:.1%}"); m[1].metric("Scenario risk", f"{p1:.1%}", f"{(p1 - p0) * 100:+.1f} pts", delta_color="inverse")
        m[2].metric("Scenario category", str(band(np.array([p1]), lo, hi)[0]))
