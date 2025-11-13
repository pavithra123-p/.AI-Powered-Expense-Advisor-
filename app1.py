### AI_POWERED EXPESNSE ADVISIOR ##
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
from io import BytesIO
from reportlab.pdfgen import canvas

from reportlab.lib.pagesizes import A4

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="AI Expense Advisor", layout="wide")
st.title("AI-Powered Expense Advisor")

# -------------------- THEME SWITCH --------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

st.sidebar.header(" Settings")
st.sidebar.toggle("Dark Mode", key="dark_mode")
PLOTLY_THEME = "plotly_dark" if st.session_state.dark_mode else "plotly_white"

# -------------------- LOAD MODEL --------------------
try:
    model = joblib.load("expense_classifier.joblib")
except:
    st.error(" Model file not found! Place `expense_classifier.joblib` in folder.")
    st.stop()

# -------------------- PDF EXPORT FUNCTION --------------------
def create_pdf(summary_text):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 40

    for line in summary_text.split("\n"):
        c.drawString(40, y, line[:110])
        y -= 18
        if y < 60:
            c.showPage()
            y = height - 40
    c.save()
    buffer.seek(0)
    return buffer

# -------------------- FILE UPLOAD --------------------
uploaded_file = st.file_uploader(" Upload Expense CSV", type=["csv"])
if uploaded_file is None:
    st.info("Upload a CSV with at least a `description` column.")
    st.stop()

df = pd.read_csv(uploaded_file)
df.columns = [c.strip().lower() for c in df.columns]

if "description" not in df.columns:
    st.error("CSV must contain a `description` column.")
    st.stop()

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

if "amount" in df.columns:
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

# -------------------- PREDICT CATEGORIES --------------------
df["predicted_category"] = model.predict(df["description"].astype(str))
use_col = "predicted_category"

st.subheader(" Uploaded Data Preview")
st.dataframe(df.head(20), use_container_width=True)

# -------------------- BAR CHART --------------------
st.subheader(" Spending by Category")
if "amount" in df.columns:
    bar = df.groupby(use_col)["amount"].sum().reset_index()
    fig = px.bar(bar, x=use_col, y="amount",
                 title="Total Spend by Category",
                 template=PLOTLY_THEME)
    st.plotly_chart(fig, use_container_width=True)

# -------------------- PIE CHART --------------------
st.subheader(" Category Spending Share")
if "amount" in df.columns:
    pie = df.groupby(use_col)["amount"].sum().reset_index()
    fig = px.pie(pie, values="amount", names=use_col,
                 hole=0.45, title="Spending Percentage",
                 template=PLOTLY_THEME)
    st.plotly_chart(fig, use_container_width=True)

# -------------------- MONTHLY TREND --------------------
if "date" in df.columns and "amount" in df.columns:
    st.subheader(" Monthly Spending Trend")
    month_data = (df.dropna(subset=["date"])
                    .assign(month=lambda x: x["date"].dt.to_period("M").astype(str))
                    .groupby("month")["amount"].sum().reset_index())

    if not month_data.empty:
        fig = px.line(month_data, x="month", y="amount", markers=True,
                      title="Total Spend by Month", template=PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

# -------------------- HEATMAP CATEGORY × MONTH --------------------
if "date" in df.columns and "amount" in df.columns:
    st.subheader(" Heatmap — Category vs Month")
    temp = df.dropna(subset=["date", "amount"])
    temp["month"] = temp["date"].dt.to_period("M").astype(str)
    pivot = temp.pivot_table(index=use_col, columns="month",
                             values="amount", aggfunc="sum", fill_value=0)

    if not pivot.empty:
        fig = px.imshow(pivot, aspect="auto",
                        labels=dict(x="Month", y="Category", color="Spend"),
                        title="Heatmap: Spend by Category per Month",
                        template=PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

# -------------------- BUDGET SUGGESTIONS --------------------
st.subheader(" Smart Budget Suggestions")
tips = []

if "amount" in df.columns:
    total = df["amount"].sum()
    share = df.groupby(use_col)["amount"].sum() / total

    if share.iloc[0] > 0.35:
        tips.append(
            f" High spend on **{share.idxmax()}**. Reduce by 10–15% next month."
        )

    if (df["amount"] < 200).sum() > 8:
        tips.append(" Too many small spends — plan daily budget / batch shopping.")

    text = " ".join(df["description"].str.lower())
    if any(word in text for word in ["netflix", "prime", "hotstar", "subscription"]):
        tips.append(" Subscription detected — cancel unused memberships.")

if tips:
    for t in tips: st.write("• " + t)
else:
    st.success(" Spending looks balanced!")

# -------------------- TOTAL NEXT-MONTH PREDICTION --------------------
if "date" in df.columns and "amount" in df.columns:
    st.subheader("🔮 Predicted Next Month Total Spend")

    temp = df.dropna(subset=["date","amount"]).copy()
    temp["month"] = temp["date"].dt.to_period("M").astype(str)
    monthly = temp.groupby("month")["amount"].sum().reset_index()
    monthly["t"] = np.arange(len(monthly))

    if len(monthly) >= 3:
        reg = LinearRegression().fit(monthly[["t"]], monthly["amount"])
        next_val = reg.predict([[monthly["t"].max() + 1]])[0]
        st.success(f"Estimated total next month spend: **₹{next_val:,.0f}**")

# -------------------- CATEGORY WISE FORECAST --------------------
if "date" in df.columns and "amount" in df.columns:
    st.subheader(" Category-wise Next Month Forecast")

    temp = df.dropna(subset=["date","amount"]).copy()
    temp["month"] = temp["date"].dt.to_period("M").astype(str)

    results = {}
    for cat in df[use_col].unique():
        cdata = temp[temp[use_col]==cat].groupby("month")["amount"].sum().reset_index()
        if len(cdata) >= 3:
            cdata["t"] = np.arange(len(cdata))
            reg = LinearRegression().fit(cdata[["t"]], cdata["amount"])
            nxt = reg.predict([[cdata["t"].max()+1]])[0]
            results[cat] = nxt

    if results:
        forecast = pd.DataFrame(results.items(), columns=["Category","Predicted_Spend"])
        forecast = forecast.sort_values(by="Predicted_Spend", ascending=False)
        st.dataframe(forecast, use_container_width=True)

        top_cat = forecast.iloc[0]["Category"]
        top_val = forecast.iloc[0]["Predicted_Spend"]
        st.success(f" Expected highest expense next month: **{top_cat} → ₹{top_val:,.0f}**")

# -------------------- ADD MANUAL EXPENSE --------------------
st.subheader("Add Manual Expense")
with st.form("add_expense", clear_on_submit=True):
    d = st.text_input("Description")
    a = st.number_input("Amount", min_value=0.0)
    dt = st.date_input("Date")
    btn = st.form_submit_button("Add & Predict")

if btn:
    new = {"description": d, "amount": a, "date": pd.to_datetime(str(dt))}
    new["predicted_category"] = model.predict([d])[0]
    df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
    st.success(f"Added  {d} → {new['predicted_category']}")
    st.dataframe(df.tail(5))

# -------------------- PDF DOWNLOAD --------------------
summary = f"""
AI Expense Report
-------------------------
Total rows        : {len(df)}
Total spending    : ₹{df['amount'].sum() if 'amount' in df.columns else 'N/A'}
Top category      : {df['predicted_category'].value_counts().idxmax()}
"""

pdf = create_pdf(summary)
st.download_button(" Download PDF Report", data=pdf,
                   file_name="expense_report.pdf", mime="application/pdf")

# -------------------- CSV DOWNLOAD --------------------
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇ Download Updated CSV", csv, "updated_expenses.csv", "text/csv")

