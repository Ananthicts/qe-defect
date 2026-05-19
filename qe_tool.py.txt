import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# ================= FILE =================
FILE = "defects.xlsx"

# ================= LOAD DATA =================
def load_data():
    if os.path.exists(FILE):
        return pd.read_excel(FILE)
    else:
        df = pd.DataFrame(columns=[
            "ID","Title","Description","Severity","Status",
            "Assignee","Module","CreatedDate","SLADays","Age","SLAStatus"
        ])
        return df

def save_data(df):
    df.to_excel(FILE, index=False)

df = load_data()

# ================= SLA =================
def calculate_sla(severity, created):
    today = datetime.now()
    sla_map = {"Low":5,"Medium":3,"High":2,"Critical":1}

    sla_days = sla_map.get(severity,3)
    created_date = datetime.strptime(created, "%Y-%m-%d")
    age = (today - created_date).days

    status = "✅ Within SLA" if age <= sla_days else "❌ Breached"
    return sla_days, age, status

# ================= AI: TRAIN =================
def train_model(df):
    if len(df) < 5:
        return None, None

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["Description"].astype(str))
    y = df["Severity"]

    model = MultinomialNB()
    model.fit(X, y)

    return model, vectorizer

def predict_severity(model, vectorizer, text):
    if model is None:
        return "N/A"

    X = vectorizer.transform([text])
    return model.predict(X)[0]

# ================= DUPLICATE =================
def find_duplicates(df, desc):
    if len(df) == 0:
        return []

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["Description"].astype(str).tolist()+[desc])

    sim = cosine_similarity(X[-1], X[:-1])[0]

    duplicates = []
    for i, s in enumerate(sim):
        if s > 0.7:
            duplicates.append(df.iloc[i]["Title"])

    return duplicates

# ================= PREDICTION =================
def prepare_trend(df):
    if len(df)==0:
        return None

    df["CreatedDate"] = pd.to_datetime(df["CreatedDate"])
    trend = df.groupby("CreatedDate").size().reset_index(name="Count")
    trend["Day"] = range(len(trend))
    return trend

def predict_future(df):
    if len(df) < 5:
        return None

    trend = prepare_trend(df)
    if trend is None:
        return None

    X = trend[["Day"]]
    y = trend["Count"]

    model = LinearRegression()
    model.fit(X, y)

    future = np.array(range(len(trend), len(trend)+5)).reshape(-1,1)
    pred = model.predict(future)

    return pred

# ================= UI =================
st.set_page_config(page_title="QE AI Dashboard", layout="wide")

st.title("🚀 QE AI Defect Management System")

menu = st.sidebar.selectbox("Menu", ["Dashboard", "Create Defect", "View Data"])

# ================= CREATE =================
if menu == "Create Defect":

    st.subheader("Create New Defect")

    title = st.text_input("Title")
    desc = st.text_area("Description")

    severity = st.selectbox("Severity", ["Low","Medium","High","Critical"])
    assignee = st.text_input("Assignee")
    module = st.text_input("Module")

    # AI Suggestions
    model, vectorizer = train_model(df)

    if desc:
        pred = predict_severity(model, vectorizer, desc)
        st.info(f"🤖 Suggested Severity: {pred}")

        duplicates = find_duplicates(df, desc)
        if duplicates:
            st.warning("Possible duplicates:")
            for d in duplicates:
                st.write("-", d)

    if st.button("Submit"):

        created = datetime.now().strftime("%Y-%m-%d")
        sla, age, status = calculate_sla(severity, created)

        new = pd.DataFrame([{
            "ID": len(df)+1,
            "Title": title,
            "Description": desc,
            "Severity": severity,
            "Status": "Open",
            "Assignee": assignee,
            "Module": module,
            "CreatedDate": created,
            "SLADays": sla,
            "Age": age,
            "SLAStatus": status
        }])

        df = pd.concat([df,new], ignore_index=True)
        save_data(df)

        st.success("✅ Defect Added")

# ================= DASHBOARD =================
elif menu == "Dashboard":

    st.subheader("Executive Dashboard")

    total = len(df)
    open_d = len(df[df["Status"]!="Closed"])
    critical = len(df[df["Severity"]=="Critical"])
    breach = len(df[df["SLAStatus"]=="❌ Breached"])

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total", total)
    c2.metric("Open", open_d)
    c3.metric("Critical", critical)
    c4.metric("Breaches", breach)

    # Charts
    fig1 = px.bar(df, x="Severity", title="Severity")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.pie(df, names="Status", title="Status")
    st.plotly_chart(fig2)

    # Prediction
    st.subheader("🔮 Prediction")

    pred = predict_future(df)

    if pred is not None:

        pred_df = pd.DataFrame({
            "Day": range(1,len(pred)+1),
            "Predicted": pred
        })

        fig_pred = px.line(pred_df, x="Day", y="Predicted", markers=True)
        st.plotly_chart(fig_pred)

# ================= VIEW =================
elif menu == "View Data":

    st.subheader("All Defects")

