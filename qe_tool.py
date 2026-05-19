
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------------- CONFIG ----------------
FILE = "defects.xlsx"

# ---------------- LOGIN ----------------
USERS = {"qe": "123"}

if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:
    st.title("🔐 QE Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if USERS.get(user) == pwd:
            st.session_state.login = True
        else:
            st.error("Invalid login")
    st.stop()

# ---------------- LOAD DATA ----------------
def load_data():
    if os.path.exists(FILE):
        try:
            return pd.read_excel(FILE)
        except:
            pass

    # sample data
    sample = pd.DataFrame([
        {"ID":1,"Title":"Login issue","Description":"Login fails","Severity":"High","Status":"Open","Assignee":"A","Module":"Auth","CreatedDate":datetime.now().strftime("%Y-%m-%d"),"SLADays":2,"Age":1,"SLAStatus":"✅ Within SLA"},
        {"ID":2,"Title":"API error","Description":"API failure","Severity":"Critical","Status":"Open","Assignee":"B","Module":"API","CreatedDate":datetime.now().strftime("%Y-%m-%d"),"SLADays":1,"Age":1,"SLAStatus":"❌ Breached"},
        {"ID":3,"Title":"UI bug","Description":"Button broken","Severity":"Medium","Status":"Open","Assignee":"C","Module":"UI","CreatedDate":datetime.now().strftime("%Y-%m-%d"),"SLADays":3,"Age":2,"SLAStatus":"✅ Within SLA"},
        {"ID":4,"Title":"Crash","Description":"App crash","Severity":"High","Status":"Open","Assignee":"D","Module":"Core","CreatedDate":datetime.now().strftime("%Y-%m-%d"),"SLADays":2,"Age":2,"SLAStatus":"❌ Breached"},
        {"ID":5,"Title":"Timeout","Description":"Slow response","Severity":"Medium","Status":"Open","Assignee":"E","Module":"Backend","CreatedDate":datetime.now().strftime("%Y-%m-%d"),"SLADays":3,"Age":1,"SLAStatus":"✅ Within SLA"}
    ])
    return sample

def save_data(df):
    df.to_excel(FILE, index=False)

df = load_data()

# ---------------- SLA ----------------
def calculate_sla(severity, created):
    sla_map = {"Low":5,"Medium":3,"High":2,"Critical":1}
    sla_days = sla_map.get(severity,3)

    created_date = datetime.strptime(created, "%Y-%m-%d")
    age = (datetime.now() - created_date).days

    status = "✅ Within SLA" if age <= sla_days else "❌ Breached"
    return sla_days, age, status

# ---------------- PREDICTION ----------------
def predict_future(df):
    if len(df) < 5:
        return None

    df["CreatedDate"] = pd.to_datetime(df["CreatedDate"])
    trend = df.groupby("CreatedDate").size().reset_index(name="Count")

    if len(trend) < 2:
        return None

    trend["Day"] = range(len(trend))

    X = trend[["Day"]]
    y = trend["Count"]

    model = LinearRegression()
    model.fit(X, y)

    future = np.array(range(len(trend), len(trend)+5)).reshape(-1,1)
    return model.predict(future)

# ---------------- UI ----------------
st.set_page_config(layout="wide")
st.title("🚀 QE AI Defect Management System")

menu = st.sidebar.radio("Navigation", [
    "📊 Dashboard",
    "➕ Create",
    "📋 View"
])

# ---------------- CREATE ----------------
if menu == "➕ Create":

    st.subheader("Create Defect")

    col1, col2 = st.columns(2)

    with col1:
        title = st.text_input("Title")
        desc = st.text_area("Description")

    with col2:
        severity = st.selectbox("Severity", ["Low","Medium","High","Critical"])
        assignee = st.text_input("Assignee")
        module = st.text_input("Module")

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

        st.success("✅ Defect Created")

# ---------------- DASHBOARD ----------------
elif menu == "📊 Dashboard":

    st.subheader("Executive Dashboard")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Total", len(df))
    col2.metric("Open", len(df[df["Status"]!="Closed"]))
    col3.metric("Critical", len(df[df["Severity"]=="Critical"]))
    col4.metric("Breach", len(df[df["SLAStatus"]=="❌ Breached"]))

    fig1 = px.bar(df, x="Severity", color="Severity")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.pie(df, names="Status")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🔮 Prediction")

    pred = predict_future(df)

    if pred is not None:
        pred_df = pd.DataFrame({
            "Day": range(1,len(pred)+1),
            "Defects": pred
        })
        fig3 = px.line(pred_df, x="Day", y="Defects", markers=True)
        st.plotly_chart(fig3)
    else:
        st.warning("Not enough data for prediction")

# ---------------- VIEW ----------------
elif menu == "📋 View":

    st.subheader("All Defects")

    def highlight(row):
        return ['background-color:#ffcccc']*len(row) if row["SLAStatus"]=="❌ Breached" else ['']*len(row)

    st.dataframe(df.style.apply(highlight, axis=1), use_container_width=True)
