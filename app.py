import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
from prophet import Prophet
import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import Literal
from pydantic import BaseModel, Field

# ------------ Structured Output Models ------------

ChartType = Literal["bar", "line", "pie", "scatter", "histogram"]

class ChartSpecification(BaseModel):
    chart_type: ChartType
    x_axis: str
    y_axis: str
    x_label: str
    y_label: str
    title: str

class AnalyticsResponse(BaseModel):
    explanation: str
    sql_query: str
    chart_spec: ChartSpecification

# ✅ ---------- Analytics Agent (Structured Output) ---------- ✅

def analytics_agent(user_query: str) -> AnalyticsResponse:
    """
    Takes a natural language question and returns a structured response:
    - explanation (text)
    - sql_query (valid PostgreSQL SQL)
    - chart_spec (instructions for visualization)
    """

    system_prompt = """
    You are an analytics assistant for a PostgreSQL database table named "sales_data".

    STRICT RULES — FOLLOW EXACTLY:
    1. Use ONLY the columns listed below.
    2. NEVER create or assume new columns like OrderDate, Month, etc.
    3. Do NOT use DATE_TRUNC or EXTRACT.
    4. "InvoiceMonth" is already a YYYY-MM formatted string. Use it directly if needed.
    5. Always wrap column names in double quotes.
    6. Always wrap the table name as "sales_data".
    7. For sums or averages of TotalPrice, use:
       ROUND(CAST(SUM("TotalPrice") AS numeric), 2)
    8. NEVER output DELETE, UPDATE, INSERT, DROP, or TRUNCATE.
    9. Only return SELECT queries.
    10. The output MUST be structured according to the AnalyticsResponse model.

    AVAILABLE COLUMNS:
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country",
    "TotalPrice", "InvoiceMonth"
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    # ✅ Structured OpenAI Call
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages,
        response_format=AnalyticsResponse,
        temperature=0.1
    )

    # ✅ Parsed structured output
    return completion.choices[0].message.parsed

# ✅ ---------- Execute SQL & Return DataFrame ---------- ✅

def run_analytics_query(sql_query: str) -> pd.DataFrame:
    """
    Runs the SQL query against the Neon database and returns a pandas DataFrame.
    """

    try:
        df_result = pd.read_sql(sql_query, engine)

        if df_result.empty:
            st.warning("✅ Query ran successfully, but returned no results.")

        return df_result

    except Exception as e:
        st.error(f"❌ SQL Execution Error: {e}")
        return pd.DataFrame()
    
# ✅ ---------- Create Chart from ChartSpecification ---------- ✅

import plotly.express as px

def create_chart_from_spec(df: pd.DataFrame, chart_spec) -> px.scatter:
    """
    Builds a Plotly chart based on the chart specification returned by the model.
    """

    # Safety checks
    if df.empty:
        return None

    if chart_spec.x_axis not in df.columns:
        st.warning(f"⚠️ '{chart_spec.x_axis}' not found in data. Using first column instead.")
        chart_spec.x_axis = df.columns[0]

    if chart_spec.y_axis not in df.columns:
        st.warning(f"⚠️ '{chart_spec.y_axis}' not found in data. Using first numeric column instead.")
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) == 0:
            return None
        chart_spec.y_axis = numeric_cols[0]

    # Build chart based on type
    if chart_spec.chart_type == "bar":
        fig = px.bar(
            df,
            x=chart_spec.x_axis,
            y=chart_spec.y_axis,
            title=chart_spec.title,
            labels={chart_spec.x_axis: chart_spec.x_label,
                    chart_spec.y_axis: chart_spec.y_label}
        )

    elif chart_spec.chart_type == "line":
        fig = px.line(
            df,
            x=chart_spec.x_axis,
            y=chart_spec.y_axis,
            title=chart_spec.title,
            markers=True,
            labels={chart_spec.x_axis: chart_spec.x_label,
                    chart_spec.y_axis: chart_spec.y_label}
        )

    elif chart_spec.chart_type == "pie":
        fig = px.pie(
            df,
            names=chart_spec.x_axis,
            values=chart_spec.y_axis,
            title=chart_spec.title
        )

    elif chart_spec.chart_type == "scatter":
        fig = px.scatter(
            df,
            x=chart_spec.x_axis,
            y=chart_spec.y_axis,
            title=chart_spec.title,
            labels={chart_spec.x_axis: chart_spec.x_label,
                    chart_spec.y_axis: chart_spec.y_label}
        )

    elif chart_spec.chart_type == "histogram":
        fig = px.histogram(
            df,
            x=chart_spec.y_axis,
            title=chart_spec.title,
            nbins=20
        )

    else:
        st.info("⚠️ Unsupported chart type.")
        return None

    return fig

# ------------------------------------------------------
# 🔐 LOAD ENVIRONMENT VARIABLES
# ------------------------------------------------------
load_dotenv()
NEON_CONN = os.getenv("NEON_CONNECTION_STRING")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not NEON_CONN:
    st.error("❌ NEON_CONNECTION_STRING missing in .env")
if not OPENAI_KEY:
    st.error("❌ OPENAI_API_KEY missing in .env")

client = OpenAI(api_key=OPENAI_KEY)

# ------------------------------------------------------
# 🗄️ DATABASE CONNECTION
# ------------------------------------------------------
engine = create_engine(NEON_CONN)

# ------------------------------------------------------
# 🎛️ PAGE SETUP
# ------------------------------------------------------
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Sales Analytics & Forecasting Dashboard")

# ------------------------------------------------------
# 📂 LOAD DATA (cached)
# ------------------------------------------------------
@st.cache_data
def load_data():
    query = """SELECT *, CAST("InvoiceDate" AS timestamp) AS invoice_ts FROM "sales_data" """
    df = pd.read_sql(query, engine)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df

df = load_data()

# ------------------------------------------------------
# 🎨 AUTO CHART GENERATOR
# ------------------------------------------------------
def auto_chart(df):
    """
    Advanced Intelligent Chart Generator:
    - Avoids ID columns
    - Prefers date → numeric (line chart)
    - Uses category → numeric (bar chart)
    - Uses 2 numeric cols (scatter)
    - Provides clean fallback
    """

    if df.empty:
        return None

    # Identify column types
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    date_cols = [
        col for col in df.columns 
        if "date" in col.lower() or "month" in col.lower() or df[col].dtype == "datetime64[ns]"
    ]

    # Remove typical ID columns
    id_keywords = ["id", "code", "number", "invoice"]
    clean_object_cols = [
        c for c in object_cols 
        if not any(k in c.lower() for k in id_keywords)
    ]
    clean_numeric_cols = [
        c for c in numeric_cols
        if not any(k in c.lower() for k in id_keywords)
    ]

    # Fallback to raw numeric if nothing is clean
    if not clean_numeric_cols:
        clean_numeric_cols = numeric_cols

    # ---------------------------------------------------
    # 1️⃣ DATE + NUMERIC → LINE CHART
    # ---------------------------------------------------
    if date_cols and clean_numeric_cols:
        x = date_cols[0]
        y = clean_numeric_cols[0]
        return px.line(df, x=x, y=y, title=f"{y} over {x}", markers=True)

    # ---------------------------------------------------
    # 2️⃣ CATEGORY + NUMERIC → BAR CHART
    # ---------------------------------------------------
    if clean_object_cols and clean_numeric_cols:
        x = clean_object_cols[0]
        y = clean_numeric_cols[0]
        return px.bar(df, x=x, y=y, title=f"{y} by {x}")

    # ---------------------------------------------------
    # 3️⃣ NUMERIC vs NUMERIC → SCATTER
    # ---------------------------------------------------
    if len(clean_numeric_cols) >= 2:
        x, y = clean_numeric_cols[:2]
        return px.scatter(df, x=x, y=y, title=f"{y} vs {x}")

    # ---------------------------------------------------
    # 4️⃣ Fallback: show single numeric bar
    # ---------------------------------------------------
    if clean_numeric_cols:
        return px.bar(df, y=clean_numeric_cols[0], title=f"{clean_numeric_cols[0]} Distribution")

    return None


# ------------------------------------------------------
# 🧭 SIDEBAR NAVIGATION
# ------------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "SQL Assistant", "Forecast", "Filters"]
)

# ------------------------------------------------------
# 📌 PAGE 1: DASHBOARD
# ------------------------------------------------------
if page == "Dashboard":
    st.header("📌 Business Summary")

    total_revenue = df["TotalPrice"].sum()
    total_customers = df["CustomerID"].nunique()
    total_items = df["Quantity"].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Total Revenue", f"${total_revenue:,.2f}")
    col2.metric("🧑‍💼 Active Customers", total_customers)
    col3.metric("📦 Total Items Sold", total_items)

    st.subheader("📅 Monthly Revenue Trend")
    monthly = (
        df.groupby(df["InvoiceDate"].dt.to_period("M"))["TotalPrice"]
        .sum()
        .reset_index()
    )
    monthly["InvoiceDate"] = monthly["InvoiceDate"].astype(str)

    fig = px.line(monthly, x="InvoiceDate", y="TotalPrice")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------
# 📌 PAGE 2: SQL ASSISTANT (ChatGPT-style)
# ------------------------------------------------------
if page == "SQL Assistant":

    st.title("🤖 Smart SQL Assistant")
    st.write("Ask anything about your database. I will convert it to SQL, run it, and show results + charts.")

    # ------------------------------------------------------
    # Initialize session state
    # ------------------------------------------------------
    if "conversation" not in st.session_state:
        st.session_state.conversation = []   # list[{ user, sql, result }]
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = 0

    st.markdown("---")

    # ------------------------------------------------------
    # Build tab labels
    # ------------------------------------------------------
    num_queries = len(st.session_state.conversation)
    tab_labels = [f"Query #{i+1}" for i in range(num_queries)] + ["➕ New Query"]

    selected_tab = st.radio(
        "Conversation History",
        options=list(range(len(tab_labels))),
        format_func=lambda i: tab_labels[i],
        index=st.session_state.active_tab,
        horizontal=True
    )

    st.session_state.active_tab = selected_tab
    st.markdown("---")

    # ------------------------------------------------------
    # CASE 1 — EXISTING QUERY TAB
    # ------------------------------------------------------
    if selected_tab < num_queries:

        entry = st.session_state.conversation[selected_tab]

        st.markdown("### 🧑 You Asked:")
        st.markdown(f"> {entry['user']}")

        st.markdown("### 📝 Generated SQL")
        st.code(entry["sql"], language="sql")

        st.markdown("### 📄 SQL Result")
        st.dataframe(entry["result"], use_container_width=True)

        chart = auto_chart(entry["result"])
        if chart:
            st.markdown("### 📊 Visualization")
            st.plotly_chart(chart, use_container_width=True)

    # ------------------------------------------------------
    # CASE 2 — NEW QUERY TAB
    # ------------------------------------------------------
    else:
        st.markdown("### 💬 Ask a new question:")

        user_prompt = st.text_area(
            "",
            key="new_query_box",
            placeholder="e.g., monthly sales trend",
            height=120
        )

        if st.button("Generate SQL & Run"):
            if not user_prompt.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Generating SQL..."):

                    # ------------------------------------------------------
                    # STRICT RULES PROMPT (your version)
                    # ------------------------------------------------------
                    prompt = f"""
                    You are a SQL generator for a PostgreSQL table named "sales_data".

                    STRICT RULES — FOLLOW EXACTLY:
                    1. Use ONLY the columns listed below.
                    2. NEVER create or assume new columns like OrderDate, OrderID, Month, etc.
                    3. NEVER use DATE_TRUNC or EXTRACT.
                    4. The column "InvoiceMonth" is already a YYYY-MM formatted string. Use it directly.
                    5. Always wrap column names in double quotes.
                    6. Always use ROUND(CAST(SUM("TotalPrice") AS numeric), 2) for sums.
                    7. Only output raw SQL. No explanation, No markdown.

                    AVAILABLE COLUMNS in "sales_data":
                    - "InvoiceNo"
                    - "StockCode"
                    - "Description"
                    - "Quantity"
                    - "InvoiceDate"      (timestamp)
                    - "UnitPrice"
                    - "CustomerID"
                    - "Country"
                    - "TotalPrice"
                    - "InvoiceMonth"     (YYYY-MM string)

                    USER QUESTION:
                    {user_prompt}
                    """

                    # ------------------------------------------------------
                    # Generate SQL from OpenAI
                    # ------------------------------------------------------
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}]
                    )

                    sql_query = (
                        response.choices[0].message.content
                        .replace("```sql", "")
                        .replace("```", "")
                        .strip()
                    )

                    # ------------------------------------------------------
                    # EXECUTE SQL
                    # ------------------------------------------------------
                    try:
                        result_df = pd.read_sql(sql_query, engine)

                        # Save new query into conversation
                        st.session_state.conversation.append({
                            "user": user_prompt,
                            "sql": sql_query,
                            "result": result_df
                        })

                        # Auto-switch to the newly created tab
                        st.session_state.active_tab = len(st.session_state.conversation) - 1
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ SQL Error: {e}")
                        st.code(sql_query, language="sql")
# ------------------------------------------------------
# 📌 PAGE 3: FORECAST
# ------------------------------------------------------
if page == "Forecast":
    st.header("📉 Prophet Forecast Model")

    # -----------------------------
    # 1. Country Selector
    # -----------------------------
    countries = sorted(df["Country"].dropna().unique().tolist())
    default_country = "United Kingdom" if "United Kingdom" in countries else countries[0]

    country = st.selectbox("🌍 Select Country for Forecasting", countries, index=countries.index(default_country))

    st.write(f"Forecasting sales for **{country}**")

    # Filter dataset by country
    df_country = df[df["Country"] == country].copy()

    # -----------------------------
    # 2. Monthly Aggregation
    # -----------------------------
    monthly = (
        df_country
        .groupby(df_country["InvoiceDate"].dt.to_period("M"))["TotalPrice"]
        .sum()
        .reset_index()
    )

    if monthly.empty:
        st.warning(f"No data available for {country}. Try another country.")
        st.stop()

    monthly["ds"] = monthly["InvoiceDate"].astype(str)
    monthly.rename(columns={"TotalPrice": "y"}, inplace=True)

    # -----------------------------
    # 3. Forecast Horizon
    # -----------------------------
    periods = st.slider(
        "Months to Forecast:",
        min_value=3,
        max_value=24,
        value=12,
        help="Choose how many future months Prophet should predict."
    )

    # -----------------------------
    # 4. Fit Prophet Model
    # -----------------------------
    model = Prophet()
    model.fit(monthly[["ds", "y"]])

    # -----------------------------
    # 5. Generate Forecast
    # -----------------------------
    future = model.make_future_dataframe(periods=periods, freq="M")
    forecast = model.predict(future)

    # -----------------------------
    # 6. Plot Forecast
    # -----------------------------
    st.subheader("📈 Forecast Chart")
    fig = model.plot(forecast)
    st.pyplot(fig)

    # -----------------------------
    # 7. Trend & Seasonality Components
    # -----------------------------
    st.subheader("📊 Trend & Seasonal Components")
    components_fig = model.plot_components(forecast)
    st.pyplot(components_fig)

    # -----------------------------
    # 8. Forecast Table
    # -----------------------------
    st.subheader("📄 Forecast Data (Last 20 rows)")
    st.dataframe(
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(20),
        use_container_width=True
    )

# ------------------------------------------------------
# 📌 PAGE 4: FILTERS
# ------------------------------------------------------
if page == "Filters":
    st.header("⚙️ Data Explorer With Filters")

    colA, colB = st.columns(2)
    with colA:
        countries = ["All"] + sorted(df["Country"].unique().tolist())
        selected_country = st.selectbox("🌍 Country", countries)

    with colB:
        date_range = st.date_input(
            "📅 Date Range",
            [df["InvoiceDate"].min(), df["InvoiceDate"].max()]
        )

    filtered = df.copy()
    start, end = date_range

    filtered = filtered[
        (filtered["InvoiceDate"] >= pd.to_datetime(start)) &
        (filtered["InvoiceDate"] <= pd.to_datetime(end))
    ]

    if selected_country != "All":
        filtered = filtered[filtered["Country"] == selected_country]

    st.subheader("📊 Filtered Sales")
    st.dataframe(filtered.head(100))

    monthly_filtered = (
        filtered.groupby(filtered["InvoiceDate"].dt.to_period("M"))["TotalPrice"]
        .sum()
        .reset_index()
    )
    monthly_filtered["InvoiceDate"] = monthly_filtered["InvoiceDate"].astype(str)

    fig = px.bar(monthly_filtered, x="InvoiceDate", y="TotalPrice")
    st.plotly_chart(fig, use_container_width=True)