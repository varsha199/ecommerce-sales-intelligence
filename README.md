📊 Ecommerce Sales Intelligence & Forecasting Dashboard

An end-to-end data analytics project that transforms raw ecommerce data into a powerful interactive dashboard featuring:

✅ Data Cleaning & Loading
✅ Exploratory Data Analysis (EDA)
✅ PostgreSQL (Neon) Data Warehouse
✅ AI-Powered SQL Assistant
✅ Time-Series Forecasting with Prophet
✅ Streamlit Web App

📁 Project Structure
ECOMMERCE-SALES-INTELLIGENCE
│
├── .streamlit/
│   └── config.toml             # UI theme & styling for Streamlit
│
├── data/
│   ├── raw/
│   │   └── data.csv            # Original dataset
│   └── cleaned/                # (Optional) processed dataset
│
├── .env                        # Environment variables (DB + API keys)
├── .gitignore                  # Git ignore rules
│
├── 0_Data_load.ipynb           # Load & inspect raw data
├── 1_EDA.ipynb                 # Exploratory Data Analysis
├── 2_load_to_neon.ipynb        # Upload data to Neon PostgreSQL
├── 3_sql_analysis.ipynb        # SQL queries & insights
├── 4_forecasting.ipynb         # Prophet-based forecasting model
│
├── app.py                      # Streamlit application
├── LICENSE                     # License information
└── README.md                   # Project documentation

🚀 Features
✅ 1. Interactive Dashboard

Revenue, customers, and items sold

Monthly revenue trends

Country & date filters

✅ 2. AI-Powered SQL Assistant 🤖

Ask questions in plain English

Converts to SQL automatically

Executes on Neon database

Displays results + charts

Remembers past conversation history

✅ 3. Forecasting 📉

Prophet time-series model

Future sales prediction

Trend & seasonality components

Country-wise forecasting (optional extension)

✅ 4. Clean, Modular Workflow

Raw → EDA → SQL Warehouse → Forecasting → UI

🔧 Setup Instructions
1️⃣ Clone the Repo
git clone https://github.com/varsha199/ecommerce-sales-intelligence.git
cd ecommerce-sales-intelligence

2️⃣ Create Virtual Environment
python3 -m venv venv
source venv/bin/activate  # Mac

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Add Environment Variables

Create a .env file:

NEON_CONNECTION_STRING=postgresql://<user>:<password>@<host>/<db>
OPENAI_API_KEY=sk-xxxx

5️⃣ Run the App
streamlit run app.py

🧠 Technologies Used
Category	Tools
Language	Python
Database	Neon PostgreSQL
Modeling	Prophet
UI	Streamlit
Visualization	Plotly
AI	OpenAI GPT
Data Handling	Pandas, SQLAlchemy
🔮 Future Enhancements

✅ Multi-country forecasting
✅ Structured output for SQL + charts
✅ User authentication
✅ Export reports as PDF

👤 Author

Name: Varsha Maurya
LinkedIn: https://www.linkedin.com/in/varsha-maurya/ 
GitHub: https://github.com/varsha199/ecommerce-sales-intelligence

✅ License

This project is licensed under the MIT License.