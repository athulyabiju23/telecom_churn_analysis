import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from dotenv import load_dotenv

st.set_page_config(page_title="Telecom Churn Analysis", page_icon="📞", layout="wide")

# loading data - tries snowflake first, falls back to csv
@st.cache_data
def load_data():
    try:
        import snowflake.connector
        load_dotenv()
        try:
            user = st.secrets["SNOWFLAKE_USER"]
            password = st.secrets["SNOWFLAKE_PASSWORD"]
            account = st.secrets["SNOWFLAKE_ACCOUNT"]
            warehouse = st.secrets["SNOWFLAKE_WAREHOUSE"]
            database = st.secrets["SNOWFLAKE_DATABASE"]
        except:
            user = os.getenv('SNOWFLAKE_USER')
            password = os.getenv('SNOWFLAKE_PASSWORD')
            account = os.getenv('SNOWFLAKE_ACCOUNT')
            warehouse = os.getenv('SNOWFLAKE_WAREHOUSE')
            database = os.getenv('SNOWFLAKE_DATABASE')
        
        conn = snowflake.connector.connect(
            user=user,
            password=password,
            account=account,
            warehouse=warehouse,
            database=database,
            schema='CLEAN'
        )
        ibm_maven = pd.read_sql('SELECT * FROM IBM_MAVEN', conn)
        kaggle = pd.read_sql('SELECT * FROM KAGGLE_TELECOMS', conn)
        st.sidebar.success("connected to snowflake")
    except Exception as e:
        ibm_maven = pd.read_csv('data/processed/scored_customers.csv')
        kaggle = pd.read_csv('data/processed/kaggle_telecoms.csv')
    
    ibm_maven['CHURN_BINARY'] = (ibm_maven['CHURN'] == 'Yes').astype(int)
    return ibm_maven, kaggle

@st.cache_resource
def load_model():
    try:
        with open('data/processed/churn_model.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

ibm_maven, kaggle = load_data()
model_data = load_model()

# sidebar
st.sidebar.title("📞 Telecom Churn")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", 
    ["Overview", "Churn Explorer", "Risk Predictor", "Playbook"],
    label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("**data sources:**")
st.sidebar.markdown("IBM Telco + Maven Analytics (6,589)")
st.sidebar.markdown("Kaggle Telecoms (3,333)")


if page == "Overview":
    st.title("Telecom Churn Overview")
    st.markdown("analyzing ~10,000 customers to understand who churns and why")
    
    total = len(ibm_maven)
    churned = (ibm_maven['CHURN'] == 'Yes').sum()
    churn_rate = churned / total * 100
    avg_monthly = ibm_maven['MONTHLYCHARGES'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total:,}")
    col2.metric("Churned", f"{churned:,}")
    col3.metric("Churn Rate", f"{churn_rate:.1f}%")
    col4.metric("Avg Monthly Charges", f"${avg_monthly:.0f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn by Contract Type")
        contract_churn = ibm_maven.groupby('CONTRACT')['CHURN_BINARY'].mean() * 100
        contract_order = contract_churn.sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(contract_order.index, contract_order.values, 
                      color=['indianred', 'sandybrown', 'mediumseagreen'])
        for bar, val in zip(bars, contract_order.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                    f'{val:.1f}%', ha='center', fontsize=10)
        ax.set_ylabel('Churn Rate (%)')
        ax.set_ylim(0, 60)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Why Customers Left")
        reasons = ibm_maven[ibm_maven['CHURN']=='Yes']['CHURN_CATEGORY'].value_counts()
        
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(reasons.index, reasons.values, color='steelblue')
        for bar, val in zip(bars, reasons.values):
            pct = val / reasons.sum() * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{pct:.0f}%', ha='center', fontsize=9)
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=20)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn by Tenure")
        ibm_maven['TENURE_COHORT'] = pd.cut(ibm_maven['TENURE'],
            bins=[0, 6, 12, 24, 100],
            labels=['0-6 mo', '6-12 mo', '12-24 mo', '24+ mo'])
        tenure_churn = ibm_maven.groupby('TENURE_COHORT', observed=True)['CHURN_BINARY'].mean() * 100
        
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#d9534f', '#f0ad4e', '#5bc0de', '#5cb85c']
        bars = ax.bar(tenure_churn.index, tenure_churn.values, color=colors)
        for bar, val in zip(bars, tenure_churn.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', fontsize=10)
        ax.set_ylabel('Churn Rate (%)')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Service Calls vs Churn (Kaggle)")
        kaggle['CHURN_BIN'] = (kaggle['CHURN_FLAG'] == 'Yes').astype(int) if 'CHURN_FLAG' in kaggle.columns else (kaggle['CHURN'] == True).astype(int)
        cs_churn = kaggle.groupby('CUSTOMER_SERVICE_CALLS')['CHURN_BIN'].mean() * 100
        cs_churn = cs_churn[cs_churn.index <= 7]
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(cs_churn.index, cs_churn.values, color='steelblue')
        for x, val in zip(cs_churn.index, cs_churn.values):
            ax.text(x, val + 1.5, f'{val:.0f}%', ha='center', fontsize=9)
        ax.set_xlabel('Customer Service Calls')
        ax.set_ylabel('Churn Rate (%)')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


elif page == "Churn Explorer":
    st.title("Churn Explorer")
    st.markdown("filter and explore churn patterns")
    
    st.sidebar.markdown("### Filters")
    
    contracts = ibm_maven['CONTRACT'].unique().tolist()
    selected_contracts = st.sidebar.multiselect("Contract Type", contracts, default=contracts)
    
    tenure_range = st.sidebar.slider("Tenure (months)", 
        int(ibm_maven['TENURE'].min()), int(ibm_maven['TENURE'].max()),
        (0, int(ibm_maven['TENURE'].max())))
    
    charge_range = st.sidebar.slider("Monthly Charges ($)",
        float(ibm_maven['MONTHLYCHARGES'].min()), float(ibm_maven['MONTHLYCHARGES'].max()),
        (float(ibm_maven['MONTHLYCHARGES'].min()), float(ibm_maven['MONTHLYCHARGES'].max())))
    
    filtered = ibm_maven[
        (ibm_maven['CONTRACT'].isin(selected_contracts)) &
        (ibm_maven['TENURE'] >= tenure_range[0]) &
        (ibm_maven['TENURE'] <= tenure_range[1]) &
        (ibm_maven['MONTHLYCHARGES'] >= charge_range[0]) &
        (ibm_maven['MONTHLYCHARGES'] <= charge_range[1])
    ]
    
    total = len(filtered)
    churned = (filtered['CHURN'] == 'Yes').sum()
    rate = churned / total * 100 if total > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Filtered Customers", f"{total:,}")
    col2.metric("Churned", f"{churned:,}")
    col3.metric("Churn Rate", f"{rate:.1f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn by Internet Service")
        if total > 0:
            internet_churn = filtered.groupby('INTERNETSERVICE')['CHURN_BINARY'].mean() * 100
            fig, ax = plt.subplots(figsize=(6, 4))
            internet_churn.plot(kind='bar', ax=ax, color='steelblue')
            ax.set_ylabel('Churn Rate (%)')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    with col2:
        st.subheader("Monthly Charges Distribution")
        if total > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            filtered[filtered['CHURN']=='No']['MONTHLYCHARGES'].hist(
                bins=25, alpha=0.5, label='Stayed', color='steelblue', ax=ax)
            filtered[filtered['CHURN']=='Yes']['MONTHLYCHARGES'].hist(
                bins=25, alpha=0.5, label='Churned', color='salmon', ax=ax)
            ax.set_xlabel('Monthly Charges ($)')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    st.subheader("Churn Reasons (filtered)")
    if churned > 0:
        reasons = filtered[filtered['CHURN']=='Yes']['CHURN_CATEGORY'].value_counts()
        fig, ax = plt.subplots(figsize=(10, 3))
        reasons.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_xlabel('Count')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("no churned customers in current filter")
    
    with st.expander("view raw data"):
        st.dataframe(filtered.head(100))


elif page == "Risk Predictor":
    st.title("Customer Churn Risk Predictor")
    st.markdown("input a customer profile to get their churn probability")
    
    if model_data is None:
        st.error("model not found. make sure churn_model.pkl is in data/processed/")
    else:
        model = model_data['model']
        le_dict = model_data['le_dict']
        feature_cols = model_data['feature_cols']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Customer Info")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
            total_charges = monthly_charges * tenure
            contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
            internet = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
            tech_support = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
            online_security = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
        
        with col2:
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ['Male', 'Female'])
            age = st.slider("Age", 18, 85, 40)
            married = st.selectbox("Married", ['Yes', 'No'])
            senior = st.selectbox("Senior Citizen", [0, 1])
            dependents = st.selectbox("Dependents", ['Yes', 'No'])
            payment = st.selectbox("Payment Method", 
                ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
            paperless = st.selectbox("Paperless Billing", ['Yes', 'No'])
        
        if st.button("predict churn risk", type="primary"):
            input_dict = {}
            for col in feature_cols:
                input_dict[col] = 0
            
            if 'TENURE' in input_dict: input_dict['TENURE'] = tenure
            if 'MONTHLYCHARGES' in input_dict: input_dict['MONTHLYCHARGES'] = monthly_charges
            if 'TOTALCHARGES' in input_dict: input_dict['TOTALCHARGES'] = total_charges
            if 'AGE' in input_dict: input_dict['AGE'] = age
            if 'SENIORCITIZEN' in input_dict: input_dict['SENIORCITIZEN'] = senior
            if 'NUMBER_OF_REFERRALS' in input_dict: input_dict['NUMBER_OF_REFERRALS'] = 0
            if 'AVG_MONTHLY_GB_DOWNLOAD' in input_dict: input_dict['AVG_MONTHLY_GB_DOWNLOAD'] = 20
            
            cat_mappings = {
                'GENDER': gender, 'CONTRACT': contract, 'INTERNETSERVICE': internet,
                'TECHSUPPORT': tech_support, 'ONLINESECURITY': online_security,
                'PAYMENTMETHOD': payment, 'PAPERLESSBILLING': paperless,
                'MARRIED': married, 'DEPENDENTS': dependents,
                'MULTIPLELINES': 'No', 'ONLINEBACKUP': 'No',
                'DEVICEPROTECTION': 'No', 'STREAMINGTV': 'No',
                'STREAMINGMOVIES': 'No', 'PHONESERVICE': 'Yes',
                'INTERNET_TYPE': 'Fiber Optic' if internet == 'Fiber optic' else 'DSL',
                'STREAMING_MUSIC': 'No', 'UNLIMITED_DATA': 'Yes',
                'OFFER': 'None',
            }
            
            for col_name, val in cat_mappings.items():
                if col_name in le_dict and col_name in input_dict:
                    le = le_dict[col_name]
                    try:
                        input_dict[col_name] = le.transform([str(val)])[0]
                    except:
                        input_dict[col_name] = 0
            
            X_input = pd.DataFrame([input_dict])[feature_cols]
            prob = model.predict_proba(X_input)[0][1]
            
            if prob < 0.3:
                tier = "🟢 Low Risk"
            elif prob < 0.6:
                tier = "🟡 Medium Risk"
            else:
                tier = "🔴 High Risk"
            
            st.markdown("---")
            st.subheader("Results")
            
            r1, r2, r3 = st.columns(3)
            r1.metric("Churn Probability", f"{prob*100:.1f}%")
            r2.metric("Risk Tier", tier)
            r3.metric("Monthly Revenue", f"${monthly_charges:.0f}")
            
            st.subheader("recommended actions")
            
            actions = []
            if contract == 'Month-to-month':
                actions.append("offer discount to switch to annual contract")
            if tenure < 6:
                actions.append("schedule onboarding check-in call within 30 days")
            if tech_support == 'No' and internet != 'No':
                actions.append("offer free tech support trial for 3 months")
            if payment == 'Electronic check':
                actions.append("incentivize switch to auto-pay ($5/mo discount)")
            if prob >= 0.6:
                actions.append("⚠️ escalate to retention team immediately")
            if prob >= 0.3 and prob < 0.6:
                actions.append("send targeted retention offer")
            if monthly_charges > 80:
                actions.append("review pricing - customer may be overpaying for usage")
            
            if not actions:
                actions.append("✅ no immediate action needed, customer looks stable")
            
            for a in actions:
                st.markdown(f"- {a}")


elif page == "Playbook":
    st.title("Churn Playbook")
    st.markdown("retention action plan based on analysis of ~10,000 customers")
    
    st.markdown("---")
    
    st.header("what the data told us")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **contract type is the biggest factor**  
        month to month customers churn at 48% vs 2.8% for two year contracts. 
        the single most important thing is getting people onto longer contracts.
        
        **first 6 months is the danger zone**  
        77% of customers who leave do it in the first 6 months. retention 
        efforts need to be front loaded in the early months.
        
        **competitors are the #1 reason**  
        45% of churners left for a competitor. only 11% cited price. 
        discounting alone wont fix this.
        """)
    
    with col2:
        st.markdown("""
        **customer service calls are a warning sign**  
        churn is normal at ~11% for 0-3 calls. at 4+ calls it jumps to 46%.
        the 4th call should trigger automatic escalation.
        
        **higher paying customers churn more**  
        customers in the $60-120/mo range churn at higher rates. 
        competitors target these high value accounts.
        
        **tenure is the strongest predictor**  
        correlation of -0.43 with churn. the longer someone stays,
        the less likely they are to leave.
        """)
    
    st.markdown("---")
    
    st.header("retention triggers")
    
    triggers = pd.DataFrame({
        'Trigger': [
            'New customer (0-6 months)',
            'Month-to-month + tenure < 12 months',
            '3+ support calls in 30 days',
            'Churn probability > 60%',
            'High value ($80+/mo) on month-to-month',
            'No tech support + has internet',
            'Electronic check payment',
        ],
        'Action': [
            '30 and 90 day check-in calls',
            'Offer 15% off to switch to annual',
            'Auto-escalate to retention specialist',
            'Immediate outreach from retention team',
            'Assign dedicated account manager',
            'Offer 3 month free tech support',
            'Offer $5/mo discount for auto-pay',
        ],
        'Priority': [
            'HIGH', 'HIGH', 'HIGH', 'CRITICAL',
            'MEDIUM', 'MEDIUM', 'LOW'
        ]
    })
    
    st.dataframe(triggers, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    st.header("revenue impact")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("5% Churn Reduction", "~$293K/year saved")
    col2.metric("10% Churn Reduction", "~$586K/year saved")
    col3.metric("15% Churn Reduction", "~$879K/year saved")
    
    st.markdown("""
    the high risk group alone is 1,149 customers paying an average of $74/month. 
    thats about $85K in monthly revenue that is likely to walk out the door.
    the cost of most retention offers is tiny compared to acquiring a replacement customer.
    """)
    
    st.markdown("---")
    
    st.header("risk tier breakdown")
    
    tiers = pd.DataFrame({
        'Tier': ['Low Risk', 'Medium Risk', 'High Risk'],
        'Churn Prob': ['0-30%', '30-60%', '60-100%'],
        'Customers': ['4,215', '1,224', '1,149'],
        'Actual Churn': ['2.6%', '54%', '95%'],
        'Avg Tenure': ['43 months', '29 months', '8 months'],
        'Avg Monthly': ['$60', '$81', '$74'],
        'Action': ['Monitor', 'Proactive offer', 'Immediate intervention']
    })
    
    st.dataframe(tiers, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    st.header("model vs reality")
    st.markdown("""
    the model says tenure and contract type are the biggest predictors. 
    but customers themselves say 45% left for a competitor and 17% were 
    unhappy with service. only 11% mentioned price.
    
    both are true at the same time. short tenure month-to-month customers 
    are the easiest targets for competitors. the retention strategy should 
    focus on making the product good enough that competitors cant steal 
    customers AND locking people into contracts early.
    """)
