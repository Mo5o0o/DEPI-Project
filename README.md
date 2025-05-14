
# Sales Forecasting and Optimization

This project aims to forecast future sales using historical data to optimize inventory and marketing strategies.

## 📊 Objectives
- Perform EDA to identify patterns and seasonality.
- Build and evaluate time-series forecasting model (XGBoost, Decision Tree, Random Forest, LR).
- Deploy the best-performing model using Streamlit.

## 🧱 Project Structure
```
Superstore-Sales-Analysis-main/
├── data/
│   ├── 1_Superstore Dataset                # Raw dataset file
│   └── 2_cleaned_superstore.csv            # Cleaned & transformed dataset
│
├── notebooks/
│   ├── 1_preprocessing.ipynb 
│   └── 2_modeling.ipynb
│ 
├── models/                                # Trained model files 
│   ├── 1_decision_tree.pkl
│   ├── 2_linear_regression_model.pkl
│   └── 3_xgboost_model.pkl             
│
├── app/
│   └── streamlit_app.py                   # Streamlit
│   └── dashboard.py                       # Dashboard
├── requirements.txt
│
└── Report/
    ├── Final_Report.pdf
    └── Final_Presentation.pptx
```

## 🧪 Installation
```bash
pip install -r requirements.txt
```

## ▶️ Running the App
```bash
streamlit run streamlit_app.py
```

### Application Description:
**Intelligent Sales Forecasting System** uses machine learning techniques to predict future sales, helping businesses forecast demand and make better decisions for inventory and marketing strategies.

<p>You can view and interact with the live application <a href="https://depiapp-vpwfjn4zcim93ut6jjpadg.streamlit.app/" target="_blank">here</a>.</p>


## 👥 Team Members
- Ahmed Mohammed Elsayed
- Ahmed Mohamed Bedair Elbhbaty
- Adel Tamer Adel Badran
- Ezz El-Deen Ashraf Mohammed
