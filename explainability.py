import joblib
import pandas as pd
import numpy as np
import shap
import dice_ml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

# Load models and scalers
model = joblib.load("Loan_model.joblib")
features = joblib.load("Loan_features.joblib")
ss = joblib.load("standard_scaler.joblib")
mms = joblib.load("minmax_scaler.joblib")

class LoanPredictorWrapper:
    def __init__(self, model, ss, mms, features):
        self.model = model
        self.ss = ss
        self.mms = mms
        self.features = features
        self.classes_ = np.array([0, 1])

    def predict(self, X):
        X_proc = self._preprocess(X)
        return self.model.predict(X_proc)

    def predict_proba(self, X):
        X_proc = self._preprocess(X)
        return self.model.predict_proba(X_proc)

    def _preprocess(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.features)
        
        df = X.copy()
        
        home_ownership_mapping = {
            "RENT": 0, "OWN": 1, "MORTGAGE": 2, "OTHER": 3,
            0: 0, 1: 1, 2: 2, 3: 3,
            "0": 0, "1": 1, "2": 2, "3": 3
        }
        previous_loan_defaults_mapping = {
            "NO": 0, "YES": 1,
            "No": 0, "Yes": 1,
            "no": 0, "yes": 1,
            0: 0, 1: 1,
            "0": 0, "1": 1
        }
        
        df['person_home_ownership'] = df['person_home_ownership'].map(home_ownership_mapping).fillna(0).astype(int)
        df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map(previous_loan_defaults_mapping).fillna(0).astype(int)
        
        df['person_income'] = pd.to_numeric(df['person_income'], errors='coerce').fillna(0.0)
        df['loan_percent_income'] = pd.to_numeric(df['loan_percent_income'], errors='coerce').fillna(0.0)
        df['loan_int_rate'] = pd.to_numeric(df['loan_int_rate'], errors='coerce').fillna(0.0)
        
        df[['person_income', 'loan_percent_income']] = self.ss.transform(df[['person_income', 'loan_percent_income']])
        df[['loan_int_rate']] = self.mms.transform(df[['loan_int_rate']])
        
        return df[self.features]

# Initialize wrapper
wrapper = LoanPredictorWrapper(model, ss, mms, features)

# Load background dataset
df_all = pd.read_csv('loan_data__2_.csv')
df_background = pd.DataFrame()
df_background['person_income'] = df_all['person_income']
df_background['person_home_ownership'] = df_all['person_home_ownership']
df_background['loan_int_rate'] = df_all['loan_int_rate']
df_background['loan_percent_income'] = df_all['loan_percent_income']
df_background['previous_loan_defaults_on_file'] = df_all['previous_loan_defaults_on_file'].map({'No': 'NO', 'Yes': 'YES'}).fillna('NO')
df_background['loan_status'] = df_all['loan_status']

# Subsample background data to keep explanation generation snappy
df_background_sample = df_background.sample(100, random_state=42).reset_index(drop=True)
X_bg = df_background_sample[features]

# 1. DiCE Setup
d = dice_ml.Data(
    dataframe=df_background_sample,
    continuous_features=['person_income', 'loan_int_rate', 'loan_percent_income'],
    outcome_name='loan_status'
)
m = dice_ml.Model(model=wrapper, backend="sklearn")
dice_explainer = dice_ml.Dice(d, m, method="random")

# 2. SHAP Setup
def predict_approved_proba(X):
    return wrapper.predict_proba(X)[:, 1]

X_bg_small = X_bg.sample(30, random_state=42).reset_index(drop=True)
shap_explainer = shap.KernelExplainer(predict_approved_proba, X_bg_small)

def get_shap_chart(input_df):
    """
    Generates SHAP impact chart for a single row and returns it as a base64 PNG string.
    """
    try:
        shap_values = shap_explainer.shap_values(input_df)
        val = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        
        # Format labels nicely
        raw_vals = input_df.iloc[0]
        labels = []
        for f in features:
            val_str = raw_vals[f]
            if f == 'person_income':
                labels.append(f"Income: ₹{int(val_str):,}")
            elif f == 'loan_int_rate':
                labels.append(f"Interest Rate: {val_str}%")
            elif f == 'loan_percent_income':
                labels.append(f"Loan/Income: {val_str}")
            elif f == 'person_home_ownership':
                labels.append(f"Home: {val_str}")
            else:
                labels.append(f"Defaults: {val_str}")
                
        y_pos = np.arange(len(features))
        
        # Stylish Plot
        fig, ax = plt.subplots(figsize=(7, 3.5), facecolor='#0b1120')
        ax.set_facecolor('#0b1120')
        
        colors = ['#10B981' if v >= 0 else '#EF4444' for v in val]
        bars = ax.barh(y_pos, val, align='center', color=colors, edgecolor='none', height=0.6)
        
        # Add labels on bars
        for bar, v in zip(bars, val):
            width = bar.get_width()
            x_pos = width + 0.005 if v >= 0 else width - 0.005
            ha = 'left' if v >= 0 else 'right'
            ax.text(x_pos, bar.get_y() + bar.get_height()/2, f"{v:+.3f}", 
                    va='center', ha=ha, color='#F8FAFC', fontname='sans-serif', fontsize=9, fontweight='semibold')
                    
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, color='#E2E8F0', fontname='sans-serif', fontsize=10, fontweight='medium')
        
        # Clean axes style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#334155')
        ax.spines['bottom'].set_color('#334155')
        ax.tick_params(axis='x', colors='#94A3B8')
        ax.xaxis.grid(True, linestyle='--', alpha=0.1, color='#F8FAFC')
        
        plt.title("Feature Impact on Approval Probability (SHAP)", color='#F8FAFC', fontsize=11, fontweight='bold', pad=15)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return img_str
    except Exception as e:
        print("SHAP Chart Error:", e)
        return ""

def get_counterfactuals(input_df, is_approved):
    """
    Generates 3 counterfactual scenarios showing how to change the model output.
    """
    desired_class = 0 if is_approved else 1
    try:
        cf = dice_explainer.generate_counterfactuals(
            input_df,
            total_CFs=3,
            desired_class=desired_class,
            features_to_vary=['person_income', 'loan_int_rate', 'loan_percent_income']
        )
        cf_df = cf.cf_examples_list[0].final_cfs_df
        if cf_df is None or cf_df.shape[0] == 0:
            return []
            
        results = []
        for _, row in cf_df.iterrows():
            if int(row['loan_status']) != desired_class:
                continue
            results.append({
                "person_income": float(row['person_income']),
                "person_home_ownership": str(row['person_home_ownership']),
                "loan_int_rate": float(row['loan_int_rate']),
                "loan_percent_income": float(row['loan_percent_income']),
                "previous_loan_defaults_on_file": str(row['previous_loan_defaults_on_file'])
            })
            if len(results) >= 3:
                break
        return results
    except Exception as e:
        print("DiCE Error:", e)
        return []
