import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split

NUMERIC_FEATURES = [
    'annual_income','loan_amount','loan_tenure_years','credit_history_years',
    'existing_emi','credit_utilization','delinquencies_12m','age'
]
CATEGORICAL_FEATURES = ['employment_type','home_ownership','loan_purpose']

FEATURE_LABELS = {
    'annual_income':'Annual income','loan_amount':'Loan amount','loan_tenure_years':'Loan tenure',
    'credit_history_years':'Credit history','existing_emi':'Existing EMI',
    'credit_utilization':'Credit utilization','delinquencies_12m':'Recent delinquencies',
    'age':'Age','employment_type':'Employment type','home_ownership':'Home ownership',
    'loan_purpose':'Loan purpose'
}

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def generate_credit_data(n=7000, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 66, n)
    annual_income = np.clip(rng.lognormal(np.log(850000), 0.55, n), 180000, 8000000)
    loan_amount = np.clip(rng.lognormal(np.log(550000), 0.75, n), 50000, 5000000)
    loan_tenure_years = rng.choice([1,2,3,4,5,7,10], size=n, p=[.08,.10,.22,.10,.30,.08,.12])
    credit_history_years = np.clip((age-18) * rng.uniform(.15,.85,n), .2, 30)
    existing_emi = np.clip(annual_income/12 * rng.beta(1.8,6.5,n), 0, 220000)
    credit_utilization = np.clip(rng.beta(2.1,3.4,n), .01, .99)
    delinquencies_12m = np.clip(rng.poisson(.28,n), 0, 6)
    employment_type = rng.choice(['Salaried','Self-employed','Contract'], size=n, p=[.68,.24,.08])
    home_ownership = rng.choice(['Owned','Mortgage','Rented'], size=n, p=[.35,.32,.33])
    loan_purpose = rng.choice(['Home','Auto','Personal','Business','Education'], size=n, p=[.22,.20,.28,.18,.12])

    monthly_income = annual_income / 12
    emi_ratio = existing_emi / np.maximum(monthly_income,1)
    loan_to_income = loan_amount / np.maximum(annual_income,1)

    logit = (
        -4.25 + 2.5*credit_utilization + 1.45*emi_ratio + 0.60*loan_to_income
        + 0.52*delinquencies_12m - 0.055*credit_history_years - 0.00000018*annual_income
        + 0.30*(employment_type=='Self-employed') + 0.58*(employment_type=='Contract')
        + 0.24*(home_ownership=='Rented') + 0.20*(loan_purpose=='Personal')
        + 0.16*(loan_purpose=='Business')
    )
    pd_true = sigmoid(logit)
    default = rng.binomial(1, pd_true)

    return pd.DataFrame({
        'annual_income':annual_income,'loan_amount':loan_amount,'loan_tenure_years':loan_tenure_years,
        'credit_history_years':credit_history_years,'existing_emi':existing_emi,
        'credit_utilization':credit_utilization,'delinquencies_12m':delinquencies_12m,'age':age,
        'employment_type':employment_type,'home_ownership':home_ownership,'loan_purpose':loan_purpose,
        'default':default
    })

def ks_statistic(y_true, scores):
    df = pd.DataFrame({'y':y_true,'score':scores}).sort_values('score', ascending=False)
    bad_total = max(df['y'].sum(), 1)
    good_total = max((1-df['y']).sum(), 1)
    df['cum_bad'] = df['y'].cumsum()/bad_total
    df['cum_good'] = (1-df['y']).cumsum()/good_total
    return float((df['cum_bad']-df['cum_good']).abs().max())

def calibration_table(y_true, scores, bins=10):
    df = pd.DataFrame({'actual':y_true,'pd':scores})
    df['bucket'] = pd.qcut(df['pd'], q=bins, duplicates='drop')
    out = df.groupby('bucket', observed=True).agg(
        borrowers=('actual','size'), predicted_pd=('pd','mean'), actual_default_rate=('actual','mean')
    ).reset_index(drop=True)
    return [
        {'borrowers':int(r.borrowers),'predicted_pd':float(r.predicted_pd),'actual_default_rate':float(r.actual_default_rate)}
        for r in out.itertuples()
    ]

def risk_grade(pd_value):
    if pd_value < .02: return 'A','Low'
    if pd_value < .05: return 'B','Moderate-Low'
    if pd_value < .10: return 'C','Moderate'
    if pd_value < .20: return 'D','High'
    return 'E','Very High'

class CreditRiskEngine:
    def __init__(self, seed=42):
        self.seed = seed
        self.data = generate_credit_data(seed=seed)
        self._fit()

    def _fit(self):
        X = self.data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        y = self.data['default']
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.30,random_state=self.seed,stratify=y)

        prep = ColumnTransformer([
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
        ])
        self.model = Pipeline([('prep',prep),('model',LogisticRegression(max_iter=1200,class_weight='balanced'))])
        self.model.fit(X_train,y_train)
        scores = self.model.predict_proba(X_test)[:,1]
        pred = (scores >= .50).astype(int)

        tree_X = pd.get_dummies(X, columns=CATEGORICAL_FEATURES, drop_first=False)
        trX, teX, trY, teY = train_test_split(tree_X,y,test_size=.30,random_state=self.seed,stratify=y)
        self.tree_columns = list(tree_X.columns)
        self.tree_model = GradientBoostingClassifier(random_state=self.seed)
        self.tree_model.fit(trX,trY)
        tree_scores = self.tree_model.predict_proba(teX)[:,1]

        tn,fp,fn,tp = confusion_matrix(y_test,pred).ravel()
        self.metrics = {
            'logistic_auc':float(roc_auc_score(y_test,scores)),
            'tree_auc':float(roc_auc_score(teY,tree_scores)),
            'ks':ks_statistic(y_test.to_numpy(),scores),
            'recall':float(recall_score(y_test,pred,zero_division=0)),
            'confusion_matrix':{'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)},
            'calibration':calibration_table(y_test.to_numpy(),scores),
            'test_size':int(len(y_test)),
            'default_rate':float(y.mean())
        }

    def _row(self, borrower):
        return pd.DataFrame([{k: borrower[k] for k in NUMERIC_FEATURES + CATEGORICAL_FEATURES}])

    def explain_logistic(self, row):
        prep = self.model.named_steps['prep']
        model = self.model.named_steps['model']
        transformed = prep.transform(row)
        if hasattr(transformed,'toarray'):
            transformed = transformed.toarray()
        names = prep.get_feature_names_out()
        contribs = transformed[0] * model.coef_[0]
        items = []
        for name,value in zip(names,contribs):
            clean = name.replace('num__','').replace('cat__','')
            label = clean
            for key,friendly in FEATURE_LABELS.items():
                if clean == key:
                    label = friendly
                    break
                if clean.startswith(key+'_'):
                    label = f"{friendly}: {clean[len(key)+1:]}"
                    break
            items.append({'feature':label,'impact':float(value),'direction':'increases risk' if value>0 else 'reduces risk'})
        items.sort(key=lambda x: abs(x['impact']), reverse=True)
        return items[:6]

    def assess(self, borrower, lgd, ead):
        row = self._row(borrower)
        pd_logistic = float(self.model.predict_proba(row)[0,1])
        tree_row = pd.get_dummies(row, columns=CATEGORICAL_FEATURES, drop_first=False).reindex(columns=self.tree_columns, fill_value=0)
        pd_tree = float(self.tree_model.predict_proba(tree_row)[0,1])
        grade,label = risk_grade(pd_logistic)
        expected_loss = pd_logistic * lgd * ead
        monthly_income = borrower['annual_income']/12
        return {
            'pd':pd_logistic,'benchmark_pd':pd_tree,'risk_grade':grade,'risk_label':label,
            'lgd':lgd,'ead':ead,'expected_loss':expected_loss,
            'credit_score':int(round(max(300,min(900,850-pd_logistic*500)))),
            'derived':{
                'emi_to_income':borrower['existing_emi']/max(monthly_income,1),
                'loan_to_income':borrower['loan_amount']/max(borrower['annual_income'],1)
            },
            'drivers':self.explain_logistic(row),
            'model_metrics':self.metrics
        }

    def portfolio_summary(self, n=500, seed=101):
        rng = np.random.default_rng(seed)
        sample = self.data.sample(n=n, random_state=seed).copy()
        X = sample[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        sample['pd'] = self.model.predict_proba(X)[:,1]
        sample['lgd'] = np.clip(rng.normal(.43,.09,n), .20, .75)
        sample['ead'] = np.clip(sample['loan_amount'] * rng.uniform(.75,1.05,n), 25000, 5000000)
        sample['expected_loss'] = sample['pd'] * sample['lgd'] * sample['ead']
        sample['grade'] = sample['pd'].map(lambda x: risk_grade(x)[0])
        grp = sample.groupby('grade').agg(
            borrowers=('pd','size'), exposure=('ead','sum'), expected_loss=('expected_loss','sum'), avg_pd=('pd','mean')
        ).reindex(['A','B','C','D','E']).dropna().reset_index()
        return {
            'borrowers':int(len(sample)),
            'total_exposure':float(sample['ead'].sum()),
            'portfolio_expected_loss':float(sample['expected_loss'].sum()),
            'weighted_pd':float(np.average(sample['pd'],weights=sample['ead'])),
            'grade_summary':[
                {'grade':str(r.grade),'borrowers':int(r.borrowers),'exposure':float(r.exposure),'expected_loss':float(r.expected_loss),'avg_pd':float(r.avg_pd)}
                for r in grp.itertuples()
            ]
        }

credit_engine = CreditRiskEngine()
