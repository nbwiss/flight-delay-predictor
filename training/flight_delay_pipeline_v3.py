# %% [markdown]
# # Flight Delay Prediction Pipeline
# **3 Models:** Cancellation (binary), Delay Y/N (binary), Delay Minutes (regression)
# **3 Algorithms each:** XGBoost (GPU), Random Forest, Logistic/Ridge baseline
# **Tuning:** RandomizedSearchCV (n_iter=10, 5-fold) on 500K subsample
# **Evaluation:** 10-fold stratified CV on full training data + held-out test set

# %% Mount Google Drive & Install
from google.colab import drive
drive.mount('/content/drive')

!pip install xgboost shap tqdm category_encoders -q

# %% Imports
import pandas as pd
import numpy as np
import time, os, gc, subprocess, warnings
warnings.filterwarnings('ignore')

from datetime import date, timedelta
from scipy import sparse

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, KFold,
    RandomizedSearchCV, cross_validate
)
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
import category_encoders as ce
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    precision_score, recall_score, accuracy_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from xgboost import XGBClassifier, XGBRegressor
import joblib
from tqdm.auto import tqdm

# GPU check
try:
    r = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    GPU_OK = r.returncode == 0
except:
    GPU_OK = False
XGB_DEVICE = 'cuda' if GPU_OK else 'cpu'
print(f"XGBoost device: {XGB_DEVICE}")

# %% Load Data
t0 = time.time()
DATA_PATH = '/content/drive/MyDrive/flight_delay_dataset_2024.parquet'
df = pd.read_parquet(DATA_PATH)
print(f"Loaded {df.shape[0]:,} x {df.shape[1]} in {time.time()-t0:.1f}s")

# Quick stats
print(f"Cancellation rate: {df['target_cancelled'].mean():.4f}")
nc = df[df['target_cancelled'] == 0]
print(f"Delay rate (non-cancelled): {nc['target_delayed'].mean():.4f}")
print(f"Mean delay (delayed only): {nc.loc[nc['target_delayed']==1, 'target_delay_minutes'].mean():.1f} min")
del nc

# %% Fix potential column name artifact
rename = {}
for c in df.columns:
    if c.endswith('visibilityl'):
        rename[c] = c.replace('visibilityl', 'visibility')
if rename:
    df.rename(columns=rename, inplace=True)
    print(f"Renamed: {rename}")

# %% Add flight ID
df['flight_id'] = np.arange(1, len(df) + 1)

# %% Feature Engineering
print("Engineering features...")

# --- Time features ---
df['dep_hour'] = (df['CRSDepTime'] // 100).clip(0, 23).astype(np.int8)
df['dep_minute'] = (df['CRSDepTime'] % 100).clip(0, 59).astype(np.int8)
df['arr_hour'] = (df['CRSArrTime'] // 100).clip(0, 23).astype(np.int8)
df['is_weekend'] = df['DayOfWeek'].isin([6, 7]).astype(np.int8)

# --- Holidays (±1 day buffer) ---
holidays_2024 = [
    date(2024,1,1), date(2024,1,15), date(2024,2,19), date(2024,3,31),
    date(2024,5,27), date(2024,6,19), date(2024,7,4), date(2024,9,2),
    date(2024,10,14), date(2024,11,11), date(2024,11,28), date(2024,11,29),
    date(2024,12,24), date(2024,12,25), date(2024,12,31),
]
holiday_window = set()
for h in holidays_2024:
    for d in range(-1, 2):
        holiday_window.add(h + timedelta(days=d))

df['_fdate'] = pd.to_datetime(df['FlightDate'])
df['is_holiday'] = df['_fdate'].dt.date.isin(holiday_window).astype(np.int8)

# --- Time block (categorical) ---
df['time_block'] = pd.cut(
    df['dep_hour'], bins=[-1, 5, 11, 17, 20, 24],
    labels=['redeye', 'morning', 'afternoon', 'evening', 'night']
).astype(str)

# --- Weather derived ---
for pfx in ['origin_wx_', 'dest_wx_']:
    df[f'{pfx}temp_dew_spread'] = df[f'{pfx}temperature_2m'] - df[f'{pfx}dew_point_2m']
    df[f'{pfx}adverse_wx'] = (df[f'{pfx}weather_code'] >= 50).astype(np.int8)
    df[f'{pfx}tstorm'] = (df[f'{pfx}weather_code'] >= 95).astype(np.int8)
    df[f'{pfx}total_precip'] = df[f'{pfx}rain'] + df[f'{pfx}snowfall']
    df[f'{pfx}low_ceiling'] = (df[f'{pfx}cloud_cover_low'] > 85).astype(np.int8)
    df[f'{pfx}low_vis'] = (df[f'{pfx}visibility'] < 15840).astype(np.int8)  # <3 miles in feet

print(f"Post-engineering shape: {df.shape}")

# %% Extract Targets & Drop Columns
y_canc = df['target_cancelled'].fillna(0).values.astype(np.int8)
y_del  = df['target_delayed'].fillna(0).values.astype(np.int8)
y_mins = df['target_delay_minutes'].fillna(0).values.astype(np.float32)

# Clip negative delay minutes to 0 (early arrivals are considered 0 minutes delayed)
y_mins = np.clip(y_mins, 0, None)

drop_cols = [
    # --- LEAKAGE: unknown at prediction time ---
    'DepTime', 'ArrTime', 'DepDelay', 'DepDel15', 'ArrDelay', 'ArrDel15',
    'ActualElapsedTime', 'AirTime',
    'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay',
    # --- REDUNDANT / ID ---
    'Year', 'DOT_ID_Reporting_Airline', 'IATA_CODE_Reporting_Airline',
    'OriginCityName', 'DestCityName', 'OriginState', 'DestState',
    'CancellationCode', 'Diverted', 'Cancelled',
    'Flight_Number_Reporting_Airline', 'DistanceGroup',
    # --- TRANSFORMED / TEMP ---
    'FlightDate', '_fdate', 'CRSDepTime', 'CRSArrTime', 'flight_id',
    # --- TARGETS ---
    'target_cancelled', 'target_delayed', 'target_delay_minutes',
]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# Identify column types
CAT_COLS = ['Origin', 'Dest', 'Reporting_Airline', 'time_block']
NUM_COLS = sorted([c for c in df.columns if c not in CAT_COLS])

print(f"Final: {len(NUM_COLS)} numeric + {len(CAT_COLS)} categorical features")
print(f"Nulls: {df.isnull().sum().sum():,}")

# %% Train / Test Split (80/20, stratified on cancellation)
# Split raw dataframe indices first, THEN fit scaler/encoder on train only (no leakage)
idx = np.arange(len(df))
idx_tr, idx_te = train_test_split(idx, test_size=0.2, stratify=y_canc, random_state=42)

y_canc_tr, y_canc_te = y_canc[idx_tr], y_canc[idx_te]
y_del_tr, y_del_te   = y_del[idx_tr],  y_del[idx_te]
y_mins_tr, y_mins_te = y_mins[idx_tr], y_mins[idx_te]

df_tr = df.iloc[idx_tr].reset_index(drop=True)
df_te = df.iloc[idx_te].reset_index(drop=True)
del df, idx
gc.collect()

# %% Build Tree Feature Matrices (TargetEncoder)
print("Building tree feature matrices (no leakage)...")
t0 = time.time()

# Impute nulls — medians from TRAIN
medians = df_tr[NUM_COLS].median()
df_tr[NUM_COLS] = df_tr[NUM_COLS].fillna(medians)
df_te[NUM_COLS] = df_te[NUM_COLS].fillna(medians)

# Scale numerics — fit on TRAIN
scaler = StandardScaler()
num_tr = scaler.fit_transform(df_tr[NUM_COLS].values)
num_te = scaler.transform(df_te[NUM_COLS].values)

# Non-cancelled masks
nc_tr = y_canc_tr == 0
nc_te = y_canc_te == 0

# --- MODEL A: Cancellation (Target = y_canc_tr) ---
te_a = ce.TargetEncoder(cols=CAT_COLS)
cat_tr_a = te_a.fit_transform(df_tr[CAT_COLS], y_canc_tr)
cat_te_a = te_a.transform(df_te[CAT_COLS])

X_train_a = np.hstack([num_tr, cat_tr_a]).astype(np.float32)
X_test_a  = np.hstack([num_te, cat_te_a]).astype(np.float32)

# --- MODEL B: Delay Y/N (Target = y_del_tr) ---
te_b = ce.TargetEncoder(cols=CAT_COLS)
cat_tr_b = te_b.fit_transform(df_tr.loc[nc_tr, CAT_COLS], y_del_tr[nc_tr])
cat_te_b = te_b.transform(df_te.loc[nc_te, CAT_COLS])

X_tr_b = np.hstack([num_tr[nc_tr], cat_tr_b]).astype(np.float32)
X_te_b = np.hstack([num_te[nc_te], cat_te_b]).astype(np.float32)
y_tr_b, y_te_b = y_del_tr[nc_tr], y_del_te[nc_te]

# --- MODEL C: Delay Minutes (Target = log1p(y_mins_tr) equivalent, but we just use y_mins_tr for the encoder) ---
te_c = ce.TargetEncoder(cols=CAT_COLS)
# Fitting the encoder on the log of minutes usually correlates better with the target
cat_tr_c = te_c.fit_transform(df_tr.loc[nc_tr, CAT_COLS], np.log1p(y_mins_tr[nc_tr]))
cat_te_c = te_c.transform(df_te.loc[nc_te, CAT_COLS])

X_tr_c = np.hstack([num_tr[nc_tr], cat_tr_c]).astype(np.float32)
X_te_c = np.hstack([num_te[nc_te], cat_te_c]).astype(np.float32)
y_tr_c, y_te_c = y_mins_tr[nc_tr], y_mins_te[nc_te]

FEATURE_NAMES = NUM_COLS + CAT_COLS
print(f"X_train_a: {X_train_a.shape} built in {time.time()-t0:.1f}s")

del df_tr, df_te, num_tr, num_te
gc.collect()

spw_canc  = (y_canc_tr == 0).sum() / max((y_canc_tr == 1).sum(), 1)
spw_delay = (y_tr_b == 0).sum() / max((y_tr_b == 1).sum(), 1)

print(f"Model A  train: {X_train_a.shape[0]:,}  test: {X_test_a.shape[0]:,}  pos: {y_canc_tr.mean():.4f}")
print(f"Model B  train: {X_tr_b.shape[0]:,}  test: {X_te_b.shape[0]:,}  pos: {y_tr_b.mean():.4f}")
print(f"Model C  train: {X_tr_c.shape[0]:,}  test: {X_te_c.shape[0]:,}  mean: {y_tr_c.mean():.1f} min")
print(f"scale_pos_weight — cancel: {spw_canc:.1f}  delay: {spw_delay:.1f}")

# %% HELPER FUNCTIONS

SAVE_DIR = '/content/drive/MyDrive/flight_models_v3'
os.makedirs(SAVE_DIR, exist_ok=True)

TUNE_SIZE = 500_000   # subsample for hyperparameter search
TUNE_CV   = 5         # CV folds during tuning
EVAL_CV   = 10        # CV folds for final evaluation
N_ITER    = 10        # random hyperparameter combos

def subsample(X, y, size=TUNE_SIZE, stratify=True, rs=42):
    """Stratified or random subsample."""
    if len(y) <= size:
        return X, y
    if stratify:
        _, Xs, _, ys = train_test_split(X, y, test_size=size, stratify=y, random_state=rs)
    else:
        idx = np.random.RandomState(rs).choice(len(y), size, replace=False)
        Xs, ys = X[idx], y[idx]
    return Xs, ys


def tune(model, params, X, y, scoring, gpu=False):
    """RandomizedSearchCV. Returns best_params, best_score."""
    search = RandomizedSearchCV(
        model, params, n_iter=N_ITER, cv=TUNE_CV,
        scoring=scoring, random_state=42,
        n_jobs=1 if gpu else -1, verbose=1
    )
    search.fit(X, y)
    return search.best_params_, search.best_score_


def cv_classify(model, X, y):
    """10-fold stratified CV for classifier."""
    skf = StratifiedKFold(n_splits=EVAL_CV, shuffle=True, random_state=42)
    sc = {'f1':'f1', 'auc':'roc_auc', 'prec':'precision',
          'rec':'recall', 'acc':'accuracy'}
    res = cross_validate(model, X, y, cv=skf, scoring=sc, n_jobs=1, verbose=1)
    return {k.replace('test_',''): v for k, v in res.items() if k.startswith('test_')}


def cv_regress(model, X, y):
    """10-fold CV for regressor, stratified on delay > 0."""
    strat = (y > 0).astype(int)
    skf = StratifiedKFold(n_splits=EVAL_CV, shuffle=True, random_state=42)
    splits = list(skf.split(X, strat))
    sc = {'mae':'neg_mean_absolute_error', 'rmse':'neg_root_mean_squared_error', 'r2':'r2'}
    res = cross_validate(model, X, y, cv=splits, scoring=sc, n_jobs=1, verbose=1)
    return {'mae': -res['test_mae'], 'rmse': -res['test_rmse'], 'r2': res['test_r2']}


def eval_clf(model, X, y, name):
    """Test set evaluation for classifier."""
    yp = model.predict(X)
    yprob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, yprob)
    print(f"\n--- {name} TEST ---")
    print(classification_report(y, yp, digits=4))
    print(f"AUC-ROC: {auc:.4f}")
    return {'f1': f1_score(y, yp), 'auc': auc}


def eval_reg(model, X, y, name):
    """Test set evaluation for regressor."""
    yp = model.predict(X)
    mae = mean_absolute_error(y, yp)
    rmse = np.sqrt(mean_squared_error(y, yp))
    r2 = r2_score(y, yp)
    n = len(y)
    p = X.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    print(f"\n--- {name} TEST ---")
    print(f"MAE: {mae:.2f} min  |  RMSE: {rmse:.2f} min  |  R2: {r2:.4f}  |  Adj R2: {adj_r2:.4f}")
    delayed = y > 0
    if delayed.sum() > 0:
        print(f"MAE delayed:  {mean_absolute_error(y[delayed], yp[delayed]):.2f} min")
        print(f"MAE on-time:  {mean_absolute_error(y[~delayed], yp[~delayed]):.2f} min")
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'adj_r2': adj_r2}


def pp_cv(cv, name):
    """Pretty-print CV results."""
    print(f"\n{'='*50}")
    print(f"  {name} — 10-Fold CV Summary")
    print(f"{'='*50}")
    for k, v in cv.items():
        print(f"  {k:>6s}: {v.mean():.4f} +/- {v.std():.4f}   (min {v.min():.4f}, max {v.max():.4f})")


def save_checkpoint(model_name, algo, result):
    """Save a single model + results to Drive immediately after training."""
    tag = f"{model_name}_{algo.lower().replace(' ','_')}"
    # Save model
    joblib.dump(result['model'], os.path.join(SAVE_DIR, f'{tag}_model.joblib'))
    # Save result summary (without the model object, for lightweight reload)
    summary = {
        'params': result['params'],
        'test': result['test'],
        'time': result['time'],
        'cv_means': {k: float(v.mean()) for k, v in result['cv'].items()},
        'cv_stds':  {k: float(v.std())  for k, v in result['cv'].items()},
    }
    joblib.dump(summary, os.path.join(SAVE_DIR, f'{tag}_summary.joblib'))
    print(f"  ✓ Checkpoint saved: {tag}")


def run_all(name, configs, Xtr, ytr, Xte, yte, task='clf'):
    """
    Train all algorithms for one model with progress bar and checkpointing.
    configs: {algo: {'model':..., 'params':..., 'gpu':bool}}
    Returns {algo: {'model':..., 'params':..., 'cv':..., 'test':..., 'time':...}}
    """
    results = {}
    pbar = tqdm(configs.items(), desc=name, unit='algo')
    for algo, cfg in pbar:
        pbar.set_postfix_str(algo)
        
        tag = f"{name}_{algo.lower().replace(' ','_')}"
        model_path = os.path.join(SAVE_DIR, f'{tag}_model.joblib')
        summary_path = os.path.join(SAVE_DIR, f'{tag}_summary.joblib')
        
        if os.path.exists(model_path) and os.path.exists(summary_path):
            print(f"\n{'#'*60}")
            print(f"  {name} > {algo}  [LOADED FROM CHECKPOINT]")
            print(f"{'#'*60}")
            try:
                m = joblib.load(model_path)
                summary = joblib.load(summary_path)
                
                class MockCVMetric:
                    def __init__(self, mean_val, std_val): 
                        self._m = mean_val
                        self._s = std_val
                    def mean(self): return self._m
                    def std(self): return self._s
                
                if 'cv_means' in summary:
                    cv = {k: MockCVMetric(v, summary['cv_stds'].get(k, 0)) for k, v in summary['cv_means'].items()}
                else:
                    cv = summary['cv']
                    
                results[algo] = {
                    'model': m, 'params': summary['params'], 
                    'cv': cv, 'test': summary['test'], 'time': summary['time']
                }
                continue
            except Exception as e:
                print(f"Failed to load checkpoint for {algo}: {e}. Retraining...")

        print(f"\n{'#'*60}")
        print(f"  {name} > {algo}")
        print(f"{'#'*60}")
        t0 = time.time()

        is_clf = task == 'clf'
        scoring_t = 'f1' if is_clf else 'neg_mean_absolute_error'

        # Subsample for tuning
        Xt, yt = subsample(Xtr, ytr, stratify=is_clf)
        print(f"Tuning on {len(yt):,} samples...")

        bp, bs = tune(cfg['model'], cfg['params'], Xt, yt, scoring_t, cfg.get('gpu', False))
        print(f"Best params: {bp}")
        print(f"Best tune score: {bs:.4f}")

        m = cfg['model'].set_params(**bp)

        # 10-fold CV full training data
        print(f"\n10-fold CV on {Xtr.shape[0]:,} samples...")
        cv = cv_classify(m, Xtr, ytr) if is_clf else cv_regress(m, Xtr, ytr)
        pp_cv(cv, f"{name} > {algo}")

        # Final train + test eval
        print(f"\nFinal train on {Xtr.shape[0]:,} samples...")
        m.fit(Xtr, ytr)
        te = eval_clf(m, Xte, yte, f"{name} > {algo}") if is_clf else eval_reg(m, Xte, yte, f"{name} > {algo}")

        elapsed = time.time() - t0
        print(f"Completed in {elapsed/60:.1f} min")

        results[algo] = {'model': m, 'params': bp, 'cv': cv, 'test': te, 'time': elapsed}

        # Save checkpoint to Drive immediately
        save_checkpoint(name, algo, results[algo])

    return results


# %% HYPERPARAMETER GRIDS

# --- XGBoost classification grid ---
XGB_CLF_PARAMS = {
    'n_estimators':      [100, 200, 300, 500],
    'max_depth':         [3, 5, 7, 9],
    'learning_rate':     [0.01, 0.05, 0.1, 0.2],
    'subsample':         [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree':  [0.6, 0.7, 0.8, 0.9],
    'min_child_weight':  [1, 3, 5, 10],
    'gamma':             [0, 0.1, 0.3],
    'reg_alpha':         [0, 0.01, 0.1],
    'reg_lambda':        [1, 2, 5],
}

# --- XGBoost regression grid ---
XGB_REG_PARAMS = {
    'n_estimators':      [100, 200, 300, 500],
    'max_depth':         [3, 5, 7, 9],
    'learning_rate':     [0.01, 0.05, 0.1, 0.2],
    'subsample':         [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree':  [0.6, 0.7, 0.8, 0.9],
    'min_child_weight':  [1, 3, 5, 10],
    'gamma':             [0, 0.1, 0.3],
    'reg_alpha':         [0, 0.01, 0.1],
    'reg_lambda':        [1, 2, 5],
}

# --- Random Forest classification grid ---
RF_CLF_PARAMS = {
    'n_estimators':      [100, 200, 300],
    'max_depth':         [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'max_features':      ['sqrt', 'log2', 0.3],
}

# --- Random Forest regression grid ---
RF_REG_PARAMS = {
    'n_estimators':      [100, 200, 300],
    'max_depth':         [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'max_features':      ['sqrt', 'log2', 0.3],
}

# --- Logistic Regression grid ---
LR_PARAMS = {
    'C':       [0.001, 0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver':  ['liblinear', 'saga'],
}

# --- Ridge Regression grid ---
RIDGE_PARAMS = {
    'alpha': [0.01, 0.1, 1, 10, 100, 1000],
}


# %% MODEL A: CANCELLATION CLASSIFIER

configs_a = {
    'XGBoost': {
        'model': XGBClassifier(
            device=XGB_DEVICE, tree_method='hist',
            scale_pos_weight=spw_canc, eval_metric='logloss',
            random_state=42
        ),
        'params': XGB_CLF_PARAMS,
        'gpu': GPU_OK,
    },
}

print("\n" + "="*60)
print("  MODEL A: CANCELLATION CLASSIFIER")
print("="*60)
results_a = run_all("ModelA_Cancel", configs_a, X_train_a, y_canc_tr, X_test_a, y_canc_te, task='clf')


# %% MODEL B: DELAY CLASSIFIER (non-cancelled flights only)

configs_b = {
    'XGBoost': {
        'model': XGBClassifier(
            device=XGB_DEVICE, tree_method='hist',
            scale_pos_weight=spw_delay, eval_metric='logloss',
            random_state=42
        ),
        'params': XGB_CLF_PARAMS,
        'gpu': GPU_OK,
    },
}

print("\n" + "="*60)
print("  MODEL B: DELAY CLASSIFIER")
print("="*60)
results_b = run_all("ModelB_Delay", configs_b, X_tr_b, y_tr_b, X_te_b, y_te_b, task='clf')


# %% MODEL C: DELAY MINUTES REGRESSOR (non-cancelled flights only)

configs_c = {
    'XGBoost': {
        'model': TransformedTargetRegressor(
            regressor=XGBRegressor(device=XGB_DEVICE, tree_method='hist', random_state=42),
            func=np.log1p, inverse_func=np.expm1
        ),
        'params': {'regressor__' + k: v for k, v in XGB_REG_PARAMS.items()},
        'gpu': GPU_OK,
    },
}

print("\n" + "="*60)
print("  MODEL C: DELAY MINUTES REGRESSOR")
print("="*60)
results_c = run_all("ModelC_Mins", configs_c, X_tr_c, y_tr_c, X_te_c, y_te_c, task='reg')


# %% RESULTS COMPARISON TABLE

def compare_classifiers(results, model_name):
    """Print comparison table for classifiers."""
    print(f"\n{'='*70}")
    print(f"  {model_name} — Algorithm Comparison")
    print(f"{'='*70}")
    print(f"{'Algorithm':<22s} {'CV F1':>8s} {'CV AUC':>8s} {'Test F1':>8s} {'Test AUC':>8s} {'Time':>8s}")
    print("-" * 70)
    for algo, r in results.items():
        cv_f1  = r['cv']['f1'].mean()
        cv_auc = r['cv']['auc'].mean()
        te_f1  = r['test']['f1']
        te_auc = r['test']['auc']
        mins   = r['time'] / 60
        print(f"{algo:<22s} {cv_f1:>8.4f} {cv_auc:>8.4f} {te_f1:>8.4f} {te_auc:>8.4f} {mins:>7.1f}m")

def compare_regressors(results, model_name):
    """Print comparison table for regressors."""
    print(f"\n{'='*95}")
    print(f"  {model_name} — Algorithm Comparison")
    print(f"{'='*95}")
    print(f"{'Algorithm':<22s} {'CV MAE':>8s} {'CV RMSE':>8s} {'CV R2':>8s} {'Test RMSE':>9s} {'Test Adj R2':>11s} {'Time':>8s}")
    print("-" * 95)
    for algo, r in results.items():
        cv_mae  = r['cv']['mae'].mean()
        cv_rmse = r['cv']['rmse'].mean()
        cv_r2   = r['cv']['r2'].mean()
        te_rmse = r['test']['rmse']
        te_adj_r2 = r['test']['adj_r2']
        mins    = r['time'] / 60
        print(f"{algo:<22s} {cv_mae:>8.2f} {cv_rmse:>8.2f} {cv_r2:>8.4f} {te_rmse:>9.2f} {te_adj_r2:>11.4f} {mins:>7.1f}m")

compare_classifiers(results_a, "Model A: Cancellation")
compare_classifiers(results_b, "Model B: Delay Y/N")
compare_regressors(results_c, "Model C: Delay Minutes")

# Identify best algorithm per model
best_a_algo = max(results_a, key=lambda k: results_a[k]['test']['f1'])
best_b_algo = max(results_b, key=lambda k: results_b[k]['test']['f1'])
best_c_algo = min(results_c, key=lambda k: results_c[k]['test']['mae'])

print(f"\nBest Model A: {best_a_algo}")
print(f"Best Model B: {best_b_algo}")
print(f"Best Model C: {best_c_algo}")


# %% SHAP ANALYSIS (best models only)

import shap

# SHAP on 1000 test samples (speed)
SHAP_SAMPLE = 1000

for label, results, Xte, best_algo in [
    ("ModelA_Cancel", results_a, X_test_a, best_a_algo),
    ("ModelB_Delay",  results_b, X_te_b, best_b_algo),
    ("ModelC_Mins",   results_c, X_te_c, best_c_algo),
]:
    print(f"\nSHAP for {label} ({best_algo})...")
    m = results[best_algo]['model']
    idx = np.random.RandomState(42).choice(Xte.shape[0], min(SHAP_SAMPLE, Xte.shape[0]), replace=False)
    X_shap = Xte[idx]

    try:
        explainer = shap.TreeExplainer(m)
        shap_vals = explainer.shap_values(X_shap)
        # Save SHAP values
        np.save(os.path.join(SAVE_DIR, f'{label}_shap_values.npy'), shap_vals)
        # Summary plot
        shap.summary_plot(
            shap_vals, X_shap,
            feature_names=FEATURE_NAMES,
            max_display=20, show=True
        )
    except Exception as e:
        print(f"SHAP failed for {label}: {e}")
        print("Falling back to built-in feature importance...")
        if hasattr(m, 'feature_importances_'):
            imp = pd.Series(m.feature_importances_, index=FEATURE_NAMES)
            top20 = imp.nlargest(20)
            print(top20.to_string())


# %% SAVE MODELS & ARTIFACTS

# Save preprocessing objects
joblib.dump(scaler, os.path.join(SAVE_DIR, 'scaler.joblib'))
joblib.dump(te_a, os.path.join(SAVE_DIR, 'target_encoder_a.joblib'))
joblib.dump(te_b, os.path.join(SAVE_DIR, 'target_encoder_b.joblib'))
joblib.dump(te_c, os.path.join(SAVE_DIR, 'target_encoder_c.joblib'))
joblib.dump(FEATURE_NAMES, os.path.join(SAVE_DIR, 'feature_names.joblib'))

# Save best model per task
for label, results, best_algo in [
    ("model_a_cancellation", results_a, best_a_algo),
    ("model_b_delay",       results_b, best_b_algo),
    ("model_c_minutes",     results_c, best_c_algo),
]:
    m = results[best_algo]['model']
    path = os.path.join(SAVE_DIR, f'{label}.joblib')
    joblib.dump(m, path)
    print(f"Saved {label} ({best_algo}) -> {path}")

# Save all models (for comparison in writeup)
for model_label, results in [("a", results_a), ("b", results_b), ("c", results_c)]:
    for algo, r in results.items():
        path = os.path.join(SAVE_DIR, f'model_{model_label}_{algo.lower().replace(" ","_")}.joblib')
        joblib.dump(r['model'], path)

# Save results summary
summary = {
    'model_a': {algo: {'params': r['params'], 'test': r['test'], 'time': r['time'],
                        'cv_means': {k: float(v.mean()) for k, v in r['cv'].items()}}
                for algo, r in results_a.items()},
    'model_b': {algo: {'params': r['params'], 'test': r['test'], 'time': r['time'],
                        'cv_means': {k: float(v.mean()) for k, v in r['cv'].items()}}
                for algo, r in results_b.items()},
    'model_c': {algo: {'params': r['params'], 'test': r['test'], 'time': r['time'],
                        'cv_means': {k: float(v.mean()) for k, v in r['cv'].items()}}
                for algo, r in results_c.items()},
    'best': {'a': best_a_algo, 'b': best_b_algo, 'c': best_c_algo},
}
joblib.dump(summary, os.path.join(SAVE_DIR, 'results_summary.joblib'))

print(f"\nAll artifacts saved to {SAVE_DIR}")
print("Done.")
