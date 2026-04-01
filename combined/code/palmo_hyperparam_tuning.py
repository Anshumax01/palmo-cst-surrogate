import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings, os, time
from math import comb
warnings.filterwarnings("ignore")

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

try:
    from cst_modeling.section import cst_foil
    CST_LIB = True
except ImportError:
    CST_LIB = False

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TRAIN_CSV  = r"C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Results CST NN\Training_NACA_data.csv (3).xlsx"
TEST_CSV   = r"C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Results CST NN\Testing_NACA_data.csv (3).xlsx"
OUTPUT_DIR = r"C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Results Hyperparameterization"

FLIGHT_COLS = ["Mach", "Re", "Alpha"]
OUTPUT_COLS = ["Cl", "Cd", "Cm"]
AIRFOIL_COL = "NACA"

N_CST       = 7        # fixed — do not change
FIXED_LR    = 1e-3     # fixed per boss instruction
FIXED_BATCH = 512      # fixed per boss instruction
MAX_ITER    = 500
PATIENCE    = 30
RANDOM_SEED = 42

MACH_PLOT = 0.25
RE_PLOT   = 1_000_000

# 14 hidden layer architectures from the screenshot
HIDDEN_CONFIGS = [
    # Shallow (2 layers)
    (50,  50),
    (100, 100),
    (200, 200),
    (400, 400),
    # Medium (3 layers)
    (50,  50,  50),
    (100, 100, 100),
    (200, 200, 200),   # ← current baseline
    (400, 400, 400),
    # Deep (4 layers)
    (100, 100, 100, 100),
    (200, 200, 200, 200),
    (400, 400, 400, 400),
    # Funnel shapes
    (100, 200, 100),
    (200, 400, 200),
    (50,  100, 200, 100, 50),
]

# ─── CST CORE ─────────────────────────────────────────────────────────────────
def naca4_coords(code, n=201):
    code = int(code)
    m    = (code // 1000)         / 100.0
    p    = ((code % 1000) // 100) / 10.0
    t    = (code % 100)           / 100.0
    beta = np.linspace(0, np.pi, n)
    x    = 0.5 * (1.0 - np.cos(beta))
    yt   = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
    if m == 0.0 or p == 0.0:
        yc  = np.zeros_like(x);  dyc = np.zeros_like(x)
    else:
        yc  = np.where(x < p, m/p**2*(2*p*x - x**2),       m/(1-p)**2*(1 - 2*p + 2*p*x - x**2))
        dyc = np.where(x < p, 2*m/p**2*(p - x),             2*m/(1-p)**2*(p - x))
    theta = np.arctan(dyc)
    return (x - yt*np.sin(theta), yc + yt*np.cos(theta),
            x + yt*np.sin(theta), yc - yt*np.cos(theta))

def _bernstein_basis(x, n_cst):
    C = np.sqrt(x) * (1.0 - x)
    n = n_cst - 1
    B = np.column_stack([comb(n, k) * x**k * (1-x)**(n-k) for k in range(n_cst)])
    return C[:, None] * B

def fit_cst(xu, yu, xl, yl, n_cst):
    xu = np.clip(xu, 1e-9, 1-1e-9);  xl = np.clip(xl, 1e-9, 1-1e-9)
    Pu = _bernstein_basis(xu, n_cst); Pl = _bernstein_basis(xl, n_cst)
    Au, _, _, _ = np.linalg.lstsq(Pu, yu, rcond=None)
    Al, _, _, _ = np.linalg.lstsq(Pl, yl, rcond=None)
    return Au, Al

def get_cst_params(code):
    xu, yu, xl, yl = naca4_coords(code)
    return fit_cst(xu, yu, xl, yl, N_CST)

def build_cst_features(df):
    u_cols = [f"cst_u{i}" for i in range(N_CST)]
    l_cols = [f"cst_l{i}" for i in range(N_CST)]
    cache  = {}
    for code in df[AIRFOIL_COL].unique():
        cu, cl       = get_cst_params(code)
        cache[code]  = np.concatenate([cu, cl])
    arr = np.array([cache[c] for c in df[AIRFOIL_COL]])
    for i, col in enumerate(u_cols + l_cols):
        df[col] = arr[:, i]
    return df, u_cols + l_cols

# ─── DATA LOADING ─────────────────────────────────────────────────────────────
def load_data():
    train_df = pd.read_excel(TRAIN_CSV)
    test_df  = pd.read_excel(TEST_CSV)
    train_df, cst_cols = build_cst_features(train_df.copy())
    test_df,  _        = build_cst_features(test_df.copy())
    input_cols = FLIGHT_COLS + cst_cols
    X_train = train_df[input_cols].values.astype(np.float64)
    y_train = train_df[OUTPUT_COLS].values.astype(np.float64)
    X_test  = test_df[input_cols].values.astype(np.float64)
    y_test  = test_df[OUTPUT_COLS].values.astype(np.float64)
    scaler_X = StandardScaler()
    X_train  = scaler_X.fit_transform(X_train)
    X_test   = scaler_X.transform(X_test)
    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train)
    print(f"  Train: {len(X_train):,}   Test: {len(X_test):,}   Inputs: {X_train.shape[1]}")
    return X_train, y_train_s, X_test, y_test, scaler_y, test_df

# ─── TRAINING & EVALUATION ────────────────────────────────────────────────────
def train_model(X_train, y_train_s, hidden):
    model = MLPRegressor(
        hidden_layer_sizes=hidden, activation="relu", solver="adam",
        learning_rate_init=FIXED_LR, batch_size=FIXED_BATCH,
        max_iter=MAX_ITER, tol=1e-5,
        n_iter_no_change=PATIENCE, random_state=RANDOM_SEED, verbose=False,
    )
    model.fit(X_train, y_train_s)
    return model, model.loss_curve_

def evaluate(model, X_test, y_test, scaler_y):
    y_pred = scaler_y.inverse_transform(model.predict(X_test))
    metrics = {}
    for i, col in enumerate(OUTPUT_COLS):
        metrics[col] = {
            "r2":   r2_score(y_test[:, i], y_pred[:, i]),
            "mae":  mean_absolute_error(y_test[:, i], y_pred[:, i]),
            "mse":  mean_squared_error(y_test[:, i], y_pred[:, i]),
            "rmse": np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i])),
        }
    return y_pred, metrics

def arch_label(h):
    return "(" + ",".join(str(x) for x in h) + ")"

def arch_label_multiline(h):
    return arch_label(h) + f"\nd={len(h)}"

# ─── ARCHITECTURE SWEEP PLOTS ─────────────────────────────────────────────────
def plot_arch_r2(df, path):
    labels = [arch_label(tuple(r["hidden"])) for _, r in df.iterrows()]
    x      = np.arange(len(df))
    w      = 0.25
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - w,   df["Cl_r2"], w, label="Cl R²",  color="steelblue", edgecolor="black", lw=0.5)
    ax.bar(x,       df["Cd_r2"], w, label="Cd R²",  color="tomato",    edgecolor="black", lw=0.5)
    ax.bar(x + w,   df["Cm_r2"], w, label="Cm R²",  color="seagreen",  edgecolor="black", lw=0.5)
    ax.axhline(0.999, color="gray", ls="--", lw=1, alpha=0.6, label="R²=0.999")
    # Highlight baseline (200,200,200)
    base_idx = next((i for i, r in df.iterrows() if tuple(r["hidden"]) == (200,200,200)), None)
    if base_idx is not None:
        ax.axvspan(base_idx - 0.45, base_idx + 0.45, alpha=0.08, color="gold", label="Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    bot = max(0.97, df[["Cl_r2","Cd_r2","Cm_r2"]].min().min() - 0.005)
    ax.set_ylim(bottom=bot)
    ax.set_ylabel("R²")
    ax.set_title(f"Architecture Sweep — R² by Config  (N_CST={N_CST}, lr={FIXED_LR}, batch={FIXED_BATCH})",
                 fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def plot_arch_mae(df, path):
    labels = [arch_label(tuple(r["hidden"])) for _, r in df.iterrows()]
    x      = np.arange(len(df))
    w      = 0.25
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - w,   df["Cl_mae"], w, label="Cl MAE", color="steelblue", edgecolor="black", lw=0.5)
    ax.bar(x,       df["Cd_mae"], w, label="Cd MAE", color="tomato",    edgecolor="black", lw=0.5)
    ax.bar(x + w,   df["Cm_mae"], w, label="Cm MAE", color="seagreen",  edgecolor="black", lw=0.5)
    if base_idx := next((i for i, r in df.iterrows() if tuple(r["hidden"]) == (200,200,200)), None):
        ax.axvspan(base_idx - 0.45, base_idx + 0.45, alpha=0.08, color="gold", label="Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("MAE")
    ax.set_title(f"Architecture Sweep — MAE by Config  (N_CST={N_CST}, lr={FIXED_LR}, batch={FIXED_BATCH})",
                 fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def plot_arch_rmse(df, path):
    labels = [arch_label(tuple(r["hidden"])) for _, r in df.iterrows()]
    x      = np.arange(len(df))
    w      = 0.25
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - w,   df["Cl_rmse"], w, label="Cl RMSE", color="steelblue", edgecolor="black", lw=0.5)
    ax.bar(x,       df["Cd_rmse"], w, label="Cd RMSE", color="tomato",    edgecolor="black", lw=0.5)
    ax.bar(x + w,   df["Cm_rmse"], w, label="Cm RMSE", color="seagreen",  edgecolor="black", lw=0.5)
    if base_idx := next((i for i, r in df.iterrows() if tuple(r["hidden"]) == (200,200,200)), None):
        ax.axvspan(base_idx - 0.45, base_idx + 0.45, alpha=0.08, color="gold", label="Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("RMSE")
    ax.set_title(f"Architecture Sweep — RMSE by Config  (N_CST={N_CST}, lr={FIXED_LR}, batch={FIXED_BATCH})",
                 fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def plot_arch_top10(df, path):
    df    = df.copy()
    df["avg_r2"] = (df["Cl_r2"] + df["Cd_r2"] + df["Cm_r2"]) / 3
    top10 = df.nlargest(10, "avg_r2").reset_index(drop=True)
    labels = [arch_label(tuple(r["hidden"])) for _, r in top10.iterrows()]
    x = np.arange(len(top10));  w = 0.25
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w,   top10["Cl_r2"], w, label="Cl R²",  color="steelblue", edgecolor="black", lw=0.5)
    ax.bar(x,       top10["Cd_r2"], w, label="Cd R²",  color="tomato",    edgecolor="black", lw=0.5)
    ax.bar(x + w,   top10["Cm_r2"], w, label="Cm R²",  color="seagreen",  edgecolor="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("R²")
    bot = max(0.97, top10[["Cl_r2","Cd_r2","Cm_r2"]].min().min() - 0.003)
    ax.set_ylim(bottom=bot)
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Architecture Sweep — Top 10 Configurations by Mean R²", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def plot_depth_sensitivity(df, path):
    depths = sorted(df["depth"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(depths)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for ax, col in zip(axes, OUTPUT_COLS):
        means = [df[df["depth"] == d][f"{col}_r2"].mean() for d in depths]
        stds  = [df[df["depth"] == d][f"{col}_r2"].std()  for d in depths]
        ax.bar([str(d) for d in depths], means, yerr=stds,
               color=colors, edgecolor="black", lw=0.6, capsize=4)
        ax.set_xlabel("Network Depth (# layers)")
        ax.set_ylabel("Mean R²")
        ax.set_title(f"{col} — R² vs Depth")
        ax.grid(True, alpha=0.3, axis="y")
        bot = max(0.97, min(means) - 0.01)
        ax.set_ylim(bottom=bot)
    plt.suptitle(f"Architecture Sweep — R² vs Network Depth  (N_CST={N_CST})", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def plot_width_sensitivity(df, path):
    widths = sorted(df["width"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(widths)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for ax, col in zip(axes, OUTPUT_COLS):
        means = [df[df["width"] == w][f"{col}_r2"].mean() for w in widths]
        stds  = [df[df["width"] == w][f"{col}_r2"].std()  for w in widths]
        ax.bar([str(w) for w in widths], means, yerr=stds,
               color=colors, edgecolor="black", lw=0.6, capsize=4)
        ax.set_xlabel("Layer Width (neurons per layer)")
        ax.set_ylabel("Mean R²")
        ax.set_title(f"{col} — R² vs Width")
        ax.grid(True, alpha=0.3, axis="y")
        bot = max(0.97, min(means) - 0.01)
        ax.set_ylim(bottom=bot)
    plt.suptitle(f"Architecture Sweep — R² vs Layer Width  (N_CST={N_CST})", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def plot_training_time(df, path):
    labels = [arch_label(tuple(r["hidden"])) for _, r in df.iterrows()]
    fig, ax = plt.subplots(figsize=(16, 5))
    colors  = ["gold" if tuple(r["hidden"]) == (200,200,200) else "steelblue"
               for _, r in df.iterrows()]
    ax.bar(range(len(df)), df["train_time_s"], color=colors, edgecolor="black", lw=0.5)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Training Time (s)")
    ax.set_title(f"Architecture Sweep — Training Time  (gold = baseline)",fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

# ─── BEST MODEL PLOTS ─────────────────────────────────────────────────────────
def plot_training_curve(train_losses, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_losses, lw=1.5, color="steelblue", label="Train Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (MSE, scaled)")
    ax.set_title("Best Model — Training Curve"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def plot_predictions_scatter(y_pred, y_test, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col, i in zip(axes, OUTPUT_COLS, range(3)):
        lo = min(y_test[:,i].min(), y_pred[:,i].min())
        hi = max(y_test[:,i].max(), y_pred[:,i].max())
        ax.scatter(y_test[:,i], y_pred[:,i], s=1, alpha=0.2, color="steelblue")
        ax.plot([lo,hi],[lo,hi],"r--",lw=1.5,label="Ideal")
        r2 = r2_score(y_test[:,i], y_pred[:,i])
        ax.set_xlabel(f"CFD {col}"); ax.set_ylabel(f"NN {col}")
        ax.set_title(f"{col}  R²={r2:.5f}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.suptitle("Best Model — Predicted vs CFD (Test Set)", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def plot_lift_curves(y_pred, y_test, test_df, path):
    airfoils = sorted(test_df[AIRFOIL_COL].unique())
    fig, axes = plt.subplots(1, len(airfoils), figsize=(5*len(airfoils), 5), sharey=True)
    if len(airfoils) == 1: axes = [axes]
    for ax, af in zip(axes, airfoils):
        mask = ((np.abs(test_df["Mach"].values - MACH_PLOT) < 1e-4) &
                (np.abs(test_df["Re"].values - RE_PLOT) < RE_PLOT*0.05) &
                (test_df[AIRFOIL_COL].values == af))
        if mask.sum() == 0: ax.set_title(f"NACA {af}\n(no data)"); continue
        aoa = test_df.loc[mask,"Alpha"].values; s = np.argsort(aoa)
        idx = OUTPUT_COLS.index("Cl")
        ax.plot(aoa[s], y_test[mask,idx][s], "k-",  lw=2.0, label="CFD")
        ax.plot(aoa[s], y_pred[mask,idx][s], "gs",  ms=5,
                markerfacecolor="none", markeredgewidth=1.2, label="Best NN")
        ax.set_xlabel("Angle of Attack (°)")
        ax.set_title(f"NACA {af}\nM={MACH_PLOT}, Re={RE_PLOT:.0e}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("$c_l$")
    plt.suptitle("Best Model — Lift Curves (Test Airfoils)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def plot_drag_polars(y_pred, y_test, test_df, path):
    airfoils = sorted(test_df[AIRFOIL_COL].unique())
    fig, axes = plt.subplots(1, len(airfoils), figsize=(5*len(airfoils), 5), sharey=True)
    if len(airfoils) == 1: axes = [axes]
    for ax, af in zip(axes, airfoils):
        mask = ((np.abs(test_df["Mach"].values - MACH_PLOT) < 1e-4) &
                (np.abs(test_df["Re"].values - RE_PLOT) < RE_PLOT*0.05) &
                (test_df[AIRFOIL_COL].values == af))
        if mask.sum() == 0: continue
        cl_cfd = y_test[mask, OUTPUT_COLS.index("Cl")]
        cd_cfd = y_test[mask, OUTPUT_COLS.index("Cd")]
        cl_nn  = y_pred[mask, OUTPUT_COLS.index("Cl")]
        cd_nn  = y_pred[mask, OUTPUT_COLS.index("Cd")]
        s = np.argsort(cl_cfd)
        ax.plot(cd_cfd[s], cl_cfd[s], "k-",  lw=2.0, label="CFD")
        ax.plot(cd_nn[s],  cl_nn[s],  "gs",  ms=5,
                markerfacecolor="none", markeredgewidth=1.2, label="Best NN")
        ax.set_xlabel("$c_d$")
        ax.set_title(f"NACA {af}\nM={MACH_PLOT}, Re={RE_PLOT:.0e}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("$c_l$")
    plt.suptitle("Best Model — Drag Polars (Test Airfoils)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def plot_error_distributions(y_pred, y_test, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = {"Cl": "steelblue", "Cd": "tomato", "Cm": "seagreen"}
    for ax, col, i in zip(axes, OUTPUT_COLS, range(3)):
        errors = y_pred[:,i] - y_test[:,i]
        ax.hist(errors, bins=80, color=colors[col], edgecolor="none", alpha=0.85)
        ax.axvline(0, color="black", ls="--", lw=1.5)
        ax.set_xlabel(f"NN − CFD  ({col})")
        ax.set_ylabel("Count")
        ax.set_title(f"{col}  μ={errors.mean():.5f}  σ={errors.std():.5f}")
        ax.grid(True, alpha=0.3)
    plt.suptitle("Best Model — Prediction Error Distributions", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def plot_per_airfoil(y_pred, y_test, test_df, path):
    airfoils = sorted(test_df[AIRFOIL_COL].unique())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    bar_colors = ["steelblue","tomato","seagreen","goldenrod"]
    for col_i, coeff in enumerate(OUTPUT_COLS):
        ax_r2  = axes[0][col_i]
        ax_mae = axes[1][col_i]
        r2s, maes, labels = [], [], []
        for af in airfoils:
            mask = test_df[AIRFOIL_COL].values == af
            if mask.sum() == 0: continue
            r2s.append(r2_score(y_test[mask,col_i], y_pred[mask,col_i]))
            maes.append(mean_absolute_error(y_test[mask,col_i], y_pred[mask,col_i]))
            labels.append(f"NACA\n{af}")
        c = [bar_colors[i % len(bar_colors)] for i in range(len(labels))]
        ax_r2.bar(labels, r2s,  color=c, edgecolor="black", lw=0.5)
        ax_r2.axhline(0.99, color="green", ls="--", lw=1, label="R²=0.99")
        ax_r2.set_ylim(min(0.95, min(r2s)-0.01), 1.001)
        ax_r2.set_title(f"{coeff} — R² per Test Airfoil")
        ax_r2.set_ylabel("R²"); ax_r2.legend(fontsize=7); ax_r2.grid(True, alpha=0.3, axis="y")
        ax_mae.bar(labels, maes, color=c, edgecolor="black", lw=0.5)
        ax_mae.set_title(f"{coeff} — MAE per Test Airfoil")
        ax_mae.set_ylabel("MAE"); ax_mae.grid(True, alpha=0.3, axis="y")
    plt.suptitle("Best Model — Per-Airfoil Performance", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def save_per_testcase(y_pred, y_test, test_df, path):
    records  = []
    airfoils = sorted(test_df[AIRFOIL_COL].unique())
    machs    = sorted(test_df["Mach"].unique())
    reynolds = sorted(test_df["Re"].unique())
    for af in airfoils:
        for mach in machs:
            for re in reynolds:
                mask = ((test_df[AIRFOIL_COL].values == af) &
                        (np.abs(test_df["Mach"].values - mach) < 1e-4) &
                        (test_df["Re"].values == re))
                if mask.sum() == 0: continue
                row = {"Airfoil": af, "Mach": mach, "Re": re, "N_points": int(mask.sum())}
                for i, col in enumerate(OUTPUT_COLS):
                    row[f"{col}_MAE"]  = round(mean_absolute_error(y_test[mask,i], y_pred[mask,i]), 6)
                    row[f"{col}_RMSE"] = round(np.sqrt(mean_squared_error(y_test[mask,i], y_pred[mask,i])), 6)
                    row[f"{col}_R2"]   = round(r2_score(y_test[mask,i], y_pred[mask,i]), 6)
                records.append(row)
    pd.DataFrame(records).to_csv(path, index=False)
    print(f"Saved: {path}")

# ─── MAIN SWEEP ───────────────────────────────────────────────────────────────
def run_architecture_sweep():
    print("=" * 65)
    print("ARCHITECTURE SWEEP")
    print(f"  N_CST   = {N_CST}  (fixed)")
    print(f"  LR      = {FIXED_LR}  (fixed)")
    print(f"  Batch   = {FIXED_BATCH}  (fixed)")
    print(f"  Configs = {len(HIDDEN_CONFIGS)}")
    print("=" * 65)

    sweep_dir = os.path.join(OUTPUT_DIR, "architecture_sweep")
    best_dir  = os.path.join(OUTPUT_DIR, "best_model")
    os.makedirs(sweep_dir, exist_ok=True)
    os.makedirs(best_dir,  exist_ok=True)

    print("\nLoading data and computing CST features...")
    X_train, y_train_s, X_test, y_test, scaler_y, test_df = load_data()

    records = []
    for run_num, hidden in enumerate(HIDDEN_CONFIGS, 1):
        print(f"\n  [{run_num:02d}/{len(HIDDEN_CONFIGS)}]  {arch_label(hidden):<30}", end="  ", flush=True)
        t0 = time.time()
        model, _ = train_model(X_train, y_train_s, hidden)
        elapsed  = time.time() - t0
        _, metrics = evaluate(model, X_test, y_test, scaler_y)

        row = {
            "hidden":       hidden,
            "depth":        len(hidden),
            "width":        max(hidden),
            "train_time_s": round(elapsed, 2),
        }
        for col in OUTPUT_COLS:
            row[f"{col}_r2"]   = round(metrics[col]["r2"],   6)
            row[f"{col}_mae"]  = round(metrics[col]["mae"],  6)
            row[f"{col}_mse"]  = round(metrics[col]["mse"],  8)
            row[f"{col}_rmse"] = round(metrics[col]["rmse"], 6)
        records.append(row)

        print(f"Cl R²={metrics['Cl']['r2']:.5f}  "
              f"Cd R²={metrics['Cd']['r2']:.5f}  "
              f"Cm R²={metrics['Cm']['r2']:.5f}  "
              f"[{elapsed:.0f}s]")

        # Save partial CSV after every run so nothing is lost if it crashes
        pd.DataFrame(records).to_csv(
            os.path.join(sweep_dir, "arch_sweep_partial.csv"), index=False)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(sweep_dir, "arch_sweep_results.csv"), index=False)
    print(f"\nSaved: arch_sweep_results.csv")

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("\nGenerating sweep plots...")
    p = lambda f: os.path.join(sweep_dir, f)
    plot_arch_r2(df,             p("arch_r2_all_configs.png"))
    plot_arch_mae(df,            p("arch_mae_all_configs.png"))
    plot_arch_rmse(df,           p("arch_rmse_all_configs.png"))
    plot_arch_top10(df,          p("arch_top10_configs.png"))
    plot_depth_sensitivity(df,   p("arch_r2_vs_depth.png"))
    plot_width_sensitivity(df,   p("arch_r2_vs_width.png"))
    plot_training_time(df,       p("arch_training_time.png"))

    # ── Best config ────────────────────────────────────────────────────────────
    df["avg_r2"] = (df["Cl_r2"] + df["Cd_r2"] + df["Cm_r2"]) / 3
    best_idx = df["avg_r2"].idxmax()
    best_row = df.loc[best_idx]
    best_hidden = tuple(best_row["hidden"])

    print("\n" + "=" * 65)
    print("BEST ARCHITECTURE")
    print(f"  Hidden  : {arch_label(best_hidden)}")
    print(f"  Depth   : {best_row['depth']}")
    print(f"  Width   : {best_row['width']}")
    print(f"  Cl R²   : {best_row['Cl_r2']:.5f}    MAE: {best_row['Cl_mae']:.5f}    RMSE: {best_row['Cl_rmse']:.5f}")
    print(f"  Cd R²   : {best_row['Cd_r2']:.5f}    MAE: {best_row['Cd_mae']:.5f}    RMSE: {best_row['Cd_rmse']:.5f}")
    print(f"  Cm R²   : {best_row['Cm_r2']:.5f}    MAE: {best_row['Cm_mae']:.5f}    RMSE: {best_row['Cm_rmse']:.5f}")
    print(f"  Mean R² : {best_row['avg_r2']:.5f}")
    print("=" * 65)

    # ── Retrain best from scratch for full evaluation ──────────────────────────
    print(f"\nRetraining best config {arch_label(best_hidden)} for full evaluation...")
    best_model, train_losses = train_model(X_train, y_train_s, best_hidden)
    y_pred, best_metrics     = evaluate(best_model, X_test, y_test, scaler_y)

    print("\n  Final test set performance:")
    for col in OUTPUT_COLS:
        print(f"    {col}  R²={best_metrics[col]['r2']:.5f}  "
              f"MAE={best_metrics[col]['mae']:.5f}  "
              f"MSE={best_metrics[col]['mse']:.7f}  "
              f"RMSE={best_metrics[col]['rmse']:.5f}")

    b = lambda f: os.path.join(best_dir, f)
    print("\nGenerating best model plots...")
    plot_training_curve(train_losses,           b("best_training_curve.png"))
    plot_predictions_scatter(y_pred, y_test,    b("best_predictions_scatter.png"))
    plot_lift_curves(y_pred, y_test, test_df,   b("best_lift_curves.png"))
    plot_drag_polars(y_pred, y_test, test_df,   b("best_drag_polars.png"))
    plot_error_distributions(y_pred, y_test,    b("best_error_distributions.png"))
    plot_per_airfoil(y_pred, y_test, test_df,   b("best_per_airfoil_metrics.png"))
    save_per_testcase(y_pred, y_test, test_df,  b("best_per_testcase_errors.csv"))

    # ── Final summary CSV ──────────────────────────────────────────────────────
    summary_rows = []
    for col in OUTPUT_COLS:
        summary_rows.append({
            "output":       col,
            "best_hidden":  arch_label(best_hidden),
            "depth":        len(best_hidden),
            "width":        max(best_hidden),
            "N_CST":        N_CST,
            "lr":           FIXED_LR,
            "batch":        FIXED_BATCH,
            "R2":           round(best_metrics[col]["r2"],   6),
            "MAE":          round(best_metrics[col]["mae"],  6),
            "MSE":          round(best_metrics[col]["mse"],  8),
            "RMSE":         round(best_metrics[col]["rmse"], 6),
        })
    summary_path = os.path.join(OUTPUT_DIR, "final_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    return df, best_row, best_metrics

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df, best_row, best_metrics = run_architecture_sweep()

    print("\n" + "=" * 65)
    print("ALL DONE")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\narchitecture_sweep/")
    print("  arch_sweep_results.csv")
    print("  arch_r2_all_configs.png")
    print("  arch_mae_all_configs.png")
    print("  arch_rmse_all_configs.png")
    print("  arch_top10_configs.png")
    print("  arch_r2_vs_depth.png")
    print("  arch_r2_vs_width.png")
    print("  arch_training_time.png")
    print("\nbest_model/")
    print("  best_training_curve.png")
    print("  best_predictions_scatter.png")
    print("  best_lift_curves.png")
    print("  best_drag_polars.png")
    print("  best_error_distributions.png")
    print("  best_per_airfoil_metrics.png")
    print("  best_per_testcase_errors.csv")
    print("\nfinal_summary.csv")
    print("=" * 65)

if __name__ == "__main__":
    main()