import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings("ignore")

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TRAIN_CSV   = r"C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Results CST NN\Training_NACA_data.csv (3).xlsx"
TEST_CSV    = r"C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Results CST NN\Testing_NACA_data.csv (3).xlsx"
OUTPUT_DIR  = r"C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Results Baseline Split\Cm"

INPUT_COLS  = ["Mach", "Re", "Alpha", "Thickness", "Camber"]
TARGET      = "Cm"
AIRFOIL_COL = "NACA"

HIDDEN_LAYERS = (200, 200, 200)
LEARNING_RATE = 1e-3
BATCH_SIZE    = 512
MAX_ITER      = 500
PATIENCE      = 30
RANDOM_SEED   = 42
VAL_FRACTION  = 0.10

# ─── DATA ─────────────────────────────────────────────────────────────────────
def load_data():
    print(f"Loading data for {TARGET}...")
    train_df = pd.read_excel(TRAIN_CSV)
    test_df  = pd.read_excel(TEST_CSV)
    print(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    X_train = train_df[INPUT_COLS].values.astype(np.float64)
    y_train = train_df[TARGET].values.astype(np.float64).reshape(-1, 1)
    X_test  = test_df[INPUT_COLS].values.astype(np.float64)
    y_test  = test_df[TARGET].values.astype(np.float64)

    scaler_X = StandardScaler()
    X_train  = scaler_X.fit_transform(X_train)
    X_test   = scaler_X.transform(X_test)

    # Cm gets its OWN scaler — no cross-contamination with Cl or Cd
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train).ravel()

    return X_train, y_train_scaled, X_test, y_test, scaler_X, scaler_y, test_df

# ─── TRAIN ────────────────────────────────────────────────────────────────────
def train_model(X_train, y_train_scaled):
    print(f"\nTraining {TARGET} model  |  arch={HIDDEN_LAYERS}  |  lr={LEARNING_RATE}")

    rng   = np.random.default_rng(RANDOM_SEED)
    n_val = max(1, int(VAL_FRACTION * len(X_train)))
    idx   = rng.permutation(len(X_train))
    Xtr, ytr = X_train[idx[n_val:]], y_train_scaled[idx[n_val:]]
    Xvl, yvl = X_train[idx[:n_val]], y_train_scaled[idx[:n_val]]

    model = MLPRegressor(
        hidden_layer_sizes=HIDDEN_LAYERS, activation="relu", solver="adam",
        learning_rate_init=LEARNING_RATE, batch_size=BATCH_SIZE,
        max_iter=1, warm_start=True, tol=0.0,
        n_iter_no_change=MAX_ITER, random_state=RANDOM_SEED, verbose=False,
    )

    train_losses, val_losses = [], []
    best_val, best_coefs, best_intercepts, no_improve = float("inf"), None, None, 0

    for epoch in range(1, MAX_ITER + 1):
        model.fit(Xtr, ytr)
        tr_loss  = mean_squared_error(ytr, model.predict(Xtr))
        val_loss = mean_squared_error(yvl, model.predict(Xvl))
        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        if val_loss < best_val - 1e-8:
            best_val        = val_loss
            best_coefs      = [c.copy() for c in model.coefs_]
            best_intercepts = [b.copy() for b in model.intercepts_]
            no_improve      = 0
        else:
            no_improve += 1

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{MAX_ITER}  "
                  f"train={tr_loss:.6f}  val={val_loss:.6f}")

        if no_improve >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch}  (best val={best_val:.6f})")
            break

    model.coefs_      = best_coefs
    model.intercepts_ = best_intercepts
    return model, train_losses, val_losses

# ─── EVALUATE ─────────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, scaler_y):
    y_pred_scaled = model.predict(X_test).reshape(-1, 1)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).ravel()

    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"  {TARGET}  →  MAE={mae:.5f}  MSE={mse:.5f}  RMSE={rmse:.5f}  R²={r2:.5f}")
    print(f"{'='*60}\n")
    return y_pred

# ─── PLOTS ────────────────────────────────────────────────────────────────────
def plot_loss_summary(train_losses, val_losses, y_pred, y_test, path_curve, path_summary):
    # Training curve
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train MSE (scaled)", lw=1.5)
    plt.plot(val_losses,   label="Val MSE (scaled)",   lw=1.5)
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title(f"PALMO {TARGET} — Training Curve")
    plt.legend(); plt.tight_layout()
    plt.savefig(path_curve, dpi=150); plt.close()
    print(f"Saved: {path_curve}")

    # Loss summary bar chart
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse}
    plt.figure(figsize=(6, 4))
    bars = plt.bar(metrics.keys(), metrics.values(),
                   color=["steelblue", "orange", "green"],
                   edgecolor="black", lw=0.5)
    for bar, val in zip(bars, metrics.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{val:.5f}", ha="center", va="bottom", fontsize=10)
    plt.title(f"PALMO {TARGET} — Final Loss Summary")
    plt.ylabel("Error"); plt.tight_layout()
    plt.savefig(path_summary, dpi=150); plt.close()
    print(f"Saved: {path_summary}")

def plot_scatter(y_pred, y_test, path):
    r2 = r2_score(y_test, y_pred)
    lo = min(y_test.min(), y_pred.min())
    hi = max(y_test.max(), y_pred.max())
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, s=1, alpha=0.2, color="steelblue")
    plt.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Ideal")
    plt.xlabel(f"CFD {TARGET}"); plt.ylabel(f"NN {TARGET}")
    plt.title(f"{TARGET}  (R²={r2:.5f})")
    plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")

def plot_error_distribution(y_pred, y_test, path):
    errors = y_pred - y_test
    plt.figure(figsize=(7, 4))
    plt.hist(errors, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
    plt.axvline(0,             color="red",    ls="--", lw=1.5)
    plt.axvline(errors.mean(), color="orange", ls="--", lw=1.2,
                label=f"μ={errors.mean():.5f}")
    plt.xlabel(f"NN − CFD  ({TARGET})"); plt.ylabel("Count")
    plt.title(f"{TARGET}  σ={errors.std():.5f}  μ={errors.mean():.5f}")
    plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")

def plot_aoa_curve(y_pred, y_test, test_df, path,
                   mach=0.25, reynolds=1_000_000):
    airfoils = sorted(test_df[AIRFOIL_COL].unique())
    fig, axes = plt.subplots(1, len(airfoils),
                             figsize=(5*len(airfoils), 5), sharey=True)
    if len(airfoils) == 1: axes = [axes]
    for ax, af in zip(axes, airfoils):
        mask = ((np.abs(test_df["Mach"].values - mach) < 1e-4) &
                (np.abs(test_df["Re"].values - reynolds) < reynolds*0.05) &
                (test_df[AIRFOIL_COL].values == af))
        if mask.sum() == 0: continue
        aoa = test_df.loc[mask, "Alpha"].values
        s   = np.argsort(aoa)
        ax.plot(aoa[s], y_test[mask][s], "k-",  lw=2.0, label="CFD")
        ax.plot(aoa[s], y_pred[mask][s], "gs",  ms=5,
                markerfacecolor="none", markeredgewidth=1.2, label="NN")
        ax.set_xlabel("AoA (deg)")
        ax.set_title(f"NACA {af}\nM={mach}, Re={reynolds:.0e}")
        ax.legend(fontsize=8)
    axes[0].set_ylabel(TARGET)
    plt.suptitle(f"PALMO {TARGET} — AoA Curve", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def plot_multi_mach(y_pred, y_test, test_df, path,
                    machs=(0.25, 0.50, 0.75), reynolds=1_000_000):
    airfoils = sorted(test_df[AIRFOIL_COL].unique())
    colors   = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    fig, axes = plt.subplots(1, len(airfoils),
                             figsize=(5*len(airfoils), 5), sharey=True)
    if len(airfoils) == 1: axes = [axes]
    for ax, af in zip(axes, airfoils):
        for mach, col in zip(machs, colors):
            mask = ((np.abs(test_df["Mach"].values - mach) < 1e-4) &
                    (np.abs(test_df["Re"].values - reynolds) < reynolds*0.05) &
                    (test_df[AIRFOIL_COL].values == af))
            if mask.sum() == 0: continue
            aoa = test_df.loc[mask, "Alpha"].values
            s   = np.argsort(aoa)
            ax.plot(aoa[s], y_test[mask][s], "-",  color=col, lw=2.0,
                    label=f"CFD M={mach}")
            ax.plot(aoa[s], y_pred[mask][s], "s",  color=col, ms=4,
                    markerfacecolor="none", markeredgewidth=1.2,
                    label=f"NN M={mach}")
        ax.set_xlabel("AoA (deg)"); ax.set_title(f"NACA {af}")
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel(TARGET)
    plt.suptitle(f"PALMO {TARGET} — Multi-Mach (Re={reynolds:.0e})",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def plot_error_vs_aoa(y_pred, y_test, test_df, path,
                      mach=0.25, reynolds=1_000_000):
    airfoils = sorted(test_df[AIRFOIL_COL].unique())
    fig, axes = plt.subplots(1, len(airfoils),
                             figsize=(5*len(airfoils), 4), sharey=True)
    if len(airfoils) == 1: axes = [axes]
    for ax, af in zip(axes, airfoils):
        mask = ((np.abs(test_df["Mach"].values - mach) < 1e-4) &
                (np.abs(test_df["Re"].values - reynolds) < reynolds*0.05) &
                (test_df[AIRFOIL_COL].values == af))
        if mask.sum() == 0: continue
        aoa = test_df.loc[mask, "Alpha"].values
        err = y_pred[mask] - y_test[mask]
        s   = np.argsort(aoa)
        ax.plot(aoa[s], err[s], "k-o", ms=3, lw=1.2)
        ax.axhline(0, color="red", ls="--", lw=1)
        ax.fill_between(aoa[s], err[s], 0, alpha=0.15, color="steelblue")
        ax.set_xlabel("AoA (deg)"); ax.set_title(f"NACA {af}")
        if ax is axes[0]: ax.set_ylabel(f"{TARGET} error (NN−CFD)")
    plt.suptitle(f"PALMO {TARGET} — Error vs AoA", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def plot_per_airfoil(y_pred, y_test, test_df, path):
    airfoils = sorted(test_df[AIRFOIL_COL].unique())
    r2s, maes, labels = [], [], []
    for af in airfoils:
        mask = test_df[AIRFOIL_COL].values == af
        r2s.append(r2_score(y_test[mask], y_pred[mask]))
        maes.append(mean_absolute_error(y_test[mask], y_pred[mask]))
        labels.append(f"NACA\n{af}")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    bar_colors = ["steelblue" if r >= 0.99 else
                  "orange"    if r >= 0.97 else "red" for r in r2s]
    ax1.bar(labels, r2s, color=bar_colors, edgecolor="black", lw=0.5)
    ax1.axhline(0.99, color="green", ls="--", lw=1, label="R²=0.99")
    ax1.set_ylim(min(0.95, min(r2s)-0.01), 1.001)
    ax1.set_title(f"{TARGET} — R² per Airfoil")
    ax1.set_ylabel("R²"); ax1.legend(fontsize=7)
    ax2.bar(labels, maes, color="steelblue", edgecolor="black", lw=0.5)
    ax2.set_title(f"{TARGET} — MAE per Airfoil"); ax2.set_ylabel("MAE")
    plt.suptitle(f"PALMO {TARGET} — Per-Airfoil Breakdown", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def save_predictions(y_pred, y_test, test_df, path):
    out = test_df.copy()
    out[f"{TARGET}_CFD"] = y_test
    out[f"{TARGET}_NN"]  = y_pred
    out[f"{TARGET}_err"] = y_pred - y_test
    out.to_csv(path, index=False)
    print(f"Saved: {path}")

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    def out(f): return os.path.join(OUTPUT_DIR, f)

    X_train, y_train_sc, X_test, y_test, scaler_X, scaler_y, test_df = load_data()
    model, tr_loss, val_loss = train_model(X_train, y_train_sc)
    y_pred = evaluate(model, X_test, y_test, scaler_y)

    plot_loss_summary(tr_loss, val_loss, y_pred, y_test,
                      path_curve=out(f"{TARGET.lower()}_training_curve.png"),
                      path_summary=out(f"{TARGET.lower()}_loss_summary.png"))
    plot_scatter(y_pred, y_test,               path=out("cm_predictions_scatter.png"))
    plot_error_distribution(y_pred, y_test,    path=out("cm_error_distribution.png"))
    plot_aoa_curve(y_pred, y_test, test_df,    path=out("cm_aoa_curve.png"))
    plot_multi_mach(y_pred, y_test, test_df,   path=out("cm_multi_mach.png"))
    plot_error_vs_aoa(y_pred, y_test, test_df, path=out("cm_error_vs_aoa.png"))
    plot_per_airfoil(y_pred, y_test, test_df,  path=out("cm_per_airfoil.png"))
    save_predictions(y_pred, y_test, test_df,  path=out("cm_predictions.csv"))

    print(f"\nDone. All {TARGET} outputs saved to:\n  {OUTPUT_DIR}")

if __name__ == "__main__":
    main()