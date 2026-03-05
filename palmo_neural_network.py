import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings("ignore")

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# CONFIGURATION
TRAIN_CSV   = r"C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Results\Training_NACA_data.csv (3).xlsx"
TEST_CSV    = r"C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Results\Testing_NACA_data.csv (3).xlsx"
OUTPUT_DIR  = r"C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Results"

INPUT_COLS  = ["Mach", "Re", "Alpha", "Thickness", "Camber"]
OUTPUT_COLS = ["Cl", "Cd", "Cm"]
AIRFOIL_COL = "NACA"

HIDDEN_LAYERS   = (200, 200, 200)   
LEARNING_RATE   = 1e-3
BATCH_SIZE      = 512
MAX_ITER        = 500
TOLERANCE       = 1e-6
PATIENCE        = 30               
RANDOM_SEED     = 42
VAL_FRACTION    = 0.10

#  DATA LOADING
def load_data():
    print("Loading data...")
    train_df = pd.read_excel(TRAIN_CSV)
    test_df  = pd.read_excel(TEST_CSV)

    print(f"  Training samples : {len(train_df):,}")
    print(f"  Testing  samples : {len(test_df):,}")
    print(f"  Columns          : {list(train_df.columns)}\n")

    X_train = train_df[INPUT_COLS].values.astype(np.float64)
    y_train = train_df[OUTPUT_COLS].values.astype(np.float64)
    X_test  = test_df[INPUT_COLS].values.astype(np.float64)
    y_test  = test_df[OUTPUT_COLS].values.astype(np.float64)

    # Scale inputs
    scaler_X = StandardScaler()
    X_train  = scaler_X.fit_transform(X_train)
    X_test   = scaler_X.transform(X_test)

    # Scale outputs
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)

    return X_train, y_train_scaled, X_test, y_test, scaler_X, scaler_y, test_df


#MANUAL EARLY-STOPPING TRAINING LOOP

def train_model(X_train, y_train_scaled):
    print(f"Training PALMO-NN  |  architecture: {HIDDEN_LAYERS}  |  Adam lr={LEARNING_RATE}")
    print(f"  {int(len(X_train)*(1-VAL_FRACTION)):,} train  /  {int(len(X_train)*VAL_FRACTION):,} val  samples\n")

    rng = np.random.default_rng(RANDOM_SEED)
    n_val   = max(1, int(VAL_FRACTION * len(X_train)))
    idx     = rng.permutation(len(X_train))
    tr_idx, val_idx = idx[n_val:], idx[:n_val]

    Xtr, ytr = X_train[tr_idx], y_train_scaled[tr_idx]
    Xvl, yvl = X_train[val_idx], y_train_scaled[val_idx]

    model = MLPRegressor(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation="relu",
        solver="adam",
        learning_rate_init=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        max_iter=1,           
        warm_start=True,
        tol=0.0,
        n_iter_no_change=MAX_ITER, 
        random_state=RANDOM_SEED,
        verbose=False,
    )

    train_losses, val_losses = [], []
    best_val   = float("inf")
    best_coefs = None
    best_intercepts = None
    no_improve  = 0

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
            print(f"\n  Early stopping at epoch {epoch}  "
                  f"(best val={best_val:.6f})")
            break

    # Restore best weights
    model.coefs_      = best_coefs
    model.intercepts_ = best_intercepts

    return model, train_losses, val_losses


# EVALUATION
def evaluate(model, X_test, y_test, scaler_y):
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    print("\n" + "=" * 55)
    print("TEST SET PERFORMANCE")
    print("=" * 55)
    results = {}
    for i, col in enumerate(OUTPUT_COLS):
        mae  = mean_absolute_error(y_test[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        r2   = r2_score(y_test[:, i], y_pred[:, i])
        print(f"  {col:4s}  MAE={mae:.5f}  RMSE={rmse:.5f}  R²={r2:.5f}")
        results[col] = dict(mae=mae, rmse=rmse, r2=r2)
    print("=" * 55 + "\n")
    return y_pred, results


# PLOTS
def plot_training_curve(train_losses, val_losses, path="training_curve.png"):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train MSE (scaled)", linewidth=1.5)
    plt.plot(val_losses,   label="Val MSE (scaled)",   linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("PALMO Neural Network – Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_predictions(y_pred, y_test, path="predictions_scatter.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col, i in zip(axes, OUTPUT_COLS, range(3)):
        lo = min(y_test[:, i].min(), y_pred[:, i].min())
        hi = max(y_test[:, i].max(), y_pred[:, i].max())
        ax.scatter(y_test[:, i], y_pred[:, i], s=1, alpha=0.2, color="steelblue")
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Ideal")
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        ax.set_xlabel(f"CFD {col}")
        ax.set_ylabel(f"NN {col}")
        ax.set_title(f"{col}  (R²={r2:.4f})")
        ax.legend(fontsize=8)
    plt.suptitle("PALMO NN – Predicted vs CFD (Test Set)", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_lift_curves(y_pred, y_test, test_df,
                     mach=0.25, reynolds=1_000_000,
                     path="lift_curves.png"):
    """Cl vs AoA for each test airfoil at fixed Mach / Re (Fig. 3 equivalent)."""
    airfoils = sorted(test_df[AIRFOIL_COL].unique())
    fig, axes = plt.subplots(1, len(airfoils), figsize=(5 * len(airfoils), 5), sharey=True)
    if len(airfoils) == 1:
        axes = [axes]

    for ax, af in zip(axes, airfoils):
        mask = (
            (np.abs(test_df["Mach"].values - mach)      < 1e-4) &
            (np.abs(test_df["Re"].values   - reynolds)  < reynolds * 0.05) &
            (test_df[AIRFOIL_COL].values == af)
        )
        n = mask.sum()
        if n == 0:
            ax.set_title(f"NACA {af}\n(no data at M={mach}, Re={reynolds:.0e})")
            continue

        aoa    = test_df.loc[mask, "Alpha"].values
        cl_cfd = y_test[mask, OUTPUT_COLS.index("Cl")]
        cl_nn  = y_pred[mask, OUTPUT_COLS.index("Cl")]
        s      = np.argsort(aoa)

        ax.plot(aoa[s], cl_cfd[s], "k-",   lw=2.0, label="OVERFLOW-CFD Data")
        ax.plot(aoa[s], cl_nn[s],  "gs",   ms=5, markerfacecolor="none",
                markeredgewidth=1.2, label="Surrogate Model Predictions")
        ax.set_xlabel("Angle of Attack (deg)")
        ax.set_title(f"NACA {af}\nM={mach}, Re={reynolds:.0e}")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("$c_l$")
    plt.suptitle("PALMO NN – Lift Curve Comparison (Test Airfoils)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_drag_polars(y_pred, y_test, test_df,
                     mach=0.25, reynolds=1_000_000,
                     path="drag_polars.png"):
    """Drag polar: Cd vs Cl for each test airfoil."""
    airfoils = sorted(test_df[AIRFOIL_COL].unique())
    fig, axes = plt.subplots(1, len(airfoils), figsize=(5 * len(airfoils), 5), sharey=True)
    if len(airfoils) == 1:
        axes = [axes]

    for ax, af in zip(axes, airfoils):
        mask = (
            (np.abs(test_df["Mach"].values - mach)     < 1e-4) &
            (np.abs(test_df["Re"].values   - reynolds) < reynolds * 0.05) &
            (test_df[AIRFOIL_COL].values == af)
        )
        if mask.sum() == 0:
            ax.set_title(f"NACA {af}\n(no data)")
            continue

        cl_cfd = y_test[mask, OUTPUT_COLS.index("Cl")]
        cd_cfd = y_test[mask, OUTPUT_COLS.index("Cd")]
        cl_nn  = y_pred[mask, OUTPUT_COLS.index("Cl")]
        cd_nn  = y_pred[mask, OUTPUT_COLS.index("Cd")]
        s = np.argsort(cl_cfd)

        ax.plot(cd_cfd[s], cl_cfd[s], "k-",  lw=2.0, label="OVERFLOW-CFD Data")
        ax.plot(cd_nn[s],  cl_nn[s],  "gs",  ms=5, markerfacecolor="none",
                markeredgewidth=1.2, label="Surrogate Model Predictions")
        ax.set_xlabel("$c_d$")
        ax.set_title(f"NACA {af}\nM={mach}, Re={reynolds:.0e}")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("$c_l$")
    plt.suptitle("PALMO NN – Drag Polars (Test Airfoils)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_aoa_coefficient(y_pred, y_test, test_df, coeff, ylabel,
                         mach=0.25, reynolds=1_000_000, path=None):
    """AoA vs a single aerodynamic coefficient for each test airfoil."""
    if path is None:
        path = f"aoa_vs_{coeff.lower()}.png"

    airfoils = sorted(test_df[AIRFOIL_COL].unique())
    fig, axes = plt.subplots(1, len(airfoils), figsize=(5 * len(airfoils), 5), sharey=True)
    if len(airfoils) == 1:
        axes = [axes]

    idx = OUTPUT_COLS.index(coeff)

    for ax, af in zip(axes, airfoils):
        mask = (
            (np.abs(test_df["Mach"].values - mach)     < 1e-4) &
            (np.abs(test_df["Re"].values   - reynolds) < reynolds * 0.05) &
            (test_df[AIRFOIL_COL].values == af)
        )
        if mask.sum() == 0:
            ax.set_title(f"NACA {af}\n(no data at M={mach}, Re={reynolds:.0e})")
            continue

        aoa   = test_df.loc[mask, "Alpha"].values
        c_cfd = y_test[mask, idx]
        c_nn  = y_pred[mask, idx]
        s     = np.argsort(aoa)

        ax.plot(aoa[s], c_cfd[s], "k-",  lw=2.0, label="OVERFLOW-CFD Data")
        ax.plot(aoa[s], c_nn[s],  "gs",  ms=5, markerfacecolor="none",
                markeredgewidth=1.2, label="Surrogate Model Predictions")
        ax.set_xlabel("Angle of Attack (deg)")
        ax.set_title(f"NACA {af}\nM={mach}, Re={reynolds:.0e}")
        ax.legend(fontsize=8)

    axes[0].set_ylabel(ylabel)
    plt.suptitle(f"PALMO NN – {ylabel} vs Angle of Attack (Test Airfoils)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_error_distributions(y_pred, y_test, path="error_distributions.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, i in zip(axes, OUTPUT_COLS, range(3)):
        errors = y_pred[:, i] - y_test[:, i]
        ax.hist(errors, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
        ax.axvline(0, color="red", linestyle="--", lw=1.5)
        ax.set_xlabel(f"NN − CFD  ({col})")
        ax.set_ylabel("Count")
        ax.set_title(f"{col}  σ={errors.std():.5f}")
    plt.suptitle("PALMO NN – Prediction Error Distributions", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# DEEPER ANALYSIS PLOTS

def plot_per_airfoil_metrics(y_pred, y_test, test_df, path="per_airfoil_metrics.png"):
    """Bar chart of R², MAE per airfoil per coefficient."""
    airfoils = sorted(test_df[AIRFOIL_COL].unique())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for col_i, coeff in enumerate(OUTPUT_COLS):
        ax_r2  = axes[0][col_i]
        ax_mae = axes[1][col_i]
        r2s, maes, labels = [], [], []

        for af in airfoils:
            mask = test_df[AIRFOIL_COL].values == af
            if mask.sum() == 0:
                continue
            r2  = r2_score(y_test[mask, col_i], y_pred[mask, col_i])
            mae = mean_absolute_error(y_test[mask, col_i], y_pred[mask, col_i])
            r2s.append(r2); maes.append(mae); labels.append(f"NACA\n{af}")

        colors = ["steelblue" if r >= 0.99 else "orange" if r >= 0.97 else "red" for r in r2s]
        ax_r2.bar(labels, r2s, color=colors, edgecolor="black", linewidth=0.5)
        ax_r2.axhline(0.99, color="green", linestyle="--", lw=1, label="R²=0.99")
        ax_r2.set_ylim(min(0.95, min(r2s) - 0.01), 1.001)
        ax_r2.set_title(f"{coeff} — R² per Airfoil")
        ax_r2.set_ylabel("R²"); ax_r2.legend(fontsize=7)

        ax_mae.bar(labels, maes, color="steelblue", edgecolor="black", linewidth=0.5)
        ax_mae.set_title(f"{coeff} — MAE per Airfoil")
        ax_mae.set_ylabel("MAE")

    plt.suptitle("PALMO NN – Per-Airfoil Performance Breakdown", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_error_vs_aoa(y_pred, y_test, test_df, path="error_vs_aoa.png",
                      mach=0.25, reynolds=1_000_000):
    """Plot prediction error (NN - CFD) vs angle of attack for each airfoil."""
    airfoils = sorted(test_df[AIRFOIL_COL].unique())
    fig, axes = plt.subplots(3, len(airfoils),
                             figsize=(5 * len(airfoils), 10), sharey="row")

    for col_i, coeff in enumerate(OUTPUT_COLS):
        for af_i, af in enumerate(airfoils):
            ax = axes[col_i][af_i]
            mask = (
                (np.abs(test_df["Mach"].values - mach)     < 1e-4) &
                (np.abs(test_df["Re"].values   - reynolds) < reynolds * 0.05) &
                (test_df[AIRFOIL_COL].values == af)
            )
            if mask.sum() == 0:
                continue
            aoa   = test_df.loc[mask, "Alpha"].values
            err   = y_pred[mask, col_i] - y_test[mask, col_i]
            s     = np.argsort(aoa)
            ax.plot(aoa[s], err[s], "k-o", ms=3, lw=1.2)
            ax.axhline(0, color="red", linestyle="--", lw=1)
            ax.fill_between(aoa[s], err[s], 0, alpha=0.15, color="steelblue")
            ax.set_xlabel("AoA (deg)")
            if af_i == 0:
                ax.set_ylabel(f"{coeff} error\n(NN − CFD)")
            if col_i == 0:
                ax.set_title(f"NACA {af}")

    plt.suptitle(f"PALMO NN – Prediction Error vs AoA  (M={mach}, Re={reynolds:.0e})",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_multi_mach(y_pred, y_test, test_df, coeff, ylabel,
                    machs=(0.25, 0.50, 0.75), reynolds=1_000_000, path=None):
    """AoA vs coefficient at multiple Mach numbers for each airfoil."""
    if path is None:
        path = f"multi_mach_{coeff.lower()}.png"

    airfoils = sorted(test_df[AIRFOIL_COL].unique())
    colors   = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    idx = OUTPUT_COLS.index(coeff)

    fig, axes = plt.subplots(1, len(airfoils), figsize=(5 * len(airfoils), 5), sharey=True)
    if len(airfoils) == 1:
        axes = [axes]

    for ax, af in zip(axes, airfoils):
        for mach, col in zip(machs, colors):
            mask = (
                (np.abs(test_df["Mach"].values - mach)     < 1e-4) &
                (np.abs(test_df["Re"].values   - reynolds) < reynolds * 0.05) &
                (test_df[AIRFOIL_COL].values == af)
            )
            if mask.sum() == 0:
                continue
            aoa   = test_df.loc[mask, "Alpha"].values
            c_cfd = y_test[mask, idx]
            c_nn  = y_pred[mask, idx]
            s     = np.argsort(aoa)
            ax.plot(aoa[s], c_cfd[s], "-",  color=col, lw=2.0, label=f"CFD M={mach}")
            ax.plot(aoa[s], c_nn[s],  "s",  color=col, ms=4,
                    markerfacecolor="none", markeredgewidth=1.2, label=f"NN M={mach}")

        ax.set_xlabel("Angle of Attack (deg)")
        ax.set_title(f"NACA {af}")
        ax.legend(fontsize=7, ncol=2)

    axes[0].set_ylabel(ylabel)
    plt.suptitle(f"PALMO NN – {ylabel} vs AoA at Multiple Mach Numbers (Re={reynolds:.0e})",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")



def save_per_testcase_errors(y_pred, y_test, test_df, path="per_testcase_errors.csv"):
    """
    Compute MAE and RMSE for every unique (Airfoil, Mach, Re) combination
    as defined by Table 1 in the PALMO paper — 4 airfoils x 10 Mach x 8 Re = 320 cases.
    Each case covers 41 AoA points (-20 to +20 degrees).
    """
    records = []
    airfoils = sorted(test_df[AIRFOIL_COL].unique())
    machs    = sorted(test_df["Mach"].unique())
    reynolds = sorted(test_df["Re"].unique())

    for af in airfoils:
        for mach in machs:
            for re in reynolds:
                mask = (
                    (test_df[AIRFOIL_COL].values == af) &
                    (np.abs(test_df["Mach"].values - mach) < 1e-4) &
                    (test_df["Re"].values == re)
                )
                if mask.sum() == 0:
                    continue

                row = {"Airfoil": af, "Mach": mach, "Re": re, "N_points": mask.sum()}
                for i, col in enumerate(OUTPUT_COLS):
                    mae  = mean_absolute_error(y_test[mask, i], y_pred[mask, i])
                    rmse = np.sqrt(mean_squared_error(y_test[mask, i], y_pred[mask, i]))
                    row[f"{col}_MAE"]  = round(mae,  6)
                    row[f"{col}_RMSE"] = round(rmse, 6)
                records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(path, index=False)

    # Print summary to terminal
    print("\n" + "=" * 75)
    print("PER TEST CASE ERRORS  (Airfoil / Mach / Re)")
    print("=" * 75)
    print(f"{'Airfoil':>8} {'Mach':>6} {'Re':>8} | "
          f"{'Cl_MAE':>8} {'Cl_RMSE':>8} | "
          f"{'Cd_MAE':>8} {'Cd_RMSE':>8} | "
          f"{'Cm_MAE':>8} {'Cm_RMSE':>8}")
    print("-" * 75)
    for _, r in df.iterrows():
        print(f"{int(r.Airfoil):>8} {r.Mach:>6.2f} {int(r.Re):>8} | "
              f"{r.Cl_MAE:>8.5f} {r.Cl_RMSE:>8.5f} | "
              f"{r.Cd_MAE:>8.5f} {r.Cd_RMSE:>8.5f} | "
              f"{r.Cm_MAE:>8.5f} {r.Cm_RMSE:>8.5f}")
    print("=" * 75)
    print(f"Saved: {path}\n")


def save_predictions(y_pred, y_test, test_df, path="palmo_predictions.csv"):
    out = test_df.copy()
    for i, col in enumerate(OUTPUT_COLS):
        out[f"{col}_CFD"] = y_test[:, i]
        out[f"{col}_NN"]  = y_pred[:, i]
        out[f"{col}_err"] = y_pred[:, i] - y_test[:, i]
    out.to_csv(path, index=False)
    print(f"Saved: {path}")


# MAIN

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    def out(filename):
        return os.path.join(OUTPUT_DIR, filename)

    # Load & scale
    (X_train, y_train_scaled,
     X_test,  y_test,
     scaler_X, scaler_y, test_df) = load_data()

    # Train
    model, train_losses, val_losses = train_model(X_train, y_train_scaled)

    # Evaluate
    y_pred, results = evaluate(model, X_test, y_test, scaler_y)

    # Plots
    plot_training_curve(train_losses, val_losses,          path=out("training_curve.png"))
    plot_predictions(y_pred, y_test,                       path=out("predictions_scatter.png"))
    plot_lift_curves(y_pred, y_test, test_df,              path=out("lift_curves.png"))
    plot_drag_polars(y_pred, y_test, test_df,              path=out("drag_polars.png"))
    plot_aoa_coefficient(y_pred, y_test, test_df, "Cl", "$c_l$", path=out("aoa_vs_cl.png"))
    plot_aoa_coefficient(y_pred, y_test, test_df, "Cd", "$c_d$", path=out("aoa_vs_cd.png"))
    plot_aoa_coefficient(y_pred, y_test, test_df, "Cm", "$c_m$", path=out("aoa_vs_cm.png"))
    plot_error_distributions(y_pred, y_test,               path=out("error_distributions.png"))
    plot_per_airfoil_metrics(y_pred, y_test, test_df,      path=out("per_airfoil_metrics.png"))
    plot_error_vs_aoa(y_pred, y_test, test_df,             path=out("error_vs_aoa.png"))
    plot_multi_mach(y_pred, y_test, test_df, "Cl", "$c_l$", path=out("multi_mach_cl.png"))
    plot_multi_mach(y_pred, y_test, test_df, "Cd", "$c_d$", path=out("multi_mach_cd.png"))

    # Save predictions CSV
    save_per_testcase_errors(y_pred, y_test, test_df,      path=out("per_testcase_errors.csv"))
    save_predictions(y_pred, y_test, test_df,              path=out("palmo_predictions.csv"))

    print(f"\nDone. All outputs saved to: {OUTPUT_DIR}")
    print("  training_curve.png, predictions_scatter.png")
    print("  lift_curves.png, drag_polars.png")
    print("  aoa_vs_cl.png, aoa_vs_cd.png, aoa_vs_cm.png")
    print("  error_distributions.png, per_airfoil_metrics.png")
    print("  error_vs_aoa.png, multi_mach_cl.png, multi_mach_cd.png")
    print("  palmo_predictions.csv")


if __name__ == "__main__":
    main()