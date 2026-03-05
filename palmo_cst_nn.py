import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import os
warnings.filterwarnings("ignore")

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

try:
    from cst_modeling.section import cst_foil, cst_foil_fit
    CST_LIB = True
    print("cst_modeling3d found — using library CST routines.")
except ImportError:
    CST_LIB = False
    print("WARNING: cst_modeling3d not found.")
    print("  Install with:  pip install cst-modeling3d")
    print("  Falling back to built-in CST implementation (Kulfan 2007).\n")

TRAIN_CSV  = r"c:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Results CST NN\Training_NACA_data.csv (3).xlsx"
TEST_CSV   = r"c:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Results CST NN\Testing_NACA_data.csv (3).xlsx"
OUTPUT_DIR = r"C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Results CST NN"

FLIGHT_COLS = ["Mach", "Re", "Alpha"]
OUTPUT_COLS = ["Cl", "Cd", "Cm"]
AIRFOIL_COL = "NACA"

N_CST         = 7
CST_ORDERS    = [3, 4, 5, 6, 7, 8, 9, 10]

HIDDEN_LAYERS = (200, 200, 200)
LEARNING_RATE = 1e-3
BATCH_SIZE    = 512
MAX_ITER      = 500
PATIENCE      = 30
RANDOM_SEED   = 42
VAL_FRACTION  = 0.10

MACH_PLOT     = 0.25
RE_PLOT       = 1_000_000

def naca4_coords(code, n=201):
    code = int(code)
    m    = (code // 1000)         / 100.0
    p    = ((code % 1000) // 100) / 10.0
    t    = (code % 100)           / 100.0

    beta = np.linspace(0, np.pi, n)
    x    = 0.5 * (1.0 - np.cos(beta))

    yt = 5*t * (0.2969*np.sqrt(x)
                - 0.1260*x
                - 0.3516*x**2
                + 0.2843*x**3
                - 0.1015*x**4)

    if m == 0.0 or p == 0.0:
        yc  = np.zeros_like(x)
        dyc = np.zeros_like(x)
    else:
        yc  = np.where(x < p,
                       m / p**2      * (2*p*x - x**2),
                       m / (1-p)**2  * (1 - 2*p + 2*p*x - x**2))
        dyc = np.where(x < p,
                       2*m / p**2      * (p - x),
                       2*m / (1-p)**2  * (p - x))

    theta = np.arctan(dyc)
    xu = x  - yt * np.sin(theta);  yu = yc + yt * np.cos(theta)
    xl = x  + yt * np.sin(theta);  yl = yc - yt * np.cos(theta)
    return xu, yu, xl, yl

def _bernstein_basis(x, n_cst):
    from math import comb
    C   = np.sqrt(x) * (1.0 - x)
    n   = n_cst - 1
    B   = np.column_stack([comb(n, k) * x**k * (1-x)**(n-k)
                           for k in range(n_cst)])
    return C[:, None] * B

def _fit_cst_builtin(xu, yu, xl, yl, n_cst):
    xu = np.clip(xu, 1e-9, 1-1e-9)
    xl = np.clip(xl, 1e-9, 1-1e-9)
    Pu = _bernstein_basis(xu, n_cst)
    Pl = _bernstein_basis(xl, n_cst)
    Au, _, _, _ = np.linalg.lstsq(Pu, yu, rcond=None)
    Al, _, _, _ = np.linalg.lstsq(Pl, yl, rcond=None)
    return Au, Al

def _reconstruct_cst_builtin(cst_u, cst_l, n_pts=201):
    beta = np.linspace(0, np.pi, n_pts)
    x    = 0.5 * (1.0 - np.cos(beta))
    xc   = np.clip(x, 1e-9, 1-1e-9)
    Pu   = _bernstein_basis(xc, len(cst_u))
    Pl   = _bernstein_basis(xc, len(cst_l))
    return x, Pu @ cst_u, Pl @ cst_l

def fit_cst(xu, yu, xl, yl, n_cst):
    return _fit_cst_builtin(xu, yu, xl, yl, n_cst)

def reconstruct_cst(cst_u, cst_l, n_pts=201):
    if CST_LIB:
        x, yu, yl, _, _ = cst_foil(n_pts, cst_u, cst_l,
                                     x=None, t=None, tail=0.0)
        return x, yu, yl
    return _reconstruct_cst_builtin(cst_u, cst_l, n_pts)

def get_cst_params(naca_code, n_cst=N_CST):
    xu, yu, xl, yl = naca4_coords(naca_code)
    return fit_cst(xu, yu, xl, yl, n_cst)

def plot_cst_convergence(all_codes, orders=CST_ORDERS, path="cst_convergence.png"):
    print("Running CST convergence study...")
    n_eval = 501
    beta   = np.linspace(0, np.pi, n_eval)
    x_eval = 0.5 * (1.0 - np.cos(beta))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_codes)))

    summary = {}
    for color, code in zip(colors, all_codes):
        xu, yu, xl, yl = naca4_coords(code, n=n_eval)
        rms_list, max_list = [], []

        for n in orders:
            cst_u, cst_l   = fit_cst(xu, yu, xl, yl, n)
            x_r, yu_r, yl_r = reconstruct_cst(cst_u, cst_l, n_pts=n_eval)

            yu_true = np.interp(x_r, xu, yu)
            yl_true = np.interp(x_r, xl, yl)
            err     = np.concatenate([np.abs(yu_r - yu_true),
                                      np.abs(yl_r - yl_true)])
            rms_list.append(np.sqrt(np.mean(err**2)))
            max_list.append(np.max(err))

        summary[code] = dict(orders=orders, rms=rms_list, max=max_list)
        label = f"NACA {code}"
        axes[0].semilogy(orders, rms_list, "o-", color=color, lw=1.5, ms=5, label=label)
        axes[1].semilogy(orders, max_list,  "o-", color=color, lw=1.5, ms=5, label=label)

    for ax, title, ylab in zip(
            axes,
            ["RMS Reconstruction Error vs CST Order",
             "Max Reconstruction Error vs CST Order"],
            ["RMS |y_true − y_CST|  (y/c)",
             "Max |y_true − y_CST|  (y/c)"]):
        ax.set_xlabel("Number of CST Parameters per Surface", fontsize=11)
        ax.set_ylabel(ylab, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(orders)
        ax.axvline(N_CST, color="crimson", linestyle="--", lw=1.5,
                   label=f"N_CST = {N_CST} (used)")
        ax.legend(fontsize=6.5, ncol=2, loc="upper right")
        ax.grid(True, which="both", alpha=0.3)

    plt.suptitle("CST Convergence Study — All Airfoils in PALMO Database",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return summary

def plot_true_vs_cst_grid(all_codes, n_cst=N_CST,
                           path="cst_true_vs_reconstructed.png"):
    print("Plotting true vs CST reconstruction grid...")
    ncols  = 4
    nrows  = int(np.ceil(len(all_codes) / ncols))
    n_eval = 401

    fig, axes = plt.subplots(nrows * 2, ncols,
                              figsize=(4.5*ncols, 3.8*nrows),
                              gridspec_kw={"hspace": 0.55, "wspace": 0.35})

    for idx, code in enumerate(all_codes):
        r_shape = (idx // ncols) * 2
        c       = idx % ncols
        ax_s    = axes[r_shape][c]
        ax_r    = axes[r_shape + 1][c]

        xu, yu, xl, yl   = naca4_coords(code, n=n_eval)
        cst_u, cst_l     = get_cst_params(code, n_cst)
        x_r, yu_r, yl_r  = reconstruct_cst(cst_u, cst_l, n_pts=n_eval)

        yu_true = np.interp(x_r, xu, yu)
        yl_true = np.interp(x_r, xl, yl)
        res_u   = yu_r - yu_true
        res_l   = yl_r - yl_true
        rms     = np.sqrt(np.mean(np.concatenate([res_u, res_l])**2))

        ax_s.plot(xu, yu, "k-",  lw=1.8)
        ax_s.plot(xl, yl, "k-",  lw=1.8, label="True NACA")
        ax_s.plot(x_r, yu_r, "r--", lw=1.1, label=f"CST n={n_cst}")
        ax_s.plot(x_r, yl_r, "r--", lw=1.1)
        ax_s.set_title(f"NACA {code}\nRMS = {rms:.2e}", fontsize=7.5)
        ax_s.set_aspect("equal"); ax_s.axis("off")
        if idx == 0:
            ax_s.legend(fontsize=6, loc="upper right")

        ax_r.plot(x_r, res_u,  color="steelblue", lw=1.1, label="Upper")
        ax_r.plot(x_r, res_l,  color="tomato",    lw=1.1, label="Lower")
        ax_r.axhline(0, color="k", lw=0.7, ls="--")
        ax_r.set_xlim(0, 1)
        ax_r.set_xlabel("x/c", fontsize=6)
        ax_r.set_ylabel("Δy/c", fontsize=6)
        ax_r.tick_params(labelsize=5.5)
        ax_r.grid(True, alpha=0.25)
        if idx == 0:
            ax_r.legend(fontsize=6)

    for idx in range(len(all_codes), nrows * ncols):
        r = (idx // ncols) * 2;  c = idx % ncols
        axes[r][c].axis("off"); axes[r+1][c].axis("off")

    plt.suptitle(f"True NACA vs CST Reconstruction  (n_cst = {n_cst} per surface)",
                 fontsize=13, fontweight="bold")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

def plot_true_vs_cst_individual(all_codes, n_cst=N_CST, out_dir=OUTPUT_DIR):
    print("Plotting individual true vs CST pages...")
    n_eval = 501

    for code in all_codes:
        xu, yu, xl, yl   = naca4_coords(code, n=n_eval)
        cst_u, cst_l     = get_cst_params(code, n_cst)
        x_r, yu_r, yl_r  = reconstruct_cst(cst_u, cst_l, n_pts=n_eval)

        yu_true = np.interp(x_r, xu, yu)
        yl_true = np.interp(x_r, xl, yl)
        res_u   = yu_r - yu_true
        res_l   = yl_r - yl_true
        all_err = np.concatenate([res_u, res_l])
        rms     = np.sqrt(np.mean(all_err**2))
        max_err = np.max(np.abs(all_err))

        fig = plt.figure(figsize=(15, 8))
        gs  = gridspec.GridSpec(2, 3, figure=fig,
                                hspace=0.45, wspace=0.38)
        ax_foil  = fig.add_subplot(gs[0, :2])
        ax_res_u = fig.add_subplot(gs[1, 0])
        ax_res_l = fig.add_subplot(gs[1, 1])
        ax_table = fig.add_subplot(gs[:, 2])

        ax_foil.plot(xu, yu, "k-",  lw=2.2, label="True NACA coords")
        ax_foil.plot(xl, yl, "k-",  lw=2.2)
        ax_foil.plot(x_r, yu_r, "r--", lw=1.6,
                     label=f"CST reconstruction  (n = {n_cst})")
        ax_foil.plot(x_r, yl_r, "r--", lw=1.6)
        ax_foil.fill_between(x_r, yu_true, yu_r,
                              alpha=0.15, color="red", label="Error region")
        ax_foil.fill_between(x_r, yl_true, yl_r,
                              alpha=0.15, color="red")
        ax_foil.set_xlabel("x/c", fontsize=11)
        ax_foil.set_ylabel("y/c", fontsize=11)
        ax_foil.set_title(
            f"NACA {code}   |   RMS error = {rms:.3e}   |   Max error = {max_err:.3e}",
            fontsize=11)
        ax_foil.legend(fontsize=9)
        ax_foil.grid(True, alpha=0.3)
        ax_foil.set_aspect("equal")

        ax_res_u.plot(x_r, res_u, color="steelblue", lw=1.4)
        ax_res_u.fill_between(x_r, res_u, 0, alpha=0.2, color="steelblue")
        ax_res_u.axhline(0, color="k", lw=0.9, ls="--")
        ax_res_u.set_xlabel("x/c"); ax_res_u.set_ylabel("Δy/c")
        ax_res_u.set_title("Upper surface residual  (CST − True)")
        ax_res_u.grid(True, alpha=0.3)
        rms_u = np.sqrt(np.mean(res_u**2))
        ax_res_u.annotate(f"RMS = {rms_u:.2e}", xy=(0.97, 0.95),
                           xycoords="axes fraction", ha="right", va="top",
                           fontsize=8, color="steelblue")

        ax_res_l.plot(x_r, res_l, color="tomato", lw=1.4)
        ax_res_l.fill_between(x_r, res_l, 0, alpha=0.2, color="tomato")
        ax_res_l.axhline(0, color="k", lw=0.9, ls="--")
        ax_res_l.set_xlabel("x/c"); ax_res_l.set_ylabel("Δy/c")
        ax_res_l.set_title("Lower surface residual  (CST − True)")
        ax_res_l.grid(True, alpha=0.3)
        rms_l = np.sqrt(np.mean(res_l**2))
        ax_res_l.annotate(f"RMS = {rms_l:.2e}", xy=(0.97, 0.95),
                           xycoords="axes fraction", ha="right", va="top",
                           fontsize=8, color="tomato")

        ax_table.axis("off")
        rows = [[str(i), f"{u:.6f}", f"{l:.6f}"]
                for i, (u, l) in enumerate(zip(cst_u, cst_l))]
        tbl = ax_table.table(
            cellText=rows,
            colLabels=["i", "A_upper[i]", "A_lower[i]"],
            loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9.5)
        tbl.scale(1.1, 1.8)
        ax_table.set_title(f"Fitted CST coefficients\n(n_cst = {n_cst} per surface)",
                            fontsize=10, pad=12)

        fig.suptitle(f"NACA {code} — True Geometry vs CST Reconstruction",
                     fontsize=14, fontweight="bold")

        fname = os.path.join(out_dir, f"cst_recon_NACA{code}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fname}")

def build_cst_dataframe(df, n_cst=N_CST):
    u_cols = [f"cst_u{i}" for i in range(n_cst)]
    l_cols = [f"cst_l{i}" for i in range(n_cst)]
    cache  = {}

    for code in sorted(df[AIRFOIL_COL].unique()):
        cst_u, cst_l = get_cst_params(code, n_cst)
        cache[code]  = np.concatenate([cst_u, cst_l])
        print(f"  NACA {code:4d} | "
              f"cst_u = {np.round(cst_u, 5)} | "
              f"cst_l = {np.round(cst_l, 5)}")

    cst_array = np.array([cache[c] for c in df[AIRFOIL_COL]])
    for i, col in enumerate(u_cols + l_cols):
        df[col] = cst_array[:, i]

    return df, u_cols + l_cols

def load_data(n_cst=N_CST):
    print("Loading Excel data...")
    train_df = pd.read_excel(TRAIN_CSV)
    test_df  = pd.read_excel(TEST_CSV)
    print(f"  Training samples : {len(train_df):,}")
    print(f"  Testing  samples : {len(test_df):,}\n")

    print("Fitting CST parameters — training airfoils:")
    train_df, cst_cols = build_cst_dataframe(train_df, n_cst)
    print("\nFitting CST parameters — test airfoils:")
    test_df, _         = build_cst_dataframe(test_df,  n_cst)

    input_cols = FLIGHT_COLS + cst_cols
    print(f"\nFinal model inputs ({len(input_cols)}): {input_cols}")
    print(f"  Flight conditions : {FLIGHT_COLS}")
    print(f"  Geometry (CST)    : {n_cst} upper + {n_cst} lower = {2*n_cst} params\n")

    X_train = train_df[input_cols].values.astype(np.float64)
    y_train = train_df[OUTPUT_COLS].values.astype(np.float64)
    X_test  = test_df[input_cols].values.astype(np.float64)
    y_test  = test_df[OUTPUT_COLS].values.astype(np.float64)

    scaler_X       = StandardScaler()
    X_train        = scaler_X.fit_transform(X_train)
    X_test         = scaler_X.transform(X_test)

    scaler_y       = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)

    return (X_train, y_train_scaled, X_test, y_test,
            scaler_X, scaler_y, test_df, input_cols)

def train_model(X_train, y_train_scaled):
    print(f"Training PALMO CST-NN")
    print(f"  Architecture  : {HIDDEN_LAYERS}  ReLU  Adam  lr={LEARNING_RATE}")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Max epochs    : {MAX_ITER}  |  Early stop patience: {PATIENCE}")
    print(f"  Train / Val   : "
          f"{int(len(X_train)*(1-VAL_FRACTION)):,} / "
          f"{int(len(X_train)*VAL_FRACTION):,} samples\n")

    rng = np.random.default_rng(RANDOM_SEED)
    n_val = max(1, int(VAL_FRACTION * len(X_train)))
    idx   = rng.permutation(len(X_train))
    tr_idx, val_idx = idx[n_val:], idx[:n_val]
    Xtr, ytr = X_train[tr_idx], y_train_scaled[tr_idx]
    Xvl, yvl = X_train[val_idx], y_train_scaled[val_idx]

    model = MLPRegressor(
        hidden_layer_sizes = HIDDEN_LAYERS,
        activation         = "relu",
        solver             = "adam",
        learning_rate_init = LEARNING_RATE,
        batch_size         = BATCH_SIZE,
        max_iter           = 1,
        warm_start         = True,
        tol                = 0.0,
        n_iter_no_change   = MAX_ITER,
        random_state       = RANDOM_SEED,
        verbose            = False,
    )

    train_losses, val_losses                 = [], []
    best_val, no_improve                     = float("inf"), 0
    best_coefs, best_intercepts              = None, None

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
                  f"(best val = {best_val:.6f})")
            break

    model.coefs_      = best_coefs
    model.intercepts_ = best_intercepts
    return model, train_losses, val_losses

def evaluate(model, X_test, y_test, scaler_y):
    y_pred = scaler_y.inverse_transform(model.predict(X_test))
    print("\n" + "=" * 58)
    print("TEST SET PERFORMANCE  (CST-parametrized NN)")
    print("=" * 58)
    results = {}
    for i, col in enumerate(OUTPUT_COLS):
        mae  = mean_absolute_error(y_test[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        r2   = r2_score(y_test[:, i], y_pred[:, i])
        print(f"  {col:4s}  MAE={mae:.5f}  RMSE={rmse:.5f}  R²={r2:.5f}")
        results[col] = dict(mae=mae, rmse=rmse, r2=r2)
    print("=" * 58 + "\n")
    return y_pred, results

def plot_training_curve(train_losses, val_losses, path):
    plt.figure(figsize=(9, 4))
    plt.plot(train_losses, label="Train MSE (scaled)", lw=1.8)
    plt.plot(val_losses,   label="Val MSE (scaled)",   lw=1.8)
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title("PALMO CST-NN — Training Curve")
    plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")

def plot_predictions(y_pred, y_test, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col, i in zip(axes, OUTPUT_COLS, range(3)):
        lo = min(y_test[:, i].min(), y_pred[:, i].min())
        hi = max(y_test[:, i].max(), y_pred[:, i].max())
        ax.scatter(y_test[:, i], y_pred[:, i], s=1, alpha=0.2, color="steelblue")
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Ideal")
        ax.set_xlabel(f"CFD {col}"); ax.set_ylabel(f"NN {col}")
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        ax.set_title(f"{col}  (R² = {r2:.4f})")
        ax.legend(fontsize=8)
    plt.suptitle("PALMO CST-NN — Predicted vs CFD (Test Set)", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def plot_lift_curves(y_pred, y_test, test_df, mach=MACH_PLOT,
                     reynolds=RE_PLOT, path="cst_lift_curves.png"):
    airfoils = sorted(test_df[AIRFOIL_COL].unique())
    fig, axes = plt.subplots(1, len(airfoils),
                              figsize=(5*len(airfoils), 5), sharey=True)
    if len(airfoils) == 1: axes = [axes]
    for ax, af in zip(axes, airfoils):
        mask = ((np.abs(test_df["Mach"].values - mach) < 1e-4) &
                (np.abs(test_df["Re"].values - reynolds) < reynolds*0.05) &
                (test_df[AIRFOIL_COL].values == af))
        if mask.sum() == 0: ax.set_title(f"NACA {af}\n(no data)"); continue
        aoa = test_df.loc[mask, "Alpha"].values; s = np.argsort(aoa)
        ci  = OUTPUT_COLS.index("Cl")
        ax.plot(aoa[s], y_test[mask, ci][s], "k-",  lw=2.0,
                label="OVERFLOW CFD")
        ax.plot(aoa[s], y_pred[mask, ci][s], "gs", ms=5,
                markerfacecolor="none", markeredgewidth=1.2,
                label="CST-NN Prediction")
        ax.set_xlabel("Angle of Attack (°)")
        ax.set_title(f"NACA {af}\nM={mach}, Re={reynolds:.0e}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("$c_l$")
    plt.suptitle("PALMO CST-NN — Lift Curves (Test Airfoils)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def plot_drag_polars(y_pred, y_test, test_df, mach=MACH_PLOT,
                     reynolds=RE_PLOT, path="cst_drag_polars.png"):
    airfoils = sorted(test_df[AIRFOIL_COL].unique())
    fig, axes = plt.subplots(1, len(airfoils),
                              figsize=(5*len(airfoils), 5), sharey=True)
    if len(airfoils) == 1: axes = [axes]
    for ax, af in zip(axes, airfoils):
        mask = ((np.abs(test_df["Mach"].values - mach) < 1e-4) &
                (np.abs(test_df["Re"].values - reynolds) < reynolds*0.05) &
                (test_df[AIRFOIL_COL].values == af))
        if mask.sum() == 0: ax.set_title(f"NACA {af}\n(no data)"); continue
        ci_cl = OUTPUT_COLS.index("Cl"); ci_cd = OUTPUT_COLS.index("Cd")
        cl_c  = y_test[mask, ci_cl]; cd_c = y_test[mask, ci_cd]
        cl_n  = y_pred[mask, ci_cl]; cd_n = y_pred[mask, ci_cd]
        s     = np.argsort(cl_c)
        ax.plot(cd_c[s], cl_c[s], "k-",  lw=2.0, label="OVERFLOW CFD")
        ax.plot(cd_n[s], cl_n[s], "gs", ms=5,
                markerfacecolor="none", markeredgewidth=1.2,
                label="CST-NN Prediction")
        ax.set_xlabel("$c_d$")
        ax.set_title(f"NACA {af}\nM={mach}, Re={reynolds:.0e}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("$c_l$")
    plt.suptitle("PALMO CST-NN — Drag Polars (Test Airfoils)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def plot_aoa_coefficient(y_pred, y_test, test_df, coeff, ylabel,
                          mach=MACH_PLOT, reynolds=RE_PLOT, path=None):
    if path is None: path = f"cst_aoa_vs_{coeff.lower()}.png"
    airfoils = sorted(test_df[AIRFOIL_COL].unique())
    fig, axes = plt.subplots(1, len(airfoils),
                              figsize=(5*len(airfoils), 5), sharey=True)
    if len(airfoils) == 1: axes = [axes]
    idx = OUTPUT_COLS.index(coeff)
    for ax, af in zip(axes, airfoils):
        mask = ((np.abs(test_df["Mach"].values - mach) < 1e-4) &
                (np.abs(test_df["Re"].values - reynolds) < reynolds*0.05) &
                (test_df[AIRFOIL_COL].values == af))
        if mask.sum() == 0: ax.set_title(f"NACA {af}\n(no data)"); continue
        aoa = test_df.loc[mask, "Alpha"].values; s = np.argsort(aoa)
        ax.plot(aoa[s], y_test[mask, idx][s], "k-",  lw=2.0, label="OVERFLOW CFD")
        ax.plot(aoa[s], y_pred[mask, idx][s], "gs", ms=5,
                markerfacecolor="none", markeredgewidth=1.2,
                label="CST-NN Prediction")
        ax.set_xlabel("Angle of Attack (°)")
        ax.set_title(f"NACA {af}\nM={mach}, Re={reynolds:.0e}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(ylabel)
    plt.suptitle(f"PALMO CST-NN — {ylabel} vs AoA (Test Airfoils)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def plot_error_distributions(y_pred, y_test, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, i in zip(axes, OUTPUT_COLS, range(3)):
        err = y_pred[:, i] - y_test[:, i]
        ax.hist(err, bins=80, color="steelblue", edgecolor="none", alpha=0.85)
        ax.axvline(0, color="red", ls="--", lw=1.5)
        ax.set_xlabel(f"NN − CFD  ({col})"); ax.set_ylabel("Count")
        ax.set_title(f"{col}   μ={err.mean():.4f}   σ={err.std():.5f}")
        ax.grid(True, alpha=0.3)
    plt.suptitle("PALMO CST-NN — Prediction Error Distributions", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def plot_per_airfoil_metrics(y_pred, y_test, test_df, path):
    airfoils = sorted(test_df[AIRFOIL_COL].unique())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ci, coeff in enumerate(OUTPUT_COLS):
        ax_r2 = axes[0][ci]; ax_mae = axes[1][ci]
        r2s, maes, labels = [], [], []
        for af in airfoils:
            mask = test_df[AIRFOIL_COL].values == af
            if mask.sum() == 0: continue
            r2s.append(r2_score(y_test[mask, ci], y_pred[mask, ci]))
            maes.append(mean_absolute_error(y_test[mask, ci], y_pred[mask, ci]))
            labels.append(f"NACA\n{af}")
        colors = ["steelblue" if r >= 0.99
                  else "orange" if r >= 0.97
                  else "red" for r in r2s]
        ax_r2.bar(labels, r2s, color=colors, edgecolor="black", lw=0.5)
        ax_r2.axhline(0.99, color="green", ls="--", lw=1, label="R²=0.99")
        ax_r2.set_ylim(min(0.95, min(r2s)-0.01), 1.001)
        ax_r2.set_title(f"{coeff} — R² per Airfoil")
        ax_r2.set_ylabel("R²"); ax_r2.legend(fontsize=7)
        ax_mae.bar(labels, maes, color="steelblue", edgecolor="black", lw=0.5)
        ax_mae.set_title(f"{coeff} — MAE per Airfoil")
        ax_mae.set_ylabel("MAE")
    plt.suptitle("PALMO CST-NN — Per-Airfoil Performance Breakdown", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def plot_error_vs_aoa(y_pred, y_test, test_df, mach=MACH_PLOT,
                       reynolds=RE_PLOT, path="cst_error_vs_aoa.png"):
    airfoils = sorted(test_df[AIRFOIL_COL].unique())
    fig, axes = plt.subplots(3, len(airfoils),
                              figsize=(5*len(airfoils), 10), sharey="row")
    for ci, coeff in enumerate(OUTPUT_COLS):
        for ai, af in enumerate(airfoils):
            ax = axes[ci][ai]
            mask = ((np.abs(test_df["Mach"].values - mach) < 1e-4) &
                    (np.abs(test_df["Re"].values - reynolds) < reynolds*0.05) &
                    (test_df[AIRFOIL_COL].values == af))
            if mask.sum() == 0: continue
            aoa = test_df.loc[mask, "Alpha"].values
            err = y_pred[mask, ci] - y_test[mask, ci]
            s   = np.argsort(aoa)
            ax.plot(aoa[s], err[s], "k-o", ms=3, lw=1.2)
            ax.axhline(0, color="red", ls="--", lw=1)
            ax.fill_between(aoa[s], err[s], 0, alpha=0.15, color="steelblue")
            ax.set_xlabel("AoA (°)")
            ax.grid(True, alpha=0.3)
            if ai == 0: ax.set_ylabel(f"{coeff} error\n(NN − CFD)")
            if ci == 0: ax.set_title(f"NACA {af}")
    plt.suptitle(f"PALMO CST-NN — Error vs AoA  (M={mach}, Re={reynolds:.0e})",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def plot_multi_mach(y_pred, y_test, test_df, coeff, ylabel,
                     machs=(0.25, 0.50, 0.75), reynolds=RE_PLOT, path=None):
    if path is None: path = f"cst_multi_mach_{coeff.lower()}.png"
    airfoils = sorted(test_df[AIRFOIL_COL].unique())
    colors   = ["tab:blue", "tab:orange", "tab:green", "tab:red",
                 "tab:purple", "tab:brown"]
    idx = OUTPUT_COLS.index(coeff)
    fig, axes = plt.subplots(1, len(airfoils),
                              figsize=(5*len(airfoils), 5), sharey=True)
    if len(airfoils) == 1: axes = [axes]
    for ax, af in zip(axes, airfoils):
        for mach, col in zip(machs, colors):
            mask = ((np.abs(test_df["Mach"].values - mach) < 1e-4) &
                    (np.abs(test_df["Re"].values - reynolds) < reynolds*0.05) &
                    (test_df[AIRFOIL_COL].values == af))
            if mask.sum() == 0: continue
            aoa = test_df.loc[mask, "Alpha"].values; s = np.argsort(aoa)
            ax.plot(aoa[s], y_test[mask, idx][s], "-",
                    color=col, lw=2.0, label=f"CFD M={mach}")
            ax.plot(aoa[s], y_pred[mask, idx][s], "s",
                    color=col, ms=4, markerfacecolor="none",
                    markeredgewidth=1.2, label=f"NN M={mach}")
        ax.set_xlabel("Angle of Attack (°)")
        ax.set_title(f"NACA {af}")
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(ylabel)
    plt.suptitle(
        f"PALMO CST-NN — {ylabel} vs AoA  Multiple Mach  (Re={reynolds:.0e})",
        fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def save_per_testcase_errors(y_pred, y_test, test_df, path):
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
                row = {"Airfoil": af, "Mach": mach, "Re": re,
                       "N_points": mask.sum()}
                for i, col in enumerate(OUTPUT_COLS):
                    row[f"{col}_MAE"] = round(
                        mean_absolute_error(y_test[mask, i], y_pred[mask, i]), 6)
                    row[f"{col}_RMSE"] = round(
                        np.sqrt(mean_squared_error(y_test[mask, i], y_pred[mask, i])), 6)
                records.append(row)
    df = pd.DataFrame(records)
    df.to_csv(path, index=False)

    print("\n" + "=" * 78)
    print("PER TEST CASE ERRORS  (Airfoil / Mach / Re)")
    print("=" * 78)
    print(f"{'Airfoil':>8} {'Mach':>6} {'Re':>9} | "
          f"{'Cl_MAE':>8} {'Cl_RMSE':>8} | "
          f"{'Cd_MAE':>8} {'Cd_RMSE':>8} | "
          f"{'Cm_MAE':>8} {'Cm_RMSE':>8}")
    print("-" * 78)
    for _, r in df.iterrows():
        print(f"{int(r.Airfoil):>8} {r.Mach:>6.2f} {int(r.Re):>9} | "
              f"{r.Cl_MAE:>8.5f} {r.Cl_RMSE:>8.5f} | "
              f"{r.Cd_MAE:>8.5f} {r.Cd_RMSE:>8.5f} | "
              f"{r.Cm_MAE:>8.5f} {r.Cm_RMSE:>8.5f}")
    print("=" * 78)
    print(f"Saved: {path}\n")

def save_predictions(y_pred, y_test, test_df, path):
    out = test_df.copy()
    for i, col in enumerate(OUTPUT_COLS):
        out[f"{col}_CFD"] = y_test[:, i]
        out[f"{col}_NN"]  = y_pred[:, i]
        out[f"{col}_err"] = y_pred[:, i] - y_test[:, i]
    out.to_csv(path, index=False)
    print(f"Saved: {path}")

def save_cst_coefficient_table(all_codes, n_cst=N_CST, path="cst_coefficients.csv"):
    rows = []
    for code in sorted(all_codes):
        cst_u, cst_l = get_cst_params(code, n_cst)
        row = {"NACA": code}
        for i in range(n_cst):
            row[f"A_upper_{i}"] = round(cst_u[i], 8)
            row[f"A_lower_{i}"] = round(cst_l[i], 8)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    def out(f): return os.path.join(OUTPUT_DIR, f)

    train_df0 = pd.read_excel(TRAIN_CSV)
    test_df0  = pd.read_excel(TEST_CSV)
    all_codes = sorted(set(train_df0[AIRFOIL_COL].unique()) |
                       set(test_df0[AIRFOIL_COL].unique()))
    train_codes = sorted(train_df0[AIRFOIL_COL].unique())
    test_codes  = sorted(test_df0[AIRFOIL_COL].unique())
    print(f"Training airfoils  : {train_codes}")
    print(f"Test airfoils      : {test_codes}")
    print(f"All airfoils total : {all_codes}\n")

    print("=" * 60)
    print("TASK 5 — CST Convergence Study")
    print("=" * 60)
    plot_cst_convergence(all_codes, orders=CST_ORDERS,
                         path=out("cst_convergence.png"))

    print("\n" + "=" * 60)
    print("TASK 6 — True vs CST Reconstructed Airfoil Plots")
    print("=" * 60)
    plot_true_vs_cst_grid(all_codes, n_cst=N_CST,
                           path=out("cst_true_vs_reconstructed.png"))
    plot_true_vs_cst_individual(all_codes, n_cst=N_CST, out_dir=OUTPUT_DIR)
    save_cst_coefficient_table(all_codes, n_cst=N_CST,
                                path=out("cst_coefficients.csv"))

    print("\n" + "=" * 60)
    print("TASKS 3 & 4 — CST Data Matrix + NN Training")
    print("=" * 60)
    (X_train, y_train_scaled,
     X_test, y_test,
     scaler_X, scaler_y,
     test_df, input_cols) = load_data(N_CST)

    model, train_losses, val_losses = train_model(X_train, y_train_scaled)
    y_pred, results = evaluate(model, X_test, y_test, scaler_y)

    print("\nGenerating all output plots...")
    plot_training_curve(train_losses, val_losses,
                        path=out("cst_training_curve.png"))
    plot_predictions(y_pred, y_test,
                     path=out("cst_predictions_scatter.png"))
    plot_lift_curves(y_pred, y_test, test_df,
                     path=out("cst_lift_curves.png"))
    plot_drag_polars(y_pred, y_test, test_df,
                     path=out("cst_drag_polars.png"))
    plot_aoa_coefficient(y_pred, y_test, test_df, "Cl", "$c_l$",
                         path=out("cst_aoa_vs_cl.png"))
    plot_aoa_coefficient(y_pred, y_test, test_df, "Cd", "$c_d$",
                         path=out("cst_aoa_vs_cd.png"))
    plot_aoa_coefficient(y_pred, y_test, test_df, "Cm", "$c_m$",
                         path=out("cst_aoa_vs_cm.png"))
    plot_error_distributions(y_pred, y_test,
                              path=out("cst_error_distributions.png"))
    plot_per_airfoil_metrics(y_pred, y_test, test_df,
                              path=out("cst_per_airfoil_metrics.png"))
    plot_error_vs_aoa(y_pred, y_test, test_df,
                      path=out("cst_error_vs_aoa.png"))
    plot_multi_mach(y_pred, y_test, test_df, "Cl", "$c_l$",
                    path=out("cst_multi_mach_cl.png"))
    plot_multi_mach(y_pred, y_test, test_df, "Cd", "$c_d$",
                    path=out("cst_multi_mach_cd.png"))

    save_per_testcase_errors(y_pred, y_test, test_df,
                              path=out("cst_per_testcase_errors.csv"))
    save_predictions(y_pred, y_test, test_df,
                     path=out("cst_palmo_predictions.csv"))

    print("\n" + "=" * 60)
    print("ALL DONE")
    print("=" * 60)
    print(f"Output directory : {OUTPUT_DIR}")
    print(f"Model inputs     : {len(input_cols)}  "
          f"({FLIGHT_COLS} + {N_CST*2} CST params)")
    print("\nGeometry outputs (Tasks 5 & 6):")
    print("  cst_convergence.png              RMS/Max error vs CST order")
    print("  cst_true_vs_reconstructed.png    Grid overview all airfoils")
    print("  cst_recon_NACA****.png           One detailed page per airfoil")
    print("  cst_coefficients.csv             All fitted CST coefficients")
    print("\nNeural network outputs (Tasks 3 & 4):")
    print("  cst_training_curve.png")
    print("  cst_predictions_scatter.png      Cl/Cd/Cm predicted vs CFD")
    print("  cst_lift_curves.png")
    print("  cst_drag_polars.png")
    print("  cst_aoa_vs_cl/cd/cm.png")
    print("  cst_error_distributions.png")
    print("  cst_per_airfoil_metrics.png      R² and MAE per test airfoil")
    print("  cst_error_vs_aoa.png")
    print("  cst_multi_mach_cl/cd.png")
    print("  cst_per_testcase_errors.csv      MAE/RMSE all 320 test cases")
    print("  cst_palmo_predictions.csv        Full prediction table")

if __name__ == "__main__":
    main()