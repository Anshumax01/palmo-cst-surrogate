
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
TRAIN_CSV   = r"C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Results CST NN\Training_NACA_data.csv (3).xlsx"
TEST_CSV    = r"C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Results CST NN\Testing_NACA_data.csv (3).xlsx"
OUTPUT_DIR  = r"C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Results Hyperparameterization\Cm"

FLIGHT_COLS = ["Mach", "Re", "Alpha"]
TARGET      = "Cm"
AIRFOIL_COL = "NACA"

N_CST       = 7
FIXED_LR    = 1e-3
FIXED_BATCH = 512
MAX_ITER    = 500
PATIENCE    = 30
RANDOM_SEED = 42

MACH_PLOT   = 0.25
RE_PLOT     = 1_000_000

HIDDEN_CONFIGS = [
    (50,  50),
    (100, 100),
    (200, 200),
    (400, 400),
    (50,  50,  50),
    (100, 100, 100),
    (200, 200, 200),        # current baseline
    (400, 400, 400),
    (100, 100, 100, 100),
    (200, 200, 200, 200),
    (400, 400, 400, 400),
    (100, 200, 100),
    (200, 400, 200),
    (50,  100, 200, 100, 50),
]

# ─── CST CORE ─────────────────────────────────────────────────────────────────
def naca4_coords(code, n=201):
    code=int(code); m=(code//1000)/100.0; p=((code%1000)//100)/10.0; t=(code%100)/100.0
    beta=np.linspace(0,np.pi,n); x=0.5*(1.0-np.cos(beta))
    yt=5*t*(0.2969*np.sqrt(x)-0.1260*x-0.3516*x**2+0.2843*x**3-0.1015*x**4)
    if m==0.0 or p==0.0:
        yc=np.zeros_like(x); dyc=np.zeros_like(x)
    else:
        yc =np.where(x<p,m/p**2*(2*p*x-x**2),      m/(1-p)**2*(1-2*p+2*p*x-x**2))
        dyc=np.where(x<p,2*m/p**2*(p-x),            2*m/(1-p)**2*(p-x))
    theta=np.arctan(dyc)
    return x-yt*np.sin(theta),yc+yt*np.cos(theta),x+yt*np.sin(theta),yc-yt*np.cos(theta)

def _bernstein_basis(x, n_cst):
    C=np.sqrt(x)*(1.0-x); n=n_cst-1
    B=np.column_stack([comb(n,k)*x**k*(1-x)**(n-k) for k in range(n_cst)])
    return C[:,None]*B

def fit_cst(xu,yu,xl,yl,n_cst):
    xu=np.clip(xu,1e-9,1-1e-9); xl=np.clip(xl,1e-9,1-1e-9)
    Pu=_bernstein_basis(xu,n_cst); Pl=_bernstein_basis(xl,n_cst)
    Au,_,_,_=np.linalg.lstsq(Pu,yu,rcond=None); Al,_,_,_=np.linalg.lstsq(Pl,yl,rcond=None)
    return Au,Al

def get_cst_params(code):
    xu,yu,xl,yl=naca4_coords(code); return fit_cst(xu,yu,xl,yl,N_CST)

def build_cst_features(df):
    u_cols=[f"cst_u{i}" for i in range(N_CST)]; l_cols=[f"cst_l{i}" for i in range(N_CST)]
    cache={}
    for code in df[AIRFOIL_COL].unique():
        cu,cl=get_cst_params(code); cache[code]=np.concatenate([cu,cl])
    arr=np.array([cache[c] for c in df[AIRFOIL_COL]])
    for i,col in enumerate(u_cols+l_cols): df[col]=arr[:,i]
    return df, u_cols+l_cols

# ─── DATA ─────────────────────────────────────────────────────────────────────
def load_data():
    train_df=pd.read_excel(TRAIN_CSV); test_df=pd.read_excel(TEST_CSV)
    train_df,cst_cols=build_cst_features(train_df.copy())
    test_df,_        =build_cst_features(test_df.copy())
    input_cols=FLIGHT_COLS+cst_cols

    X_train=train_df[input_cols].values.astype(np.float64)
    y_train=train_df[TARGET].values.astype(np.float64).reshape(-1,1)
    X_test =test_df[input_cols].values.astype(np.float64)
    y_test =test_df[TARGET].values.astype(np.float64)

    scaler_X=StandardScaler(); X_train=scaler_X.fit_transform(X_train); X_test=scaler_X.transform(X_test)
    scaler_y=StandardScaler(); y_train_s=scaler_y.fit_transform(y_train).ravel()

    print(f"  Train: {len(X_train):,}   Test: {len(X_test):,}   Inputs: {X_train.shape[1]}")
    return X_train,y_train_s,X_test,y_test,scaler_y,test_df

# ─── TRAIN & EVALUATE ─────────────────────────────────────────────────────────
def train_model(X_train, y_train_s, hidden):
    model=MLPRegressor(
        hidden_layer_sizes=hidden, activation="relu", solver="adam",
        learning_rate_init=FIXED_LR, batch_size=FIXED_BATCH,
        max_iter=MAX_ITER, tol=1e-5,
        n_iter_no_change=PATIENCE, random_state=RANDOM_SEED, verbose=False,
    )
    model.fit(X_train,y_train_s)
    return model, model.loss_curve_

def evaluate(model, X_test, y_test, scaler_y):
    y_pred_s=model.predict(X_test).reshape(-1,1)
    y_pred=scaler_y.inverse_transform(y_pred_s).ravel()
    return y_pred, {
        "r2":   r2_score(y_test,y_pred),
        "mae":  mean_absolute_error(y_test,y_pred),
        "mse":  mean_squared_error(y_test,y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test,y_pred)),
    }

def arch_label(h): return "("+",".join(str(x) for x in h)+")"

# ─── SWEEP PLOTS ──────────────────────────────────────────────────────────────
def plot_arch_r2(df, path):
    labels=[arch_label(tuple(r["hidden"])) for _,r in df.iterrows()]
    x=np.arange(len(df))
    fig,ax=plt.subplots(figsize=(16,6))
    colors=["gold" if tuple(r["hidden"])==(200,200,200) else "steelblue" for _,r in df.iterrows()]
    bars=ax.bar(x,df["r2"],color=colors,edgecolor="black",lw=0.5)
    for bar,val in zip(bars,df["r2"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.00005,
                f"{val:.5f}", ha="center", va="bottom", fontsize=6, rotation=90)
    ax.axhline(0.999,color="red",ls="--",lw=1,label="R²=0.999")
    best_idx=df["r2"].idxmax()
    ax.axvspan(best_idx-0.45,best_idx+0.45,alpha=0.12,color="limegreen",label="Best")
    ax.set_xticks(x); ax.set_xticklabels(labels,rotation=35,ha="right",fontsize=8)
    bot=max(0.97,df["r2"].min()-0.005); ax.set_ylim(bottom=bot)
    ax.set_ylabel("R²"); ax.set_title(f"{TARGET} Architecture Sweep — R²  (N_CST={N_CST}, lr={FIXED_LR}, batch={FIXED_BATCH})",fontsize=12)
    ax.legend(fontsize=9); ax.grid(True,alpha=0.3,axis="y")
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def plot_arch_mae(df, path):
    labels=[arch_label(tuple(r["hidden"])) for _,r in df.iterrows()]
    x=np.arange(len(df))
    fig,ax=plt.subplots(figsize=(16,6))
    colors=["gold" if tuple(r["hidden"])==(200,200,200) else "tomato" for _,r in df.iterrows()]
    ax.bar(x,df["mae"],color=colors,edgecolor="black",lw=0.5)
    best_idx=df["mae"].idxmin()
    ax.axvspan(best_idx-0.45,best_idx+0.45,alpha=0.12,color="limegreen",label="Best (lowest MAE)")
    ax.set_xticks(x); ax.set_xticklabels(labels,rotation=35,ha="right",fontsize=8)
    ax.set_ylabel("MAE"); ax.set_title(f"{TARGET} Architecture Sweep — MAE",fontsize=12)
    ax.legend(fontsize=9); ax.grid(True,alpha=0.3,axis="y")
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def plot_arch_rmse(df, path):
    labels=[arch_label(tuple(r["hidden"])) for _,r in df.iterrows()]
    x=np.arange(len(df))
    fig,ax=plt.subplots(figsize=(16,6))
    colors=["gold" if tuple(r["hidden"])==(200,200,200) else "seagreen" for _,r in df.iterrows()]
    ax.bar(x,df["rmse"],color=colors,edgecolor="black",lw=0.5)
    best_idx=df["rmse"].idxmin()
    ax.axvspan(best_idx-0.45,best_idx+0.45,alpha=0.12,color="limegreen",label="Best (lowest RMSE)")
    ax.set_xticks(x); ax.set_xticklabels(labels,rotation=35,ha="right",fontsize=8)
    ax.set_ylabel("RMSE"); ax.set_title(f"{TARGET} Architecture Sweep — RMSE",fontsize=12)
    ax.legend(fontsize=9); ax.grid(True,alpha=0.3,axis="y")
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def plot_arch_top(df, path):
    top=df.nlargest(min(10,len(df)),"r2").reset_index(drop=True)
    labels=[arch_label(tuple(r["hidden"])) for _,r in top.iterrows()]
    fig,ax=plt.subplots(figsize=(14,6))
    ax.bar(range(len(top)),top["r2"],color="seagreen",edgecolor="black",lw=0.5)
    ax.set_xticks(range(len(top))); ax.set_xticklabels(labels,rotation=30,ha="right",fontsize=8)
    ax.set_ylabel("R²"); bot=max(0.97,top["r2"].min()-0.003); ax.set_ylim(bottom=bot)
    ax.set_title(f"{TARGET} — Top Configurations by R²",fontsize=12)
    ax.grid(True,alpha=0.3,axis="y")
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def plot_depth_sensitivity(df, path):
    depths=sorted(df["depth"].unique()); colors=plt.cm.tab10(np.linspace(0,1,len(depths)))
    fig,ax=plt.subplots(figsize=(8,5))
    means=[df[df["depth"]==d]["r2"].mean() for d in depths]
    stds =[df[df["depth"]==d]["r2"].std()  for d in depths]
    ax.bar([str(d) for d in depths],means,yerr=stds,color=colors,edgecolor="black",lw=0.6,capsize=4)
    ax.set_xlabel("Depth (# layers)"); ax.set_ylabel("Mean R²")
    ax.set_title(f"{TARGET} — R² vs Network Depth"); ax.grid(True,alpha=0.3,axis="y")
    bot=max(0.97,min(means)-0.01); ax.set_ylim(bottom=bot)
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def plot_width_sensitivity(df, path):
    widths=sorted(df["width"].unique()); colors=plt.cm.tab10(np.linspace(0,1,len(widths)))
    fig,ax=plt.subplots(figsize=(8,5))
    means=[df[df["width"]==w]["r2"].mean() for w in widths]
    stds =[df[df["width"]==w]["r2"].std()  for w in widths]
    ax.bar([str(w) for w in widths],means,yerr=stds,color=colors,edgecolor="black",lw=0.6,capsize=4)
    ax.set_xlabel("Width (neurons/layer)"); ax.set_ylabel("Mean R²")
    ax.set_title(f"{TARGET} — R² vs Layer Width"); ax.grid(True,alpha=0.3,axis="y")
    bot=max(0.97,min(means)-0.01); ax.set_ylim(bottom=bot)
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def plot_training_time(df, path):
    labels=[arch_label(tuple(r["hidden"])) for _,r in df.iterrows()]
    colors=["gold" if tuple(r["hidden"])==(200,200,200) else "steelblue" for _,r in df.iterrows()]
    fig,ax=plt.subplots(figsize=(16,5))
    ax.bar(range(len(df)),df["train_time_s"],color=colors,edgecolor="black",lw=0.5)
    ax.set_xticks(range(len(df))); ax.set_xticklabels(labels,rotation=35,ha="right",fontsize=8)
    ax.set_ylabel("Training Time (s)"); ax.set_title(f"{TARGET} Architecture Sweep — Training Time  (gold=baseline)")
    ax.grid(True,alpha=0.3,axis="y")
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

# ─── BEST MODEL PLOTS ─────────────────────────────────────────────────────────
def plot_training_curve(train_losses, path):
    fig,ax=plt.subplots(figsize=(9,5))
    ax.plot(train_losses,lw=1.5,color="seagreen",label="Train Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (MSE, scaled)")
    ax.set_title(f"Best {TARGET} Model — Training Curve"); ax.legend(); ax.grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def plot_scatter(y_pred, y_test, path):
    r2=r2_score(y_test,y_pred)
    lo=min(y_test.min(),y_pred.min()); hi=max(y_test.max(),y_pred.max())
    fig,ax=plt.subplots(figsize=(6,6))
    ax.scatter(y_test,y_pred,s=1,alpha=0.2,color="seagreen")
    ax.plot([lo,hi],[lo,hi],"r--",lw=1.5,label="Ideal")
    ax.set_xlabel(f"CFD {TARGET}"); ax.set_ylabel(f"NN {TARGET}")
    ax.set_title(f"{TARGET}  R²={r2:.5f}"); ax.legend(); ax.grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def plot_error_distribution(y_pred, y_test, path):
    errors=y_pred-y_test
    fig,ax=plt.subplots(figsize=(7,4))
    ax.hist(errors,bins=80,color="seagreen",edgecolor="none",alpha=0.85)
    ax.axvline(0,color="red",ls="--",lw=1.5)
    ax.axvline(errors.mean(),color="orange",ls="--",lw=1.2,label=f"μ={errors.mean():.5f}")
    ax.set_xlabel(f"NN − CFD ({TARGET})"); ax.set_ylabel("Count")
    ax.set_title(f"{TARGET}  μ={errors.mean():.5f}  σ={errors.std():.5f}")
    ax.legend(); ax.grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def plot_aoa_curve(y_pred, y_test, test_df, path):
    airfoils=sorted(test_df[AIRFOIL_COL].unique())
    fig,axes=plt.subplots(1,len(airfoils),figsize=(5*len(airfoils),5),sharey=True)
    if len(airfoils)==1: axes=[axes]
    for ax,af in zip(axes,airfoils):
        mask=((np.abs(test_df["Mach"].values-MACH_PLOT)<1e-4)&
              (np.abs(test_df["Re"].values-RE_PLOT)<RE_PLOT*0.05)&
              (test_df[AIRFOIL_COL].values==af))
        if mask.sum()==0: continue
        aoa=test_df.loc[mask,"Alpha"].values; s=np.argsort(aoa)
        ax.plot(aoa[s],y_test[mask][s],"k-",lw=2.0,label="CFD")
        ax.plot(aoa[s],y_pred[mask][s],"gs",ms=5,markerfacecolor="none",markeredgewidth=1.2,label="Best NN")
        ax.set_xlabel("AoA (°)"); ax.set_title(f"NACA {af}\nM={MACH_PLOT}, Re={RE_PLOT:.0e}")
        ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
    axes[0].set_ylabel(TARGET)
    plt.suptitle(f"Best {TARGET} Model — AoA Curve",fontsize=13,y=1.02)
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def plot_per_airfoil(y_pred, y_test, test_df, path):
    airfoils=sorted(test_df[AIRFOIL_COL].unique())
    r2s,maes,labels=[],[],[]
    for af in airfoils:
        mask=test_df[AIRFOIL_COL].values==af
        r2s.append(r2_score(y_test[mask],y_pred[mask]))
        maes.append(mean_absolute_error(y_test[mask],y_pred[mask]))
        labels.append(f"NACA\n{af}")
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4))
    bar_colors=["steelblue" if r>=0.99 else "orange" if r>=0.97 else "red" for r in r2s]
    ax1.bar(labels,r2s,color=bar_colors,edgecolor="black",lw=0.5)
    ax1.axhline(0.99,color="green",ls="--",lw=1,label="R²=0.99")
    ax1.set_ylim(min(0.95,min(r2s)-0.01),1.001); ax1.set_ylabel("R²")
    ax1.set_title(f"{TARGET} — R² per Airfoil"); ax1.legend(fontsize=7); ax1.grid(True,alpha=0.3,axis="y")
    ax2.bar(labels,maes,color="seagreen",edgecolor="black",lw=0.5)
    ax2.set_ylabel("MAE"); ax2.set_title(f"{TARGET} — MAE per Airfoil"); ax2.grid(True,alpha=0.3,axis="y")
    plt.suptitle(f"Best {TARGET} Model — Per-Airfoil Breakdown",fontsize=13)
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def save_per_testcase(y_pred, y_test, test_df, path):
    records=[]; airfoils=sorted(test_df[AIRFOIL_COL].unique())
    machs=sorted(test_df["Mach"].unique()); reynolds=sorted(test_df["Re"].unique())
    for af in airfoils:
        for mach in machs:
            for re in reynolds:
                mask=((test_df[AIRFOIL_COL].values==af)&
                      (np.abs(test_df["Mach"].values-mach)<1e-4)&
                      (test_df["Re"].values==re))
                if mask.sum()==0: continue
                records.append({
                    "Airfoil":af,"Mach":mach,"Re":re,"N_points":int(mask.sum()),
                    f"{TARGET}_MAE":  round(mean_absolute_error(y_test[mask],y_pred[mask]),6),
                    f"{TARGET}_MSE":  round(mean_squared_error(y_test[mask],y_pred[mask]),8),
                    f"{TARGET}_RMSE": round(np.sqrt(mean_squared_error(y_test[mask],y_pred[mask])),6),
                    f"{TARGET}_R2":   round(r2_score(y_test[mask],y_pred[mask]),6),
                })
    pd.DataFrame(records).to_csv(path,index=False); print(f"Saved: {path}")

# ─── MAIN SWEEP ───────────────────────────────────────────────────────────────
def run_sweep():
    print("="*65)
    print(f"ARCHITECTURE SWEEP — {TARGET}")
    print(f"  N_CST={N_CST}  LR={FIXED_LR}  Batch={FIXED_BATCH}  Configs={len(HIDDEN_CONFIGS)}")
    print("="*65)

    sweep_dir=os.path.join(OUTPUT_DIR,"sweep"); best_dir=os.path.join(OUTPUT_DIR,"best")
    os.makedirs(sweep_dir,exist_ok=True); os.makedirs(best_dir,exist_ok=True)

    print("\nLoading data...")
    X_train,y_train_s,X_test,y_test,scaler_y,test_df=load_data()

    records=[]
    for run_num,hidden in enumerate(HIDDEN_CONFIGS,1):
        print(f"\n  [{run_num:02d}/{len(HIDDEN_CONFIGS)}]  {arch_label(hidden):<30}",end="  ",flush=True)
        t0=time.time()
        model,_=train_model(X_train,y_train_s,hidden)
        elapsed=time.time()-t0
        y_pred,metrics=evaluate(model,X_test,y_test,scaler_y)
        records.append({
            "hidden":hidden,"depth":len(hidden),"width":max(hidden),
            "train_time_s":round(elapsed,2),
            "r2":  round(metrics["r2"],  6),
            "mae": round(metrics["mae"], 6),
            "mse": round(metrics["mse"], 8),
            "rmse":round(metrics["rmse"],6),
        })
        print(f"R²={metrics['r2']:.5f}  MAE={metrics['mae']:.5f}  RMSE={metrics['rmse']:.5f}  [{elapsed:.0f}s]")
        pd.DataFrame(records).to_csv(os.path.join(sweep_dir,"sweep_partial.csv"),index=False)

    df=pd.DataFrame(records) 
    df.to_csv(os.path.join(sweep_dir,"sweep_results.csv"),index=False)

    # Sweep plots
    p=lambda f: os.path.join(sweep_dir,f)
    plot_arch_r2(df,          p(f"{TARGET.lower()}_arch_r2.png"))
    plot_arch_mae(df,         p(f"{TARGET.lower()}_arch_mae.png"))
    plot_arch_rmse(df,        p(f"{TARGET.lower()}_arch_rmse.png"))
    plot_arch_top(df,         p(f"{TARGET.lower()}_arch_top.png"))
    plot_depth_sensitivity(df,p(f"{TARGET.lower()}_depth_sensitivity.png"))
    plot_width_sensitivity(df,p(f"{TARGET.lower()}_width_sensitivity.png"))
    plot_training_time(df,    p(f"{TARGET.lower()}_training_time.png"))

    # Best config
    best_idx=df["r2"].idxmax(); best_row=df.loc[best_idx]
    best_hidden=tuple(best_row["hidden"])
    print("\n"+"="*65)
    print(f"BEST {TARGET} ARCHITECTURE")
    print(f"  Hidden : {arch_label(best_hidden)}")
    print(f"  Depth  : {best_row['depth']}   Width: {best_row['width']}")
    print(f"  R²     : {best_row['r2']:.5f}")
    print(f"  MAE    : {best_row['mae']:.5f}")
    print(f"  MSE    : {best_row['mse']:.7f}")
    print(f"  RMSE   : {best_row['rmse']:.5f}")
    print("="*65)

    # Retrain best
    print(f"\nRetraining best {arch_label(best_hidden)}...")
    best_model,train_losses=train_model(X_train,y_train_s,best_hidden)
    y_pred,best_metrics=evaluate(best_model,X_test,y_test,scaler_y)
    print(f"  Final → R²={best_metrics['r2']:.5f}  MAE={best_metrics['mae']:.5f}  "
          f"MSE={best_metrics['mse']:.7f}  RMSE={best_metrics['rmse']:.5f}")

    b=lambda f: os.path.join(best_dir,f)
    plot_training_curve(train_losses,    b(f"best_{TARGET.lower()}_training_curve.png"))
    plot_scatter(y_pred,y_test,          b(f"best_{TARGET.lower()}_scatter.png"))
    plot_error_distribution(y_pred,y_test,b(f"best_{TARGET.lower()}_error_dist.png"))
    plot_aoa_curve(y_pred,y_test,test_df,b(f"best_{TARGET.lower()}_aoa_curve.png"))
    plot_per_airfoil(y_pred,y_test,test_df,b(f"best_{TARGET.lower()}_per_airfoil.png"))
    save_per_testcase(y_pred,y_test,test_df,b(f"best_{TARGET.lower()}_per_testcase.csv"))

    # Summary CSV
    pd.DataFrame([{
        "output":TARGET,"best_hidden":arch_label(best_hidden),
        "depth":len(best_hidden),"width":max(best_hidden),
        "N_CST":N_CST,"lr":FIXED_LR,"batch":FIXED_BATCH,
        "R2":round(best_metrics["r2"],6),"MAE":round(best_metrics["mae"],6),
        "MSE":round(best_metrics["mse"],8),"RMSE":round(best_metrics["rmse"],6),
    }]).to_csv(os.path.join(OUTPUT_DIR,f"{TARGET.lower()}_summary.csv"),index=False)

    return df,best_row,best_metrics

def main():
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    run_sweep()
    print(f"\nDone. All {TARGET} outputs → {OUTPUT_DIR}")

if __name__=="__main__":
    main()
