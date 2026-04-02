"""
palmo_pytorch_nn_cd_400_4.py
============================
PALMO CST surrogate — Cd ONLY (PyTorch version)
Full 400x400x400x400 dedicated to drag coefficient.
Own StandardScaler, own loss — no Cd/Cm contamination.
Run palmo_cst_geometry.py first for geometry plots.
"""

import numpy as np
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from cst_modeling.section import cst_foil
    CST_LIB = True
except ImportError:
    CST_LIB = False

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TRAIN_CSV   = r"C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Results CST NN\Training_NACA_data.csv (3).xlsx"
TEST_CSV    = r"C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Results CST NN\Testing_NACA_data.csv (3).xlsx"
OUTPUT_DIR  = r"C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Results CST NN Split\Cd_PyTorch_400_4"

FLIGHT_COLS = ["Mach", "Re", "Alpha"]
TARGET      = "Cd"
AIRFOIL_COL = "NACA"
N_CST       = 7
TAIL        = 0.004

HIDDEN_LAYERS = (400, 400, 400, 400)
LEARNING_RATE = 1e-3
BATCH_SIZE    = 512
MAX_ITER      = 500
PATIENCE      = 30
RANDOM_SEED   = 42
VAL_FRACTION  = 0.10
MACH_PLOT     = 0.25
RE_PLOT       = 1_000_000

# ─── UTILITIES ──────────────────────────────────────────────────────────────
def compute_r2(y_true, y_pred):
    y_true = torch.tensor(y_true, dtype=torch.float32)
    y_pred = torch.tensor(y_pred, dtype=torch.float32)
    ss_res = torch.sum((y_pred - y_true)**2)
    ss_tot = torch.sum((y_true - torch.mean(y_true))**2)
    return (1 - ss_res / ss_tot).item()
class CSTNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=HIDDEN_LAYERS):
        super(CSTNet, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ─── CST UTILITIES ────────────────────────────────────────────────────────────
def naca4_coords(code, n=201):
    code=int(code); m=(code//1000)/100.0; p=((code%1000)//100)/10.0; t=(code%100)/100.0
    beta=np.linspace(0,np.pi,n); x=0.5*(1.0-np.cos(beta))
    yt=5*t*(0.2969*np.sqrt(x)-0.1260*x-0.3516*x**2+0.2843*x**3-0.1015*x**4)
    if m==0.0 or p==0.0: yc=np.zeros_like(x); dyc=np.zeros_like(x)
    else:
        yc =np.where(x<p,m/p**2*(2*p*x-x**2),m/(1-p)**2*(1-2*p+2*p*x-x**2))
        dyc=np.where(x<p,2*m/p**2*(p-x),2*m/(1-p)**2*(p-x))
    theta=np.arctan(dyc)
    return x-yt*np.sin(theta),yc+yt*np.cos(theta),x+yt*np.sin(theta),yc-yt*np.cos(theta)

def _bernstein_basis(x, n_cst):
    from math import comb
    C=np.sqrt(x)*(1.0-x); n=n_cst-1
    return C[:,None]*np.column_stack([comb(n,k)*x**k*(1-x)**(n-k) for k in range(n_cst)])

def fit_cst(xu, yu, xl, yl, n_cst):
    xu=np.clip(xu,1e-9,1-1e-9); xl=np.clip(xl,1e-9,1-1e-9)
    Au,_,_,_=np.linalg.lstsq(_bernstein_basis(xu,n_cst),yu,rcond=None)
    Al,_,_,_=np.linalg.lstsq(_bernstein_basis(xl,n_cst),yl,rcond=None)
    return Au, Al

def reconstruct_cst(cst_u, cst_l, n_pts=201, tail=TAIL):
    if CST_LIB:
        x,yu,yl,_,_=cst_foil(n_pts,cst_u,cst_l,x=None,t=None,tail=tail); return x,yu,yl
    beta=np.linspace(0,np.pi,n_pts); x=0.5*(1.0-np.cos(beta)); xc=np.clip(x,1e-9,1-1e-9)
    Pu=_bernstein_basis(xc,len(cst_u)); Pl=_bernstein_basis(xc,len(cst_l))
    return x,Pu@cst_u+0.5*tail*x,Pl@cst_l-0.5*tail*x

def get_cst_params(code, n_cst=N_CST):
    xu,yu,xl,yl=naca4_coords(code); return fit_cst(xu,yu,xl,yl,n_cst)

def build_cst_dataframe(df, n_cst=N_CST):
    u_cols=[f"cst_u{i}" for i in range(n_cst)]; l_cols=[f"cst_l{i}" for i in range(n_cst)]
    cache={}
    for code in sorted(df[AIRFOIL_COL].unique()):
        cu,cl=get_cst_params(code,n_cst); cache[code]=np.concatenate([cu,cl])
        print(f"  NACA {code:4d} | cst_u={np.round(cu,4)}")
    cst_arr=np.array([cache[c] for c in df[AIRFOIL_COL]])
    for i,col in enumerate(u_cols+l_cols): df[col]=cst_arr[:,i]
    return df, u_cols+l_cols

# ─── DATA ─────────────────────────────────────────────────────────────────────
def load_data():
    print(f"Loading data for {TARGET}...")
    train_df=pd.read_excel(TRAIN_CSV); test_df=pd.read_excel(TEST_CSV)
    print(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")
    print("Fitting CST — train:"); train_df,cst_cols=build_cst_dataframe(train_df)
    print("Fitting CST — test:");  test_df,_=build_cst_dataframe(test_df)
    input_cols=FLIGHT_COLS+cst_cols
    print(f"  Inputs ({len(input_cols)}): {FLIGHT_COLS} + {N_CST*2} CST params\n")
    X_train=train_df[input_cols].values.astype(np.float64)
    y_train=train_df[TARGET].values.astype(np.float64).reshape(-1,1)
    X_test =test_df[input_cols].values.astype(np.float64)
    y_test =test_df[TARGET].values.astype(np.float64)
    # Manual scaling for X
    mean_X = np.mean(X_train, axis=0)
    std_X = np.std(X_train, axis=0)
    X_train_scaled = (X_train - mean_X) / std_X
    X_test_scaled = (X_test - mean_X) / std_X
    # Manual scaling for y
    mean_y = np.mean(y_train)
    std_y = np.std(y_train)
    y_train_scaled = (y_train.ravel() - mean_y) / std_y
    return X_train_scaled, y_train_scaled, X_test_scaled, y_test, mean_X, std_X, mean_y, std_y, test_df

# ─── TRAIN ────────────────────────────────────────────────────────────────────
def train_model(X_train, y_train_scaled):
    print(f"\nTraining {TARGET} model  |  arch={HIDDEN_LAYERS}  |  lr={LEARNING_RATE}")
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    rng=np.random.default_rng(RANDOM_SEED); n_val=max(1,int(VAL_FRACTION*len(X_train)))
    idx=rng.permutation(len(X_train))
    Xtr,ytr=X_train[idx[n_val:]],y_train_scaled[idx[n_val:]]
    Xvl,yvl=X_train[idx[:n_val]],y_train_scaled[idx[:n_val]]

    input_dim = Xtr.shape[1]
    model = CSTNet(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(Xvl, dtype=torch.float32), torch.tensor(yvl, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(1, MAX_ITER + 1):
        model.train()
        epoch_train_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(x_batch).squeeze()
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                pred = model(x_batch).squeeze()
                loss = criterion(pred, y_batch)
                epoch_val_loss += loss.item()
        epoch_val_loss /= len(val_loader)
        val_losses.append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{MAX_ITER}  train={epoch_train_loss:.6f}  val={epoch_val_loss:.6f}")

        if patience_counter >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch}  (best val={best_val_loss:.6f})")
            break

    model.load_state_dict(best_state)
    return model, train_losses, val_losses

# ─── EVALUATE ─────────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, mean_y, std_y):
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_pred_scaled = model(X_test_tensor).squeeze().numpy()
    y_pred = y_pred_scaled * std_y + mean_y
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)
    mae = torch.mean(torch.abs(y_pred_tensor - y_test_tensor)).item()
    mse = torch.mean((y_pred_tensor - y_test_tensor)**2).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()
    r2 = compute_r2(y_test, y_pred)
    print(f"\n{'='*60}")
    print(f"  {TARGET}  →  MAE={mae:.5f}  MSE={mse:.5f}  RMSE={rmse:.5f}  R²={r2:.5f}")
    print(f"{'='*60}\n")
    return y_pred

# ─── PLOTS ────────────────────────────────────────────────────────────────────
def plot_loss_summary(train_losses, val_losses, y_pred, y_test, path_curve, path_summary):
    plt.figure(figsize=(8,4))
    plt.plot(train_losses,label="Train MSE (scaled)",lw=1.5)
    plt.plot(val_losses,label="Val MSE (scaled)",lw=1.5)
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title(f"PALMO CST {TARGET} — Training Curve"); plt.legend(); plt.tight_layout()
    plt.savefig(path_curve,dpi=150); plt.close(); print(f"Saved: {path_curve}")
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    y_pred_t = torch.tensor(y_pred, dtype=torch.float32)
    mae = torch.mean(torch.abs(y_pred_t - y_test_t)).item()
    mse = torch.mean((y_pred_t - y_test_t)**2).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()
    metrics={"MAE":mae,"MSE":mse,"RMSE":rmse}
    plt.figure(figsize=(6,4))
    bars=plt.bar(metrics.keys(),metrics.values(),color=["steelblue","orange","green"],edgecolor="black",lw=0.5)
    for bar,val in zip(bars,metrics.values()):
        plt.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f"{val:.5f}",ha="center",va="bottom",fontsize=10)
    plt.title(f"PALMO CST {TARGET} — Final Loss Summary"); plt.ylabel("Error"); plt.tight_layout()
    plt.savefig(path_summary,dpi=150); plt.close(); print(f"Saved: {path_summary}")

def plot_scatter(y_pred, y_test, path):
    r2 = compute_r2(y_test, y_pred)
    lo=min(y_test.min(),y_pred.min()); hi=max(y_test.max(),y_pred.max())
    plt.figure(figsize=(6,6)); plt.scatter(y_test,y_pred,s=1,alpha=0.2,color="steelblue")
    plt.plot([lo,hi],[lo,hi],"r--",lw=1.5,label="Ideal")
    plt.xlabel(f"CFD {TARGET}"); plt.ylabel(f"NN {TARGET}"); plt.title(f"{TARGET}  (R²={r2:.5f})")
    plt.legend(); plt.tight_layout(); plt.savefig(path,dpi=150); plt.close(); print(f"Saved: {path}")

def plot_error_distribution(y_pred, y_test, path):
    errors=y_pred-y_test; plt.figure(figsize=(7,4))
    plt.hist(errors,bins=80,color="steelblue",edgecolor="none",alpha=0.8)
    plt.axvline(0,color="red",ls="--",lw=1.5)
    plt.axvline(errors.mean(),color="orange",ls="--",lw=1.2,label=f"μ={errors.mean():.5f}")
    plt.xlabel(f"NN − CFD  ({TARGET})"); plt.ylabel("Count")
    plt.title(f"{TARGET}  σ={errors.std():.5f}  μ={errors.mean():.5f}")
    plt.legend(); plt.tight_layout(); plt.savefig(path,dpi=150); plt.close(); print(f"Saved: {path}")

def plot_aoa_curve(y_pred, y_test, test_df, path, mach=MACH_PLOT, reynolds=RE_PLOT):
    airfoils=sorted(test_df[AIRFOIL_COL].unique())
    fig,axes=plt.subplots(1,len(airfoils),figsize=(5*len(airfoils),5),sharey=True)
    if len(airfoils)==1: axes=[axes]
    for ax,af in zip(axes,airfoils):
        mask=((np.abs(test_df["Mach"].values-mach)<1e-4)&(np.abs(test_df["Re"].values-reynolds)<reynolds*0.05)&(test_df[AIRFOIL_COL].values==af))
        if mask.sum()==0: continue
        aoa=test_df.loc[mask,"Alpha"].values; s=np.argsort(aoa)
        ax.plot(aoa[s],y_test[mask][s],"k-",lw=2.0,label="CFD")
        ax.plot(aoa[s],y_pred[mask][s],"gs",ms=5,markerfacecolor="none",markeredgewidth=1.2,label="CST-NN")
        ax.set_xlabel("AoA (°)"); ax.set_title(f"NACA {af}\nM={mach}, Re={reynolds:.0e}")
        ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
    axes[0].set_ylabel(TARGET)
    plt.suptitle(f"PALMO CST {TARGET} — AoA Curve",fontsize=13,y=1.02)
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close(); print(f"Saved: {path}")

def plot_multi_mach(y_pred, y_test, test_df, path, machs=(0.25,0.50,0.75), reynolds=RE_PLOT):
    airfoils=sorted(test_df[AIRFOIL_COL].unique()); colors=["tab:blue","tab:orange","tab:green","tab:red"]
    fig,axes=plt.subplots(1,len(airfoils),figsize=(5*len(airfoils),5),sharey=True)
    if len(airfoils)==1: axes=[axes]
    for ax,af in zip(axes,airfoils):
        for mach,col in zip(machs,colors):
            mask=((np.abs(test_df["Mach"].values-mach)<1e-4)&(np.abs(test_df["Re"].values-reynolds)<reynolds*0.05)&(test_df[AIRFOIL_COL].values==af))
            if mask.sum()==0: continue
            aoa=test_df.loc[mask,"Alpha"].values; s=np.argsort(aoa)
            ax.plot(aoa[s],y_test[mask][s],"-",color=col,lw=2.0,label=f"CFD M={mach}")
            ax.plot(aoa[s],y_pred[mask][s],"s",color=col,ms=4,markerfacecolor="none",markeredgewidth=1.2,label=f"NN M={mach}")
        ax.set_xlabel("AoA (°)"); ax.set_title(f"NACA {af}"); ax.legend(fontsize=7,ncol=2); ax.grid(True,alpha=0.3)
    axes[0].set_ylabel(TARGET)
    plt.suptitle(f"PALMO CST {TARGET} — Multi-Mach",fontsize=12,y=1.02)
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close(); print(f"Saved: {path}")

def plot_error_vs_aoa(y_pred, y_test, test_df, path, mach=MACH_PLOT, reynolds=RE_PLOT):
    airfoils=sorted(test_df[AIRFOIL_COL].unique())
    fig,axes=plt.subplots(1,len(airfoils),figsize=(5*len(airfoils),4),sharey=True)
    if len(airfoils)==1: axes=[axes]
    for ax,af in zip(axes,airfoils):
        mask=((np.abs(test_df["Mach"].values-mach)<1e-4)&(np.abs(test_df["Re"].values-reynolds)<reynolds*0.05)&(test_df[AIRFOIL_COL].values==af))
        if mask.sum()==0: continue
        aoa=test_df.loc[mask,"Alpha"].values; err=y_pred[mask]-y_test[mask]; s=np.argsort(aoa)
        ax.plot(aoa[s],err[s],"k-o",ms=3,lw=1.2); ax.axhline(0,color="red",ls="--",lw=1)
        ax.fill_between(aoa[s],err[s],0,alpha=0.15,color="steelblue")
        ax.set_xlabel("AoA (°)"); ax.set_title(f"NACA {af}"); ax.grid(True,alpha=0.3)
        if ax is axes[0]: ax.set_ylabel(f"{TARGET} error (NN−CFD)")
    plt.suptitle(f"PALMO CST {TARGET} — Error vs AoA",fontsize=13)
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close(); print(f"Saved: {path}")

def plot_per_airfoil(y_pred, y_test, test_df, path):
    airfoils=sorted(test_df[AIRFOIL_COL].unique()); r2s,maes,labels=[],[],[]
    for af in airfoils:
        mask=test_df[AIRFOIL_COL].values==af
        r2s.append(compute_r2(y_test[mask], y_pred[mask]))
        maes.append(torch.mean(torch.abs(torch.tensor(y_pred[mask], dtype=torch.float32) - torch.tensor(y_test[mask], dtype=torch.float32))).item()); labels.append(f"NACA\n{af}")
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4))
    bar_colors=["steelblue" if r>=0.99 else "orange" if r>=0.97 else "red" for r in r2s]
    ax1.bar(labels,r2s,color=bar_colors,edgecolor="black",lw=0.5)
    ax1.axhline(0.99,color="green",ls="--",lw=1,label="R²=0.99"); ax1.set_ylim(min(0.95,min(r2s)-0.01),1.001)
    ax1.set_title(f"{TARGET} — R² per Airfoil"); ax1.set_ylabel("R²"); ax1.legend(fontsize=7)
    ax2.bar(labels,maes,color="steelblue",edgecolor="black",lw=0.5)
    ax2.set_title(f"{TARGET} — MAE per Airfoil"); ax2.set_ylabel("MAE")
    plt.suptitle(f"PALMO CST {TARGET} — Per-Airfoil Breakdown",fontsize=13)
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close(); print(f"Saved: {path}")

def save_predictions(y_pred, y_test, test_df, path):
    out=test_df.copy(); out[f"{TARGET}_CFD"]=y_test; out[f"{TARGET}_NN"]=y_pred; out[f"{TARGET}_err"]=y_pred-y_test
    out.to_csv(path,index=False); print(f"Saved: {path}")

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    def out(f): return os.path.join(OUTPUT_DIR,f)
    X_train,y_train_sc,X_test,y_test,mean_X,std_X,mean_y,std_y,test_df=load_data()
    model,tr_loss,val_loss=train_model(X_train,y_train_sc)
    y_pred=evaluate(model,X_test,y_test,mean_y,std_y)
    plot_loss_summary(tr_loss,val_loss,y_pred,y_test,
                      path_curve=out("cd_training_curve.png"),path_summary=out("cd_loss_summary.png"))
    plot_scatter(y_pred,y_test,               path=out("cd_predictions_scatter.png"))
    plot_error_distribution(y_pred,y_test,    path=out("cd_error_distribution.png"))
    plot_aoa_curve(y_pred,y_test,test_df,     path=out("cd_aoa_curve.png"))
    plot_multi_mach(y_pred,y_test,test_df,    path=out("cd_multi_mach.png"))
    plot_error_vs_aoa(y_pred,y_test,test_df,  path=out("cd_error_vs_aoa.png"))
    plot_per_airfoil(y_pred,y_test,test_df,   path=out("cd_per_airfoil.png"))
    save_predictions(y_pred,y_test,test_df,   path=out("cd_predictions.csv"))
    print(f"\nDone. All {TARGET} outputs → {OUTPUT_DIR}")

if __name__=="__main__":
    main()