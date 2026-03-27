"""
palmo_cst_geometry.py
=====================
Run this FIRST — generates all CST geometry outputs:
  - Convergence study
  - Reconstruction grid + individual pages
  - Tail comparison plots (tail=0.0 vs tail=TAIL)
  - Geometric feature plots
  - CST coefficient table CSV
"""

import numpy as np
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, os
warnings.filterwarnings("ignore")

try:
    from cst_modeling.section import cst_foil
    CST_LIB = True
    print("cst_modeling3d found.")
except ImportError:
    CST_LIB = False
    print("WARNING: cst_modeling3d not found — using built-in fallback.\n")

TRAIN_CSV   = r"C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Results CST NN\Training_NACA_data.csv (3).xlsx"
TEST_CSV    = r"C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Results CST NN\Testing_NACA_data.csv (3).xlsx"
OUTPUT_DIR  = r"C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Results CST NN Split\Geometry"
AIRFOIL_COL = "NACA"
N_CST       = 7
TAIL        = 0.004
CST_ORDERS  = [3, 4, 5, 6, 7, 8, 9, 10]

def naca4_coords(code, n=201):
    code=int(code); m=(code//1000)/100.0; p=((code%1000)//100)/10.0; t=(code%100)/100.0
    beta=np.linspace(0,np.pi,n); x=0.5*(1.0-np.cos(beta))
    yt=5*t*(0.2969*np.sqrt(x)-0.1260*x-0.3516*x**2+0.2843*x**3-0.1015*x**4)
    if m==0.0 or p==0.0:
        yc=np.zeros_like(x); dyc=np.zeros_like(x)
    else:
        yc =np.where(x<p,m/p**2*(2*p*x-x**2),m/(1-p)**2*(1-2*p+2*p*x-x**2))
        dyc=np.where(x<p,2*m/p**2*(p-x),2*m/(1-p)**2*(p-x))
    theta=np.arctan(dyc)
    return x-yt*np.sin(theta),yc+yt*np.cos(theta),x+yt*np.sin(theta),yc-yt*np.cos(theta)

def _bernstein_basis(x, n_cst):
    from math import comb
    C=np.sqrt(x)*(1.0-x); n=n_cst-1
    B=np.column_stack([comb(n,k)*x**k*(1-x)**(n-k) for k in range(n_cst)])
    return C[:,None]*B

def fit_cst(xu, yu, xl, yl, n_cst):
    xu=np.clip(xu,1e-9,1-1e-9); xl=np.clip(xl,1e-9,1-1e-9)
    Pu=_bernstein_basis(xu,n_cst); Pl=_bernstein_basis(xl,n_cst)
    Au,_,_,_=np.linalg.lstsq(Pu,yu,rcond=None); Al,_,_,_=np.linalg.lstsq(Pl,yl,rcond=None)
    return Au, Al

def reconstruct_cst(cst_u, cst_l, n_pts=201, tail=TAIL):
    if CST_LIB:
        x,yu,yl,_,_=cst_foil(n_pts,cst_u,cst_l,x=None,t=None,tail=tail); return x,yu,yl
    beta=np.linspace(0,np.pi,n_pts); x=0.5*(1.0-np.cos(beta)); xc=np.clip(x,1e-9,1-1e-9)
    Pu=_bernstein_basis(xc,len(cst_u)); Pl=_bernstein_basis(xc,len(cst_l))
    return x,Pu@cst_u+0.5*tail*x,Pl@cst_l-0.5*tail*x

def get_cst_params(code, n_cst=N_CST):
    xu,yu,xl,yl=naca4_coords(code); return fit_cst(xu,yu,xl,yl,n_cst)

def plot_cst_convergence(all_codes, path):
    print("Running CST convergence study...")
    n_eval=501; fig,axes=plt.subplots(1,2,figsize=(14,5))
    colors=plt.cm.tab20(np.linspace(0,1,len(all_codes)))
    for color,code in zip(colors,all_codes):
        xu,yu,xl,yl=naca4_coords(code,n=n_eval); rms_list,max_list=[],[]
        for n in CST_ORDERS:
            cu,cl=fit_cst(xu,yu,xl,yl,n); xr,yur,ylr=reconstruct_cst(cu,cl,n_pts=n_eval)
            yut=np.interp(xr,xu,yu); ylt=np.interp(xr,xl,yl)
            err=np.concatenate([np.abs(yur-yut),np.abs(ylr-ylt)])
            rms_list.append(np.sqrt(np.mean(err**2))); max_list.append(np.max(err))
        axes[0].semilogy(CST_ORDERS,rms_list,"o-",color=color,lw=1.5,ms=5,label=f"NACA {code}")
        axes[1].semilogy(CST_ORDERS,max_list, "o-",color=color,lw=1.5,ms=5,label=f"NACA {code}")
    for ax,title,ylab in zip(axes,
            ["RMS Error vs CST Order","Max Error vs CST Order"],
            ["RMS |y_true-y_CST| (y/c)","Max |y_true-y_CST| (y/c)"]):
        ax.set_xlabel("CST Parameters per Surface",fontsize=11); ax.set_ylabel(ylab,fontsize=10)
        ax.set_title(title,fontsize=11); ax.set_xticks(CST_ORDERS)
        ax.axvline(N_CST,color="crimson",ls="--",lw=1.5,label=f"N_CST={N_CST} (used)")
        ax.legend(fontsize=6.5,ncol=2,loc="upper right"); ax.grid(True,which="both",alpha=0.3)
    plt.suptitle(f"CST Convergence Study — All PALMO Airfoils  (tail={TAIL})",fontsize=13,fontweight="bold")
    plt.tight_layout(); plt.savefig(path,dpi=300,bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def plot_tail_comparison(cst_u, cst_l, code, out_dir):
    if CST_LIB:
        x1,yu1,yl1,tmax1,_=cst_foil(201,cst_u,cst_l,x=None,t=None,tail=0.0)
        x2,yu2,yl2,tmax2,_=cst_foil(201,cst_u,cst_l,x=None,t=None,tail=TAIL)
        x3,yu3,yl3,tmax3,_=cst_foil(201,cst_u,cst_l,x=None,t=0.11,tail=TAIL)
    else:
        beta=np.linspace(0,np.pi,201); x1=0.5*(1-np.cos(beta))
        xc=np.clip(x1,1e-9,1-1e-9); Pu=_bernstein_basis(xc,len(cst_u)); Pl=_bernstein_basis(xc,len(cst_l))
        x2=x3=x1; yu1=Pu@cst_u; yl1=Pl@cst_l
        yu2=yu1+0.5*TAIL*x1; yl2=yl1-0.5*TAIL*x1
        yu3=yu2.copy(); yl3=yl2.copy(); tmax1=tmax2=tmax3=0.0
    print(f"  NACA {code} | tmax: {tmax1:.3f}  {tmax2:.3f}  {tmax3:.3f}")

    for xa,yua,yla,xb,yub,ylb,lbl_b,suffix,title in [
        (x1,yu1,yl1,x2,yu2,yl2,f'tail={TAIL:.3f}',"",
         r'NACA %s — $t_\text{TE}$ without keeping $t_\text{max}$'%code),
        (x1,yu1,yl1,x3,yu3,yl3,f'tail={TAIL:.3f}, tmax=0.11',"_tmax",
         r'NACA %s — $t_\text{TE}$ while keeping $t_\text{max}$'%code)]:
        fig,ax=plt.subplots(figsize=(10,5))
        ax.plot(xa,yua,'b',lw=1.8,label='tail=0.000'); ax.plot(xa,yla,'b',lw=1.8,label='_nolegend_')
        ax.plot(xb,yub,'r--',lw=1.2,label=lbl_b);      ax.plot(xb,ylb,'r--',lw=1.2,label='_nolegend_')
        ax.set_xlim(-0.05,1.05); ax.set_ylim(-0.07,0.07)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.legend(); ax.grid(True,alpha=0.3); ax.set_title(title)
        fname=os.path.join(out_dir,f'cst_tail{suffix}_NACA{code}.png')
        fig.savefig(fname,dpi=300,bbox_inches='tight'); plt.close(fig); print(f"  Saved: {fname}")

def _compute_geo(x, yu, yl):
    th=yu-yl; cam=(yu+yl)/2.0; i_t=np.argmax(th)
    t_max,x_t=float(th[i_t]),float(x[i_t])
    t_20=float(np.interp(0.2,x,th)); t_70=float(np.interp(0.7,x,th))
    i_c=np.argmax(np.abs(cam)); c_max,x_c=float(cam[i_c]),float(x[i_c])
    volume=float(np.trapezoid(th,x))
    m60=x<=0.6; m40=x>=0.6
    c_f60=float(np.trapezoid(cam[m60],x[m60]))/0.6 if m60.sum()>1 else 0.0
    c_r40=float(np.trapezoid(cam[m40],x[m40]))/0.4 if m40.sum()>1 else 0.0
    n_le=max(5,len(x)//20); dy_le=np.gradient(yu[:n_le],x[:n_le]); d2y=np.gradient(dy_le,x[:n_le])
    kap=abs(d2y[0])/(1+dy_le[0]**2)**1.5; r_le=1.0/kap if kap>1e-9 else 0.0
    dyu_te=(yu[-1]-yu[-2])/(x[-1]-x[-2]+1e-12); dyl_te=(yl[-1]-yl[-2])/(x[-1]-x[-2]+1e-12)
    wedge=float(np.degrees(np.arctan(dyu_te)-np.arctan(dyl_te)))
    slope_te=float(np.degrees((np.arctan(dyu_te)+np.arctan(dyl_te))/2.0))
    slope_le=float(np.degrees(np.arctan(dy_le[0])))
    i_uc=np.argmax(yu); i_lc=np.argmin(yl)
    d2yu=np.gradient(np.gradient(yu,x),x); c_mean=float(np.sqrt(np.mean(d2yu**2)))
    return dict(th=th,cam=cam,t_max=t_max,x_t=x_t,t_20=t_20,t_70=t_70,
                c_max=c_max,x_c=x_c,volume=volume,c_f60=c_f60,c_r40=c_r40,
                r_le=r_le,wedge=wedge,slope_te=slope_te,slope_le=slope_le,
                x_uc=float(x[i_uc]),y_uc=float(yu[i_uc]),
                x_lc=float(x[i_lc]),y_lc=float(yl[i_lc]),c_mean=c_mean)

def plot_geometric_features(cst_u, cst_l, code, out_dir):
    for tail in [0.0, TAIL]:
        if CST_LIB:
            x,yu,yl,_,_=cst_foil(1001,cst_u,cst_l,x=None,t=None,tail=tail)
        else:
            beta=np.linspace(0,np.pi,1001); x=0.5*(1-np.cos(beta)); xc=np.clip(x,1e-9,1-1e-9)
            Pu=_bernstein_basis(xc,len(cst_u)); Pl=_bernstein_basis(xc,len(cst_l))
            yu=Pu@cst_u+0.5*tail*x; yl=Pl@cst_l-0.5*tail*x
        g=_compute_geo(x,yu,yl)
        fig,ax=plt.subplots(figsize=(16,8))
        ax.plot(x,yu,'k',lw=1.5); ax.plot(x,yl,'k',lw=1.5,label='_nolegend_')
        ax.plot(x,g['th']/2,'b--',lw=0.8,label='Thickness'); ax.plot(x,-g['th']/2,'b--',lw=0.8,label='_nolegend_')
        ax.plot(x,g['cam'],'r--',lw=0.8,label='Camber')
        ax.set_xlim(-0.2,1.2); ax.set_ylim(-0.2,0.2); ax.set_aspect('equal')
        ax.legend(['Airfoil surface','Thickness','Camber'],fontsize=9); ax.grid(True,alpha=0.25)

        def lbl(xp,yp,text,dx=0.01,dy=0.01,color='b'):
            ax.plot(xp,yp,color+'*',ms=6); ax.text(xp+dx,yp+dy,text,color=color,fontsize=8)

        lbl(g['x_t'],0.0,r'$t_{max}$: %.3f'%g['t_max'])
        lbl(0.2,0.0,r'$t_{0.2}$: %.3f'%g['t_20'])
        lbl(0.7,0.0,r'$t_{0.7}$: %.3f'%g['t_70'])
        lbl(g['x_c'],0.0,r'$c_{max}$: %.3f'%g['c_max'],dx=-0.02,dy=-0.04,color='r')
        lbl(g['x_uc'],g['y_uc'],r'$y_{uc}$: %.3f'%g['y_uc'])
        lbl(g['x_lc'],g['y_lc'],r'$y_{lc}$: %.3f'%g['y_lc'])
        lbl(0.78,0.19,r'Volume: %.4f'%g['volume'],dy=0)
        lbl(0.78,0.17,r'Mean curvature: %.3f'%g['c_mean'],dy=0)
        lbl(0.3,0.0,r'$c_{f60}$: %.4f'%g['c_f60'],dx=-0.02,dy=-0.03,color='r')
        lbl(0.78,0.15,r'$c_{r40}$: %.4f'%g['c_r40'],dy=0)
        lbl(0.78,0.13,r'LE radius: %.4f'%g['r_le'],dy=0)
        lbl(0.78,0.11,r'LE slope: %.2f°'%g['slope_le'],dy=0)
        lbl(0.78,0.09,r'TE wedge: %.2f°'%g['wedge'],dy=0)
        lbl(0.78,0.07,r'TE slope: %.2f°'%g['slope_te'],dy=0)
        ax.set_title(f'NACA {code} — Geometric Features  (tail={tail:.3f})',fontsize=12)
        fname=os.path.join(out_dir,f'cst_geo_NACA{code}_tail{tail:.3f}.png')
        fig.savefig(fname,dpi=300,bbox_inches='tight'); plt.close(fig); print(f"  Saved: {fname}")

def plot_reconstruction_grid(all_codes, path):
    print("Plotting reconstruction grid...")
    ncols=4; nrows=int(np.ceil(len(all_codes)/ncols)); n_eval=401
    fig,axes=plt.subplots(nrows*2,ncols,figsize=(4.5*ncols,3.8*nrows),
                           gridspec_kw={"hspace":0.55,"wspace":0.35})
    for idx,code in enumerate(all_codes):
        r=(idx//ncols)*2; c=idx%ncols; ax_s=axes[r][c]; ax_r=axes[r+1][c]
        xu,yu,xl,yl=naca4_coords(code,n=n_eval); cu,cl=get_cst_params(code)
        xr,yur,ylr=reconstruct_cst(cu,cl,n_pts=n_eval)
        yut=np.interp(xr,xu,yu); ylt=np.interp(xr,xl,yl)
        res_u=yur-yut; res_l=ylr-ylt; rms=np.sqrt(np.mean(np.concatenate([res_u,res_l])**2))
        ax_s.plot(xu,yu,'b',lw=1.8,label="True NACA"); ax_s.plot(xl,yl,'b',lw=1.8,label='_nolegend_')
        ax_s.plot(xr,yur,'r--',lw=1.1,label=f"CST n={N_CST}"); ax_s.plot(xr,ylr,'r--',lw=1.1)
        ax_s.set_title(f"NACA {code}\nRMS={rms:.2e}",fontsize=7.5); ax_s.set_aspect("equal"); ax_s.axis("off")
        if idx==0: ax_s.legend(fontsize=6,loc="upper right")
        ax_r.plot(xr,res_u,color="steelblue",lw=1.1,label="Upper"); ax_r.fill_between(xr,res_u,0,alpha=0.2,color="steelblue")
        ax_r.plot(xr,res_l,color="tomato",lw=1.1,label="Lower");    ax_r.fill_between(xr,res_l,0,alpha=0.2,color="tomato")
        ax_r.axhline(0,color="k",lw=0.7,ls="--"); ax_r.set_xlim(0,1)
        ax_r.set_xlabel("x/c",fontsize=6); ax_r.set_ylabel("Δy/c",fontsize=6); ax_r.tick_params(labelsize=5.5); ax_r.grid(True,alpha=0.25)
        if idx==0: ax_r.legend(fontsize=6)
    for idx in range(len(all_codes),nrows*ncols):
        r=(idx//ncols)*2; c=idx%ncols; axes[r][c].axis("off"); axes[r+1][c].axis("off")
    plt.suptitle(f"True NACA vs CST Reconstruction  (n_cst={N_CST}, tail={TAIL})",fontsize=13,fontweight="bold")
    plt.savefig(path,dpi=300,bbox_inches="tight"); plt.close(); print(f"Saved: {path}")

def plot_reconstruction_individual(all_codes, out_dir):
    print("Plotting individual reconstruction pages...")
    n_eval=501
    for code in all_codes:
        xu,yu,xl,yl=naca4_coords(code,n=n_eval); cu,cl=get_cst_params(code)
        xr,yur,ylr=reconstruct_cst(cu,cl,n_pts=n_eval)
        yut=np.interp(xr,xu,yu); ylt=np.interp(xr,xl,yl)
        res_u=yur-yut; res_l=ylr-ylt; all_err=np.concatenate([res_u,res_l])
        rms=np.sqrt(np.mean(all_err**2)); max_err=np.max(np.abs(all_err))
        rms_u=np.sqrt(np.mean(res_u**2)); rms_l=np.sqrt(np.mean(res_l**2))

        fig=plt.figure(figsize=(16,10))
        gs=gridspec.GridSpec(3,3,figure=fig,hspace=0.5,wspace=0.38)
        ax_foil=fig.add_subplot(gs[0,:2]); ax_tail=fig.add_subplot(gs[1,:2])
        ax_res_u=fig.add_subplot(gs[2,0]); ax_res_l=fig.add_subplot(gs[2,1])
        ax_table=fig.add_subplot(gs[:,2])

        ax_foil.plot(xu,yu,'b',lw=2.2,label="True NACA coords"); ax_foil.plot(xl,yl,'b',lw=2.2,label='_nolegend_')
        ax_foil.plot(xr,yur,'r--',lw=1.6,label=f"CST (n={N_CST}, tail={TAIL})"); ax_foil.plot(xr,ylr,'r--',lw=1.6)
        ax_foil.fill_between(xr,yut,yur,alpha=0.15,color="red",label="Error region")
        ax_foil.fill_between(xr,ylt,ylr,alpha=0.15,color="red")
        ax_foil.set_xlim(-0.05,1.05); ax_foil.set_ylim(-0.07,0.07)
        ax_foil.set_xlabel("x/c"); ax_foil.set_ylabel("y/c")
        ax_foil.set_title(f"NACA {code}  |  RMS={rms:.3e}  |  Max={max_err:.3e}  |  tail={TAIL}")
        ax_foil.legend(fontsize=8); ax_foil.grid(True,alpha=0.3)

        if CST_LIB:
            x0,yu0,yl0,_,_=cst_foil(n_eval,cu,cl,x=None,t=None,tail=0.0)
            xt,yut2,ylt2,_,_=cst_foil(n_eval,cu,cl,x=None,t=None,tail=TAIL)
        else:
            beta=np.linspace(0,np.pi,n_eval); x0=0.5*(1-np.cos(beta)); xc=np.clip(x0,1e-9,1-1e-9)
            Pu=_bernstein_basis(xc,len(cu)); Pl=_bernstein_basis(xc,len(cl))
            yu0=Pu@cu; yl0=Pl@cl; xt=x0; yut2=yu0+0.5*TAIL*x0; ylt2=yl0-0.5*TAIL*x0
        ax_tail.plot(x0,yu0,'b',label='tail=0.000'); ax_tail.plot(x0,yl0,'b',label='_nolegend_')
        ax_tail.plot(xt,yut2,'r--',lw=1,label=f'tail={TAIL}'); ax_tail.plot(xt,ylt2,'r--',lw=1,label='_nolegend_')
        ax_tail.set_xlim(-0.05,1.05); ax_tail.set_ylim(-0.07,0.07)
        ax_tail.set_xlabel("x"); ax_tail.set_ylabel("y")
        ax_tail.set_title(r'Adding $t_\text{TE}$ — NACA %s'%code)
        ax_tail.legend(fontsize=8); ax_tail.grid(True,alpha=0.3)

        ax_res_u.plot(xr,res_u,color="steelblue",lw=1.4); ax_res_u.fill_between(xr,res_u,0,alpha=0.2,color="steelblue")
        ax_res_u.axhline(0,color="k",lw=0.9,ls="--"); ax_res_u.set_xlabel("x/c"); ax_res_u.set_ylabel("Δy/c")
        ax_res_u.set_title("Upper surface residual  (CST − True)"); ax_res_u.grid(True,alpha=0.3)
        ax_res_u.annotate(f"RMS={rms_u:.2e}",xy=(0.97,0.95),xycoords="axes fraction",ha="right",va="top",fontsize=8,color="steelblue")

        ax_res_l.plot(xr,res_l,color="tomato",lw=1.4); ax_res_l.fill_between(xr,res_l,0,alpha=0.2,color="tomato")
        ax_res_l.axhline(0,color="k",lw=0.9,ls="--"); ax_res_l.set_xlabel("x/c"); ax_res_l.set_ylabel("Δy/c")
        ax_res_l.set_title("Lower surface residual  (CST − True)"); ax_res_l.grid(True,alpha=0.3)
        ax_res_l.annotate(f"RMS={rms_l:.2e}",xy=(0.97,0.95),xycoords="axes fraction",ha="right",va="top",fontsize=8,color="tomato")

        ax_table.axis("off")
        rows=[[str(i),f"{u:.6f}",f"{l:.6f}"] for i,(u,l) in enumerate(zip(cu,cl))]
        tbl=ax_table.table(cellText=rows,colLabels=["i","A_upper[i]","A_lower[i]"],loc="center",cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1.1,1.8)
        ax_table.set_title(f"CST coefficients\n(n={N_CST}, tail={TAIL})",fontsize=10,pad=12)
        fig.suptitle(f"NACA {code} — True Geometry vs CST Reconstruction",fontsize=14,fontweight="bold")
        fname=os.path.join(out_dir,f"cst_recon_NACA{code}.png")
        plt.savefig(fname,dpi=300,bbox_inches="tight"); plt.close(); print(f"  Saved: {fname}")

        plot_tail_comparison(cu,cl,code,out_dir=out_dir)
        plot_geometric_features(cu,cl,code,out_dir=out_dir)

def save_cst_table(all_codes, path):
    rows=[]
    for code in sorted(all_codes):
        cu,cl=get_cst_params(code); row={"NACA":code}
        for i in range(N_CST): row[f"A_upper_{i}"]=round(cu[i],8); row[f"A_lower_{i}"]=round(cl[i],8)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path,index=False); print(f"Saved: {path}")

def main():
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    def out(f): return os.path.join(OUTPUT_DIR,f)
    train_df=pd.read_excel(TRAIN_CSV); test_df=pd.read_excel(TEST_CSV)
    all_codes=sorted(set(train_df[AIRFOIL_COL].unique())|set(test_df[AIRFOIL_COL].unique()))
    print(f"All airfoils: {all_codes}  |  N_CST={N_CST}  |  TAIL={TAIL}\n")
    plot_cst_convergence(all_codes,path=out("cst_convergence.png"))
    plot_reconstruction_grid(all_codes,path=out("cst_reconstruction_grid.png"))
    plot_reconstruction_individual(all_codes,out_dir=OUTPUT_DIR)
    save_cst_table(all_codes,path=out("cst_coefficients.csv"))
    print("\nGeometry done. Now run palmo_cst_nn_cl/cd/cm.py")

if __name__=="__main__":
    main()