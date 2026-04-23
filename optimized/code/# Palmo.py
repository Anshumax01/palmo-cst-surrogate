"""Palmo.py
Merged inverse-design notebook using the three CST-NN surrogates
(Cl, Cd, Cm) instead of the LAM / Cp pipeline.

Structure mirrors the original notebook:
  Section 1 - imports and settings
  Section 2 - NN infrastructure and evaluation model
  Section 3 - NACA 3415 target setup
  Section 4 - design input file
  Section 5 - run inverse design
  Section 6 - visualize results
"""

## SECTION 1 – Imports & settings

import sys, time, pickle
from math import comb
 
import numpy as np
import torch
import torch.nn as nn
import gpytorch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from matplotlib import pyplot as plt
 
try:
    import tqdm.notebook as tn
except ImportError:
    import tqdm as tn
 
# ---- import relevant functions from the LAM suite (boss's code, untouched) ----
sys.path.append(r'C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Python')
 
from lam import visualize
from lam.utils import initialize_devices, initialize_plot_settings
from lam.geometry import lam_pca_transformer
from lam.design import inverse
 
# ---- Initialize settings ----
torch.set_default_dtype(torch.float64)
output_device = initialize_devices(use_gpu=False, gpu_id=0)
initialize_plot_settings()
 
## SECTION 2 – NN infrastructure

class CSTNet(nn.Module):
    """
    Generic MLP used for all three surrogates.
      Cl  →  hidden_dims = (400, 400, 400)
      Cd  →  hidden_dims = (400, 400, 400, 400)
      Cm  →  hidden_dims = (400, 400, 400, 400)
    """
    def __init__(self, input_dim: int, hidden_dims: tuple):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
 
 
# ── 2b.  Differentiable CST fitting from LAM y/c coords ─────────────────────
 
N_CST = 7       # must match your trained models
TAIL  = 0.004   # trailing-edge thickness (matches training scripts)
 
# LAM x/c grid (28 points) – matches airfoil_design_input.lam_xc
LAM_XC = torch.tensor([
    0.0,    0.0025, 0.0075, 0.01,  0.015,  0.02,
    0.025,  0.05,   0.075,  0.1,   0.15,   0.2,
    0.25,   0.3,    0.35,   0.4,   0.45,   0.5,   0.55,
    0.6,    0.65,   0.70,   0.75,  0.8,    0.85,  0.90,
    0.95,   1.0
], dtype=torch.float64)
 
 
class BernsteinFitter:
    """
    Pre-computes the Moore-Penrose pseudo-inverse of the Bernstein basis at the
    interior LAM stations so CST fitting is a single differentiable matmul.
    Skips x=0 and x=1 (singular points of the basis).
    """
    def __init__(self, xc: torch.Tensor = LAM_XC, n_cst: int = N_CST):
        xc_inner = xc[1:-1].double().clamp(1e-9, 1 - 1e-9)   # (26,)
        B        = self._basis(xc_inner, n_cst)                # (26, 7)
        self.B_pinv   = torch.linalg.pinv(B)                   # (7, 26)
        self.xc_inner = xc_inner
 
    @staticmethod
    def _basis(x: torch.Tensor, n_cst: int) -> torch.Tensor:
        n = n_cst - 1
        C = torch.sqrt(x) * (1.0 - x)
        cols = [C * x.pow(k) * (1 - x).pow(n - k) * comb(n, k)
                for k in range(n_cst)]
        return torch.stack(cols, dim=1)
 
    def fit(self, ycu: torch.Tensor, ycl: torch.Tensor) -> tuple:
        """
        ycu, ycl : (28,) y/c values at LAM_XC stations.
        Returns (cst_u, cst_l), each (N_CST,) — fully differentiable.
        """
        xc_i  = self.xc_inner.to(ycu.device)
        B_inv = self.B_pinv.to(ycu.device)
        yu_i  = ycu[1:-1] - 0.5 * TAIL * xc_i   # tail correction
        yl_i  = ycl[1:-1] + 0.5 * TAIL * xc_i
        return B_inv @ yu_i, B_inv @ yl_i          # (7,), (7,)
 
 
_BERNSTEIN_FITTER = BernsteinFitter()   # singleton
 
 
# ── 2c.  Differentiable standard scaler ─────────────────────────────────────
 
class TorchStandardScaler:
    """
    Differentiable feature scaler.  Build via:
      • TorchStandardScaler.from_sklearn(sk_scaler)      – for Cl (sklearn)
      • TorchStandardScaler(mean_np, std_np)             – for Cd / Cm (manual)
    """
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        # Keep tensors in default precision; cast to input dtype at transform time.
        self.mean = torch.as_tensor(mean)
        self.std  = torch.as_tensor(std)
 
    @classmethod
    def from_sklearn(cls, sk_scaler):
        return cls(sk_scaler.mean_, sk_scaler.scale_)
 
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std  = self.std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std
 
 
# ── 2d.  NN evaluation model (replaces large_airfoil_model / LAM) ───────────
 
class NNCSTEvaluationModel:
    """
    Wraps the three trained CSTNet surrogates.
 
    Converts a PCA sample (15-D) to Normal distributions over Cl, Cd, Cm:
        PCA sample → y/c coords (via pca_transformer)
                   → CST params (via BernsteinFitter)
                   → [Mach, Re, Alpha, cst_u×7, cst_l×7]  (17-D)
                   → CSTNet forward → scalar → Normal(μ, σ_nn)
 
    Parameters
    ----------
    model_cl / cd / cm   : trained, eval()-mode CSTNet
    scaler_X_*           : TorchStandardScaler for features
    mean_y_* / std_y_*   : output unscaling constants from training
    mach, reynolds, alpha: fixed flight condition
    nn_std_*             : epistemic std added to each NN output.
                           Controls how tightly SVI matches the target.
                           Typical: Cl ≈ 0.02, Cd ≈ 0.001, Cm ≈ 0.001
    """
 
    def __init__(self,
                 model_cl: CSTNet,  model_cd: CSTNet,  model_cm: CSTNet,
                 scaler_X_cl: TorchStandardScaler,
                 scaler_X_cd: TorchStandardScaler,
                 scaler_X_cm: TorchStandardScaler,
                 mean_y_cl: float, std_y_cl: float,
                 mean_y_cd: float, std_y_cd: float,
                 mean_y_cm: float, std_y_cm: float,
                 mach:      float,
                 reynolds:  float,
                 alpha:     float,
                 output_device: str = 'cpu',
                 nn_std_cl: float = 0.02,
                 nn_std_cd: float = 0.001,
                 nn_std_cm: float = 0.001):
 
        self.model_cl, self.model_cd, self.model_cm = model_cl, model_cd, model_cm
        self.scaler_cl, self.scaler_cd, self.scaler_cm = scaler_X_cl, scaler_X_cd, scaler_X_cm
        self.mean_y_cl, self.std_y_cl = mean_y_cl, std_y_cl
        self.mean_y_cd, self.std_y_cd = mean_y_cd, std_y_cd
        self.mean_y_cm, self.std_y_cm = mean_y_cm, std_y_cm
        self.mach, self.reynolds, self.alpha = mach, reynolds, alpha
        self.output_device = output_device
        self.nn_std_cl, self.nn_std_cd, self.nn_std_cm = nn_std_cl, nn_std_cd, nn_std_cm
 
        for m in [model_cl, model_cd, model_cm]:
            m.eval()
 
    # ─── internal helpers ───────────────────────────────────────────────────
 
    def _pca_to_cst(self, pca_transformer, pca_vec: torch.Tensor):
        """PCA (15,) → physical coords (56,) → CST upper & lower (7, 7)."""
        phys  = pca_transformer.inverse(
            pca_vec.to(self.output_device), get_physical_coords=True)   # (56,)
        cst_u, cst_l = _BERNSTEIN_FITTER.fit(phys[:28], phys[28:])
        return cst_u, cst_l                                             # (7,), (7,)
 
    def _build_input(self, cst_u: torch.Tensor, cst_l: torch.Tensor) -> torch.Tensor:
        """Assemble [Mach, Re, Alpha, cst_u×7, cst_l×7] → (17,) model dtype/device."""
        p = next(self.model_cl.parameters())
        mdl_dtype = p.dtype
        mdl_device = p.device
        flight = torch.tensor([self.mach, self.reynolds, self.alpha],
                              dtype=mdl_dtype, device=mdl_device)
        return torch.cat([
            flight,
            cst_u.to(device=mdl_device, dtype=mdl_dtype),
            cst_l.to(device=mdl_device, dtype=mdl_dtype),
        ])
 
    def _forward(self, nn_model, scaler, mean_y, std_y,
                 x_raw: torch.Tensor) -> torch.Tensor:
        """Scale → NN → unscale → differentiable scalar."""
        p = next(nn_model.parameters())
        x_in = x_raw.unsqueeze(0).to(device=p.device, dtype=p.dtype)
        x_sc = scaler.transform(x_in).to(device=p.device, dtype=p.dtype)  # (1, 17)
        y_sc  = nn_model(x_sc).squeeze()                # scalar
        mean_y_t = torch.as_tensor(mean_y, device=p.device, dtype=p.dtype)
        std_y_t  = torch.as_tensor(std_y,  device=p.device, dtype=p.dtype)
        return y_sc * std_y_t + mean_y_t
 
    # ─── public API ─────────────────────────────────────────────────────────
 
    def predict(self, airfoil_input, sample_tensor: torch.Tensor, **kwargs) -> dict:
        """
        Returns {'cl': Normal, 'cd': Normal, 'cm': Normal}.
        Fully differentiable w.r.t. sample_tensor.
        """
        cst_u, cst_l = self._pca_to_cst(airfoil_input.pca_transformer, sample_tensor)
        x_raw = self._build_input(cst_u, cst_l)
 
        cl_val = self._forward(self.model_cl, self.scaler_cl,
                               self.mean_y_cl, self.std_y_cl, x_raw)
        cd_val = self._forward(self.model_cd, self.scaler_cd,
                               self.mean_y_cd, self.std_y_cd, x_raw)
        cm_val = self._forward(self.model_cm, self.scaler_cm,
                               self.mean_y_cm, self.std_y_cm, x_raw)
 
        mk_n = lambda v, s: torch.distributions.Normal(
            v, torch.as_tensor(s, device=v.device, dtype=v.dtype))
 
        return {
            'cl': mk_n(cl_val, self.nn_std_cl),
            'cd': mk_n(cd_val, self.nn_std_cd),
            'cm': mk_n(cm_val, self.nn_std_cm),
        }
 
    def __call__(self, airfoil_input, sample_tensor: torch.Tensor, **kwargs):
        return self.predict(airfoil_input, sample_tensor, **kwargs)
## SECTION 3 – NACA 3415 target

def _naca4_coords_np(code: int, n: int = 201):
    """Pure-numpy NACA 4-digit coordinates (upper/lower surfaces)."""
    code = int(code)
    m    = (code // 1000) / 100.0
    p    = ((code % 1000) // 100) / 10.0
    t    = (code % 100) / 100.0
    beta = np.linspace(0, np.pi, n)
    x    = 0.5 * (1.0 - np.cos(beta))
    yt   = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2
                + 0.2843*x**3 - 0.1015*x**4)
    if m == 0.0 or p == 0.0:
        yc, dyc = np.zeros_like(x), np.zeros_like(x)
    else:
        yc  = np.where(x < p,
                       m/p**2 * (2*p*x - x**2),
                       m/(1-p)**2 * (1 - 2*p + 2*p*x - x**2))
        dyc = np.where(x < p,
                       2*m/p**2 * (p - x),
                       2*m/(1-p)**2 * (p - x))
    theta = np.arctan(dyc)
    return (x - yt*np.sin(theta), yc + yt*np.cos(theta),
            x + yt*np.sin(theta), yc - yt*np.cos(theta))
 
 
def _bernstein_np(x: np.ndarray, n_cst: int) -> np.ndarray:
    n = n_cst - 1
    C = np.sqrt(x) * (1.0 - x)
    return C[:, None] * np.column_stack(
        [comb(n, k) * x**k * (1-x)**(n-k) for k in range(n_cst)])
 
 
def naca3415_cst_np(n_cst: int = N_CST) -> tuple:
    """Returns (cst_u, cst_l) as numpy arrays for NACA 3415."""
    xu, yu, xl, yl = _naca4_coords_np(3415)
    xu = np.clip(xu, 1e-9, 1 - 1e-9)
    xl = np.clip(xl, 1e-9, 1 - 1e-9)
    cu, _, _, _ = np.linalg.lstsq(_bernstein_np(xu, n_cst), yu, rcond=None)
    cl, _, _, _ = np.linalg.lstsq(_bernstein_np(xl, n_cst), yl, rcond=None)
    return cu, cl
 
 
def build_naca3415_targets(nn_model: NNCSTEvaluationModel,
                           target_mach:  float,
                           target_re:    float,
                           target_alpha: float,
                           # ── override with XFoil/CFD values if available ──
                           target_cl: float = None,
                           target_cd: float = None,
                           target_cm: float = None,
                           # ── target uncertainty ──
                           std_cl: float = 0.03,
                           std_cd: float = 0.002,
                           std_cm: float = 0.002) -> dict:
    """
    Build Normal target distributions for NACA 3415.
 
    Mode A (default, target_* = None):
        Runs the NNs on NACA 3415's CST params at the given flight condition.
        Targets are whatever the NN predicts for that airfoil.
 
    Mode B (pass explicit floats):
        Uses those values directly (e.g. from XFoil / your training data).
    """
    cst_u_np, cst_l_np = naca3415_cst_np(N_CST)
 
    if any(v is None for v in [target_cl, target_cd, target_cm]):
        p0 = next(nn_model.model_cl.parameters())
        x_dtype = p0.dtype
        x_device = p0.device
        np_dtype = np.float64 if x_dtype == torch.float64 else np.float32
        flight = np.array([target_mach, target_re, target_alpha], dtype=np_dtype)
        x_np   = np.concatenate([flight, cst_u_np.astype(np_dtype), cst_l_np.astype(np_dtype)])
        x_t    = torch.as_tensor(x_np, dtype=x_dtype, device=x_device).unsqueeze(0)
 
        def _eval(mdl, scl, my, sy):
            p = next(mdl.parameters())
            with torch.no_grad():
                x_eval = x_t.to(device=p.device, dtype=p.dtype)
                x_sc = scl.transform(x_eval).to(device=p.device, dtype=p.dtype)
                y = mdl(x_sc).squeeze()
                my_t = torch.as_tensor(my, device=p.device, dtype=p.dtype)
                sy_t = torch.as_tensor(sy, device=p.device, dtype=p.dtype)
                return float((y * sy_t + my_t).item())
 
        if target_cl is None:
            target_cl = _eval(nn_model.model_cl, nn_model.scaler_cl,
                              nn_model.mean_y_cl, nn_model.std_y_cl)
        if target_cd is None:
            target_cd = _eval(nn_model.model_cd, nn_model.scaler_cd,
                              nn_model.mean_y_cd, nn_model.std_y_cd)
        if target_cm is None:
            target_cm = _eval(nn_model.model_cm, nn_model.scaler_cm,
                              nn_model.mean_y_cm, nn_model.std_y_cm)
 
    print("=" * 55)
    print("  NACA 3415 design targets")
    print(f"    Mach={target_mach}  Re={target_re:.0e}  Alpha={target_alpha}°")
    print(f"    Cl = {target_cl:.4f}  ±{std_cl}")
    print(f"    Cd = {target_cd:.5f}  ±{std_cd}")
    print(f"    Cm = {target_cm:.5f}  ±{std_cm}")
    print("=" * 55)
 
    mk = lambda v, s: torch.distributions.Normal(
        torch.as_tensor(v, dtype=x_dtype if 'x_dtype' in locals() else torch.get_default_dtype()),
        torch.as_tensor(s, dtype=x_dtype if 'x_dtype' in locals() else torch.get_default_dtype()))
 
    return {
        'target_cl': mk(target_cl, std_cl),
        'target_cd': mk(target_cd, std_cd),
        'target_cm': mk(target_cm, std_cm),
        'values': {'cl': target_cl, 'cd': target_cd, 'cm': target_cm},
    }
## SECTION 4 – NNProbabilisticDesigner

class NNProbabilisticDesigner(inverse.probabilistic_designer):
    """
    Drop-in replacement for inverse.probabilistic_designer.
 
    Changes versus parent:
      • pyro_model uses NNCSTEvaluationModel → Normal(Cl), Normal(Cd), Normal(Cm)
      • gradient covariance uses the same NN call (fully differentiable)
      • Cp is completely absent; target_cp must be None
 
    Everything else (active-space decomposition, tolerance characterisation,
    Gaussian-mixture post-processing, save/load helpers) is inherited unchanged.
 
    Parameters
    ----------
    nn_eval_model : NNCSTEvaluationModel
    user_input    : inverse.airfoil_design_input
                    Set target_cp=None; fill target_cl / target_cd / target_cm.
    betas         : [beta_cl, beta_cd, beta_cm]  –  loss weights
    (remaining args identical to parent)
    """
 
    def __init__(self,
                 nn_eval_model: NNCSTEvaluationModel,
                 user_input,
                 betas:         list = [100, 1000, 100],
                 optimizer=None,
                 max_iters:     int  = 3000,
                 output_device: str  = 'cpu',
                 verbose:       bool = True):
 
        # Pass a dummy evaluation_model to the parent just to satisfy its __init__
        # We never call self.model() from the parent's code paths we override
        super().__init__(
            evaluation_model=_DummyEvalModel(),
            user_input=user_input,
            optimizer=optimizer,
            max_iters=max_iters,
            output_device=output_device,
            verbose=verbose,
        )
        self.nn_model = nn_eval_model
        self.betas    = betas
 
    # ── symmetrised KL (Normal × Normal) ────────────────────────────────────
 
    @staticmethod
    def _kl_symm(p: torch.distributions.Normal,
                 q: torch.distributions.Normal,
                 jitter: float = 1e-4) -> torch.Tensor:
        p_j = torch.distributions.Normal(p.loc, p.scale + jitter)
        q_j = torch.distributions.Normal(q.loc, q.scale + jitter)
        return 0.5 * (torch.distributions.kl_divergence(p_j, q_j) +
                      torch.distributions.kl_divergence(q_j, p_j))
 
    # ── run_design (override to wire NN gradient cov) ───────────────────────
 
    def run_design(self, num_particles: int = 1):
        assert self.airfoil.initial_design is not None, \
            "Provide initial_design in airfoil_design_input."
        assert any(x is not None for x in [self.airfoil.target_cl,
                                           self.airfoil.target_cd,
                                           self.airfoil.target_cm]), \
            "Provide at least one of target_cl / target_cd / target_cm."
        assert self.airfoil.target_cp is None, \
            "target_cp must be None when using NNProbabilisticDesigner."
 
        self.airfoil.pods_sample = self.airfoil.initial_design.clone()
 
        # ── SVI ──
        result, history = self.svi(num_particles=num_particles)
 
        # ── Tolerance / active-inactive space (inherited math, NN gradients) ──
        start = time.time()
        H = self._nn_gradient_cov(self.airfoil.num_gradient_samples)
        _, _, W, V = self._probabilistic_designer__partition_active_space(
            H, self.airfoil.num_active_dims)
        tol_pca, tol_geom = self._probabilistic_designer__determine_inactive_space(
            self.airfoil.pods_distrib.mean, W, V)
        self.airfoil.design_pca_distrib     = tol_pca
        self.airfoil.design_airfoil_distrib = tol_geom
        print(f'Tolerance analysis runtime: {time.time()-start:.1f}s'
              if self.verbose else '')
 
        return result, history
 
    # ── SVI loop (fully NN-based, no Cp) ────────────────────────────────────
 
    def svi(self,
            num_particles:          int   = 1,
            prior_loss_scale:       float = 0.1,
            variational_init_scale: float = 0.1):
 
        pyro.clear_param_store()
 
        target_cl = self.airfoil.target_cl
        target_cd = self.airfoil.target_cd
        target_cm = self.airfoil.target_cm
        beta_cl, beta_cd, beta_cm = self.betas
 
        nn_model = self.nn_model
        airfoil  = self.airfoil
 
        def pyro_model():
            # Prior: Normal around the initial airfoil in PCA space
            initial_af = airfoil.initial_design.clone().detach()
            vars_scale  = (torch.ones(15, dtype=initial_af.dtype) * 0.01
                           if airfoil.design_variance is None
                           else airfoil.design_variance)
            vars_prior = dist.Normal(initial_af, vars_scale).to_event(1)
 
            with pyro.poutine.scale(scale=prior_loss_scale):
                vars_sample = pyro.sample("vars_sample", vars_prior)
 
            # NN forward: PCA sample → Cl / Cd / Cm Normal distributions
            coeffs = nn_model(airfoil, sample_tensor=vars_sample)
 
            if target_cl is not None:
                pyro.factor("loss_cl",
                    -beta_cl * self._kl_symm(coeffs['cl'], target_cl))
            if target_cd is not None:
                pyro.factor("loss_cd",
                    -beta_cd * self._kl_symm(coeffs['cd'], target_cd))
            if target_cm is not None:
                pyro.factor("loss_cm",
                    -beta_cm * self._kl_symm(coeffs['cm'], target_cm))
 
        guide = pyro.infer.autoguide.AutoMultivariateNormal(
            pyro_model, init_scale=variational_init_scale)
 
        if self.optimizer is None:
            self.optimizer = pyro.optim.PyroLRScheduler(
                torch.optim.lr_scheduler.StepLR,
                optim_args={
                    "optimizer":  torch.optim.Adam,
                    "optim_args": {"lr": 0.01},
                    "step_size":  300,
                    "gamma":      0.5,
                }
            )
 
        svi_obj = SVI(pyro_model, guide, self.optimizer,
                      loss=Trace_ELBO(num_particles=num_particles))
 
        start = time.time()
        print("Narrowing the design space via SVI..." if self.verbose else "")
 
        with tn.tqdm(range(self.max_iters), desc="Iterations") as pbar:
            for _ in pbar:
                loss = svi_obj.step()
                pbar.set_postfix(loss=f"{loss:.4f}")
 
                with torch.no_grad():
                    posterior = guide.get_posterior()
                    posterior = torch.distributions.MultivariateNormal(
                        posterior.mean.detach(),
                        posterior.covariance_matrix.detach())
 
                    self.airfoil.pods_distrib = posterior
                    self.airfoil.pods_sample  = posterior.mean
 
                    mean_coeffs = self.nn_model(self.airfoil,
                                               sample_tensor=posterior.mean)
                    self.history['cl'].append(mean_coeffs['cl'].loc.item())
                    self.history['cd'].append(mean_coeffs['cd'].loc.item())
                    self.history['cm'].append(mean_coeffs['cm'].loc.item())
                    self.history['loss'].append(loss)
                    self.history['iters'].append(self.history['total_iter'])
                    self.history['pods_distrib'].append(
                        posterior.mean.detach().cpu().clone())
                    self.history['delta_distrib'].append(
                        posterior.mean.detach().cpu().clone())
 
                    # ── convergence check (same as parent) ──
                    if self.history['total_iter'] >= self.airfoil.mavg_window + 1:
                        delta = torch.stack(self.history['delta_distrib']).numpy()
                        w     = self.airfoil.mavg_window
                        prev  = delta[-w-1:-2, :self.airfoil.tracking_num].mean(0)
                        curr  = delta[-w:-1,   :self.airfoil.tracking_num].mean(0)
                        diff  = (curr - prev) / (np.abs(prev) + 1e-12)
                        if (np.all(np.abs(diff) <= self.airfoil.tolerance)
                                and np.all(np.abs(diff) > 0.0)):
                            print("  Convergence achieved." if self.verbose else "")
                            self.history['total_iter'] += 1
                            break
 
                self.history['total_iter'] += 1
 
        print(f'SVI runtime: {time.time()-start:.1f}s' if self.verbose else '')
 
        # Move posterior to CPU
        self.airfoil.pods_distrib = torch.distributions.MultivariateNormal(
            posterior.mean.cpu(), posterior.covariance_matrix.cpu())
 
        # Print final NN predictions vs targets
        with torch.no_grad():
            final = self.nn_model(self.airfoil,
                                  sample_tensor=self.airfoil.pods_distrib.mean)
        print("\n  Final design NN predictions vs NACA 3415 targets:")
        if target_cl is not None:
            print(f"    Cl = {final['cl'].loc.item():.4f}  "
                  f"(target {target_cl.loc.item():.4f})")
        if target_cd is not None:
            print(f"    Cd = {final['cd'].loc.item():.5f}  "
                  f"(target {target_cd.loc.item():.5f})")
        if target_cm is not None:
            print(f"    Cm = {final['cm'].loc.item():.5f}  "
                  f"(target {target_cm.loc.item():.5f})")
 
        return self.airfoil, self.history
 
    # ── gradient covariance via NN (replaces LAM-based version in parent) ───
 
    def _nn_gradient_cov(self, num_samples: int = 1000) -> torch.Tensor:
        print(f"  Approximating gradient covariance from {num_samples} samples..."
              if self.verbose else "")
 
        candidates  = self.airfoil.pods_distrib.sample(torch.Size([num_samples]))
        target_list = [self.airfoil.target_cl,
                       self.airfoil.target_cd,
                       self.airfoil.target_cm]
        key_list    = ['cl', 'cd', 'cm']
        H = torch.zeros(15, 15, dtype=torch.float64)
 
        for beta, tgt, key in zip(self.betas, target_list, key_list):
            if tgt is None:
                continue
            print(f"    w.r.t. {key.upper()}..." if self.verbose else "")
            C = torch.zeros(15, 15, dtype=torch.float64)
 
            for cand in candidates:
                cand_leaf = cand.clone().detach().requires_grad_(True)
                coeffs    = self.nn_model(self.airfoil, sample_tensor=cand_leaf)
                loss      = self._kl_symm(tgt, coeffs[key]) * beta
                loss.backward()
 
                if cand_leaf.grad is not None:
                    g  = cand_leaf.grad.detach().double().cpu()
                    C += torch.outer(g, g)
 
            C /= num_samples
            H += beta * C
 
        return H
 
 
# ── dummy eval model to satisfy parent __init__ ─────────────────────────────
 
class _DummyEvalModel(inverse.evaluation_model):
    """Placeholder so the parent __init__ does not crash. Never called."""
    def __init__(self):
        # parent stores self.model; give it something harmless
        self.model = None
        self.output_device = 'cpu'
 
    def predict(self, *args, **kwargs):
        raise RuntimeError(
            "_DummyEvalModel.predict() should never be called. "
            "NNProbabilisticDesigner always uses nn_model instead.")
## SECTION 5 – Main: load models, set up design, run inverse design

import torch
import pickle
import pyro
import os

if __name__ == '__main__':

    # ── Model & scaler paths ─────────────────────────────────────────────
    MODEL_CL_PATH  = r"C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Results CST NN Split\Cl_PyTorch_400_3\cl_model.pt"
    MODEL_CD_PATH  = r"C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Results CST NN Split\Cd_PyTorch_400_4\cd_model.pt"
    MODEL_CM_PATH  = r"C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Results CST NN Split\Cm_PyTorch_400_4\cm_model.pt"
    SCALER_CL_PATH = r"C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Results CST NN Split\Cl_PyTorch_400_3\cl_scalers.pkl"
    SCALER_CD_PATH = r"C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Results CST NN Split\Cd_PyTorch_400_4\cd_scalers.pkl"
    SCALER_CM_PATH = r"C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Results CST NN Split\Cm_PyTorch_400_4\cm_scalers.pkl"

    # ── Results folder ───────────────────────────────────────────────────────
    RESULTS_PATH = r'C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\CSTNN Results'
    os.makedirs(RESULTS_PATH, exist_ok=True)

    # ── Setup ────────────────────────────────────────────────────────────────
    output_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float64
    print(f"Running on: {output_device}")


    # ── Load models ─────────────────────────────────────────────────────────
    INPUT_DIM = 17 

    model_cl = CSTNet(INPUT_DIM, (400, 400, 400)).to(output_device, dtype=torch_dtype)
    model_cl.load_state_dict(torch.load(MODEL_CL_PATH, map_location=output_device))
    model_cl.eval()

    model_cd = CSTNet(INPUT_DIM, (400, 400, 400, 400)).to(output_device, dtype=torch_dtype)
    model_cd.load_state_dict(torch.load(MODEL_CD_PATH, map_location=output_device))
    model_cd.eval()

    model_cm = CSTNet(INPUT_DIM, (400, 400, 400, 400)).to(output_device, dtype=torch_dtype)
    model_cm.load_state_dict(torch.load(MODEL_CM_PATH, map_location=output_device))
    model_cm.eval()

    # ── Load scalers ────────────────────────────────────────────────────────
    def load_scaler_tensors(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return {
            k: torch.as_tensor(v, dtype=torch_dtype).to(output_device)
            if isinstance(v, (list, tuple, float, int, np.ndarray)) else v
            for k, v in data.items()
        }

    cl_sc = load_scaler_tensors(SCALER_CL_PATH)
    cd_sc = load_scaler_tensors(SCALER_CD_PATH)
    cm_sc = load_scaler_tensors(SCALER_CM_PATH)

    # ── Flight condition ────────────────────────────────────────────────────
    DESIGN_ALPHA = 0.0
    DESIGN_MACH  = 0.14
    DESIGN_RE    = 1_000_000

    # ── NN evaluation model ─────────────────────────────────────────────────
    nn_eval = NNCSTEvaluationModel(
        model_cl, model_cd, model_cm,
        TorchStandardScaler(cl_sc['mean_X'], cl_sc['std_X']),
        TorchStandardScaler(cd_sc['mean_X'], cd_sc['std_X']),
        TorchStandardScaler(cm_sc['mean_X'], cm_sc['std_X']),
        cl_sc['mean_y'], cl_sc['std_y'],
        cd_sc['mean_y'], cd_sc['std_y'],
        cm_sc['mean_y'], cm_sc['std_y'],
        DESIGN_MACH, DESIGN_RE, DESIGN_ALPHA,
        output_device
    )

    # ── PCA ─────────────────────────────────────────────────────────────────
    pca_transformer = lam_pca_transformer(output_device)

    # ── Targets ─────────────────────────────────────────────────────────────
    targets = build_naca3415_targets(
        nn_eval,
        DESIGN_MACH,
        DESIGN_RE,
        DESIGN_ALPHA
    )

    # ── Initial guess (FIXED) ───────────────────────────────────────────────
    naca3415_pca = (
        torch.tensor(
            [-0.9670, 0.5160, -0.1630, -0.0167, -0.0205, -0.0373, 0.0565,
             -0.0364, -0.0415, 0.0298, -0.0258, 0.0208, -0.0113, -0.0055, 0.0032],
            dtype=torch_dtype
        )
        /
        torch.tensor(
            [4.4, 1.4, 1.0, 0.54, 0.46, 0.3, 0.3, 0.3, 0.14,
             0.08, 0.08, 0.08, 0.08, 0.08, 0.08],
            dtype=torch_dtype
        )
    )

    initial_guess = naca3415_pca.to(output_device) + \
        0.02 * torch.randn(15, dtype=torch_dtype, device=output_device)

    # ── Input file ──────────────────────────────────────────────────────────
    input_file = inverse.airfoil_design_input(
        alpha=[DESIGN_ALPHA],
        mach=[DESIGN_MACH],
        initial_design=initial_guess,
        num_active_dims='auto',
        num_gradient_samples=500,
        target_cp=None,
        target_cl=targets['target_cl'],
        target_cd=targets['target_cd'],
        target_cm=targets['target_cm'],
        target_design=None,
        pca_transformer=pca_transformer,
        tolerance=5e-4,
    )

    # ── Designer ────────────────────────────────────────────────────────────
    airfoil_designer = NNProbabilisticDesigner(
        nn_eval_model=nn_eval,
        user_input=input_file,
        betas=[100, 1000, 100],
        max_iters=2000,
        output_device=output_device,
        verbose=True,
    )

    design, hist = airfoil_designer.run_design(num_particles=1)

    # ======================================================================
    # SECTION 6 – VISUALIZATION
    # ======================================================================

    dgn_af = inverse.helper_airfoil_plotting(design, return_dict=True)

    # ✅ USE SAME PCA (no duplicate definition)
    tgt_af = inverse.helper_airfoil_plotting(
        design,
        manual_distrib=naca3415_pca.to(output_device),
        return_dict=True
    )

    # ── Plot airfoils ────────────────────────────────────────────────────────
    f, ax = plt.subplots(figsize=(8, 6))
    f, ax, leg = visualize.plot_airfoil(
        f, ax,
        plotting_elements=[dgn_af, tgt_af],
        kwargs_list=[
            [{'color': 'tab:red', 'linewidth': 2.0},
             {'color': 'tab:red', 'alpha': 0.3}, {}],
            [{'color': 'k', 'marker': 'o', 'linestyle': 'None'}],
        ],
        legend_handle=[
            ("tab:red", "tab:red", "-", "None", 1.0),
            ("None", "k", "None", "o", 1.0),
        ],
        legend_labels=['Designed airfoil ± tolerance', 'NACA 3415'],
    )
    ax.set_ylim([-0.12, 0.12])
    leg.set_loc('lower right')
    ax.set_title('NN Inverse Design — NACA 3415 Target')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'airfoil.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # ── Loss ─────────────────────────────────────────────────────────────────
    plt.figure()
    plt.semilogy(hist['loss'])
    plt.title("Loss Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_PATH, 'loss.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # ── Cl / Cd / Cm convergence ─────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(hist['iters'], hist['cl'], color='tab:blue')
    axes[0].axhline(targets['values']['cl'], color='k', linestyle='--', label='Target')
    axes[0].set_ylabel('Cl')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(hist['iters'], hist['cd'], color='tab:orange')
    axes[1].axhline(targets['values']['cd'], color='k', linestyle='--', label='Target')
    axes[1].set_ylabel('Cd')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(hist['iters'], hist['cm'], color='tab:green')
    axes[2].axhline(targets['values']['cm'], color='k', linestyle='--', label='Target')
    axes[2].set_ylabel('Cm')
    axes[2].set_xlabel('Iteration')
    axes[2].legend()
    axes[2].grid(True)

    fig.suptitle('Aerodynamic Coefficient Convergence')
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_PATH, 'coefficients.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # ── Final values ─────────────────────────────────────────────────────────
    print("\nFinal:")
    print(f"Cl: {hist['cl'][-1]:.4f}")
    print(f"Cd: {hist['cd'][-1]:.5f}")
    print(f"Cm: {hist['cm'][-1]:.5f}")