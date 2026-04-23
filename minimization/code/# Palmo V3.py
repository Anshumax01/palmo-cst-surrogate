"""Palmo.py"""

import sys, time, pickle, os
from math import comb

import numpy as np
import torch
import torch.nn as nn
import gpytorch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

try:
    import tqdm.notebook as tn
except ImportError:
    import tqdm as tn

sys.path.append(r'C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Python')

from lam import visualize
from lam.utils import initialize_devices, initialize_plot_settings
from lam.geometry import lam_pca_transformer
from lam.design import inverse

torch.set_default_dtype(torch.float64)
output_device = initialize_devices(use_gpu=False, gpu_id=0)
initialize_plot_settings()


class CSTNet(nn.Module):
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


N_CST = 7
TAIL  = 0.004

LAM_XC = torch.tensor([
    0.0,    0.0025, 0.0075, 0.01,  0.015,  0.02,
    0.025,  0.05,   0.075,  0.1,   0.15,   0.2,
    0.25,   0.3,    0.35,   0.4,   0.45,   0.5,   0.55,
    0.6,    0.65,   0.70,   0.75,  0.8,    0.85,  0.90,
    0.95,   1.0
], dtype=torch.float64)


class BernsteinFitter:
    def __init__(self, xc: torch.Tensor = LAM_XC, n_cst: int = N_CST):
        xc_inner = xc[1:-1].double().clamp(1e-9, 1 - 1e-9)
        B        = self._basis(xc_inner, n_cst)
        self.B_pinv   = torch.linalg.pinv(B)
        self.xc_inner = xc_inner

    @staticmethod
    def _basis(x: torch.Tensor, n_cst: int) -> torch.Tensor:
        n = n_cst - 1
        C = torch.sqrt(x) * (1.0 - x)
        cols = [C * x.pow(k) * (1 - x).pow(n - k) * comb(n, k)
                for k in range(n_cst)]
        return torch.stack(cols, dim=1)

    def fit(self, ycu: torch.Tensor, ycl: torch.Tensor) -> tuple:
        xc_i  = self.xc_inner.to(ycu.device)
        B_inv = self.B_pinv.to(ycu.device)
        yu_i  = ycu[1:-1] - 0.5 * TAIL * xc_i
        yl_i  = ycl[1:-1] + 0.5 * TAIL * xc_i
        return B_inv @ yu_i, B_inv @ yl_i


_BERNSTEIN_FITTER = BernsteinFitter()


class TorchStandardScaler:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = torch.as_tensor(mean)
        self.std  = torch.as_tensor(std)

    @classmethod
    def from_sklearn(cls, sk_scaler):
        return cls(sk_scaler.mean_, sk_scaler.scale_)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std  = self.std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std


class NNCSTEvaluationModel:
    def __init__(self,
                 model_cl, model_cd, model_cm,
                 scaler_X_cl, scaler_X_cd, scaler_X_cm,
                 mean_y_cl, std_y_cl,
                 mean_y_cd, std_y_cd,
                 mean_y_cm, std_y_cm,
                 mach, reynolds, alpha,
                 output_device='cpu',
                 nn_std_cl=0.02, nn_std_cd=0.001, nn_std_cm=0.001):

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

    def _pca_to_cst(self, pca_transformer, pca_vec):
        phys  = pca_transformer.inverse(pca_vec.to(self.output_device), get_physical_coords=True)
        cst_u, cst_l = _BERNSTEIN_FITTER.fit(phys[:28], phys[28:])
        return cst_u, cst_l

    def _build_input(self, cst_u, cst_l):
        p = next(self.model_cl.parameters())
        flight = torch.tensor([self.mach, self.reynolds, self.alpha],
                              dtype=p.dtype, device=p.device)
        return torch.cat([flight,
                          cst_u.to(device=p.device, dtype=p.dtype),
                          cst_l.to(device=p.device, dtype=p.dtype)])

    def _forward(self, nn_model, scaler, mean_y, std_y, x_raw):
        p = next(nn_model.parameters())
        x_in = x_raw.unsqueeze(0).to(device=p.device, dtype=p.dtype)
        x_sc = scaler.transform(x_in).to(device=p.device, dtype=p.dtype)
        y_sc = nn_model(x_sc).squeeze()
        return y_sc * torch.as_tensor(std_y, device=p.device, dtype=p.dtype) \
                   + torch.as_tensor(mean_y, device=p.device, dtype=p.dtype)

    def predict(self, airfoil_input, sample_tensor, **kwargs):
        cst_u, cst_l = self._pca_to_cst(airfoil_input.pca_transformer, sample_tensor)
        x_raw = self._build_input(cst_u, cst_l)

        cl_val = self._forward(self.model_cl, self.scaler_cl, self.mean_y_cl, self.std_y_cl, x_raw)
        cd_val = self._forward(self.model_cd, self.scaler_cd, self.mean_y_cd, self.std_y_cd, x_raw)
        cm_val = self._forward(self.model_cm, self.scaler_cm, self.mean_y_cm, self.std_y_cm, x_raw)

        mk_n = lambda v, s: torch.distributions.Normal(
            v, torch.as_tensor(s, device=v.device, dtype=v.dtype))

        return {
            'cl': mk_n(cl_val, self.nn_std_cl),
            'cd': mk_n(cd_val, self.nn_std_cd),
            'cm': mk_n(cm_val, self.nn_std_cm),
        }

    def __call__(self, airfoil_input, sample_tensor, **kwargs):
        return self.predict(airfoil_input, sample_tensor, **kwargs)


def _naca4_coords_np(code: int, n: int = 201):
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


def naca_cst_np(naca_code: int, n_cst: int = N_CST) -> tuple:
    xu, yu, xl, yl = _naca4_coords_np(naca_code)
    xu = np.clip(xu, 1e-9, 1 - 1e-9)
    xl = np.clip(xl, 1e-9, 1 - 1e-9)
    cu, _, _, _ = np.linalg.lstsq(_bernstein_np(xu, n_cst), yu, rcond=None)
    cl, _, _, _ = np.linalg.lstsq(_bernstein_np(xl, n_cst), yl, rcond=None)
    return cu, cl


def naca_to_pca_approx(naca_code: int, pca_transformer, torch_dtype, device) -> torch.Tensor:
    xu, yu, xl, yl = _naca4_coords_np(naca_code, n=28)
    coords = np.concatenate([yu, yl]).astype(
        np.float64 if torch_dtype == torch.float64 else np.float32)
    coords_t = torch.as_tensor(coords, dtype=torch_dtype, device=device)
    pca_vec  = pca_transformer.forward(coords_t)
    return pca_vec


def build_targets_from_naca(nn_model, naca_code: int,
                             target_mach, target_re, target_alpha,
                             std_cl=0.03, std_cd=0.002, std_cm=0.002) -> dict:
    cst_u_np, cst_l_np = naca_cst_np(naca_code)

    p0       = next(nn_model.model_cl.parameters())
    x_dtype  = p0.dtype
    x_device = p0.device
    np_dtype = np.float64 if x_dtype == torch.float64 else np.float32
    flight   = np.array([target_mach, target_re, target_alpha], dtype=np_dtype)
    x_np     = np.concatenate([flight, cst_u_np.astype(np_dtype), cst_l_np.astype(np_dtype)])
    x_t      = torch.as_tensor(x_np, dtype=x_dtype, device=x_device).unsqueeze(0)

    def _eval(mdl, scl, my, sy):
        p = next(mdl.parameters())
        with torch.no_grad():
            x_e  = x_t.to(device=p.device, dtype=p.dtype)
            x_sc = scl.transform(x_e).to(device=p.device, dtype=p.dtype)
            y    = mdl(x_sc).squeeze()
            return float((y * torch.as_tensor(sy, device=p.device, dtype=p.dtype)
                            + torch.as_tensor(my, device=p.device, dtype=p.dtype)).item())

    target_cl = _eval(nn_model.model_cl, nn_model.scaler_cl, nn_model.mean_y_cl, nn_model.std_y_cl)
    target_cd = _eval(nn_model.model_cd, nn_model.scaler_cd, nn_model.mean_y_cd, nn_model.std_y_cd)
    target_cm = _eval(nn_model.model_cm, nn_model.scaler_cm, nn_model.mean_y_cm, nn_model.std_y_cm)

    print("=" * 55)
    print(f"  NACA {naca_code} baseline aerodynamics")
    print(f"    Mach={target_mach}  Re={target_re:.0e}  Alpha={target_alpha}°")
    print(f"    Cl = {target_cl:.4f}  (fixed target,  ±{std_cl})")
    print(f"    Cd = {target_cd:.5f}  (baseline to beat)")
    print(f"    Cm = {target_cm:.5f}  (fixed target,  ±{std_cm})")
    print("=" * 55)

    mk = lambda v, s: torch.distributions.Normal(
        torch.as_tensor(v, dtype=x_dtype),
        torch.as_tensor(s, dtype=x_dtype))

    return {
        'target_cl': mk(target_cl, std_cl),
        'target_cm': mk(target_cm, std_cm),
        'values': {'cl': target_cl, 'cd': target_cd, 'cm': target_cm},
    }


class NNDragMinimizationDesigner(inverse.probabilistic_designer):
    """
    Minimizes Cd while keeping Cl and Cm close to the baseline NACA airfoil.

    Loss:
        L = beta_cd * cd                           (linear: drives Cd down)
          + beta_barrier * relu(-cd)^2             (soft barrier: penalizes Cd < 0)
          + beta_cl * KL_symm(cl_pred, cl_target)
          + beta_cm * KL_symm(cm_pred, cm_target)
          + prior_loss_scale * KL(prior, guide)

    The linear term strictly minimizes Cd in the positive region.
    The barrier term heavily penalizes any negative Cd prediction,
    preventing the optimizer from exploiting physically invalid solutions.
    """

    def __init__(self, nn_eval_model, user_input,
                 beta_cl=100, beta_cd=1000, beta_cm=100, beta_barrier=1e6,
                 optimizer=None, max_iters=3000,
                 output_device='cpu', verbose=True):

        super().__init__(
            evaluation_model=_DummyEvalModel(),
            user_input=user_input,
            optimizer=optimizer,
            max_iters=max_iters,
            output_device=output_device,
            verbose=verbose,
        )
        self.nn_model     = nn_eval_model
        self.beta_cl      = beta_cl
        self.beta_cd      = beta_cd
        self.beta_cm      = beta_cm
        self.beta_barrier = beta_barrier

    @staticmethod
    def _kl_symm(p, q, jitter=1e-4):
        p_j = torch.distributions.Normal(p.loc, p.scale + jitter)
        q_j = torch.distributions.Normal(q.loc, q.scale + jitter)
        return 0.5 * (torch.distributions.kl_divergence(p_j, q_j) +
                      torch.distributions.kl_divergence(q_j, p_j))

    def run_design(self, num_particles=1):
        assert self.airfoil.initial_design is not None
        assert self.airfoil.target_cp is None

        self.airfoil.pods_sample = self.airfoil.initial_design.clone()

        result, history = self.svi(num_particles=num_particles)

        start = time.time()
        H = self._nn_gradient_cov(self.airfoil.num_gradient_samples)
        _, _, W, V = self._probabilistic_designer__partition_active_space(
            H, self.airfoil.num_active_dims)
        tol_pca, tol_geom = self._probabilistic_designer__determine_inactive_space(
            self.airfoil.pods_distrib.mean, W, V)
        self.airfoil.design_pca_distrib     = tol_pca
        self.airfoil.design_airfoil_distrib = tol_geom
        if self.verbose:
            print(f'Tolerance analysis runtime: {time.time()-start:.1f}s')

        return result, history

    def svi(self, num_particles=1, prior_loss_scale=0.1, variational_init_scale=0.1):
        pyro.clear_param_store()

        target_cl = self.airfoil.target_cl
        target_cm = self.airfoil.target_cm
        beta_cl   = self.beta_cl
        beta_cd   = self.beta_cd
        beta_cm   = self.beta_cm
        nn_model  = self.nn_model
        airfoil   = self.airfoil

        def pyro_model():
            initial_af = airfoil.initial_design.clone().detach()
            vars_scale = (torch.ones(15, dtype=initial_af.dtype) * 0.01
                          if airfoil.design_variance is None
                          else airfoil.design_variance)

            with pyro.poutine.scale(scale=prior_loss_scale):
                vars_sample = pyro.sample(
                    "vars_sample",
                    dist.Normal(initial_af, vars_scale).to_event(1))

            coeffs   = nn_model(airfoil, sample_tensor=vars_sample)
            cd_val   = coeffs['cd'].loc
            barrier  = self.beta_barrier * torch.relu(-cd_val) ** 2

            pyro.factor("loss_cd",      -beta_cd * cd_val)
            pyro.factor("loss_barrier", -barrier)

            if target_cl is not None:
                pyro.factor("loss_cl",
                    -beta_cl * self._kl_symm(coeffs['cl'], target_cl))
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
        if self.verbose:
            print("Minimizing Cd via SVI (Cl & Cm constrained)...")

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

                    if self.history['total_iter'] >= self.airfoil.mavg_window + 1:
                        delta = torch.stack(self.history['delta_distrib']).numpy()
                        w     = self.airfoil.mavg_window
                        prev  = delta[-w-1:-2, :self.airfoil.tracking_num].mean(0)
                        curr  = delta[-w:-1,   :self.airfoil.tracking_num].mean(0)
                        diff  = (curr - prev) / (np.abs(prev) + 1e-12)
                        if (np.all(np.abs(diff) <= self.airfoil.tolerance)
                                and np.all(np.abs(diff) > 0.0)):
                            if self.verbose:
                                print("  Convergence achieved.")
                            self.history['total_iter'] += 1
                            break

                self.history['total_iter'] += 1

        if self.verbose:
            print(f'SVI runtime: {time.time()-start:.1f}s')

        self.airfoil.pods_distrib = torch.distributions.MultivariateNormal(
            posterior.mean.cpu(), posterior.covariance_matrix.cpu())

        with torch.no_grad():
            final = self.nn_model(self.airfoil,
                                  sample_tensor=self.airfoil.pods_distrib.mean)
        print(f"\n  Final design NN predictions:")
        print(f"    Cl = {final['cl'].loc.item():.4f}  "
              f"(target {target_cl.loc.item():.4f})" if target_cl is not None
              else f"    Cl = {final['cl'].loc.item():.4f}")
        print(f"    Cd = {final['cd'].loc.item():.5f}  (minimized)")
        print(f"    Cm = {final['cm'].loc.item():.5f}  "
              f"(target {target_cm.loc.item():.5f})" if target_cm is not None
              else f"    Cm = {final['cm'].loc.item():.5f}")

        return self.airfoil, self.history

    def _nn_gradient_cov(self, num_samples=1000):
        if self.verbose:
            print(f"  Approximating gradient covariance from {num_samples} samples...")

        candidates = self.airfoil.pods_distrib.sample(torch.Size([num_samples]))
        H = torch.zeros(15, 15, dtype=torch.float64)

        target_list = [self.airfoil.target_cl, None, self.airfoil.target_cm]
        beta_list   = [self.beta_cl, self.beta_cd, self.beta_cm]
        key_list    = ['cl', 'cd', 'cm']

        for beta, tgt, key in zip(beta_list, target_list, key_list):
            if self.verbose:
                print(f"    w.r.t. {key.upper()}...")
            C = torch.zeros(15, 15, dtype=torch.float64)

            for cand in candidates:
                cand_leaf = cand.clone().detach().requires_grad_(True)
                coeffs    = self.nn_model(self.airfoil, sample_tensor=cand_leaf)
                cd_val    = coeffs['cd'].loc

                if key == 'cd':
                    cd_val  = coeffs['cd'].loc
                    barrier = self.beta_barrier * torch.relu(-cd_val) ** 2
                    loss    = beta * cd_val + barrier
                elif tgt is not None:
                    loss = self._kl_symm(tgt, coeffs[key]) * beta
                else:
                    continue

                loss.backward()
                if cand_leaf.grad is not None:
                    g  = cand_leaf.grad.detach().double().cpu()
                    C += torch.outer(g, g)

            C /= num_samples
            H += beta * C

        return H


class _DummyEvalModel(inverse.evaluation_model):
    def __init__(self):
        self.model = None
        self.output_device = 'cpu'

    def predict(self, *args, **kwargs):
        raise RuntimeError("_DummyEvalModel should never be called.")
    
def report_negative_drag_iterations(hist, results_path):
    cd_arr = np.array(hist['cd'])
    neg_idx = np.where(cd_arr < 0)[0]

    print("\n" + "=" * 60)
    print("NEGATIVE DRAG REPORT")
    print("=" * 60)

    print(f"Total iterations: {len(cd_arr)}")
    print(f"Negative Cd iterations: {len(neg_idx)}")

    if len(neg_idx) == 0:
        print("No negative drag iterations found.")
        return

    neg_iters = np.array(hist['iters'])[neg_idx]
    neg_cd = cd_arr[neg_idx]

    print(f"First negative iteration: {neg_iters[0]}")
    print(f"Minimum Cd: {neg_cd.min():.6f}")

    # Save csv
    summary = np.column_stack([neg_iters, neg_cd])

    np.savetxt(
        os.path.join(results_path, "negative_drag_iterations.csv"),
        summary,
        delimiter=",",
        header="iteration,cd",
        comments=""
    )

    # Plot
    plt.figure(figsize=(10,5))

    plt.plot(hist['iters'], cd_arr, label='Cd History')
    plt.scatter(
        neg_iters,
        neg_cd,
        color='red',
        s=25,
        label='Negative Cd'
    )

    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("Iteration")
    plt.ylabel("Cd")
    plt.title("Negative Drag Iterations")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        os.path.join(results_path, "negative_drag_history.png"),
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()

    print("Saved:")
    print("  negative_drag_history.png")
    print("  negative_drag_iterations.csv")


if __name__ == '__main__':

    NACA_CODE = 2412

    MODEL_CL_PATH  = r"C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Results CST NN Split\Cl_PyTorch_400_3\cl_model.pt"
    MODEL_CD_PATH  = r"C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Results CST NN Split\Cd_PyTorch_400_4\cd_model.pt"
    MODEL_CM_PATH  = r"C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Results CST NN Split\Cm_PyTorch_400_4\cm_model.pt"
    SCALER_CL_PATH = r"C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Results CST NN Split\Cl_PyTorch_400_3\cl_scalers.pkl"
    SCALER_CD_PATH = r"C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Results CST NN Split\Cd_PyTorch_400_4\cd_scalers.pkl"
    SCALER_CM_PATH = r"C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\Results CST NN Split\Cm_PyTorch_400_4\cm_scalers.pkl"

    RESULTS_PATH = r'C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO\CSTNN Results'
    os.makedirs(RESULTS_PATH, exist_ok=True)

    output_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype   = torch.float64
    print(f"Running on: {output_device}")

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

    DESIGN_ALPHA = 0.0
    DESIGN_MACH  = 0.14
    DESIGN_RE    = 1_000_000

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

    pca_transformer = lam_pca_transformer(output_device)

    targets = build_targets_from_naca(
        nn_eval, NACA_CODE, DESIGN_MACH, DESIGN_RE, DESIGN_ALPHA)

    baseline_pca = naca_to_pca_approx(NACA_CODE, pca_transformer, torch_dtype, output_device)

    initial_guess = baseline_pca + \
        0.02 * torch.randn(15, dtype=torch_dtype, device=output_device)

    input_file = inverse.airfoil_design_input(
        alpha=[DESIGN_ALPHA],
        mach=[DESIGN_MACH],
        initial_design=initial_guess,
        num_active_dims='auto',
        num_gradient_samples=500,
        target_cp=None,
        target_cl=targets['target_cl'],
        target_cd=None,
        target_cm=targets['target_cm'],
        target_design=None,
        pca_transformer=pca_transformer,
        tolerance=5e-4,
    )

    airfoil_designer = NNDragMinimizationDesigner(
        nn_eval_model=nn_eval,
        user_input=input_file,
        beta_cl=100,
        beta_cd=1000,
        beta_cm=100,
        beta_barrier=1e6,
        max_iters=2000,
        output_device=output_device,
        verbose=True,
    )

    design, hist = airfoil_designer.run_design(num_particles=1)
    report_negative_drag_iterations(hist, RESULTS_PATH)
    neg_idx = np.where(np.array(hist['cd']) < 0)[0]

if len(neg_idx) > 0:

    fig, ax = plt.subplots(figsize=(10,6))

    # baseline
    xu_b, yu_b, xl_b, yl_b = _naca4_coords_np(NACA_CODE)

    ax.plot(
        np.concatenate([xu_b, xl_b[::-1]]),
        np.concatenate([yu_b, yl_b[::-1]]),
        'k--',
        linewidth=2,
        label=f'NACA {NACA_CODE}'
    )

    xc = LAM_XC.cpu().numpy()

    # plot up to 5 suspicious airfoils
    for idx in neg_idx[:5]:

        pca_vec = hist['pods_distrib'][idx].to(output_device)

        phys = pca_transformer.inverse(
            pca_vec,
            get_physical_coords=True
        ).detach().cpu().numpy()

        yu = phys[:28]
        yl = phys[28:]

        iter_num = hist['iters'][idx]
        cd_val = hist['cd'][idx]

        ax.plot(
            xc, yu,
            alpha=0.8,
            label=f'Iter {iter_num}, Cd={cd_val:.4f}'
        )

        ax.plot(
            xc, yl,
            alpha=0.8
        )

    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    ax.set_title("Negative Drag Airfoil Candidates")

    plt.tight_layout()

    plt.savefig(
        os.path.join(RESULTS_PATH, "negative_drag_airfoils.png"),
        dpi=300
    )

    plt.show()

    dgn_af = inverse.helper_airfoil_plotting(design, return_dict=True)
    tgt_af = inverse.helper_airfoil_plotting(
        design, manual_distrib=baseline_pca.to(output_device), return_dict=True)

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
        legend_labels=[f'Drag-minimized design ± tolerance', f'NACA {NACA_CODE}'],
    )
    ax.set_ylim([-0.12, 0.12])
    leg.set_loc('lower right')
    ax.set_title(f'NN Drag Minimization — NACA {NACA_CODE} Baseline')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'airfoil.png'), dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.semilogy(hist['loss'])
    plt.title("Loss Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_PATH, 'loss.png'), dpi=300, bbox_inches='tight')
    plt.show()

    baseline_cd = targets['values']['cd']
    baseline_cl = targets['values']['cl']
    baseline_cm = targets['values']['cm']

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(hist['iters'], hist['cl'], color='tab:blue', label='Design')
    axes[0].axhline(baseline_cl, color='k', linestyle='--', label=f'NACA {NACA_CODE} baseline')
    axes[0].set_ylabel('Cl')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(hist['iters'], hist['cd'], color='tab:orange', label='Design')
    axes[1].axhline(baseline_cd, color='k', linestyle='--', label=f'NACA {NACA_CODE} baseline')
    axes[1].set_ylabel('Cd  (minimized)')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(hist['iters'], hist['cm'], color='tab:green', label='Design')
    axes[2].axhline(baseline_cm, color='k', linestyle='--', label=f'NACA {NACA_CODE} baseline')
    axes[2].set_ylabel('Cm')
    axes[2].set_xlabel('Iteration')
    axes[2].legend()
    axes[2].grid(True)

    fig.suptitle(f'Aerodynamic Coefficient Convergence  —  NACA {NACA_CODE} baseline')
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_PATH, 'coefficients.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nBaseline NACA {NACA_CODE}:")
    print(f"  Cl = {baseline_cl:.4f}")
    print(f"  Cd = {baseline_cd:.5f}")
    print(f"  Cm = {baseline_cm:.5f}")
    print(f"\nFinal optimized design:")
    print(f"  Cl = {hist['cl'][-1]:.4f}")
    print(f"  Cd = {hist['cd'][-1]:.5f}  (Δ = {hist['cd'][-1] - baseline_cd:+.5f})")
    print(f"  Cm = {hist['cm'][-1]:.5f}")