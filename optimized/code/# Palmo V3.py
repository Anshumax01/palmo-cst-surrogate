import sys, time, pickle, os, subprocess, tempfile
from math import comb

import numpy as np
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from matplotlib import pyplot as plt

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
_INIT_DEVICE = initialize_devices(use_gpu=False, gpu_id=0)
initialize_plot_settings()


class CSTNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple):
        super().__init__()
        layers, prev = [], input_dim
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
        xc_inner      = xc[1:-1].double().clamp(1e-9, 1 - 1e-9)
        self.B_pinv   = torch.linalg.pinv(self._basis(xc_inner, n_cst))
        self.xc_inner = xc_inner

    @staticmethod
    def _basis(x: torch.Tensor, n_cst: int) -> torch.Tensor:
        n = n_cst - 1
        C = torch.sqrt(x) * (1.0 - x)
        return torch.stack(
            [C * x.pow(k) * (1 - x).pow(n - k) * comb(n, k) for k in range(n_cst)],
            dim=1)

    def fit(self, ycu: torch.Tensor, ycl: torch.Tensor) -> tuple:
        xc_i = self.xc_inner.to(ycu.device)
        B_iv = self.B_pinv.to(ycu.device)
        return (B_iv @ (ycu[1:-1] - 0.5 * TAIL * xc_i),
                B_iv @ (ycl[1:-1] + 0.5 * TAIL * xc_i))


_BERNSTEIN_FITTER = BernsteinFitter()


class TorchStandardScaler:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = torch.as_tensor(mean)
        self.std  = torch.as_tensor(std)

    @classmethod
    def from_sklearn(cls, sk_scaler):
        return cls(sk_scaler.mean_, sk_scaler.scale_)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        m = self.mean.to(device=x.device, dtype=x.dtype)
        s = self.std.to(device=x.device, dtype=x.dtype)
        return (x - m) / s


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

        self.model_cl,  self.model_cd,  self.model_cm  = model_cl,   model_cd,   model_cm
        self.scaler_cl, self.scaler_cd, self.scaler_cm = scaler_X_cl, scaler_X_cd, scaler_X_cm
        self.mean_y_cl, self.std_y_cl  = mean_y_cl, std_y_cl
        self.mean_y_cd, self.std_y_cd  = mean_y_cd, std_y_cd
        self.mean_y_cm, self.std_y_cm  = mean_y_cm, std_y_cm
        self.mach,      self.reynolds, self.alpha      = mach, reynolds, alpha
        self.output_device                             = output_device
        self.nn_std_cl, self.nn_std_cd, self.nn_std_cm = nn_std_cl, nn_std_cd, nn_std_cm
        for m in [model_cl, model_cd, model_cm]:
            m.eval()

    def _pca_to_cst(self, pca_transformer, pca_vec):
        phys = pca_transformer.inverse(pca_vec.to(self.output_device),
                                       get_physical_coords=True)
        return _BERNSTEIN_FITTER.fit(phys[:28], phys[28:])

    def _build_input(self, cst_u, cst_l):
        p      = next(self.model_cl.parameters())
        flight = torch.tensor([self.mach, self.reynolds, self.alpha],
                              dtype=p.dtype, device=p.device)
        return torch.cat([flight,
                          cst_u.to(device=p.device, dtype=p.dtype),
                          cst_l.to(device=p.device, dtype=p.dtype)])

    def _forward(self, nn_model, scaler, mean_y, std_y, x_raw):
        p    = next(nn_model.parameters())
        x_in = x_raw.unsqueeze(0).to(device=p.device, dtype=p.dtype)
        x_sc = scaler.transform(x_in).to(device=p.device, dtype=p.dtype)
        y_sc = nn_model(x_sc).squeeze()
        return (y_sc
                * torch.as_tensor(std_y, device=p.device, dtype=p.dtype)
                + torch.as_tensor(mean_y, device=p.device, dtype=p.dtype))

    def predict(self, airfoil_input, sample_tensor, **kwargs):
        cst_u, cst_l = self._pca_to_cst(airfoil_input.pca_transformer, sample_tensor)
        x_raw        = self._build_input(cst_u, cst_l)

        cl_val = self._forward(self.model_cl, self.scaler_cl,
                               self.mean_y_cl, self.std_y_cl, x_raw)
        cd_val = self._forward(self.model_cd, self.scaler_cd,
                               self.mean_y_cd, self.std_y_cd, x_raw)
        cm_val = self._forward(self.model_cm, self.scaler_cm,
                               self.mean_y_cm, self.std_y_cm, x_raw)

        mk_n = lambda v, s: torch.distributions.Normal(
            v, torch.as_tensor(s, device=v.device, dtype=v.dtype))

        return {
            'cl':  mk_n(cl_val, self.nn_std_cl),
            'cd':  mk_n(cd_val, self.nn_std_cd),
            'cm':  mk_n(cm_val, self.nn_std_cm),
            '_cl': cl_val,
            '_cd': cd_val,
            '_cm': cm_val,
        }

    def __call__(self, airfoil_input, sample_tensor, **kwargs):
        return self.predict(airfoil_input, sample_tensor, **kwargs)


def naca4_to_pca_approx(code: int,
                         pca_transformer,
                         device,
                         dtype=torch.float64,
                         n_iter: int = 500,
                         lr: float = 5e-3) -> torch.Tensor:
    print(f"  Projecting NACA {code:04d} into PCA space ({n_iter} steps)...")

    code = int(code)
    m = (code // 1000) / 100.0
    p = ((code % 1000) // 100) / 10.0
    t = (code % 100) / 100.0

    x  = LAM_XC.numpy()
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x
              - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)

    if m == 0.0 or p == 0.0:
        yc = np.zeros_like(x)
    else:
        yc = np.where(x < p,
                      m/p**2 * (2*p*x - x**2),
                      m/(1-p)**2 * (1 - 2*p + 2*p*x - x**2))

    yu_np = yc + yt
    yl_np = yc - yt

    yc_target = torch.tensor(
        np.concatenate([yu_np, yl_np]), dtype=dtype, device=device)

    z   = torch.zeros(15, dtype=dtype, device=device, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)

    for _ in range(n_iter):
        opt.zero_grad()
        phys = pca_transformer.inverse(z, get_physical_coords=True)
        loss = ((phys - yc_target) ** 2).mean()
        loss.backward()
        opt.step()

    z_final = z.detach()
    print(f"  Projection RMSE (y/c): {loss.item():.2e}")

    with torch.no_grad():
        phys_final = pca_transformer.inverse(z_final, get_physical_coords=True)
        phys_np    = phys_final.cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(x, yu_np,        'k-',  label=f'NACA {code:04d} upper')
    ax.plot(x, yl_np,        'k-',  label=f'NACA {code:04d} lower')
    ax.plot(x, phys_np[:28], 'r--', label='PCA projection upper')
    ax.plot(x, phys_np[28:], 'r--', label='PCA projection lower')
    ax.set_title(f'NACA {code:04d} vs PCA projection  (RMSE = {loss.item():.2e})')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.legend(fontsize=8)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, f'naca{code:04d}_projection_check.png'),
                dpi=150, bbox_inches='tight')
    plt.show()

    return z_final


def evaluate_pca_vec(nn_eval: NNCSTEvaluationModel,
                     pca_vec: torch.Tensor,
                     pca_transformer) -> dict:
    class _Stub:
        pass
    stub = _Stub()
    stub.pca_transformer = pca_transformer

    with torch.no_grad():
        out = nn_eval(stub, pca_vec.to(nn_eval.output_device))

    return {'cl': out['_cl'].item(),
            'cd': out['_cd'].item(),
            'cm': out['_cm'].item()}


class _DummyEvalModel(inverse.evaluation_model):
    def __init__(self):
        self.model         = None
        self.output_device = 'cpu'

    def predict(self, *args, **kwargs):
        raise RuntimeError("_DummyEvalModel.predict() should never be called.")


class NNProbabilisticDesigner(inverse.probabilistic_designer):
    def __init__(self,
                 nn_eval_model,
                 user_input,
                 betas         = (100, 1000, 100),
                 optimizer     = None,
                 max_iters     = 3000,
                 output_device = 'cpu',
                 verbose       = True,
                 lock_cm       = True):

        super().__init__(
            evaluation_model=_DummyEvalModel(),
            user_input=user_input,
            optimizer=optimizer,
            max_iters=max_iters,
            output_device=output_device,
            verbose=verbose,
        )
        self.nn_model = nn_eval_model
        self.betas    = list(betas)
        self.lock_cm  = lock_cm

    @staticmethod
    def _kl_symm(p, q, jitter=1e-4):
        p_j = torch.distributions.Normal(p.loc, p.scale + jitter)
        q_j = torch.distributions.Normal(q.loc, q.scale + jitter)
        return 0.5 * (torch.distributions.kl_divergence(p_j, q_j)
                      + torch.distributions.kl_divergence(q_j, p_j))

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
            print(f'  Tolerance analysis: {time.time()-start:.1f}s')

        return result, history

    def svi(self,
            num_particles          = 1,
            prior_loss_scale       = 0.1,
            variational_init_scale = 0.1):

        pyro.clear_param_store()

        target_cl = self.airfoil.target_cl
        target_cm = self.airfoil.target_cm
        beta_cl, beta_cd, beta_cm = self.betas
        nn_model = self.nn_model
        airfoil  = self.airfoil
        lock_cm  = self.lock_cm

        def pyro_model():
            initial_af = airfoil.initial_design.clone().detach()
            vars_scale = (torch.ones(15, dtype=initial_af.dtype) * 0.01
                          if airfoil.design_variance is None
                          else airfoil.design_variance)

            with pyro.poutine.scale(scale=prior_loss_scale):
                vars_sample = pyro.sample(
                    "vars_sample",
                    dist.Normal(initial_af, vars_scale).to_event(1))

            coeffs = nn_model(airfoil, sample_tensor=vars_sample)

            if target_cl is not None:
                pyro.factor("loss_cl",
                    -beta_cl * self._kl_symm(coeffs['cl'], target_cl))

            pyro.factor("loss_cd", -beta_cd * coeffs['_cd'] ** 2)

            if lock_cm and target_cm is not None:
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
                })

        svi_obj = SVI(pyro_model, guide, self.optimizer,
                      loss=Trace_ELBO(num_particles=num_particles))

        start = time.time()
        if self.verbose:
            print("  Running SVI (Cd minimisation)...")

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

                    mc = self.nn_model(self.airfoil, sample_tensor=posterior.mean)
                    self.history['cl'].append(mc['cl'].loc.item())
                    self.history['cd'].append(mc['cd'].loc.item())
                    self.history['cm'].append(mc['cm'].loc.item())
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
            print(f'  SVI runtime: {time.time()-start:.1f}s')

        self.airfoil.pods_distrib = torch.distributions.MultivariateNormal(
            posterior.mean.cpu(), posterior.covariance_matrix.cpu())

        with torch.no_grad():
            final = self.nn_model(self.airfoil,
                                  sample_tensor=self.airfoil.pods_distrib.mean)

        cl_t = target_cl.loc.item() if target_cl is not None else float('nan')
        cm_t = (self.airfoil.target_cm.loc.item()
                if self.airfoil.target_cm is not None else float('nan'))

        print("\n  Final NN predictions (designed shape):")
        print(f"    Cl = {final['cl'].loc.item():.4f}   (target {cl_t:.4f})")
        print(f"    Cd = {final['cd'].loc.item():.5f}   minimised")
        print(f"    Cm = {final['cm'].loc.item():.5f}   (target {cm_t:.5f})")

        return self.airfoil, self.history

    def _nn_gradient_cov(self, num_samples=1000):
        if self.verbose:
            print(f"  Computing gradient covariance ({num_samples} samples)...")

        candidates = self.airfoil.pods_distrib.sample(torch.Size([num_samples]))
        H = torch.zeros(15, 15, dtype=torch.float64)

        for beta, tgt, key, is_min in zip(
                self.betas,
                [self.airfoil.target_cl, None, self.airfoil.target_cm],
                ['cl', 'cd', 'cm'],
                [False, True, False]):

            if tgt is None and not is_min:
                continue

            if self.verbose:
                suffix = " (minimisation)" if is_min else ""
                print(f"    w.r.t. {key.upper()}{suffix}...")

            C = torch.zeros(15, 15, dtype=torch.float64)
            for cand in candidates:
                cand_leaf = cand.clone().detach().requires_grad_(True)
                coeffs    = self.nn_model(self.airfoil, sample_tensor=cand_leaf)

                if is_min:
                    loss_val = beta * coeffs['_cd'] ** 2
                else:
                    loss_val = self._kl_symm(tgt, coeffs[key]) * beta

                loss_val.backward()
                if cand_leaf.grad is not None:
                    g  = cand_leaf.grad.detach().double().cpu()
                    C += torch.outer(g, g)

            H += beta * (C / num_samples)

        return H


def _write_xfoil_coords(xu, yu, xl, yl, path: str):
    xu_r, yu_r = xu[::-1], yu[::-1]
    xs = np.concatenate([xu_r, xl[1:]])
    ys = np.concatenate([yu_r, yl[1:]])
    with open(path, 'w') as f:
        f.write("DESIGNED AIRFOIL\n")
        for xi, yi in zip(xs, ys):
            f.write(f"  {xi:.6f}  {yi:.6f}\n")


def run_xfoil(xu, yu, xl, yl,
              alpha: float, re: float, mach: float,
              n_crit: float = 9.0,
              xfoil_exe: str = 'xfoil') -> dict | None:
    with tempfile.TemporaryDirectory() as tmpdir:
        coord_file = os.path.join(tmpdir, 'airfoil.dat')
        polar_file = os.path.join(tmpdir, 'polar.txt')
        _write_xfoil_coords(xu, yu, xl, yl, coord_file)

        cmds = (
            f"LOAD {coord_file}\n"
            "PANE\n"
            "OPER\n"
            f"VISC {re:.0f}\n"
            f"MACH {mach:.4f}\n"
            "VPAR\n"
            f"N {n_crit}\n"
            "\n"
            f"PACC {polar_file} /dev/null\n"
            f"ALFA {alpha:.2f}\n"
            "PACC\n"
            "\n"
            "QUIT\n"
        )
        try:
            proc = subprocess.run(
                [xfoil_exe],
                input=cmds, capture_output=True, text=True, timeout=30)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"  [XFoil] Not available or timed out: {e}")
            return None

        if not os.path.exists(polar_file):
            print("  [XFoil] No polar output produced.")
            return None

        with open(polar_file) as pf:
            lines = [l for l in pf if not l.startswith('#')]

        data_lines = [l for l in lines if len(l.split()) >= 6
                      and not l.strip()[0].isalpha()]
        if not data_lines:
            return None

        cols = data_lines[-1].split()
        return {'cl': float(cols[1]),
                'cd': float(cols[2]),
                'cm': float(cols[4])}


def pca_to_physical_coords(pca_vec, pca_transformer):
    phys = pca_transformer.inverse(pca_vec, get_physical_coords=True)
    phys = phys.detach().cpu().numpy()
    n    = len(phys) // 2
    yu, yl = phys[:n], phys[n:]
    xc = LAM_XC.numpy()
    return xc, yu, xc, yl


def validate_with_cfd(pca_vec,
                      pca_transformer,
                      alpha: float,
                      mach:  float,
                      re:    float,
                      label: str = "shape",
                      xfoil_exe: str = 'xfoil') -> dict | None:
    print(f"\n  CFD validation: {label}")
    xc, yu, _, yl = pca_to_physical_coords(pca_vec.cpu(), pca_transformer)
    result = run_xfoil(xc, yu, xc, yl,
                       alpha=alpha, re=re, mach=mach,
                       xfoil_exe=xfoil_exe)
    if result:
        print(f"    Cl  = {result['cl']:.4f}")
        print(f"    Cd  = {result['cd']:.5f}")
        print(f"    Cm  = {result['cm']:.5f}")
        print(f"    L/D = {result['cl']/result['cd']:.1f}")
    else:
        print("    CFD run failed or XFoil not found.")
    return result


if __name__ == '__main__':

    NACA_CODE  = 2412
    PCA_VECTOR = None

    DESIGN_ALPHA = 0.0
    DESIGN_MACH  = 0.14
    DESIGN_RE    = 1_000_000

    LOCK_CM   = True
    BETA_CL   = 100
    BETA_CD   = 1000
    BETA_CM   = 100
    MAX_ITERS = 2000
    STD_CL    = 0.03
    STD_CM    = 0.002

    RUN_CFD   = True
    XFOIL_EXE = 'xfoil'

    BASE = r'C:\Users\anshu\Documents\Georgia Tech 26\Research\PALMO'
    MODEL_CL_PATH  = rf'{BASE}\Results CST NN Split\Cl_PyTorch_400_3\cl_model.pt'
    MODEL_CD_PATH  = rf'{BASE}\Results CST NN Split\Cd_PyTorch_400_4\cd_model.pt'
    MODEL_CM_PATH  = rf'{BASE}\Results CST NN Split\Cm_PyTorch_400_4\cm_model.pt'
    SCALER_CL_PATH = rf'{BASE}\Results CST NN Split\Cl_PyTorch_400_3\cl_scalers.pkl'
    SCALER_CD_PATH = rf'{BASE}\Results CST NN Split\Cd_PyTorch_400_4\cd_scalers.pkl'
    SCALER_CM_PATH = rf'{BASE}\Results CST NN Split\Cm_PyTorch_400_4\cm_scalers.pkl'
    RESULTS_PATH   = rf'{BASE}\CSTNN Results'

    os.makedirs(RESULTS_PATH, exist_ok=True)
    output_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype   = torch.float64
    print(f"\nRunning on: {output_device}")

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

    def _load_scaler(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return {k: (torch.as_tensor(v, dtype=torch_dtype).to(output_device)
                    if isinstance(v, (list, tuple, float, int, np.ndarray)) else v)
                for k, v in data.items()}

    cl_sc = _load_scaler(SCALER_CL_PATH)
    cd_sc = _load_scaler(SCALER_CD_PATH)
    cm_sc = _load_scaler(SCALER_CM_PATH)

    nn_eval = NNCSTEvaluationModel(
        model_cl, model_cd, model_cm,
        TorchStandardScaler(cl_sc['mean_X'], cl_sc['std_X']),
        TorchStandardScaler(cd_sc['mean_X'], cd_sc['std_X']),
        TorchStandardScaler(cm_sc['mean_X'], cm_sc['std_X']),
        cl_sc['mean_y'], cl_sc['std_y'],
        cd_sc['mean_y'], cd_sc['std_y'],
        cm_sc['mean_y'], cm_sc['std_y'],
        DESIGN_MACH, DESIGN_RE, DESIGN_ALPHA,
        output_device,
    )

    pca_transformer = lam_pca_transformer(output_device)

    if NACA_CODE is not None:
        print(f"\n{'='*55}")
        print(f"  Reference airfoil: NACA {NACA_CODE:04d}")
        ref_pca = naca4_to_pca_approx(
            NACA_CODE, pca_transformer, output_device, torch_dtype)
    elif PCA_VECTOR is not None:
        print(f"\n{'='*55}")
        print("  Reference airfoil: user-supplied PCA vector")
        ref_pca = torch.as_tensor(PCA_VECTOR, dtype=torch_dtype, device=output_device)
    else:
        raise ValueError("Set either NACA_CODE or PCA_VECTOR.")

    ref_perf = evaluate_pca_vec(nn_eval, ref_pca, pca_transformer)

    print(f"\n  NN predictions for reference airfoil "
          f"(Mach={DESIGN_MACH}, Re={DESIGN_RE:.0e}, α={DESIGN_ALPHA}°):")
    print(f"    Cl = {ref_perf['cl']:.4f}  will be LOCKED as target")
    print(f"    Cd = {ref_perf['cd']:.5f}  will be MINIMISED")
    print(f"    Cm = {ref_perf['cm']:.5f}  " + ("will be LOCKED" if LOCK_CM else "free"))
    print('='*55)

    mk = lambda v, s: torch.distributions.Normal(
        torch.as_tensor(v, dtype=torch_dtype),
        torch.as_tensor(s, dtype=torch_dtype))

    target_cl = mk(ref_perf['cl'], STD_CL)
    target_cm = mk(ref_perf['cm'], STD_CM) if LOCK_CM else None

    initial_guess = ref_pca + 0.02 * torch.randn(15, dtype=torch_dtype, device=output_device)

    input_file = inverse.airfoil_design_input(
        alpha=[DESIGN_ALPHA],
        mach=[DESIGN_MACH],
        initial_design=initial_guess,
        num_active_dims='auto',
        num_gradient_samples=500,
        target_cp=None,
        target_cl=target_cl,
        target_cd=None,
        target_cm=target_cm,
        target_design=None,
        pca_transformer=pca_transformer,
        tolerance=5e-4,
    )

    airfoil_designer = NNProbabilisticDesigner(
        nn_eval_model=nn_eval,
        user_input=input_file,
        betas=[BETA_CL, BETA_CD, BETA_CM],
        max_iters=MAX_ITERS,
        output_device=output_device,
        verbose=True,
        lock_cm=LOCK_CM,
    )

    design, hist = airfoil_designer.run_design(num_particles=1)

    designed_pca = design.pods_distrib.mean.to(output_device)

    cfd_ref, cfd_des = None, None

    if RUN_CFD:
        print(f"\n{'='*55}")
        print("  CFD Validation (XFoil)")
        print(f"  Condition: Mach={DESIGN_MACH}, Re={DESIGN_RE:.0e}, α={DESIGN_ALPHA}°")
        print('='*55)

        cfd_ref = validate_with_cfd(
            ref_pca, pca_transformer,
            DESIGN_ALPHA, DESIGN_MACH, DESIGN_RE,
            label=f"Reference (NACA {NACA_CODE:04d})" if NACA_CODE else "Reference",
            xfoil_exe=XFOIL_EXE)

        cfd_des = validate_with_cfd(
            designed_pca, pca_transformer,
            DESIGN_ALPHA, DESIGN_MACH, DESIGN_RE,
            label="Designed (drag-minimised)",
            xfoil_exe=XFOIL_EXE)

    run_label = (f"NACA{NACA_CODE:04d}" if NACA_CODE else "custom")

    dgn_af = inverse.helper_airfoil_plotting(design, return_dict=True)
    ref_af = inverse.helper_airfoil_plotting(
        design, manual_distrib=ref_pca.to(output_device), return_dict=True)

    f, ax = plt.subplots(figsize=(9, 4))
    f, ax, leg = visualize.plot_airfoil(
        f, ax,
        plotting_elements=[dgn_af, ref_af],
        kwargs_list=[
            [{'color': 'tab:red', 'linewidth': 2.0},
             {'color': 'tab:red', 'alpha': 0.25}, {}],
            [{'color': 'k', 'marker': 'o', 'linestyle': 'None', 'markersize': 3}],
        ],
        legend_handle=[
            ("tab:red", "tab:red", "-", "None", 1.0),
            ("None", "k", "None", "o", 1.0),
        ],
        legend_labels=['Drag-minimised design ± tolerance',
                       f'Reference ({run_label})'],
    )
    ax.set_ylim([-0.15, 0.15])
    ax.set_title(f'Drag Minimisation — Reference: {run_label}')
    leg.set_loc('lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, f'airfoil_{run_label}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(7, 3))
    plt.semilogy(hist['loss'])
    plt.title("SVI Loss Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, f'loss_{run_label}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(hist['iters'], hist['cl'], 'tab:blue')
    axes[0].axhline(ref_perf['cl'], color='k', ls='--', label='Target Cl (locked)')
    axes[0].set_ylabel('Cl')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(hist['iters'], hist['cd'], 'tab:orange')
    axes[1].axhline(ref_perf['cd'], color='k', ls='--', label='Ref Cd (baseline)')
    axes[1].set_ylabel('Cd')
    axes[1].set_title('Cd — being minimised', fontsize=9)
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(hist['iters'], hist['cm'], 'tab:green')
    axes[2].axhline(ref_perf['cm'], color='k', ls='--',
                    label='Target Cm (locked)' if LOCK_CM else 'Ref Cm (free)')
    axes[2].set_ylabel('Cm')
    axes[2].set_xlabel('Iteration')
    axes[2].legend()
    axes[2].grid(True)

    fig.suptitle('Aerodynamic Coefficient Convergence')
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_PATH, f'coefficients_{run_label}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    nn_des_cl = hist['cl'][-1]
    nn_des_cd = hist['cd'][-1]
    nn_des_cm = hist['cm'][-1]

    delta_cd_nn = nn_des_cd - ref_perf['cd']
    pct_nn      = 100 * delta_cd_nn / (abs(ref_perf['cd']) + 1e-12)

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS — {run_label}  "
          f"(Mach={DESIGN_MACH}, Re={DESIGN_RE:.0e}, α={DESIGN_ALPHA}°)")
    print(f"{'='*60}")
    print(f"  {'Quantity':<10}  {'Reference':>12}  {'Design (NN)':>12}", end="")
    if cfd_ref and cfd_des:
        print(f"  {'Ref (CFD)':>12}  {'Design (CFD)':>14}")
    else:
        print()

    for qty, rv, dv_nn, rv_cfd, dv_cfd in [
        ('Cl', ref_perf['cl'], nn_des_cl,
         cfd_ref['cl'] if cfd_ref else None,
         cfd_des['cl'] if cfd_des else None),
        ('Cd', ref_perf['cd'], nn_des_cd,
         cfd_ref['cd'] if cfd_ref else None,
         cfd_des['cd'] if cfd_des else None),
        ('Cm', ref_perf['cm'], nn_des_cm,
         cfd_ref['cm'] if cfd_ref else None,
         cfd_des['cm'] if cfd_des else None),
    ]:
        row = f"  {qty:<10}  {rv:>12.5f}  {dv_nn:>12.5f}"
        if rv_cfd is not None:
            row += f"  {rv_cfd:>12.5f}  {dv_cfd:>14.5f}"
        print(row)

    print(f"\n  NN surrogate ΔCd : {delta_cd_nn:+.5f}  ({pct_nn:+.1f}%)")
    if cfd_ref and cfd_des:
        delta_cd_cfd = cfd_des['cd'] - cfd_ref['cd']
        pct_cfd      = 100 * delta_cd_cfd / (abs(cfd_ref['cd']) + 1e-12)
        print(f"  XFoil CFD   ΔCd : {delta_cd_cfd:+.5f}  ({pct_cfd:+.1f}%)")
        ld_ref = cfd_ref['cl'] / cfd_ref['cd']
        ld_des = cfd_des['cl'] / cfd_des['cd']
        print(f"\n  L/D  Reference : {ld_ref:.2f}")
        print(f"  L/D  Design    : {ld_des:.2f}  (Δ = {ld_des - ld_ref:+.2f})")
    print('='*60)
    print("\n  Plots and figures saved to:", RESULTS_PATH)