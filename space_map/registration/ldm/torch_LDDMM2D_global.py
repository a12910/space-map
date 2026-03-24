"""
Global LDDMM registration for N 2D images with Z-axis smoothness.

Jointly optimises N velocity fields so that adjacent deformed images match,
with a smoothness constraint on the velocity fields along the Z-axis.

Energy functional
-----------------
E = Σ_{i} ‖φ_i·I_i − φ_{i+1}·I_{i+1}‖² / (2σ²)      [matching]
  + Σ_{i} ⟨v_i, Av_i⟩ / (2σR²)                         [spatial reg]
  + λ_z Σ_{i} ‖v_i − v_{i+1}‖²                          [Z-axis smoothness]
"""
from __future__ import annotations
import torch
import numpy as np
import time
import space_map
from . import torch_LDDMMBase as root
from .torch_LDDMM2D_fast import (
    _unsqueeze2,
    _update_phiinv_dispatch,
    _compute_detjac_2d,
    _gradient2d_fast,
)

grid_sample = root.grid_sample
irfft = root.irfft
rfft = root.rfft


class LDDMM2DGlobal:
    """Joint LDDMM for a stack of *N* 2D images.

    Each slice owns an independent velocity field ``{vt0_k, vt1_k}``.
    One slice (default: the centre) is kept fixed (identity transform) so
    the whole stack is anchored.

    Typical usage::

        g = LDDMM2DGlobal(device='cuda')
        g.load_images([img0, img1, …, imgN])
        g.run(niter=3000)

        for k in range(N):
            ps_new = g.apply_points2d(k, ps_old, xyd)
    """

    # -------------------------------------------------------------- #
    #  Construction / setup                                            #
    # -------------------------------------------------------------- #

    def __init__(
        self,
        device=None,
        nt: int = 5,
        a: float = 5.0,
        p: float = 2.0,
        sigma: float = 20.0,
        sigmaR: float = 80.0,
        lambda_z: float = 0.0005,
        adam_alpha: float = 0.5,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8,
        verbose: int = 10,
    ):
        if device is None:
            device = space_map.DEVICE
        if device == "cpu" or device is None:
            self._dev = torch.device("cpu")
        elif device == -1 or device == "mps":
            self._dev = torch.device("mps")
        elif isinstance(device, int):
            self._dev = torch.device(f"cuda:{device}")
        else:
            self._dev = torch.device(str(device))

        self.nt = nt
        self.dt = 1.0 / nt
        self.a = float(a)
        self.p = float(p)
        self.sigma = float(sigma)
        self.sigmaR = float(sigmaR)
        self.lambda_z = float(lambda_z)
        self.adam_alpha = float(adam_alpha)
        self.adam_beta1 = float(adam_beta1)
        self.adam_beta2 = float(adam_beta2)
        self.adam_eps = float(adam_eps)
        self.verbose = int(verbose)

        self.N: int = 0
        self.images: list[torch.Tensor] = []
        self.fixed_slice: int = -1

    # ---- tensor helper ---- #

    def _t(self, x) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        elif isinstance(x, (int, float)):
            return torch.tensor(float(x), device=self._dev)
        return x.float().to(self._dev)

    # ---- load images ---- #

    def load_images(self, images: list[np.ndarray], fixed_slice: int | None = None):
        """Load *N* images (each ``(H, W)`` numpy array) for joint registration.

        Parameters
        ----------
        images : list[np.ndarray]
            The slice images, ordered along the Z-axis.
        fixed_slice : int, optional
            Index of the slice that stays at identity.  Default: centre.
        """
        self.N = len(images)
        assert self.N >= 2, "Need ≥ 2 images"
        self.fixed_slice = self.N // 2 if fixed_slice is None else fixed_slice

        self.nx = tuple(images[0].shape)
        H, W = self.nx
        self.dx = [1.0, 1.0]

        self.images = [self._t(img) for img in images]

        x0 = np.arange(H, dtype=np.float32)
        x1 = np.arange(W, dtype=np.float32)
        X0, X1 = np.meshgrid(x0, x1, indexing="ij")
        self.X0 = self._t(X0 - X0.mean())
        self.X1 = self._t(X1 - X1.mean())
        self._inv0 = 2.0 / (H * self.dx[0] - self.dx[0])
        self._inv1 = 2.0 / (W * self.dx[1] - self.dx[1])

        _z = lambda: self._t(torch.zeros(self.nx))
        self.vt0 = [[_z() for _ in range(self.nt)] for _ in range(self.N)]
        self.vt1 = [[_z() for _ in range(self.nt)] for _ in range(self.N)]
        self._am0 = [[_z() for _ in range(self.nt)] for _ in range(self.N)]
        self._am1 = [[_z() for _ in range(self.nt)] for _ in range(self.N)]
        self._av0 = [[_z() for _ in range(self.nt)] for _ in range(self.N)]
        self._av1 = [[_z() for _ in range(self.nt)] for _ in range(self.N)]

        self.It: list[list[torch.Tensor | None]] = [
            [None] * (self.nt + 1) for _ in range(self.N)
        ]
        for k in range(self.N):
            self.It[k][0] = self.images[k]
            for t in range(1, self.nt + 1):
                self.It[k][t] = self.images[k].clone()

        self._init_kernels()

    def _init_kernels(self):
        H, W = self.nx
        f0 = np.arange(H, dtype=np.float64) / (self.dx[0] * H)
        f1 = np.arange(W, dtype=np.float64) / (self.dx[1] * W)
        F0, F1 = np.meshgrid(f0, f1, indexing="ij")
        adx = self.a * self.dx[0]
        Ahat = (
            1.0
            - 2.0
            * adx**2
            * (
                (np.cos(2 * np.pi * self.dx[0] * F0) - 1.0) / self.dx[0] ** 2
                + (np.cos(2 * np.pi * self.dx[1] * F1) - 1.0) / self.dx[1] ** 2
            )
        ) ** (2.0 * self.p)
        self.Khat = self._t(np.float32(1.0 / Ahat))
        if root._TORCH_GE_18:
            self.Khat = self.Khat.reshape(H, W)

    # -------------------------------------------------------------- #
    #  Low-level helpers                                               #
    # -------------------------------------------------------------- #

    def _grid(self, phi0, phi1):
        return torch.stack(
            (phi1 * self._inv1, phi0 * self._inv0), dim=2
        ).unsqueeze(0)

    # -------------------------------------------------------------- #
    #  Forward deformation                                             #
    # -------------------------------------------------------------- #

    @torch.no_grad()
    def _fwd(self, k):
        phi = torch.stack((self.X0.clone(), self.X1.clone()), dim=0)
        for t in range(self.nt):
            phi = _update_phiinv_dispatch(
                phi, self.X0, self.X1,
                self.vt0[k][t], self.vt1[k][t],
                self.dt, self._inv0, self._inv1,
            )
            g = self._grid(phi[0], phi[1])
            self.It[k][t + 1] = torch.squeeze(
                grid_sample(_unsqueeze2(self.It[k][0]), g, padding_mode="zeros")
            )

    # -------------------------------------------------------------- #
    #  Energy                                                          #
    # -------------------------------------------------------------- #

    @torch.no_grad()
    def _match_energy_lambda(self):
        lams = [torch.zeros_like(self.X0) for _ in range(self.N)]
        EM = 0.0
        inv_s2 = 1.0 / self.sigma**2
        dA = self.dx[0] * self.dx[1]
        for i in range(self.N - 1):
            diff = self.It[i][-1] - self.It[i + 1][-1]
            EM += (torch.sum(diff**2) * 0.5 * inv_s2 * dA).item()
            lams[i] = lams[i] - diff * inv_s2
            lams[i + 1] = lams[i + 1] + diff * inv_s2
        return EM, lams

    @torch.no_grad()
    def _reg_energy(self):
        ER = 0.0
        inv_K = 1.0 / self.Khat
        c = 0.5 / self.sigmaR**2 * self.dx[0] * self.dx[1] * self.dt
        for k in range(self.N):
            if k == self.fixed_slice:
                continue
            for t in range(self.nt):
                Av0 = irfft(rfft(self.vt0[k][t], 2, onesided=False) * inv_K,
                            2, onesided=False)
                Av1 = irfft(rfft(self.vt1[k][t], 2, onesided=False) * inv_K,
                            2, onesided=False)
                ER += (torch.sum(self.vt0[k][t] * Av0
                                 + self.vt1[k][t] * Av1) * c).item()
        return ER

    @torch.no_grad()
    def _z_energy(self):
        EZ = 0.0
        for k in range(self.N - 1):
            for t in range(self.nt):
                EZ += torch.sum(
                    (self.vt0[k][t] - self.vt0[k + 1][t]) ** 2
                ).item()
                EZ += torch.sum(
                    (self.vt1[k][t] - self.vt1[k + 1][t]) ** 2
                ).item()
        return self.lambda_z * EZ

    # -------------------------------------------------------------- #
    #  Gradient computation (adjoint method) & velocity update         #
    # -------------------------------------------------------------- #

    def _grad_step(self, k, lam, t, pi0, pi1):
        """One adjoint backward step at time *t* for slice *k*."""
        Xb0 = self.X0 + self.vt0[k][t] * self.dt
        Xb1 = self.X1 + self.vt1[k][t] * self.dt
        bg = self._grid(Xb0, Xb1)
        pi0 = torch.squeeze(
            grid_sample(_unsqueeze2(pi0 - self.X0), bg, padding_mode="border")
        ) + Xb0
        pi1 = torch.squeeze(
            grid_sample(_unsqueeze2(pi1 - self.X1), bg, padding_mode="border")
        ) + Xb1

        detjac = _compute_detjac_2d(
            torch.stack((pi0, pi1), dim=0),
            float(self.dx[0]), float(self.dx[1]),
        )
        lam_t = torch.squeeze(
            grid_sample(
                _unsqueeze2(lam), self._grid(pi0, pi1), padding_mode="zeros"
            )
        ) * detjac

        gx, gy = _gradient2d_fast(
            self.It[k][t], float(self.dx[0]), float(self.dx[1])
        )
        return [gx * lam_t, gy * lam_t], pi0, pi1

    @torch.no_grad()
    def _update_slice(self, k, lam, it):
        pi0 = self.X0.clone()
        pi1 = self.X1.clone()

        bc1 = 1.0 - self.adam_beta1 ** (it + 1)
        bc2s = (1.0 - self.adam_beta2 ** (it + 1)) ** 0.5
        alpha = self.adam_alpha * bc2s / bc1
        b1, b2, eps = self.adam_beta1, self.adam_beta2, self.adam_eps
        sr2 = self.sigmaR**2
        lz2 = 2.0 * self.lambda_z

        for t in range(self.nt - 1, -1, -1):
            g, pi0, pi1 = self._grad_step(k, lam, t, pi0, pi1)

            self._am0[k][t] = b1 * self._am0[k][t] + (1.0 - b1) * g[0]
            self._am1[k][t] = b1 * self._am1[k][t] + (1.0 - b1) * g[1]
            self._av0[k][t] = b2 * self._av0[k][t] + (1.0 - b2) * g[0] ** 2
            self._av1[k][t] = b2 * self._av1[k][t] + (1.0 - b2) * g[1] ** 2

            mh0 = self._am0[k][t] / (torch.sqrt(self._av0[k][t]) + eps)
            mh1 = self._am1[k][t] / (torch.sqrt(self._av1[k][t]) + eps)
            u0 = irfft(rfft(mh0, 2, onesided=False) * self.Khat,
                        2, onesided=False)
            u1 = irfft(rfft(mh1, 2, onesided=False) * self.Khat,
                        2, onesided=False)

            self.vt0[k][t] -= alpha * u0 + self.vt0[k][t] / sr2
            self.vt1[k][t] -= alpha * u1 + self.vt1[k][t] / sr2

            if lz2 > 0:
                zg0 = torch.zeros_like(self.vt0[k][t])
                zg1 = torch.zeros_like(self.vt1[k][t])
                if k > 0:
                    zg0 += self.vt0[k][t] - self.vt0[k - 1][t]
                    zg1 += self.vt1[k][t] - self.vt1[k - 1][t]
                if k < self.N - 1:
                    zg0 += self.vt0[k][t] - self.vt0[k + 1][t]
                    zg1 += self.vt1[k][t] - self.vt1[k + 1][t]
                self.vt0[k][t] -= alpha * lz2 * zg0
                self.vt1[k][t] -= alpha * lz2 * zg1

            del g

    # -------------------------------------------------------------- #
    #  Main optimisation loop                                          #
    # -------------------------------------------------------------- #

    def run(self, niter: int = 3000, target_err_skip: float = 0.0001):
        """Run the joint registration.

        Returns
        -------
        list[float]
            Energy history (one value per iteration).
        """
        space_map.Info(
            "LDDMM2DGlobal: start  N=%d  fixed=%d  lz=%.5f"
            % (self.N, self.fixed_slice, self.lambda_z)
        )
        t0 = time.time()
        hist: list[float] = []

        for it in range(niter):
            for k in range(self.N):
                self._fwd(k)

            EM, lams = self._match_energy_lambda()
            ER = self._reg_energy()
            EZ = self._z_energy()
            E = EM + ER + EZ
            hist.append(E)

            if self.verbose > 0 and it % self.verbose == 0:
                space_map.Info(
                    "  iter %5d  E=%.4f  EM=%.4f  ER=%.4f  EZ=%.4f  %.1fs"
                    % (it, E, EM, ER, EZ, time.time() - t0)
                )
                t0 = time.time()

            if len(hist) > 200:
                ref = hist[-200]
                if abs(ref - E) / max(abs(ref), 1e-12) < target_err_skip:
                    space_map.Info(
                        "LDDMM2DGlobal: converged at iter %d  E=%.4f" % (it, E)
                    )
                    break

            for k in range(self.N):
                if k == self.fixed_slice:
                    continue
                self._update_slice(k, lams[k], it)

        space_map.Info("LDDMM2DGlobal: done  E=%.4f" % hist[-1])
        return hist

    # -------------------------------------------------------------- #
    #  Inference: grid / image / points                                #
    # -------------------------------------------------------------- #

    @torch.no_grad()
    def generate_grid(self, k: int) -> np.ndarray:
        """Backward sampling grid for slice *k*.

        Returns ``(1, H, W, 2)`` numpy array in normalised ``[-1, 1]``
        coordinates (same format as ``LDDMM2DFast.generateTransFormGridImg``).
        """
        phi = torch.stack((self.X0.clone(), self.X1.clone()), dim=0)
        for t in range(self.nt):
            phi = _update_phiinv_dispatch(
                phi, self.X0, self.X1,
                self.vt0[k][t], self.vt1[k][t],
                self.dt, self._inv0, self._inv1,
            )
        return self._grid(phi[0], phi[1]).cpu().numpy()

    generate_img_grid = generate_grid

    @torch.no_grad()
    def apply_img(self, k: int, img: np.ndarray | None = None) -> np.ndarray:
        """Warp an image with the transform of slice *k*."""
        src = self.images[k] if img is None else self._t(img)
        g = torch.from_numpy(self.generate_grid(k)).to(self._dev)
        return torch.squeeze(
            grid_sample(_unsqueeze2(src), g, padding_mode="zeros")
        ).cpu().numpy()

    @torch.no_grad()
    def apply_points2d(
        self, k: int, ps: np.ndarray, xyd: float
    ) -> np.ndarray:
        """Forward-map points ``(M, 2)`` for slice *k*.

        Parameters
        ----------
        k : int
            Slice index.
        ps : ndarray (M, 2)
            Points in world coordinates ``(x, y)``.
        xyd : float
            Pixel spacing (``space_map.XYD``).

        Returns
        -------
        ndarray (M, 2)
            Transformed world coordinates.
        """
        H, W = self.nx
        dev = self._dev
        pts = torch.from_numpy(np.asarray(ps, dtype=np.float32)).to(dev)
        col = pts[:, 0] / xyd
        row = pts[:, 1] / xyd

        bwd = torch.from_numpy(self.generate_grid(k)).to(dev)  # (1,H,W,2)

        id_x = torch.linspace(-1, 1, W, device=dev)
        id_y = torch.linspace(-1, 1, H, device=dev)
        igx, igy = torch.meshgrid(id_x, id_y, indexing="xy")
        identity = torch.stack((igx, igy), dim=-1).unsqueeze(0)

        disp = (bwd - identity).squeeze(0).permute(2, 0, 1).unsqueeze(0)

        nc = 2.0 * col / (W - 1) - 1.0
        nr = 2.0 * row / (H - 1) - 1.0
        sg = torch.stack((nc, nr), dim=1).unsqueeze(0).unsqueeze(0)

        pm = "zeros" if dev.type == "mps" else "border"
        sd = torch.nn.functional.grid_sample(
            disp, sg, align_corners=True, padding_mode=pm, mode="bilinear"
        )
        dx_ = sd[0, 0, 0, :]
        dy_ = sd[0, 1, 0, :]

        tc = (nc - dx_ + 1.0) * (W - 1) / 2.0
        tr = (nr - dy_ + 1.0) * (H - 1) / 2.0

        return torch.stack((tc * xyd, tr * xyd), dim=1).cpu().numpy()

    # -------------------------------------------------------------- #
    #  Serialisation                                                   #
    # -------------------------------------------------------------- #

    def output_transforms(self) -> dict:
        """Return all velocity fields as a dict of numpy arrays."""
        out: dict = {"N": self.N, "fixed_slice": self.fixed_slice}
        for k in range(self.N):
            out[f"vt0_{k}"] = np.stack(
                [t.cpu().numpy() for t in self.vt0[k]]
            )
            out[f"vt1_{k}"] = np.stack(
                [t.cpu().numpy() for t in self.vt1[k]]
            )
        return out

    def save(self, path: str):
        np.savez_compressed(path, **self.output_transforms())

    def load_transforms(self, path: str):
        d = np.load(path)
        for k in range(self.N):
            v0 = d[f"vt0_{k}"]
            v1 = d[f"vt1_{k}"]
            for t in range(self.nt):
                self.vt0[k][t] = self._t(v0[t])
                self.vt1[k][t] = self._t(v1[t])
