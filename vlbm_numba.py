import numpy as np
from tqdm import tqdm
from numba import jit, njit, prange
import os 
import argparse


class LBMParams:
    def __init__(self, lx, ly, stencil, dt=1e-3, tau=1e-2, R=1.0):
        self.lx   = lx
        self.ly   = ly
        self.hx   = 3  # halo size
        self.hy   = 3
        self.nx   = lx + 2 * self.hx
        self.ny   = ly + 2 * self.hy

        # Physical parameters
        self.dx  = 2.0 * np.pi / self.lx
        self.dy  = np.pi / self.ly
        self.dt  = dt
        self.tau = tau
        self.R   = R
        
        # Distribution functions
        self.f1  = np.zeros((self.nx, self.ny, stencil.npop), dtype=np.float64)
        self.f2  = np.zeros((self.nx, self.ny, stencil.npop), dtype=np.float64)
        self.frk = np.zeros((self.nx, self.ny, stencil.npop), dtype=np.float64)

        # Macroscopic variables
        self.rho = np.zeros((self.nx, self.ny), dtype=np.float64)
        self.ux  = np.zeros((self.nx, self.ny), dtype=np.float64)
        self.uy  = np.zeros((self.nx, self.ny), dtype=np.float64)


class Geometry:
    def __init__(self, params):
        self.R = params.R
        self.costh = np.zeros(params.ny, dtype=np.float64)
        self.sinth = np.zeros(params.ny, dtype=np.float64)
        
        # Initialize trigonometric arrays
        theta = ((np.arange(params.ny) - params.hy) + 0.5) * params.dy
        self.costh = np.cos(theta)
        self.sinth = np.sin(theta)


def hermite(n, x, q_order):
    """Compute Hermite polynomials"""
    if n == 0: 
        return 1.0
    elif n == 1: 
        return x
    elif n == 2: 
        return x*x - 1.0
    elif n == 3: 
        return 0.0 if q_order == 3 else x * (x*x - 3.0)
    else:
        return 0.0


def get_feq_fun(q_order):
    """Help function to create equilibrium distribution function"""
    if q_order == 3:
        @jit(nopython=True)
        def compute_feq(vx, vy, weight, rho, ux, uy, q_order):
            usq = ux*ux + uy*uy
            vu  = vx*ux + vy*uy
            feq = rho * weight * (1.0 + vu + 0.5 * (vu*vu - usq))
            return feq
    elif q_order == 4:
        @jit(nopython=True)
        def compute_feq(vx, vy, weight, rho, ux, uy, q_order):
            usq = ux*ux + uy*uy
            vu  = vx*ux + vy*uy
            vu_usq = vu*vu - usq
            feq = rho * weight * (1.0 + vu + 0.5 * vu_usq + vu/6.0 * (vu*vu - 3.0*usq))
            return feq
    else:
        raise ValueError("Only q_order 3 and 4 supported")

    return compute_feq


class Stencil:
    """Velocity stencil and quadrature rules"""
    def __init__(self, q_order=3):
        self.q_order = q_order
        self.npop    = q_order * q_order
        
        # Quadrature points and weights
        if q_order == 3:
            c = np.array([-np.sqrt(3), 0, np.sqrt(3)], dtype=np.float64)
            w = np.array([1/6, 2/3, 1/6], dtype=np.float64)
        elif q_order == 4:
            sqrt6 = np.sqrt(6)
            c = np.array([-np.sqrt(3 + sqrt6), -np.sqrt(3 - sqrt6), 
                           np.sqrt(3 - sqrt6),  np.sqrt(3 + sqrt6)], dtype=np.float64)
            w = np.array([(3 - sqrt6)/12, (3 + sqrt6)/12, 
                          (3 + sqrt6)/12, (3 - sqrt6)/12], dtype=np.float64)
        else:
            raise ValueError("Only q_order 3 and 4 supported")
        
        self.compute_feq = get_feq_fun(q_order)

        # Initialize arrays with explicit dtypes
        self.vx  = np.zeros(self.npop, dtype=np.float64)
        self.vy  = np.zeros(self.npop, dtype=np.float64)
        self.sx  = np.zeros(self.npop, dtype=np.int32)
        self.sy  = np.zeros(self.npop, dtype=np.int32)
        self.w   = np.zeros(self.npop, dtype=np.float64)
        self.kd  = np.zeros((q_order, q_order), dtype=np.float64)
        self.kdp = np.zeros((q_order, q_order), dtype=np.float64)

        self._initialize_stencil_arrays(c, w, q_order)

    def _initialize_stencil_arrays(self, c, w, q_order):
        """Initialize stencil velocity and derivative matrices"""
        for ix in range(q_order):
            for iy in range(q_order):
                ivel = ix * q_order + iy
                
                # Velocities and signs
                self.vx[ivel] = c[ix]
                self.vy[ivel] = c[iy]
                self.sx[ivel] = 1 if c[ix] > 0 else -1
                self.sy[ivel] = 1 if c[iy] > 0 else -1
                self.w[ivel]  = w[ix] * w[iy]
                
                # Derivative matrices
                lfact = 1.0
                for l in range(q_order):
                    h1 = hermite(l + 1, c[ix], q_order)
                    h2 = hermite(l, c[iy], q_order)
                    h3 = hermite(l + 1, c[iy], q_order)
                    h4 = hermite(l - 1, c[iy], q_order) if l > 0 else 0.0
                    
                    inv_lfact = 1.0 / lfact
                    self.kd[ix, iy]  += h1 * h2 * inv_lfact
                    self.kdp[ix, iy] += h1 * (h3 + l * h4) * inv_lfact
                    
                    lfact *= (l + 1)
                
                self.kd[ix, iy]  *= -w[ix]
                self.kdp[ix, iy] *= -w[ix]


def init_state(params, geom, stencil, key_init=0, rho_zero=1.0, ampl=1e-5):
    """Initialize macroscopic fields and distribution functions"""
    
    # Initialize basic fields
    params.rho.fill(rho_zero)
    params.ux.fill(0.0)
    params.uy.fill(0.0)
    
    # Create coordinate grids
    hx, hy = params.hx, params.hy
    ix_range = slice(hx, params.nx - hx)
    iy_range = slice(hy, params.ny - hy)
    
    if key_init == 0:  # Sound wave
        params.uy[ix_range, iy_range] = ampl * geom.sinth[iy_range, np.newaxis].T
        
    elif key_init == 1:  # Sound wave (parabolic)
        iy_vals = np.arange(hy, params.ny - hy)
        th = (iy_vals - hy + 0.5) * params.dy
        uy_profile = ampl * th * (np.pi - th)
        params.uy[ix_range, iy_range] = uy_profile[np.newaxis, :]
        
    elif key_init == 2:  # Shear wave
        params.ux[ix_range, iy_range] = ampl
        
    else:
        raise ValueError("Only key_init 0, 1 and 2 implemented")
    
    # Initialize distribution functions
    for ip in range(stencil.npop):
        for ix in range(hx, params.nx - hx):
            for iy in range(hy, params.ny - hy):
                params.f1[ix, iy, ip] = stencil.compute_feq(
                    stencil.vx[ip], stencil.vy[ip], stencil.w[ip],
                    params.rho[ix, iy], params.ux[ix, iy], params.uy[ix, iy],
                    stencil.q_order)


@njit(parallel=True)
def compute_macro_parallel(f, rho, ux, uy, vx, vy, nx, ny, hx, hy, npop):
    """Compute macroscopic variables"""
    for ix in prange(hx, nx - hx):
        for iy in range(hy, ny - hy):
            rho_local = 0.0
            ux_local = 0.0
            uy_local = 0.0
            
            for ip in range(npop):
                f_val = f[ix, iy, ip]
                rho_local += f_val
                ux_local  += f_val * vx[ip]
                uy_local  += f_val * vy[ip]
            
            rho[ix, iy] = rho_local
            inv_rho     = 1.0 / rho_local
            ux[ix, iy]  = ux_local * inv_rho
            uy[ix, iy]  = uy_local * inv_rho


def compute_macro(param, stencil):
    """Wrapper for macroscopic computation"""
    compute_macro_parallel(param.f1, param.rho, param.ux, param.uy,
                      stencil.vx, stencil.vy, 
                      param.nx, param.ny, param.hx, param.hy, 
                      stencil.npop)


@njit(parallel=False)
def pbc_parallel(f, nx, ny, hx, hy, lx, ly, npop):
    """Apply periodic boundary conditions"""
    # Y-planes (periodic in x-direction)
    for iy in prange(hy, ny - hy):
        for ii in range(hx):
            # Left boundary
            src_x  = hx + ii
            dest_x = nx - hx + ii
            for ip in range(npop):
                f[dest_x, iy, ip] = f[src_x, iy, ip]
            
            # Right boundary
            src_x  = hx + lx - 1 - ii
            dest_x = hx - 1 - ii
            for ip in range(npop):
                f[dest_x, iy, ip] = f[src_x, iy, ip]
    
    # X-planes (bounce-back)
    for ix in prange(hx, nx - hx):
        for ii in range(hy):
            src_x_idx = hx + (ix - hx + (lx - 1) // 2) % lx
            
            # Bottom boundary
            dest_y = hy - 1 - ii
            src_y  = hy + ii
            for ip in range(npop):
                f[ix, dest_y, ip] = f[src_x_idx, src_y, npop - ip - 1]
            
            # Top boundary
            dest_y = ny - hy + ii
            src_y = hy + ly - 1 - ii
            for ip in range(npop):
                f[ix, dest_y, ip] = f[src_x_idx, src_y, npop - ip - 1]


def pbc(param, stencil, f):
    """Periodic boundary conditions wrapper"""
    pbc_parallel(f, param.nx, param.ny, param.hx, param.hy, 
             param.lx, param.ly, stencil.npop)


@njit(parallel=True)
def stream_and_collide_parallel(f1, f2, rho, ux, uy,
                            vx, vy, sx, sy, w,
                            kd, kdp,
                            R, sinth, costh,
                            nx, ny, hx, hy, lx, ly,
                            dx, dy, dt, tau, q_order, npop, compute_feq):
    """Stream and collide step"""
    dt_tau = dt / tau
  
    for ix in prange(hx, nx - hx):
        for iy in range(hy, ny - hy):
            # Compute local macroscopic fields
            rho_local = 0.0
            ux_local = 0.0
            uy_local = 0.0

            for ip in range(npop):
                f = f1[ix, iy, ip]
                rho_local += f
                ux_local  += f * vx[ip]
                uy_local  += f * vy[ip]

            rho[ix, iy] = rho_local
            inv_rho = 1.0 / rho_local
            ux_local *= inv_rho
            uy_local *= inv_rho 
            ux[ix, iy] = ux_local 
            uy[ix, iy] = uy_local 

            # Precompute geometry-dependent scalars
            theta = (iy - hy) * dy
            theta_p1 = theta + dy

            sin_0  = np.sin(theta)
            sin_p1 = np.sin(theta_p1)
            cos_0  = np.cos(theta)
            cos_p1 = np.cos(theta_p1)

            denom_theta = R * (cos_0 - cos_p1)
            cdp         = costh[iy] / (R * sinth[iy])

            for ip in range(npop):
                vx_val = vx[ip]
                vy_val = vy[ip]
                w_val  = w[ip]
                sx_val = sx[ip]
                sy_val = sy[ip]

                aux = 0.0

                # Phi-direction UPWIND2
                pos_phi = (1 - sx_val) // 2
                ix_pos  = ix + pos_phi
                ix_m1   = ix_pos - sx_val

                fp = vx_val * 0.5 * (3.0 * f1[ix_pos, iy, ip] - f1[ix_m1, iy, ip])

                pos_phi_m = pos_phi - 1
                ix_pos_m  = ix + pos_phi_m
                ix_m1_m   = ix_pos_m - sx_val

                fm = vx_val * 0.5 * (3.0 * f1[ix_pos_m, iy, ip] - f1[ix_m1_m, iy, ip])

                aux += (fp - fm) / dx

                # Theta-direction UPWIND2 with Komissarov
                pos_th = (1 - sy_val) // 2
                iy_pos = iy + pos_th
                iy_m1  = iy_pos - sy_val

                fp_th = vy_val * 0.5 * (3.0 * f1[ix, iy_pos, ip] - f1[ix, iy_m1, ip])

                pos_th_m = pos_th - 1
                iy_pos_m = iy + pos_th_m
                iy_m1_m  = iy_pos_m - sy_val

                fm_th = vy_val * 0.5 * (3.0 * f1[ix, iy_pos_m, ip] - f1[ix, iy_m1_m, ip])

                aux += (sin_p1 * fp_th - sin_0 * fm_th) / denom_theta

                # Forcing / geometric connection coefficients
                fx = 0.0
                fy = 0.0

                sx_idx = ip // q_order
                sy_idx = ip % q_order

                for s in range(q_order):
                    idx1 = q_order * s + sy_idx
                    idx2 = q_order * sx_idx + s

                    val1 = f1[ix, iy, idx1]
                    val2 = f1[ix, iy, idx2]
                    fx += (-vy_val * cdp) * kdp[sx_idx, s] * val1
                    fy += (vx_val * vx_val * cdp) * kd[sy_idx, s] * val2

                feq = compute_feq(vx_val, vy_val, w_val, rho_local, ux_local, uy_local, q_order)

                f2[ix, iy, ip] = (f1[ix, iy, ip] -
                                  dt_tau * (f1[ix, iy, ip] - feq) -
                                  dt * aux -
                                  dt * (fx + fy))


def stream_and_collide(param, geom, stencil):
    """Stream and collide wrapper"""
    stream_and_collide_parallel(param.f1, param.f2, param.rho, param.ux, param.uy,
                            stencil.vx, stencil.vy, stencil.sx, stencil.sy, stencil.w,
                            stencil.kd, stencil.kdp,
                            geom.R, geom.sinth, geom.costh,
                            param.nx, param.ny, param.hx, param.hy, param.lx, param.ly,
                            param.dx, param.dy, param.dt, param.tau,
                            stencil.q_order, stencil.npop, stencil.compute_feq)


@njit(parallel=True)
def rk3_step_parallel(f1, f2, frk, c1, c2, nx, ny, hx, hy, npop):
    """Runge-Kutta 3rd order step"""
    for ix in prange(hx, nx - hx):
        for iy in range(hy, ny - hy):
            for ip in range(npop):
                f1[ix, iy, ip] = c1 * f2[ix, iy, ip] + c2 * frk[ix, iy, ip]


def rk3_step(param, stencil, c1, c2):
    """RK3 step wrapper"""
    rk3_step_parallel(param.f1, param.f2, param.frk, c1, c2,
                 param.nx, param.ny, param.hx, param.hy, stencil.npop)


def iter_rk3(param, geom, stencil):
    """Perform RK3 integration step"""
    np.copyto(param.frk, param.f1)
    
    # RK3 Step 1
    pbc(param, stencil, param.f1)
    stream_and_collide(param, geom, stencil)
    rk3_step(param, stencil, 1.0, 0.0)
    
    # RK3 Step 2
    pbc(param, stencil, param.f1)
    stream_and_collide(param, geom, stencil)
    rk3_step(param, stencil, 0.25, 0.75)
    
    # RK3 Step 3
    pbc(param, stencil, param.f1)
    stream_and_collide(param, geom, stencil)
    rk3_step(param, stencil, 2.0/3.0, 1.0/3.0)


def dump_profile(param, geom, stencil, niter, dump_dir="dumps",
                 rho_zero=1.0, ampl=1e-5):
    """Output simulation snapshot to file"""
    
    # Ensure output directory exists
    os.makedirs(dump_dir, exist_ok=True)

    # Construct filename prefix
    dump_fname = f"SPHQ{stencil.q_order}_U2_X1R{geom.R:.2f}Y{param.ly}" \
                 f"_dt{param.dt:g}_Ta{param.tau:g}" \
                 f"rho{rho_zero:.2f}a{ampl:g}"

    fname = os.path.join(dump_dir, f"{dump_fname}.{niter:09d}")

    output_data = []
    ntot = 0.0
    
    for ix_aux in range(param.hx, param.hx + param.lx + 1):
        ix = (ix_aux - param.hx) % param.lx
        x = param.dx * (ix + 0.5)

        for iy_aux in range(param.hy, param.hy + param.ly + 1):
            iy = (iy_aux - param.hy) % param.ly
            y  = param.dy * (iy + 0.5)

            rho_val = param.rho[ix_aux, iy_aux]
            ux_val  = param.ux[ix_aux, iy_aux]
            uy_val  = param.uy[ix_aux, iy_aux]

            # Metric factor
            factor = geom.R * geom.R * geom.sinth[iy_aux]

            if (iy_aux - param.hy < param.ly) and (ix_aux - param.hx < param.lx):
                ntot += rho_val * factor * param.dx * param.dy

            output_data.append(f"{x:.15e} {y:.15e} {rho_val:.15e} {ux_val:.15e} {uy_val:.15e}\n")
        
        output_data.append("\n")

    # Write all data at once
    with open(fname, "w") as fp:
        fp.writelines(output_data)

    print(f"iter={niter:010d}  ntot={ntot:25.20e}  file={fname}")


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Vielbein LBM Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python script.py --lx 8 --ly 64 --nsteps 10000 --ampl 1e-4"
    )
    
    parser.add_argument("--lx", type=int, default=4,
                        help="Domain size in x-direction (default: 4)")
    parser.add_argument("--ly", type=int, default=32,
                        help="Domain size in y-direction (default: 32)")
    parser.add_argument("--q-order", type=int, default=4, choices=[3, 4],
                        help="Quadrature order (default: 4)")
    parser.add_argument("--nsteps", type=int, default=6000,
                        help="Number of simulation steps (default: 6000)")
    parser.add_argument("--dump-interval", type=int, default=1000,
                        help="Output dump interval (default: 1000)")
    parser.add_argument("--key-init", type=int, default=1, choices=[0, 1, 2],
                        help="Initialization type: 0=sound wave, 1=parabolic, 2=shear (default: 1)")
    parser.add_argument("--ampl", type=float, default=1e-5,
                        help="Initial amplitude (default: 1e-5)")
    parser.add_argument("--rho-zero", type=float, default=1.0,
                        help="Reference density (default: 1.0)")
    parser.add_argument("--dt", type=float, default=1e-3,
                        help="Time step (default: 1e-3)")
    parser.add_argument("--tau", type=float, default=1e-2,
                        help="Relaxation time (default: 1e-2)")
    parser.add_argument("--R", type=float, default=1.0,
                        help="Spherical radius (default: 1.0)")
    parser.add_argument("--dump-dir", type=str, default="output",
                        help="Output directory (default: output)")
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Initialize stencil and parameters
    stencil = Stencil(q_order=args.q_order)
    param   = LBMParams(args.lx, args.ly, stencil, dt=args.dt, tau=args.tau, R=args.R)
    geom    = Geometry(param)

    # Initialize distribution functions
    init_state(param, geom, stencil, key_init=args.key_init, 
               rho_zero=args.rho_zero, ampl=args.ampl)

    print("\n" + "="*60)
    print("Vielbein LBM Simulation")
    print("="*60)
    print(f"  Domain:              {args.lx} × {args.ly}")
    print(f"  Quadrature order:    {args.q_order}")
    print(f"  Total steps:         {args.nsteps}")
    print(f"  Dump interval:       {args.dump_interval}")
    print(f"  Time step (dt):      {args.dt:g}")
    print(f"  Relaxation (tau):    {args.tau:g}")
    print(f"  Spherical radius:    {args.R:.2f}")
    print(f"  Initial condition:   {args.key_init} (amplitude: {args.ampl:g})")
    print(f"  Density (rho):       {args.rho_zero:.2f}")
    print(f"  Output directory:    {args.dump_dir}")
    print("="*60 + "\n")

    # Run simulation
    for step in tqdm(range(args.nsteps + 1), desc="Simulation"):
        if step % args.dump_interval == 0:
            compute_macro(param, stencil)
            dump_profile(param, geom, stencil, step, dump_dir=args.dump_dir,
                        rho_zero=args.rho_zero, ampl=args.ampl)

        iter_rk3(param, geom, stencil)


if __name__ == "__main__":
    main()