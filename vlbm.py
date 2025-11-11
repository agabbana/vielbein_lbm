import numpy as np
from tqdm import tqdm
import os 
import argparse

class LBMParams:
    """Parameters for LBM simulation"""
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
    """Spherical coordinate geometry"""
    def __init__(self, params):
        self.R = params.R
        self.costh = np.zeros(params.ny, dtype=np.float64)
        self.sinth = np.zeros(params.ny, dtype=np.float64)
        
        # Initialize trigonometric arrays for spherical coordinates
        theta = ((np.arange(params.ny) - params.hy) + 0.5) * params.dy
        self.costh = np.cos(theta)
        self.sinth = np.sin(theta)


def hermite(n, x, q_order):
    """Compute Hermite polynomial H_n(x)"""
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


class Stencil:
    """Velocity stencil and quadrature rules"""
    def __init__(self, q_order=3):
        self.q_order = q_order
        self.npop = q_order * q_order
        
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


def compute_feq(stencil, ip, rho, ux, uy):
    """Compute equilibrium distribution function"""
    usq = ux*ux + uy*uy
    vu = stencil.vx[ip]*ux + stencil.vy[ip]*uy
    
    if stencil.q_order == 3:
        return rho * stencil.w[ip] * (1.0 + vu + 0.5 * (vu*vu - usq))
    elif stencil.q_order == 4:
        return rho * stencil.w[ip] * (1.0 + vu + 0.5 * (vu*vu - usq) + vu/6.0 * (vu*vu - 3.0*usq))
    else:
        raise ValueError("Q_order not supported")


def init_state(params, geom, stencil, key_init=0, rho_zero=1.0, ampl=1e-5):
    """Initialize macroscopic fields and distribution functions"""
    
    hx, hy = params.hx, params.hy
    
    for ix in range(hx, params.nx - hx):
        for iy in range(hy, params.ny - hy):
            
            # Initialize density and velocities
            params.rho[ix, iy] = rho_zero
            params.ux[ix, iy] = 0.0
            params.uy[ix, iy] = 0.0
            
            if key_init == 0:  # Sound wave
                params.uy[ix, iy] = ampl * geom.sinth[iy]

            elif key_init == 1:  # Parabolic sound wave
                th = (iy - hy + 0.5) * params.dy
                params.uy[ix, iy] = ampl * th * (np.pi - th)

            elif key_init == 2:  # Shear wave
                params.ux[ix, iy] = ampl
                
            else:
                raise ValueError("Only key_init 0, 1, and 2 implemented")
            
            # Initialize distribution functions
            for ip in range(stencil.npop):
                params.f1[ix, iy, ip] = compute_feq(stencil, ip, 
                                                    params.rho[ix, iy], 
                                                    params.ux[ix, iy], 
                                                    params.uy[ix, iy])


def df_dphi_upwind2(param, f_slice, c, ix, iy, pos, sgn):
    """UPWIND2 scheme for phi derivative"""
    ix_m1 = max(0, min(param.nx - 1, ix + pos - 1 * sgn))
    ix_0  = max(0, min(param.nx - 1, ix + pos))
    
    return c * 0.5 * (3.0 * f_slice[ix_0, iy] - f_slice[ix_m1, iy])


def df_dtheta_upwind2(param, f_slice, c, ix, iy, pos, sgn):
    """UPWIND2 scheme for theta derivative (Komissarov version)"""
    iy_m1 = max(0, min(param.ny - 1, iy + pos - 1 * sgn))
    iy_0  = max(0, min(param.ny - 1, iy + pos))
    
    return c * 0.5 * (3.0 * f_slice[ix, iy_0] - f_slice[ix, iy_m1])


def compute_macro(param, stencil):
    """Compute macroscopic variables"""
    hx, hy = param.hx, param.hy
    
    for ix in range(hx, param.nx - hx):
        for iy in range(hy, param.ny - hy):
            param.rho[ix, iy] = 0.0
            param.ux[ix, iy] = 0.0
            param.uy[ix, iy] = 0.0
            
            for ip in range(stencil.npop):
                f_val = param.f1[ix, iy, ip]
                param.rho[ix, iy] += f_val
                param.ux[ix, iy] += f_val * stencil.vx[ip]
                param.uy[ix, iy] += f_val * stencil.vy[ip]
            
            inv_rho = 1.0 / param.rho[ix, iy]
            param.ux[ix, iy] *= inv_rho
            param.uy[ix, iy] *= inv_rho


def pbc(param, stencil, f):
    """Apply periodic boundary conditions"""
    hx, hy = param.hx, param.hy
    
    # Update Y-planes (periodic in x-direction)
    for iy in range(hy, param.ny - hy):
        for ii in range(hx):
            # Left boundary
            src_x  = hx + ii
            dest_x = param.nx - hx + ii
            f[dest_x, iy, :] = f[src_x, iy, :]
            
            # Right boundary
            src_x  = hx + param.lx - 1 - ii
            dest_x = hx - 1 - ii
            f[dest_x, iy, :] = f[src_x, iy, :]
    
    # Update X-planes (bounce-back)
    for ix in range(hx, param.nx - hx):
        for ii in range(hy):
            # Bottom boundary
            src_x_idx = hx + (ix - hx + (param.lx - 1) // 2) % param.lx
            dest_y = hy - 1 - ii
            src_y  = hy + ii
            
            for ip in range(stencil.npop):
                f[ix, dest_y, ip] = f[src_x_idx, src_y, stencil.npop - ip - 1]
            
            # Top boundary
            dest_y = param.ny - hy + ii
            src_y = hy + param.ly - 1 - ii
            
            for ip in range(stencil.npop):
                f[ix, dest_y, ip] = f[src_x_idx, src_y, stencil.npop - ip - 1]


def stream_and_collide(param, geom, stencil):
    """Streaming and collision step"""
    hx, hy = param.hx, param.hy
    dt_tau = param.dt / param.tau
    
    for ix in range(hx, param.nx - hx):
        for iy in range(hy, param.ny - hy):
            # Spherical coordinate prefactors
            denom = geom.R * geom.sinth[iy]
            cdp   = geom.costh[iy] / denom

            # Compute local macroscopic variables
            rho_local = 0.0
            ux_local = 0.0
            uy_local = 0.0
            
            for ip in range(stencil.npop):
                fi = param.f1[ix, iy, ip]
                rho_local += fi
                ux_local  += fi * stencil.vx[ip]
                uy_local  += fi * stencil.vy[ip]
            
            inv_rho = 1.0 / rho_local
            ux_local *= inv_rho
            uy_local *= inv_rho
            
            # Process each population
            for ip in range(stencil.npop):
                aux = 0.0
                
                # Advection: Phi direction
                pos = (1 - stencil.sx[ip]) // 2
                fp = df_dphi_upwind2(param, param.f1[:, :, ip], 
                                     stencil.vx[ip], ix, iy, pos, stencil.sx[ip])
                fm = df_dphi_upwind2(param, param.f1[:, :, ip], 
                                     stencil.vx[ip], ix, iy, pos - 1, stencil.sx[ip])
                aux += (fp - fm) / param.dx / denom
                
                # Advection: Theta direction (Komissarov)
                pos = (1 - stencil.sy[ip]) // 2
                fp = df_dtheta_upwind2(param, param.f1[:, :, ip], 
                                       stencil.vy[ip], ix, iy, pos, stencil.sy[ip])
                fm = df_dtheta_upwind2(param, param.f1[:, :, ip], 
                                       stencil.vy[ip], ix, iy, pos - 1, stencil.sy[ip])
                
                # Komissarov spherical coordinate correction
                theta_p1 = (iy - hy + 1) * param.dy
                theta    = (iy - hy) * param.dy
                
                sin_p1 = np.sin(theta_p1)
                sin_0  = np.sin(theta)
                cos_p1 = np.cos(theta_p1)
                cos_0  = np.cos(theta)
                
                denom_theta = geom.R * (cos_0 - cos_p1)
                aux += (sin_p1 * fp - sin_0 * fm) / denom_theta
                
                # Collision: Forcing terms (connection coefficients)
                fx = 0.0
                fy = 0.0

                sx = ip // stencil.q_order
                sy = ip % stencil.q_order
                
                for s in range(stencil.q_order):
                    idx1 = stencil.q_order * s + sy
                    idx2 = stencil.q_order * sx + s
                    
                    fx += (-stencil.vy[ip] * cdp) * stencil.kdp[sx, s] * param.f1[ix, iy, idx1]
                    fy += (stencil.vx[ip] * stencil.vx[ip] * cdp) * stencil.kd[sy, s] * param.f1[ix, iy, idx2]
                
                # Equilibrium distribution
                feq = compute_feq(stencil, ip, rho_local, ux_local, uy_local)
                
                # Update distribution function
                param.f2[ix, iy, ip] = (param.f1[ix, iy, ip] - 
                                        dt_tau * (param.f1[ix, iy, ip] - feq) -
                                        param.dt * aux -
                                        param.dt * (fx + fy))


def rk3_step(param, stencil, c1, c2):
    """Runge-Kutta 3rd order update step"""
    hx, hy = param.hx, param.hy
    
    for ix in range(hx, param.nx - hx):
        for iy in range(hy, param.ny - hy):
            for ip in range(stencil.npop):
                param.f1[ix, iy, ip] = c1 * param.f2[ix, iy, ip] + c2 * param.frk[ix, iy, ip]


def iter_rk3(param, geom, stencil):
    """Runge-Kutta 3rd order iteration"""
    # Store initial state
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

    ntot = 0.0
    output_data = []
    
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
        description="LBM Simulation with Spherical Coordinates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python script.py --lx 8 --ly 64 --nsteps 5000 --ampl 1e-4"
    )
    
    parser.add_argument("--lx", type=int, default=4,
                        help="Domain size in x-direction (default: 4)")
    parser.add_argument("--ly", type=int, default=32,
                        help="Domain size in y-direction (default: 32)")
    parser.add_argument("--q-order", type=int, default=4, choices=[3, 4],
                        help="Quadrature order (default: 4)")
    parser.add_argument("--nsteps", type=int, default=1000,
                        help="Number of simulation steps (default: 1000)")
    parser.add_argument("--dump-interval", type=int, default=100,
                        help="Output dump interval (default: 100)")
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