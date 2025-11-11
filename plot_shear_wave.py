import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse
from utils import get_sim_params, sol_shear, shear_F, shear_a

#import scienceplots
#plt.style.use(['science', 'bright'])
plt.rcParams.update({'figure.dpi': 150})

FIGSIZE = (3.375, 2.53)  # width, height (aspect ratio 4/3)
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Plot LBM simulation results")
parser.add_argument("--dump-dir", type=str, default="./output/",
                    help="Directory containing dump files (default: ./output/)")
parser.add_argument("--times", type=int, nargs="+", default=[0, 200, 400, 1000],
                    help="Iteration numbers to plot (default: 0 200 400 1000)")
parser.add_argument("--output", type=str, default="shear_wave_velocity_profile.pdf",
                    help="Output filename (default: shear_wave_velocity_profile.pdf)")
args = parser.parse_args()

# Load dump files
flist = []
for t in args.times:
    files = glob.glob(f"{args.dump_dir}/SPH*.{t:09d}")
    if not files:
        raise FileNotFoundError(f"No dump file found for iteration {t}")
    flist.append(files[0])

# Read data
rho_t = []
uph_t = []
uth_t = []

for idx, file in enumerate(flist):
    data = np.loadtxt(file).T
    
    if idx == 0:
        Q, SCHEME, nph, nth, R, dt, tau, V0, it = get_sim_params(file)
        phi = data[0]
        theta = data[1]
    
    rho_t.append(data[2])
    uph_t.append(data[3])
    uth_t.append(data[4])

print(f"Q={Q}, scheme={SCHEME}, domain=({nph}, {nth}), R={R}, dt={dt}, tau={tau}, V0={V0}")

# Create plot
fig, ax = plt.subplots(figsize=FIGSIZE)

# Numerical results
xx = theta[:nth] / np.pi
skip = 1

for i, t in enumerate(args.times):
    ax.plot(xx[::skip], uph_t[i][:nth:skip] / V0, ".", 
            color=COLORS[i], ms=3, zorder=100)

# Analytical results
th = np.linspace(theta.min(), theta.max(), 300)
ax.axhline(y=1, c = "k")
for t in args.times[1:]:
    ax.plot(th / np.pi, sol_shear(R, t * dt, th, tau, n=100), 
            color="black", linewidth=1)

# ax.plot(th/np.pi, shear_F(th,n=0)*shear_a(n=0)*np.sin(th), c="tab:gray",label=r"$\mathrm{Asymptotic}$")

# Formatting
ax.set_xlabel(r"$\theta/\pi$")
ax.set_ylabel(r"$u^{\hat{\theta}}/U_0$")
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.5)
ax.tick_params(axis="both")

for i, t in enumerate(args.times):
    ax.plot(np.nan, np.nan, "o", ms=3, c=COLORS[i], label=rf"$t={t * dt:.3g}$")
ax.plot(np.nan, np.nan, "-", c="k"             , label=r"$\mathrm{Analytic}$")

ax.legend(frameon = False, loc="upper right", ncol=3, fontsize=7.5)


# Save
plt.savefig(args.output, bbox_inches='tight')
print(f"Plot saved to {args.output}")
plt.close()