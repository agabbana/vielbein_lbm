import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse
from utils import get_sim_params, sol_sound

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
parser.add_argument("--output", type=str, default="sound_wave_velocity_profile.pdf",
                    help="Output filename (default: sound_wave_velocity_profile.pdf)")
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
    ax.plot(xx[::skip], uth_t[i][:nth:skip] / V0, ".", 
            color=COLORS[i], ms=3, zorder=100)

# Analytical results
th = np.linspace(theta.min(), theta.max(), 300)
for t in args.times:
    ax.plot(th / np.pi, sol_sound(R, t * dt, th, tau, n=3, key_init=1), 
            color="black", linewidth=1)

# Formatting
ax.set_xlabel(r"$\theta/\pi$")
ax.set_ylabel(r"$u^{\hat{\theta}}/U_0$")
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-3.0, 4.0)
ax.tick_params(axis="both")

# Legend
legend_lines = []
for i, t in enumerate(args.times):
    legend_lines.append(plt.Line2D([0], [0], marker="o", color="w", 
                                   markerfacecolor=COLORS[i], markersize=3, 
                                   label=rf"$t={t * dt:.3g}$"))
legend_lines.append(plt.Line2D([0], [0], color="black", linewidth=1, 
                               label=r"Analytic"))

ax.legend(handles=legend_lines, frameon=False, loc="upper right", 
          ncol=3, fontsize=7.5)

for i, t in enumerate(args.times):
    ax.plot(np.nan, np.nan, "o", ms=3, c=COLORS[i], label=rf"$t={t * dt:.3g}$")
ax.plot(np.nan, np.nan, "-", c="k"             , label=r"$\mathrm{Analytic}$")

ax.legend(frameon = False, loc="upper right", ncol=3, fontsize=7.5)


# Save
plt.savefig(args.output, bbox_inches='tight')
print(f"Plot saved to {args.output}")
plt.close()