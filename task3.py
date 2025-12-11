
import numpy as np
import pandas as pd
import covasim as cv
import math
import random
import os
import matplotlib.pyplot as plt

# -------------------------
# User-configurable options
# -------------------------
r = 20            # number of Morris trajectories (recommend 10-50)
p = 4             # grid levels per parameter (>=4)
m = 30             # number of stochastic replicates per design point
sim_days = 100    # simulation horizon (days)
pop_size = 20000  # Covasim population size (tune for speed vs realism)
seed_base = 12345 # base RNG seed

# Output files
out_dir = "morris_covasim_output"
os.makedirs(out_dir, exist_ok=True)
results_csv = os.path.join(out_dir, "morris_results.csv")
ees_csv = os.path.join(out_dir, "elementary_effects.csv")

# -------------------------
# Parameter ranges (min, max)
# -------------------------
param_ranges = {
    # beta: Covasim default beta is 0.016.
    # Here we let it vary between 0.004 (low) and 0.03 (high) — adjust to your context.
    "beta": (0.004, 0.03),

    # asymp_factor: multiplier applied to beta for asymptomatic infections
    # e.g., 0.2 (much less infectious) to 1.0 (as infectious as symptomatic)
    "asymp_factor": (0.2, 1.0),

    # rel_symp_prob: multiplier on baseline symptomatic probability (0.5 -> fewer symptomatic)
    # choose range e.g., 0.5 to 2.0 to allow reductions and increases
    "rel_symp_prob": (0.5, 2.0),

    # rel_death_prob: multiplier on baseline death probability (lethality multiplier)
    # e.g., from 0.5 (less deadly) to 3.0 (more deadly)
    "rel_death_prob": (0.5, 2.0),
}

param_names = list(param_ranges.keys())
k = len(param_names)

# -------------------------
# Morris helper functions
# -------------------------
def denormalize(x_norm, name):
    """Map normalized [0,1] to actual parameter value for parameter `name`."""
    lo, hi = param_ranges[name]
    return lo + x_norm * (hi - lo)

def random_base_grid(p, k):
    """Random base point on grid levels {0, 1/(p-1), ..., 1}."""
    levels = np.linspace(0, 1, p)
    return np.random.choice(levels, size=k)

def ensure_in_grid(x, p):
    levels = np.linspace(0, 1, p)
    x_clipped = np.clip(x, 0.0, 1.0)
    idx = np.round(x_clipped * (p - 1)).astype(int)
    return levels[idx]

# recommended delta for Morris (Campolongo et al. style)
delta = 1.0 / (p - 1)


# -------------------------
# Build Morris trajectories
# -------------------------
traj_records = []
for t in range(r):
    success = False
    attempts = 0
    while not success and attempts < 200:
        attempts += 1
        x0 = random_base_grid(p, k)
        order = np.random.permutation(k)
        x_current = x0.copy()
        seq = [{'x': x_current.copy(), 'changed': None}]
        valid = True
        for j in range(k):
            i = order[j]
            # Prefer stepping up if possible, else down
            if x_current[i] + delta <= 1.0:
                x_new = x_current.copy()
                x_new[i] = x_new[i] + delta
            elif x_current[i] - delta >= 0.0:
                x_new = x_current.copy()
                x_new[i] = x_new[i] - delta
            else:
                valid = False
                break
            x_current = ensure_in_grid(x_new, p)
            seq.append({'x': x_current.copy(), 'changed': i})
        if valid:
            success = True
            traj_records.append({'base': x0.copy(), 'order': order, 'seq': seq})
    if not success:
        raise RuntimeError("Failed to generate Morris trajectory. Try different p or delta.")

# Collect unique design points so that the same points arent calculated twice.
point_list = []
point_to_idx = {}
for tr in traj_records:
    for item in tr['seq']:
        tup = tuple(np.round(item['x'], 8))
        if tup not in point_to_idx:
            point_to_idx[tup] = len(point_list)
            point_list.append(np.array(item['x']))

n_points = len(point_list)
print(f"[INFO] Generated {len(traj_records)} trajectories with {n_points} unique design points.")

# -------------------------
# Function to create Covasim sim for a given param dict
# -------------------------
def make_sim_from_params(params_dict, n_days=sim_days, pop_size=pop_size, seed=None):
    """
    Build a Covasim Sim with the given parameters (params_dict contains actual values).
    We set a small, simple set of simulation parameters; feel free to expand with
    location, age distribution, layers, interventions, etc.
    """
    pars = {
        'pop_size': pop_size,
        'n_days': n_days,
        # core epidemic parameters
        'beta': params_dict['beta'],
        # Asymptomatic infectiousness factor
        'asymp_factor': params_dict['asymp_factor'],
        # relative symptomatic probability multiplier
        'rel_symp_prob': params_dict['rel_symp_prob'],
        # relative death probability multiplier (prognoses)
        'rel_death_prob': params_dict['rel_death_prob'],
        # reproducibility/verbosity options
        'rand_seed': seed if seed is not None else None,
        'verbose': 0,
    }
    sim = cv.Sim(pars)
    return sim

# -------------------------
# Run simulations (m replicates per design point) and store average output
# -------------------------
# Output metric: cumulative deaths at end of sim (sim.results['cum_deaths'] or sim.summary)
# Covasim's results: sim.results['cum_deaths'] is time series; final value is [-1] or sim.summary['cum_deaths']
# We'll record final cumulative deaths.

outputs = np.full(n_points, np.nan)           # average over m replicates
all_reps = {}                                 # store replicates if you want to analyze variance (optional)

import covasim as cv

for idx, x in enumerate(point_list):

    #Denormalize parameters from Morris sample
    pars = {name: denormalize(xi, name) for xi, name in zip(x, param_names)}

    #Build a list of sims to feed to MultiSim
    sim_list = []
    for rep in range(m):
        seed = seed_base + idx*m + rep
        sim = make_sim_from_params(pars, n_days=sim_days, pop_size=pop_size, seed=seed)
        sim_list.append(sim)

    #Run all replications in parallel using MultiSim
    msim = cv.MultiSim(sim_list)
    msim.run(parallel=True)      
    msim.reduce()                

    #Extract final cumulative deaths for each replication
    rep_deaths = []
    for sim in msim.sims:
        try:
            rep_deaths.append(sim.results['cum_deaths'][-1])
        except:
            rep_deaths.append(sim.summary.get('cum_deaths', np.nan))

    #Store results 
    outputs[idx] = float(np.mean(rep_deaths))
    all_reps[idx] = rep_deaths

    if (idx + 1) % 20 == 0 or idx == n_points - 1:
        print(f"[INFO] Completed {idx+1}/{n_points} design points.")

print("[INFO] All simulations finished (parallel MultiSim).")
# Save raw replicate table (long format)
rows = []
for idx in range(n_points):
    for rep_i, val in enumerate(all_reps[idx]):
        rows.append({'point_idx': idx, 'replicate': rep_i, 'deaths': val})
df_reps = pd.DataFrame(rows)
df_reps.to_csv(ees_csv, index=False)
print(f"[INFO] Saved replicate results to {ees_csv}")

# -------------------------
# Compute Elementary Effects
# -------------------------
# For each trajectory, each step gives an EE for the changed parameter:
#   EE = (Y_next - Y_current) / delta_normalized
ee_records = {name: [] for name in param_names}

for tr in traj_records:
    seq = tr['seq']
    for j in range(len(seq) - 1):
        x_curr = seq[j]['x']
        x_next = seq[j+1]['x']
        changed_idx = seq[j+1]['changed']
        tup_curr = tuple(np.round(x_curr, 8))
        tup_next = tuple(np.round(x_next, 8))
        idx_curr = point_to_idx[tup_curr]
        idx_next = point_to_idx[tup_next]
        y_curr = outputs[idx_curr]
        y_next = outputs[idx_next]
        # Elementary effect w.r.t normalized parameter change (delta)
        ee = (y_next - y_curr) / delta
        ee_records[param_names[changed_idx]].append(ee)

# Compute Morris statistics
morris_list = []
ee_rows = []
for name in param_names:
    ees = np.array(ee_records[name])
    mu = np.mean(ees) if ees.size > 0 else np.nan
    mu_star = np.mean(np.abs(ees)) if ees.size > 0 else np.nan
    sigma = np.std(ees, ddof=1) if ees.size > 1 else np.nan
    morris_list.append({
        'parameter': name,
        'mu': mu,
        'mu_star': mu_star,
        'sigma': sigma,
        'n_ee': len(ees),
        'range_min': param_ranges[name][0],
        'range_max': param_ranges[name][1],
    })
    # store EEs
    for v in ees:
        ee_rows.append({'parameter': name, 'ee': v})

df_morris = pd.DataFrame(morris_list).sort_values('mu_star', ascending=False).reset_index(drop=True)
df_ee = pd.DataFrame(ee_rows)

# Add mu_star per unit (normalize by absolute parameter range)
df_morris['mu_star_per_unit'] = df_morris.apply(lambda r: r['mu_star'] / (r['range_max'] - r['range_min']) if not np.isnan(r['mu_star']) else np.nan, axis=1)

# Save results
df_morris.to_csv(results_csv, index=False)
print(f"[INFO] Morris indices saved to {results_csv}")

# -------------------------
# Simple plot: mu_star vs sigma
# -------------------------
try:
    plt.figure(figsize=(7, 5))
    plt.scatter(df_morris['mu_star'], df_morris['sigma'])
    for i, row in df_morris.iterrows():
        plt.text(row['mu_star'] * 1.01, row['sigma'] * 1.01, row['parameter'])
    plt.xlabel('mu* (mean absolute elementary effect)')
    plt.ylabel('sigma (std of EE)')
    plt.title('Morris sensitivity: mu* vs sigma (output = cumulative deaths)')
    plt.grid(True)
    plt.tight_layout()
    plotfile = os.path.join(out_dir, 'morris_mu_star_sigma.png')
    plt.savefig(plotfile, dpi=150)
    print(f"[INFO] Plot saved to {plotfile}")
except Exception as e:
    print(f"[WARN] Could not create plot: {e}")

# -------------------------
# Print summary to console
# -------------------------
print("\nMorris sensitivity summary (sorted by mu*):")
print(df_morris[['parameter', 'mu_star', 'sigma', 'mu_star_per_unit', 'n_ee']])

# End of script
