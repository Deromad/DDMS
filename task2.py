import covasim as cv
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import pandas as pd
import time
from functools import partial
import os

# -----------------------------
# Experiment configuration
# -----------------------------
POP_SIZE = 20000        # population size for each sim (use scaling if you need larger effective pop)
N_DAYS = 180
N_RUNS = 100            # stochastic replicates per scenario
N_CPUS = max(1, mp.cpu_count() - 1)
BASE_SEED = 12345
OUTPUT_DIR = "results_covasim"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Base sim parameters
# -----------------------------
base_pars = dict(
    pop_size = POP_SIZE,
    pop_infected = 20,
    n_days = N_DAYS,
    rand_seed = None,   # will be set per-run
    verbose = 0,
    pop_type='hybrid',
)


# -----------------------------
# Define interventions
# -----------------------------

class SchoolEdgeClosure(cv.Intervention):   # custom intervention class

    def __init__(self, thresh=300):
        super().__init__()                  # initialize parent class
        self.thresh = thresh                # threshold of daily new infections
        self.active = False                 # whether schools are currently closed

    def apply(self, sim):                   # this runs every simulation timestep
        s_contacts = sim.people.contacts['s']
        inf = sim.people.infectious.sum()

        # CLOSE SCHOOLS
        if inf > self.thresh and not self.active:
            # save current beta values so we can restore later
            self.saved_beta = s_contacts['beta'].copy()

            # set ALL school edge weights to 0
            s_contacts['beta'][:] = 0.0

            self.active = True

        # OPEN SCHOOLS
        elif inf <= self.thresh and self.active:
            # restore original beta values
            s_contacts['beta'][:] = self.saved_beta

            self.active = False
# -----------------------------
# Define scenarios
# -----------------------------

scenarios = {
    "baseline": {
        "interventions": [],
        "label": "Baseline (no intervention)",
    },
    "lockdown_schools": {
        "interventions": [SchoolEdgeClosure(thresh=300)],
        "label": "close schools (at infection threshhold 500)",
    },
}

# -----------------------------
# Worker: create and run a single sim with a given seed and return results + runtime
# -----------------------------
def run_single_covasim_run(seed, pars, interventions, label=None):
    """Create a cv.Sim with the given pars and interventions, run it, and return
    the new_infections time series, runtime, and seed used.
    """
    sim_pars = dict(pars)
    sim_pars['rand_seed'] = int(seed)
    sim_label = f"{label} | seed={seed}" if label else None
    sim = cv.Sim(sim_pars, interventions=interventions, label=sim_label)
    t0 = time.time()
    sim.run()
    t1 = time.time()
    runtime = t1 - t0
    # Covasim stores daily new infections in sim.results['new_infections'] per docs
    new_inf = np.array(sim.results['new_infections'])
    cum_inf = np.array(sim.results['cum_infections'])
    cum_deaths = np.array(sim.results['cum_deaths'])
    return {
        'seed': seed,
        'runtime': runtime,
        'new_infections': new_inf,
        'cum_infections': cum_inf,
        'cum_deaths': cum_deaths,

        'sim': sim,  # keep sim object if user wants to inspect later (be mindful of memory)
    }

# -----------------------------
# Helper: run N_RUNS in parallel and collect outputs
# -----------------------------
def run_scenario_parallel(scenario_name, scenario_def, n_runs=N_RUNS, n_cpus=N_CPUS):
    print(f"Running scenario '{scenario_name}' with {n_runs} replicates on {n_cpus} CPUs...")
    # Pre-build interventions list (they can be the same object reused safely)
    interventions = scenario_def.get('interventions', [])
    label = scenario_def.get('label', scenario_name)

    seeds = [BASE_SEED + i for i in range(n_runs)]
    worker = partial(run_single_covasim_run, pars=base_pars, interventions=interventions, label=label)

    start_wall = time.time()
    # Use multiprocessing Pool to parallelize independent runs
    with mp.Pool(processes=n_cpus) as pool:
        results = pool.map(worker, seeds)
    total_wall = time.time() - start_wall

    # Extract arrays and timings
    all_new_inf = np.vstack([r['new_infections'] for r in results])  # shape (n_runs, n_days)
    all_cum_inf = np.vstack([r['cum_infections'] for r in results])
    all_cum_deaths = np.vstack([r['cum_deaths'] for r in results])
    runtimes = np.array([r['runtime'] for r in results])

    summary = {
        'scenario': scenario_name,
        'label': label,
        'n_runs': n_runs,
        'n_days': all_new_inf.shape[1],
        'new_infections_matrix': all_new_inf,
        'cum_infections_matrix': all_cum_inf,
        'cum_deaths_matrix': all_cum_deaths,
        'runtimes': runtimes,
        'total_wall_time_s': total_wall,
        'seeds': [r['seed'] for r in results],
        # optionally store sims: [r['sim'] for r in results]
    }
    print(f"Completed '{scenario_name}': wall_time={total_wall:.2f}s, mean_run_time={runtimes.mean():.3f}s")
    return summary

# -----------------------------
# Run all scenarios
# -----------------------------
all_summaries = {}
exp_start = time.time()
for sc_name, sc_def in scenarios.items():
    summ = run_scenario_parallel(sc_name, sc_def, n_runs=N_RUNS, n_cpus=N_CPUS)
    all_summaries[sc_name] = summ
exp_total = time.time() - exp_start
print(f"\nAll scenarios finished. Total experiment wall-clock time: {exp_total:.2f} s")

# -----------------------------
# Analysis: compute mean and 90% CI and plot
# -----------------------------
import matplotlib
matplotlib.use('Agg')  # safe backend for headless servers; remove if running interactively

for name in ['new_infections', 'cum_infections', 'cum_deaths']:
  plt.figure(figsize=(10,6))
  for sc_name, summ in all_summaries.items():
      mat = summ[name+ '_matrix']  # (n_runs, n_days)
      mean_ts = mat.mean(axis=0)
      p5 = np.percentile(mat, 5, axis=0)
      p95 = np.percentile(mat, 95, axis=0)
      days = np.arange(len(mean_ts))
      plt.plot(days, mean_ts, label=summ['label'])
      plt.fill_between(days, p5, p95, alpha=0.15)

  plt.xlabel('Day')
  plt.ylabel(name+' per day')
  plt.title(f'Covasim what-if scenarios — mean (n={N_RUNS}) with 90% CI '+name)
  plt.legend(fontsize='small')
  plt.grid(True)
  plot_path = os.path.join(OUTPUT_DIR, name+'_whatif_mean_90ci.png')
  plt.tight_layout()
  plt.savefig(plot_path)
  print(f"Saved plot to: {plot_path}")

# -----------------------------
# Save timing summary CSV
# -----------------------------
rows = []
for sc_name, summ in all_summaries.items():
    runtimes = summ['runtimes']
    rows.append({
        'scenario': sc_name,
        'label': summ['label'],
        'n_runs': summ['n_runs'],
        'mean_runtime_s': float(runtimes.mean()),
        'std_runtime_s': float(runtimes.std()),
        'min_runtime_s': float(runtimes.min()),
        'max_runtime_s': float(runtimes.max()),
        'total_wall_time_s': float(summ['total_wall_time_s']),
    })

timing_df = pd.DataFrame(rows)
csv_path = os.path.join(OUTPUT_DIR, 'whatif_timings.csv')
timing_df.to_csv(csv_path, index=False)
print(f"Saved timing summary to: {csv_path}")



print('\nDone.')
