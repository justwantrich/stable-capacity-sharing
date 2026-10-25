# Stable Capacity Sharing

Research code for **Sustaining Cooperation in Grid Capacity Sharing: Tariff Design for Net-Zero Industrial Parks**.

The code implements coalition transport bills, Shapley and least-core allocations, composition-neutral tariffs,
revenue calibration, storage and electrification sensitivity analyses, and capacity-hosting assessment.

## Installation

Use Python with the dependencies listed in `requirements.txt`. Run commands from the repository root.

```bash
python -m pip install -r requirements.txt
```

## Data

Download the VEA industrial load-profile dataset, version 1.0.1, from
[Zenodo](https://doi.org/10.5281/zenodo.13910298) (Paul Hendrik Tiemann, 2024; CC BY 4.0).
Extract these two original tab-separated files into `data/vea_raw/`:

```text
data/vea_raw/
  master_data_tabsep.csv
  load_profiles_tabsep.csv
```

No raw data, processed data, experimental results or execution logs are included in this repository.
The original dataset's high-load-time file is not required. Dutch tariff coefficients from Liander's 2026
schedule are defined in `nzip/tariffs.py`; German tariff parameters come from the VEA master data.

```bash
python -m nzip.data --convert
```

This generates the quality-controlled inputs in `data/vea_processed/`.
To use other directories, set `VEA_RAW_DIR` and `VEA_PROCESSED_DIR`.

## Code structure

```text
nzip/
  data.py, parks.py          input loading, quality control and park construction
  tariffs.py, kernels.py     tariff inputs, storage dispatch and bill kernels
  game.py, designs.py        coalition models, tariff variants and calibration
  cooperative.py, dual.py    cooperative allocations and LP reference models
  evaluation.py             core feasibility and allocation stability
  accounting.py             network revenue and enterprise bills
  hosting.py                additional-load hosting under fixed capacity limits
scripts/                    experiment entry points and numerical checks
```

## Running the analyses

After preparing the data:

```bash
python -m scripts.validate_kernels
python -m scripts.check_properties
python -m scripts.run_baseline_designs --out outputs/baseline_designs.parquet
python -m scripts.run_economic_accounting --baseline outputs/baseline_designs.parquet --out outputs/accounting
python -m scripts.run_local_remedies --exp enriched --out outputs/remedies_enriched.parquet
python -m scripts.run_local_remedies --exp insert --out outputs/remedies_insert.parquet
python -m scripts.run_local_remedies --exp rates --out outputs/remedies_rates.parquet
python -m scripts.run_storage_sensitivity --country nl --out outputs/storage_nl.parquet
python -m scripts.run_storage_sensitivity --country de --out outputs/storage_de.parquet
python -m scripts.run_electrification_sensitivity --out outputs/electrification.parquet --workers 4
python -m scripts.run_hosting --out outputs/hosting.parquet
```

Experiment scripts accept `--n_parks`, for example `--n_parks 5` for a smaller run.
Full runs enumerate all coalitions and can take hours. The electrification script's parallel mode uses POSIX `fork`;
use `--workers 1` on Windows. Generated files stay in `data/` and `outputs/`, both excluded from Git.

## Model settings

- Baseline park construction uses seed 1; local-remedy scenarios use seed 7; split-half calibration uses seed 0.
- Core and allocation stability use a tolerance of one millionth of summed enterprise standalone bills.
- Revenue-matched scales are fitted on the no-storage baseline. Reference-rate variants use a scale of 1.
  The electrification script selects the revenue-matched Dutch variant and the reference-rate German variant.
  The storage script evaluates both neutral variants; the local-remedy script uses the reference-rate variant.
- German original tariffs with storage use the specified peak-shaving policy, excluding deliberate additional
  cycling or peak raising solely to change tariff eligibility.
- The closest-core allocation minimises L1 distance to Shapley; when several optima exist, the HiGHS solution is used.
- Hosting uses storage power equal to 20% of baseline aggregate park peak, 2 h duration and 90% round-trip efficiency.
  Capacity limits are contractual assumptions, not power-flow validation.
