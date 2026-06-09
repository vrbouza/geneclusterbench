# Simulation To Analysis Workflow

This document describes the current intended workflow: start from one annotated genome, create evolved pangenome simulations, benchmark gene clustering tools on a cluster with Slurm, and analyse the outputs.

If this repository was cloned or copied to a new environment, adjust the local hardcoded paths before submitting jobs.

## 1. Prepare Inputs

You need one GFF3 file with embedded FASTA. The simulator expects a `##FASTA` section in the GFF3 file and uses CDS features as the source genes.

Create a project data directory with a seed file:

```text
my_benchmark_data/
└── random_numbers.txt
```

Example `random_numbers.txt`:

```text
34
1001
1002
```

## 2. Run One Simulation Locally

Use this for a quick check before submitting many jobs:

```bash
uv run python -m geneclusterbench.simulate_full_pangenome \
  --gff /path/to/genome.gff \
  --out /path/to/my_benchmark_data/simulations/genome/34 \
  --seed 34
```

The simulator writes outputs inside the selected assembly and seed folder. If the GFF is `/path/to/genome.gff`, the canonical local output folder is:

```text
/path/to/my_benchmark_data/simulations/genome/34
```

The simulator writes outputs with a prefix such as:

```text
sim_gr_1e-12_lr_1e-12_mu_1e-14
```

Important outputs include:

- `*_iso_*.fasta`: simulated isolate assemblies.
- `*_iso_*.gff`: simulated isolate annotations.
- `*_for_clustering.fasta`: nucleotide gene sequences for clustering.
- `*_for_clustering_aa.fasta`: amino-acid gene sequences for clustering.
- `*_truth_matrix.tsv`: truth mapping used for cluster evaluation.
- `*_presence_absence.csv`: simulated presence/absence matrix.
- `*_sim_tree.nwk`: simulated phylogeny.

## 3. Submit Many Simulations With Slurm

The simulation launcher reads the seed file and submits one Slurm job per seed:

```bash
uv run python -m geneclusterbench.submit_simulations \
  --outdir-sims /path/to/my_benchmark_data \
  --seeds /path/to/my_benchmark_data/random_numbers.txt \
  --gff /path/to/genome.gff \
  --simulator /path/to/geneclusterbench/src/geneclusterbench/simulate_full_pangenome.py \
  --python-env /path/to/environment/bin/activate \
  --assembly-name genome \
  --pretend
```

Remove `--pretend` to submit jobs.

The expected simulation layout is:

```text
my_benchmark_data/
├── random_numbers.txt
└── simulations/
    └── genome/
        ├── 34/
        ├── 1001/
        └── 1002/
```

By default, the assembly folder is derived from the GFF basename without extension. Use `--assembly-name` to set that folder name explicitly.

## 4. Submit Gene-Clustering Jobs

The clustering launcher discovers simulation outputs under `simulations/<assembly>/<seed>/`, then submits CD-HIT and MMseqs2 jobs for nucleotide and amino-acid FASTAs.

```bash
uv run python -m geneclusterbench.submit_gene_clustering \
  --datapath /path/to/my_benchmark_data \
  --seeds /path/to/my_benchmark_data/random_numbers.txt \
  --outdir /path/to/final_outputs \
  --temp-outdir /path/to/scratch \
  --softwaredir /hps/software/users/jlees/vrbouza/projects/clustering_benchmark/software \
  --benchmark-runner /path/to/run_benchmark.py \
  --process cdhit,mmseqs2 \
  --sequence-type nt,aa \
  --pretend
```

Remove `--pretend` to submit jobs.

The current executable paths are built from `softwaredir`:

```python
mmseqs2exec = os.path.join(softwaredir, "mmseqs2/MMseqs2/build/bin/mmseqs")
cdhitexec = os.path.join(
    softwaredir,
    "cdhit/cdhit/cd-hit-est" if seqtype == "nt" else "cdhit/cdhit/cd-hit",
)
```

The `c` values are identity thresholds. CD-HIT uses `-c` and a sequence-type-specific `-n`; nucleotide CD-HIT uses `cd-hit-est`, so thresholds below `0.8` are skipped. MMseqs2 uses `--min-seq-id`.

The launcher writes `execcommands.tsv` in the current working directory and submits it as a Slurm array through the benchmark runner.

## 5. Analyse Clustering Outputs

After clustering jobs finish, run:

```bash
uv run python -m geneclusterbench.analyse_gene_clustering \
  /path/to/clustering_benchmark_TIMESTAMP \
  --datapath /path/to/my_benchmark_data \
  --seeds /path/to/my_benchmark_data/random_numbers.txt \
  --out-folder /path/to/analysis_plots \
  --nthreads 4
```

The analyser expects benchmark result folders under:

```text
clustering_benchmark_TIMESTAMP/
└── simulations/
    └── ASSEMBLY/
        └── SEED/
            ├── cdhit_st-nt/
            ├── cdhit_st-aa/
            ├── mmseqs2_st-nt/
            └── mmseqs2_st-aa/
```

It compares predicted clusters to `*_truth_matrix.tsv` and plots:

- Adjusted Rand index.
- Purity.
- Adjusted mutual information.
- V-measure.
- Runtime.

Plots are written as PNG, PDF, and SVG.
