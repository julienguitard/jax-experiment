# jax-experiment

A sandbox project (circa 2023–2024) for experimenting with [JAX](https://github.com/google/jax)
on a real-world dataset. It pulls the live Paris **Vélib'** bike-sharing
availability feed into Postgres, runs an ETL pipeline, and trains a small
feed-forward neural network with custom-built JAX primitives — architecture
generation, batch normalization, penalized loss, and an SGD-with-momentum
optimizer — all assembled by hand rather than through a high-level framework
like Flax or Haiku.

> Status: archived experiment. Kept around for reference; not actively
> maintained.

## What's in the box

```
sources/
├── app/         FastAPI stub (placeholder)
├── common/      Functional helpers (collections, decorators, types, strings)
├── database/    Postgres image + ETL pipeline + SQL templates
│   ├── connectors/   DB client / buffer abstractions
│   ├── etl/          Extract from Vélib API → transform → load
│   ├── server/       CLI for running ad-hoc queries
│   └── sql/          Templated SQL (clusters, buffers, merges, fits…)
└── ml_server/   JAX training pipeline
    ├── data/         Batchers, dispatchers, variance helpers
    ├── models_/      Layer formulas, architectures, jit/grad decorators
    └── trainers/     Initialization, iteration step, training passes
tests/           Unit tests mirroring the sources/ tree
```

The data flow is roughly:

```
Vélib Open Data API  ──►  Postgres (raw + buffer tables)
                              │
                              ▼
                     ETL: cluster + transform
                              │
                              ▼
                JAX training loop (ml_server)
                              │
                              ▼
            persistent_data_path/fits/logs.json
```

The model takes normalized position / day-of-week / hour / "duration-from"
features and predicts station availability targets (docks available, mechanical
bikes, e-bikes).

## Requirements

- Python 3.7+ (the ML server Dockerfile pins `python:3.7`; the code itself runs
  on newer Python with current JAX)
- Postgres (provided via Docker)
- Conda is assumed for environment management — see `generate_requirements.sh`

A persistent data path is hardcoded in [sources/common/constants.py](sources/common/constants.py)
(`PERSISTENT_DATA_PATH = "/Users/julienguitard/local_data/velib/"`). Override
this before running anywhere else.

## Quick start

### 1. Bring up Postgres

```bash
docker compose up -d db
```

Default credentials (see [sources/database/DOCKERFILE](sources/database/DOCKERFILE)):

| key      | value      |
|----------|------------|
| user     | `username` |
| password | `secret`   |
| database | `database` |

### 2. Install Python deps

```bash
conda create -n jax-experiment python=3.11
conda activate jax-experiment
pip install -r requirements.txt
```

### 3. Pull data from the Vélib API

```bash
python -m sources.database.etl.main velib
```

This polls the [open data endpoint](https://data.opendatasoft.com/explore/dataset/velib-disponibilite-en-temps-reel/)
every 5 minutes (288 times ≈ one day) and writes CSVs to the persistent data
path.

### 4. Run an ad-hoc query

```bash
python -m sources.database.server.main "SELECT count(*) FROM velib_raw;"
```

### 5. Train the model

```bash
python -m sources.ml_server.example
```

Trains for 300 epochs, mini-batch size 32, single hidden layer of width 12,
batch-norm enabled, MSE loss with L1 penalization. Loss history is dumped to
`{PERSISTENT_DATA_PATH}/fits/logs.json`.

## Tests

```bash
python -m unittest discover -s tests -p 'unittest_*.py'
```

Coverage is partial — focused on `ml_server/models`, `ml_server/data`, and
`database/connectors`.

## Tooling

- `format_all_files.sh` — runs `autopep8` + `black -l 79` over the tree
- `generate_requirements.sh` — exports `requirements.txt` / `requirements.yml`
  from the active conda env

## License

[MIT](LICENSE) © 2023 Julien Guitard
