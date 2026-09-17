# work/

The production experiments behind the preprint, one subdirectory per experiment. Each one is self-contained: an entry point that runs the campaign, a `common.py` where there is shared setup, and the result scripts that turn what the campaign wrote into the figures and tables of the document.

The GPU campaigns ran on kuma, the EPFL SCITAS cluster, on NVIDIA H100 nodes, submitted through slurm wrappers that are site-specific and not included. Where the raw outputs are too large to move, the extract scripts ran there as well. In records from cluster runs the `deps` shas are null: the dependency checkouts on the cluster carried no git metadata, so the run recorded a wrong sha, which the archive build blanks. The `versions` block in the same record and `pyproject.toml` carry the versions. Everything else runs on a laptop CPU. Each README says which.

Every experiment directory has some of three output directories.

- `output/`, what the run wrote, one directory or file per condition. Conditions resume by output existence, so an interrupted campaign continues where it stopped.
- `results/`, extracted, structured, small, committed. Every figure and table rebuilds from the archive alone, reading `results/` plus the small shipped inputs such as the roster ranking and the relaxed geometries.
- `figures/`, PDF and PNG figures, and `.tex` fragments for the tables.

Result scripts are named `<topic>_<role>.py`, topic first, so everything behind one figure groups together. `<topic>_extract.py` reads `output/` and writes `results/`. `<topic>_load.py` reads `results/` and returns arrays for every consumer. `<topic>_figure.py` and `<topic>_table.py` read `results/` and write into `figures/`. Some experiments add a `<topic>_results.py`, which re-derives the numbers the preprint quotes from `results/` and prints them as JSON, exiting nonzero when a quoted claim no longer holds.

Extraction is a separate stage from plotting because extraction often has to run next to outputs too large to move, while plotting does not. That is also why `results/` is committed while `output/` is only partly included: the Hessians of the large frameworks stay on the cluster they were computed on. Each experiment's README says what is in its `output/` here and what is not.

Figure and table scripts import shared style, palette and table helpers from `sadmof.tbx`. matplotlib is an optional dependency, so run those scripts as `uv run --extra plots python ...`.
