# Launch — three ways to run the pipeline over SSH

The pipeline (`pipeline/run_all.sh`) takes ~24 hours end-to-end. There are
three ways to launch it depending on how you want to interact with it.

## TL;DR — just tell me what to do

```bash
# best for most cases: submit and forget
ssh user@cluster
cd ~/meld_emotion
qsub launch/submit.pbs
qstat -u $USER                    # check status
tail -f logs/pipeline_*.log       # watch progress (optional)
```

Disconnect, do other work, come back tomorrow.

---

## The three methods

### 1. `submit.pbs` — qsub (RECOMMENDED)

Submits the pipeline as a PBS batch job. The cluster scheduler runs it on
a compute node when one's free. **Survives SSH disconnect, login node
reboot, your laptop sleeping** — anything short of a cluster outage.

```bash
qsub launch/submit.pbs
```

Walltime is 48 hours; if the pipeline finishes early it just exits.
Resource request: 1 GPU, 16 CPUs, 96 GB RAM. Edit the `#PBS -l select=`
line if your cluster has different resource names.

**Use when:** you want to start training and walk away.

### 2. `interactive.sh` — tmux on a compute node

Requests an interactive GPU node, opens a tmux session inside it, runs
the pipeline. You can watch logs scroll past, kill it on demand,
reattach after disconnects.

```bash
bash launch/interactive.sh
```

If you're not on a GPU node already, it calls `qsub -I` to request one.
Once attached, detach without killing via `Ctrl-b d`. Reattach later
from a new SSH session with:

```bash
ssh user@cluster
tmux attach -t meld
```

**Use when:** you're debugging a fresh issue and want to see live output,
or when you want to kill the run immediately if something looks wrong.

### 3. `setup_and_run.sh` — paste-and-go for fresh installs

Does first-time setup (venv, deps, model check, manifest check, GPU
check) AND launches the pipeline. **Does NOT survive SSH disconnect.**

```bash
bash launch/setup_and_run.sh
```

**Use when:** you've just rsynced/git-cloned the project to a new
machine and want one command that handles install + verify + launch.
For long runs, after this verifies setup, kill it and use `submit.pbs`.

---

## Quick reference

| | survives disconnect | watch live | first-time setup | best for |
|---|:-:|:-:|:-:|---|
| `submit.pbs`     | ✅ | – | – | unattended runs |
| `interactive.sh` | ✅ (via tmux) | ✅ | – | debugging |
| `setup_and_run.sh` | ❌ | ✅ | ✅ | first install / verification |

---

## Resuming after a crash

Every pipeline stage is resumable from its `run_dir`. If walltime hits
or the node crashes mid-run, just resubmit:

```bash
# the latest run dir is auto-detected by every stage
qsub launch/submit.pbs
```

Or rerun a specific stage manually:

```bash
# tuning was killed mid-way → resume the optuna study
python pipeline/02_tune.py --run-dir runs/full_20260425_103022 --resume

# training crashed at epoch 7 → just rerun, it picks up at epoch 7
python pipeline/03_train_full.py --run-dir runs/full_20260425_103022
```

---

## Monitoring

```bash
# job status (PBS)
qstat -u $USER
qstat -f <jobid>                  # detail

# latest pipeline log
tail -f $(ls -t logs/pipeline_*.log | head -1)

# what stage is currently running
tail -50 $(ls -dt runs/*/ | head -1)/*.log

# watch GPU
ssh <compute_node>
nvidia-smi -l 5

# from anywhere — cancel
qdel <jobid>
```

## Troubleshooting

**"venv/ not found"** — the launch scripts expect a Python venv at
`~/meld_emotion/venv/`. Create it on the login node first:
```bash
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

**"MISSING: Llama-3.2-3B"** — run `scripts/download_models.sh` on a
node with internet. Compute nodes are usually offline.

**"MISSING: data/train.json"** — generate your manifest with the
`{video_path, utterance, emotion}` schema. Use MELD-native labels
(`neutral, joy, anger, surprise, sadness, fear, disgust`).

**job pending forever** — the cluster is busy. `qstat -f <jobid>`
shows the reason. You can sometimes lower the walltime to get
scheduled sooner.

**OOM in stage 3** — the pipeline catches and skips OOM batches, but
if it's persistent, drop `cfg.batch_size` to 8 in
`pipeline/common.py`. Grad accumulation will keep effective batch
the same.
