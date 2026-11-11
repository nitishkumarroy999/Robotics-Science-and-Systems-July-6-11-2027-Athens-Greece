# PhysTwin: Physics-Constrained Latent World Models for Certifiable Online Adaptation in Humanoid Locomotion

Official code release accompanying the RSS 2027 Stage-1 Extended Abstract.

**PhysTwin** embeds differentiable rigid-body + simplified contact dynamics as *hard* constraints inside a residual latent world model. The residual is trained so that the composite dynamics obey discrete energy and linear-momentum balance up to a quantifiable bound. Online residual-bound certificates enable safety-filtered adaptation on humanoid platforms.

## Key Features
- Hybrid dynamics: `f_phys(x,u) + r_θ(z,u)` with hard energy/momentum constraints
- Residual-bound certificates with formal violation probability bounds (Proposition 1)
- Deep-ensemble epistemic + heteroscedastic aleatoric uncertainty
- Online constrained residual adaptation under a residual-bound guard
- Edge-friendly inference path (Jetson Orin class)

## Installation
```bash
conda create -n phystwin python=3.10
conda activate phystwin
pip install -r requirements.txt
```

## Quick Start (Simulation)
```bash
# Train residual world model with physics constraints
python scripts/train.py --config configs/g1_phystwin.yaml

# Evaluate residual bounds and multi-step prediction
python scripts/eval_bounds.py --checkpoint runs/latest/model.pt

# Run online adaptation experiment under payload/friction shift
python scripts/online_adapt.py --config configs/g1_adapt.yaml
```

## Repository Structure
```
phys_twin/          # core library
  dynamics.py       # physics backbone + residual
  uncertainty.py    # ensemble + aleatoric heads
  certificates.py   # residual-bound certificates
  losses.py         # energy/momentum hinge + prediction loss
experiments/        # reproduction scripts for paper tables
configs/            # experiment configurations
scripts/            # training / evaluation entry points
docs/               # additional notes and proof sketches
```

## Citation
If you use this code, please cite the accompanying RSS 2027 paper (anonymous during review).

## License
MIT (see LICENSE)
