#!/usr/bin/env python3
"""
Training entry point for PhysTwin residual world model.
Produces checkpoints used for the residual-bound experiments in the paper.
"""

import argparse
import yaml
import torch
from pathlib import Path
from tqdm import tqdm

# Local imports (assume PYTHONPATH includes repo root)
from phys_twin.dynamics import PhysTwinDynamics
from phys_twin.uncertainty import EnsembleResidual, AleatoricHead
from phys_twin.losses import total_loss
from phys_twin.certificates import ResidualBoundCertificate


def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["hardware"]["target_device"] if torch.cuda.is_available() else "cpu")
    state_dim = cfg["env"]["state_dim"]
    action_dim = cfg["env"]["action_dim"]
    latent_dim = cfg["model"]["latent_dim"]

    model = PhysTwinDynamics(state_dim, action_dim, latent_dim).to(device)
    ensemble = EnsembleResidual(latent_dim, action_dim, cfg["model"]["n_ensemble"]).to(device)
    aleatoric = AleatoricHead(latent_dim, action_dim).to(device)
    cert = ResidualBoundCertificate(kappa=cfg["model"]["kappa"])

    opt = torch.optim.Adam(
        list(model.parameters()) + list(ensemble.parameters()) + list(aleatoric.parameters()),
        lr=cfg["training"]["lr"],
    )

    # Placeholder data loader – replace with real MuJoCo / hardware trajectories
    def sample_batch(B: int):
        x = torch.randn(B, state_dim, device=device)
        u = torch.randn(B, action_dim, device=device)
        x_next = x + 0.02 * torch.randn_like(x)  # fake next state
        return x, u, x_next

    for step in tqdm(range(cfg["training"]["total_steps"])):
        x, u, x_next = sample_batch(cfg["training"]["batch_size"])
        dx, z = model(x, u)
        # Simple Euler step for demonstration
        pred = x + cfg["env"]["dt"] * dx

        energy_res = model.energy_residual(x, pred, u, cfg["env"]["dt"])
        mom_res = model.momentum_residual(x, pred, cfg["env"]["dt"])

        losses = total_loss(
            pred, x_next, energy_res, mom_res,
            lambda_phys=cfg["training"]["lambda_phys"],
            eps=cfg["training"]["eps_energy"],
        )

        opt.zero_grad()
        losses["total"].backward()
        opt.step()

        if step % cfg["training"]["log_every"] == 0:
            print(f"step {step}: total={losses['total'].item():.4f} "
                  f"pred={losses['prediction'].item():.4f} "
                  f"phys={losses['physics'].item():.4f}")

    # Save
    out = Path("runs/latest")
    out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "ensemble": ensemble.state_dict(),
        "aleatoric": aleatoric.state_dict(),
        "cfg": cfg,
    }, out / "model.pt")
    print(f"Saved checkpoint to {out / 'model.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/g1_phystwin.yaml")
    args = parser.parse_args()
    main(args.config)
