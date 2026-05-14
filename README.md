# Event-SAE

Codebase for "Event-Grounded Sparse Autoencoders for
Vision-Language-Action Policies". The project scales sparse-autoencoder
feature labeling by anchoring candidate features in SAE-independent
kinematic events from closed-loop rollouts, ranking them against
VLM-labeled event clusters, and validating each ranking with
residual-preserving zero-out interventions.

Two VLA backbones are covered: **openVLA** and **openpi (π₀.₅)**, on
the LIBERO simulation suites.

## Pipeline

Four stages plus a ranking bridge, 11 numbered steps in total:

| Stage | Steps | Produces |
|---|---|---|
| 1. SAE training | a, b | activation shards → trained SAE |
| 2. Kinematic keyframes | c | AWE waypoints per episode |
| 3. Event clustering + VLM annotation | d–g | labeled event clusters |
| Feature ranking (bridge) | h, i, j | top-K candidate features per ranking |
| 4. Closed-loop intervention | k | per-feature ΔSR |

Stages 1–4 are shared across backbones; activation collection (a) and
the intervention hook (k) are backbone-specific. Per-backbone guides:

- **openVLA** — [docs/openvla.md](docs/openvla.md)
- **openpi (π₀.₅)** — [docs/openpi.md](docs/openpi.md)

Each guide includes installation, the full pipeline (steps a–k),
pretrained SAE checkpoints from the paper (on the Hugging Face Hub),
and a reproducibility check against the original research artifacts.

## License

MIT (see `LICENSE`). External libraries cloned under `external/` at
install time retain their own licenses.
