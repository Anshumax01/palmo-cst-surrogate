# PALMO Airfoil Surrogate Models

Neural network surrogate models for airfoil aerodynamic performance 
(Cl, Cd, Cm) trained on the NASA PALMO OVERFLOW CFD database.

## Branches

| Branch | Description | Status |
|--------|-------------|--------|
| `baseline-surrogate` | NN with NACA digit inputs | ✅ Complete |
| `cst-surrogate` | NN with CST geometry params | ✅ Complete |
| `neuron-optimization` | Architecture search | 🔜 Summer 2026 |

## Background
- NASA/TM-20240014546 — PALMO database (Cornelius 2024)
- AIAA-2007-0062 — CST parametrization (Kulfan 2007)
