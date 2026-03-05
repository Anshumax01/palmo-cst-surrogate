# PALMO Airfoil Surrogate Models

Neural network surrogate models for predicting airfoil aerodynamic 
coefficients (Cl, Cd, Cm) trained on the NASA PALMO OVERFLOW CFD database.

Developed at the **CEREAL (Computational and Experimental Rotorcraft Engineering 
and Aerodynamics) Lab, Georgia Institute of Technology**.

## Branches

| Branch | Description | Status |
|--------|-------------|--------|
| [`baseline-surrogate`](../../tree/baseline-surrogate) | MLP with NACA 4-digit inputs (5 inputs) | ✅ Complete |
| [`cst-surrogate`](../../tree/cst-surrogate) | MLP with CST geometry parametrization (17 inputs) | ✅ Complete |
| [`hyperparameter-tuning`](../../tree/hyperparameter-tuning) | Hyperparameter finetuning for current model | 🔜 Spring/Summer 2026 |
| [`architecture-search`](../../tree/architecture-search) | Explore alternative architectures | 🔜 Spring/Summer 2026 |
| [`airfoil-design`](../../tree/airfoil-design) | Airfoil design using surrogate model | 🔜 Spring/Summer 2026 |
| [`rotor-design`](../../tree/rotor-design) | Rotor design using surrogate model | 🔜 Spring/Summer 2026 |

## Future Work

1. **Hyperparameter Tuning** — Finetune the current surrogate model for improved performance
2. **Architecture Search** — Explore alternative neural network architectures if needed
3. **Airfoil Design** — Use the surrogate model to drive airfoil shape optimization
4. **Rotor Design** — Extend surrogate-based design to full rotor configurations

## References

> Cornelius, J. (2024). *PALMO: An OVERFLOW Machine Learning Airfoil 
> Performance Database Version 1.0 NACA 4-Series.* NASA/TM-20240014546.
> [ntrs.nasa.gov](https://ntrs.nasa.gov/citations/20240014546)

> Kulfan, B. M. (2007). *A Universal Parametric Geometry Representation 
> Method – CST.* AIAA-2007-0062.
> [arc.aiaa.org](https://arc.aiaa.org/doi/10.2514/6.2007-62)
```

Commit message:
```
Finalize main README with full CEREAL lab name
