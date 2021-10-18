
<img src="docs/survey_structure.svg">


# Installing dependencies

``` bash
python3 -m pip install -r requirements.txt
```

# Checklist

| Name                                         | Section | Command                                                         |
| -------------------------------------------- | ------- | --------------------------------------------------------------- |
| Vanilla Direct-Solution                      | 3.2     | python3 experiments/direct_solution.py --model vanilla          |
| Automatic Differentiation in Direct-Solution | 3.3     | python3 experiments/direct_solution.py --model autodiff         |
| Physics Informed Neural Networks             | 3.4     | python3 experiments/direct_solution.py --model pinn             |
| Hidden Physics Networks                      | 3.5     | python3 experiments/direct_solution.py --model hnn              |
| Direct Time-Stepper                          | 4.2.1   | python3 experiments/time_stepper.py --solver direct             |
| Residual Time-Stepper                        | 4.2.2   | python3 experiments/time_stepper.py --solver resnet             |
| Euler Time-Stepper                           | 4.2.3   | python3 experiments/time_stepper.py --solver euler              |
| Neural ODEs Time-Stepper                     | 4.2.4   | python3 experiments/time_stepper.py --solver {rk4,dopri5,tsit5} |
| Neural State-Space Model                     | 4.3.1   | ...                                                             |
| Neural ODEs with input                       | 4.3.2-3 | ...                                                             |
| Lagrangian Time-Stepper                      | 4.4.1   | ...                                                             |
| Hamiltonian Time-Stepper                     | 4.4.1   | ...                                                             |
| Deep Potential Time-Stepper                  | 4.4.2   | ...                                                             |
| Deep Markov-Model                            | 4.5.1   | ...                                                             |
| Latent Neural ODEs                           | 4.5.2   | python3 experiments/latent_neural_odes.py                       |
| Bayesian Neural ODEs                         | 4.5.3   | ...                                                             |
| Neural SDEs                                  | 4.5.4   | ...                                                             |




## Latent-Time Stepper

```
py
```