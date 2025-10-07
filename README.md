# Utility-Learning-Tension

[![arXiv](https://img.shields.io/badge/arXiv-2510.04399-b31b1b.svg)](https://arxiv.org/abs/2510.04399)

This repository contains code for investigating the tension between utility maximization and generalization in model selection, particularly focusing on the comparison between VC-theory based model selection and more permissive selection policies.

## Overview

The project explores how different model selection policies affect the generalization performance of polynomial regression models. It implements two main policies:

1. **TwoGate Policy**: A VC-theory inspired approach that controls model capacity and enforces a conservative acceptance criterion based on validation performance.
2. **Destructive Policy**: A more permissive baseline that accepts models that don't significantly worsen validation performance.

For more details on the theoretical foundations and experimental results, see our paper: [**Utility-Learning Tension in Self-Modifying Agents**](https://arxiv.org/abs/2510.04399)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Utility-Learning-Tension
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running Experiments

The main experiment script is `run_h.py`. Here's an example command to run the destructive policy with specific parameters:

```bash
python run_h.py \
  --policy destructive \
  --destructive_slack 0.003 \
  --seeds 50 \
  --m 500 1000 2000 5000 \
  --n_v 1000 \
  --n_test 50000 \
  --K_max 60 \
  --K_mult 0.5 \
  --c0 2.0 \
  --tau_mult 0.5 \
  --delta_v 0.05 \
  --sigma 0.6 \
  --flip 0.15 \
  --output_dir experiments/outputs
```

### Key Parameters

- `--policy`: Selection policy to use (`twogate` or `destructive`)
- `--seeds`: Number of random seeds to run
- `--m`: Training set sizes to evaluate
- `--n_v`: Size of validation set
- `--n_test`: Size of test set
- `--K_max`: Maximum model capacity to consider
- `--K_mult`: Multiplier for capacity threshold K(m) = K_mult * sqrt(m)
- `--c0`, `--tau_mult`: Parameters controlling the acceptance threshold
- `--sigma`: Noise level in data generation
- `--flip`: Label flip probability

### Visualization

After running experiments, use the `visualize.py` script to generate plots:

```bash
python visualize.py --input_dir experiments/outputs --output_dir figures
```

## Project Structure

- `core.py`: Core model and policy implementations
- `run_h.py`: Main experiment script
- `visualize.py`: Visualization utilities
- `config.py`: Configuration dataclasses
- `experiments/`: Output directory for experiment results
- `figures/`: Directory for generated plots

## Data Generation

The synthetic data is generated with the following characteristics:

- Mixture of two Gaussians for input distribution
- Piecewise cubic latent function with a kink
- Heteroskedastic noise (higher variance in the tails)
- Optional label flips
- Support for adding nuisance dimensions

## License

This project is licensed under the Creative Commons Attribution 4.0 International License - see the [LICENSE](LICENSE) file for details or visit [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{wang2025utilitylearningtensionselfmodifyingagents,
      title={Utility-Learning Tension in Self-Modifying Agents}, 
      author={Charles L. Wang and Keir Dorchen and Peter Jin},
      year={2025},
      eprint={2510.04399},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.04399}, 
}
```
