# STOAT: Spatial-Temporal Causal Inference for Epidemic Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GluonTS](https://img.shields.io/badge/GluonTS-0.11+-green.svg)](https://github.com/awslabs/gluon-ts)

STOAT (Spatial-Temporal Causal Inference for Epidemic Forecasting) is a novel framework that combines spatial causal inference with deep probabilistic forecasting for epidemic prediction. The framework extends the classical Difference-in-Differences (DiD) approach by incorporating spatial dependencies and leverages deep neural networks for uncertainty quantification.

## 🎯 Key Features

- **Spatial Causal Inference**: Incorporates spatial dependencies through spatial relation matrices for region-aware causal adjustment
- **Multi-dimensional Covariates**: Supports epidemiological covariates including reproduction numbers, mitigation stringency, vaccination coverage, and ICU capacity
- **Deep Probabilistic Forecasting**: Leverages neural encoder-decoder architecture with multiple output distributions (Gaussian, Laplace, Student's-t)
- **Uncertainty Quantification**: Provides calibrated uncertainty estimates through probabilistic forecasting
- **Interpretable Parameters**: Offers interpretable causal parameters for policy analysis


## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/STOAT.git
cd STOAT

# Install dependencies
pip install -r requirements.txt

# Install STOAT in development mode
pip install -e .
```

### Dependencies

The main dependencies include:
- `gluonts>=0.11.0` - Probabilistic time series forecasting
- `mxnet>=1.9.0` - Deep learning framework
- `numpy>=1.21.0` - Numerical computing
- `pandas>=1.3.0` - Data manipulation
- `scikit-learn>=1.0.0` - Machine learning utilities
- `matplotlib>=3.5.0` - Plotting
- `scipy>=1.7.0` - Scientific computing

## 🏃‍♂️ Quick Start

```python
import numpy as np
import pandas as pd
from stoat import STOAT
from stoat.data import load_epidemic_data
from stoat.utils import create_spatial_matrix

# Load epidemic data
data = load_epidemic_data('path/to/your/data.csv')

# Create spatial relation matrix
spatial_matrix = create_spatial_matrix(data.regions)

# Initialize STOAT model
model = STOAT(
    prediction_length=10,
    context_length=50,
    spatial_matrix=spatial_matrix,
    distr_output='laplace'
)

# Train the model
model.fit(data.train_data)

# Make predictions
forecasts = model.predict(data.test_data)

# Evaluate results
metrics = model.evaluate(forecasts, data.test_targets)
print(f"RMSE: {metrics['RMSE']:.4f}")
```

## 🏗️ Architecture

STOAT consists of two main modules:

### 1. Spatial Causal Inference Module

The spatial causal inference mechanism extends the classical DiD framework:

```
y_{i,t} = ρ Σ_{j=1}^N S_{i,j} y_{j,t} + β₀ + β₁ T_i + β₂ Post_t + δ(T_i · Post_t) + γᵀ c_{i,t} + ε_{i,t}
```

Where:
- `y_{i,t}`: Observed outcome for region i at time t
- `S_{i,j}`: Spatial relation matrix
- `T_i`: Treatment indicator
- `Post_t`: Post-treatment period indicator
- `c_{i,t}`: Multi-dimensional covariate vector
- `δ`: Treatment effect parameter

### 2. Deep Probabilistic Forecasting Module

The forecasting module uses a neural encoder-decoder architecture:

```
P(y_{t₀+1:t₀+m} | y_{1:t₀}, z_{i,1:t₀}, Θ)
```

Where `z_{i,t}` represents the causally adjusted spatial representations.

## 📊 Usage

### Data Preparation

```python
from stoat.data import EpidemicDataProcessor

# Initialize data processor
processor = EpidemicDataProcessor()

# Load and preprocess data
processed_data = processor.load_and_preprocess(
    data_path='data/epidemic_data.csv',
    target_columns=['new_cases'],
    covariate_columns=['reproduction_rate', 'stringency_index', 'vaccination_share', 'icu_patients']
)
```

### Model Training

```python
from stoat import STOAT
from stoat.models import LaplaceOutput

# Initialize model
model = STOAT(
    prediction_length=14,
    context_length=56,
    spatial_matrix=spatial_matrix,
    distr_output=LaplaceOutput(),
    num_cells=64,
    trainer_config={
        'epochs': 100,
        'learning_rate': 0.001,
        'batch_size': 32
    }
)

# Train model
model.fit(train_data)
```

### Prediction and Evaluation

```python
# Generate forecasts
forecasts = model.predict(test_data, num_samples=100)

# Evaluate performance
evaluator = model.get_evaluator()
metrics = evaluator.evaluate(forecasts, test_targets)

# Plot results
model.plot_forecasts(forecasts, test_targets)
```

##  Project Structure

```
STOAT/
├── stoat/                    # Main package
│   ├── __init__.py
│   ├── core/                 # Core functionality
│   │   ├── spatial_causal.py
│   │   ├── forecasting.py
│   │   └── stoat.py
│   ├── models/               # Model implementations
│   │   ├── distributions.py
│   │   ├── neural_networks.py
│   │   └── estimators.py
│   ├── data/                 # Data processing
│   │   ├── processors.py
│   │   └── loaders.py
│   └── utils/                # Utilities
│       ├── spatial.py
│       ├── evaluation.py
│       └── visualization.py
├── examples/                 # Example scripts
│   ├── basic_usage.py
│   ├── covid_forecasting.py
│   └── spatial_analysis.py
├── tests/                    # Test suite
│   ├── test_models.py
│   ├── test_data.py
│   └── test_utils.py
├── docs/                     # Documentation
│   ├── api_reference.md
│   └── tutorials.md
├── requirements.txt
└── README.md
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


**Note**: This is a research implementation. For production use, please ensure proper validation and testing.

