

## Implementation

- Two recommendation models are implemented:
  - **Popularity-Biased Safe Model** (deterministic, low-risk)
  - **Utility-Based Probabilistic Model** (personalized, stochastic)

- The system automatically selects which model to use based on user state:
  - Cold-start or unsatisfied users → Safe Model
  - Experienced users with prior 5★ ratings → Probabilistic Model

- **Part 1** Conditional probabilities and Bayesian updates are computed.
- **Part 2** User patience and behavior patterns are analyzed.
- **Part 3** Both models are designed and implemented.
- **Part 4** A Monte Carlo simulation framework is used for fair evaluation.

## Monte Carlo Evaluation Summary
- User histories are randomly sampled (history length = 10)
- The same history are used for both models
- Metrics are averaged over multiple trials:
  - **Hit@k**
  - **Average Rating**
  - **Time-to-5★ (Tu)**

- 95% confidence intervals are computed for the difference between models

## Key Findings
- The Safe model achieves higher Hit@10 and Average Rating
- Differences are statistically significant based on confidence intervals

## Usage Instructions
- Ensure the dataset files are placed under the `data/` directory:
- Navigate to the project root directory and run the scripts as follows:
python src/part1.py
python src/part2.py
python src/part3.py
python src/part4.py

## Gİthub link
- https://github.com/bouncmpe343/project-2g1f
