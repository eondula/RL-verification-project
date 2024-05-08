# RL-verification-project
Class project for CSCI 699: Safe-Learning Enabled Autonomous Systems Spring 2024

## Project Overview

This project develops and evaluates a reinforcement learning model for managing classroom attendance during an ongoing epidemic. The model aims to balance the dual objectives of minimizing infection risks and maximizing in-person educational activities. The project employs PyTorch for model development and the Z3 theorem prover for verification of the learned policies.

## Dependencies

- Python 3.8+
- PyTorch
- tqdm
- matplotlib
- numpy
- z3-solver
- pandas

## Key Components

1. **Model Definition (`Model` class)**: A neural network model defined using PyTorch. It predicts policies based on the current state, which includes community risk and the number of infected individuals.

2. **Infection Dynamics (`get_infected_students_apprx_sir` function)**: Simulates the number of new infections based on current policies and epidemiological parameters.

3. **Policy Evaluation (`get_label` function)**: Generates labels for training by evaluating the consequences of different allowed attendance levels on infection spread.

4. **Training Loop (`train` function)**: Executes the training process over a specified number of epochs, adjusting model weights based on observed rewards and the effectiveness of actions taken under various simulated conditions.

5. **Verification (`verify` function)**: Uses the Z3 solver to check if the learned model adheres to desired safety conditions under high-risk scenarios.

6. **Visualization (`visualize_model_behavior` function)**: Visualizes the model's behavior over a range of inputs to assess policy consistency and effectiveness.

## Setup and Running

### Installation

Install the required packages using pip:

```bash
pip install torch tqdm matplotlib numpy z3-solver pandas
