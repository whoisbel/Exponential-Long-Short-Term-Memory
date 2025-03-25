EPOCHS: 50
LR: 0.01
Purpose: Stress-test ELU’s ability to learn volatile patterns quickly while relying on early stopping to prevent instability.

Explanation:

Why 50 Epochs?
A shorter training window leverages ELU’s non-saturating gradients, which allow faster convergence compared to tanh. Early stopping (patience=15) ensures training halts if validation loss plateaus early, avoiding wasted computation.

Why LR=0.01?
ELU’s smooth gradient flow for negative inputs and non-saturation for positives reduces the risk of divergence, making this aggressive LR viable. This tests whether the model can rapidly adapt to sudden market shifts (e.g., crashes).