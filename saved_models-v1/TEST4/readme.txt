EPOCHS: 80
LR: 0.0001
Purpose: Stress-test ELU’s capacity to learn nuanced trends (e.g., dividend effects, low-volatility periods).

Explanation:

Why 80 Epochs?
Allows extended training for ultra-low LR to refine subtle patterns (e.g., intra-month seasonality). Early stopping may still terminate early if no progress.

Why LR=0.0001?
Tests ELU’s ability to handle fine-grained adjustments. While ELU mitigates vanishing gradients, this LR ensures minimal weight swings, reducing overfitting risk.