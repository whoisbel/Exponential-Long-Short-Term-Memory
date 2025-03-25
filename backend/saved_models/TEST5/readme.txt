EPOCHS: 90
LR: 0.001
Purpose: Optimize for precision in long-term forecasts (e.g., annual returns).

Explanation:

Why 90 Epochs?
Maximizes training time for long-term trend refinement (e.g., yearly market cycles). Early stopping may cap effective training at ~75 epochs.

Why LR=0.001?
Balances Adam’s default stability with ELU’s gradient advantages. Avoids the noise of higher LRs while still converging faster than tanh.