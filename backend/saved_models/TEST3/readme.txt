EPOCHS: 64
LR: 0.001
Purpose: Control: Validate if default LR + ELU provides better stability than aggressive pairings, especially for noisy data.

Explanation:

Why 64 Epochs?
A power-of-two value (64 = 2⁶) aligns with batch size=32 for hardware efficiency (GPU memory/cache optimization).

Why LR=0.001?
Adam’s default LR acts as a control to compare against aggressive settings. ELU’s improved gradient flow still aids stability, even at this conservative rate.
