import matplotlib.pyplot as plt

def plot_results_comparison(test_tickers, results, X_test_tensors, y_test_tensors, scalers, sequence_lengths):
    """
    Plots the comparison of predicted and actual values for different activation functions and sequence lengths.
    Args:
        test_tickers (list): List of ticker symbols to be tested.
        results (dict): Dictionary containing the predictions and targets for each combination of activation function, sequence length, and ticker.
        X_test_tensors (list): List of test input tensors.
        y_test_tensors (list): List of test output tensors.
        scalers (list): List of scalers used for normalizing the data.
        sequence_lengths (list): List of sequence lengths to be tested.
    Returns:
        None
    """
    for seq_length in sequence_lengths:
        plt.figure(figsize=(15, 10))
        for activation_name in ['tanh', 'elu']:
            for ticker in test_tickers:
                key = (activation_name, seq_length, ticker)
                if key in results:
                    preds = results[key]["predictions"]
                    targets = results[key]["targets"]
                    plt.plot(preds, label=f'{activation_name.capitalize()} - {ticker} - Seq Length {seq_length}')
                    plt.plot(targets, label=f'{ticker} Actual - Seq Length {seq_length}', linestyle='--')
        
        plt.title(f'Predictions vs Actual - Seq Length {seq_length}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
