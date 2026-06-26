import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

def main():
    print("Fetching EUR/USD daily data from 2020 to 2024...")
    # Fetch 5 years of daily data (2020-01-01 to 2024-12-31)
    data = yf.download("EURUSD=X", start="2020-01-01", end="2024-12-31", progress=False)
    
    # Check if we got the data
    if data.empty:
        print("Failed to download data.")
        return
        
    print(f"Data fetched: {len(data)} rows.")
    
    # Calculate log returns
    # We use log returns because they are stationary and additive
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
    data = data.dropna()
    
    # Extract year from index for splitting
    data['Year'] = data.index.year
    
    # Split into even years (Train) and odd years (Test)
    even_years = [2020, 2022, 2024]
    odd_years = [2021, 2023]
    
    train_data = data[data['Year'].isin(even_years)].copy()
    test_data = data[data['Year'].isin(odd_years)].copy()
    
    print(f"Training data (Even years): {len(train_data)} rows.")
    print(f"Testing data (Odd years): {len(test_data)} rows.")
    
    # Scale returns by 100 for better numerical stability during optimization
    train_returns = train_data['Log_Return'] * 100
    test_returns = test_data['Log_Return'] * 100
    
    print("\nFitting Markov Switching Model on Even Years (2020, 2022, 2024)...")
    # Fit Markov Regression with 2 states. 
    # We assume the variance also switches between states (k_regimes=2, switching_variance=True)
    try:
        mod_train = sm.tsa.MarkovRegression(
            train_returns, 
            k_regimes=2, 
            trend='c', 
            switching_variance=True
        )
        res_train = mod_train.fit(search_reps=20)
        print(res_train.summary())
    except Exception as e:
        print(f"Error fitting model: {e}")
        return
        
    print("\nApplying learned parameters to predict states in Odd Years (2021, 2023)...")
    
    # To test on odd years, we construct a model on the test data 
    # and filter it using the parameters learned from the training data.
    mod_test = sm.tsa.MarkovRegression(
        test_returns, 
        k_regimes=2, 
        trend='c', 
        switching_variance=True
    )
    
    # Smooth the test data using the training parameters
    try:
        res_test = mod_test.smooth(res_train.params)
        
        # Analyze the states in the test period
        test_data['State_0_Prob'] = res_test.smoothed_marginal_probabilities[0].values
        test_data['State_1_Prob'] = res_test.smoothed_marginal_probabilities[1].values
        
        # Determine predicted state (0 or 1) based on highest probability
        test_data['Predicted_State'] = np.where(test_data['State_0_Prob'] > 0.5, 0, 1)
        
        print("\n--- Conclusion & Analysis on Odd Years ---")
        state_0_mean = res_train.params.filter(like='const[0]').values[0]
        state_1_mean = res_train.params.filter(like='const[1]').values[0]
        state_0_var = res_train.params.filter(like='sigma2[0]').values[0]
        state_1_var = res_train.params.filter(like='sigma2[1]').values[0]
        
        print(f"Learned State 0 (Low Volatility): Mean Return = {state_0_mean:.4f}, Variance = {state_0_var:.4f}")
        print(f"Learned State 1 (High Volatility): Mean Return = {state_1_mean:.4f}, Variance = {state_1_var:.4f}")
        
        print(f"\nIn the odd years (2021, 2023), the model predicts:")
        state_counts = test_data['Predicted_State'].value_counts()
        print(f"- Days in State 0: {state_counts.get(0, 0)}")
        print(f"- Days in State 1: {state_counts.get(1, 0)}")
        
        # Save a plot of the results
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        axes[0].plot(test_data.index, test_returns, label='EUR/USD Returns (Odd Years)', color='black', alpha=0.6)
        axes[0].set_title('EUR/USD Log Returns in Odd Years (2021, 2023)')
        axes[0].set_ylabel('Returns (%)')
        axes[0].legend()
        
        axes[1].plot(test_data.index, test_data['State_1_Prob'], label='Prob of High Volatility (State 1)', color='red')
        axes[1].set_title('Smoothed Probability of High Volatility Regime in Odd Years')
        axes[1].set_ylabel('Probability')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('markov_results_odd_years.png')
        print("\nPlot saved to 'markov_results_odd_years.png'.")
        
    except Exception as e:
        print(f"Error filtering test data: {e}")

if __name__ == "__main__":
    main()
