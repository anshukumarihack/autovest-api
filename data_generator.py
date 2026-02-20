import pandas as pd
import numpy as np

def generate_financial_data(n_samples: int = 10000) -> pd.DataFrame:
    """Generates synthetic but realistic user transaction data for AutoVest."""
    np.random.seed(42) # For reproducibility
    
    # 1. Generate Features
    # Daily spending: Log-normal distribution (most spend little, few spend a lot)
    daily_spending = np.random.lognormal(mean=3.5, sigma=0.8, size=n_samples)
    
    # Spare change: Roughly 2% to 8% of daily spending
    spare_change_pct = np.random.uniform(0.02, 0.08, size=n_samples)
    spare_change_total = daily_spending * spare_change_pct
    
    # Spending variance: Normal distribution around 1.0
    spending_variance = np.random.normal(loc=1.0, scale=0.4, size=n_samples)
    spending_variance = np.clip(spending_variance, 0.1, 3.0)
    
    # Emergency balance ratio: 1.0 means fully funded (3-6 months expenses)
    emergency_balance_ratio = np.random.normal(loc=0.8, scale=0.5, size=n_samples)
    emergency_balance_ratio = np.clip(emergency_balance_ratio, 0.0, 2.0)
    
    # Market risk score: 0.0 (safe) to 1.0 (volatile/risky)
    market_risk_score = np.random.uniform(0.0, 1.0, size=n_samples)
    
    # User type: 60% professionals, 40% students
    user_types = np.random.choice(['professional', 'student'], size=n_samples, p=[0.6, 0.4])
    
    df = pd.DataFrame({
        'daily_spending': daily_spending,
        'spare_change_total': spare_change_total,
        'spending_variance': spending_variance,
        'emergency_balance_ratio': emergency_balance_ratio,
        'market_risk_score': market_risk_score,
        'user_type': user_types
    })
    
    # 2. Rule-based Label Generation (Simulating expert financial logic)
    # Default to Invest (1)
    invest_today = np.ones(n_samples, dtype=int)
    
    # Rule A: If emergency balance is critically low, pause investing
    invest_today[df['emergency_balance_ratio'] < 0.3] = 0
    
    # Rule B: If market is highly volatile/risky, pause
    invest_today[df['market_risk_score'] > 0.85] = 0
    
    # Rule C: If user's spending is highly erratic (variance > 1.8), pause to prevent overdrafts
    invest_today[df['spending_variance'] > 1.8] = 0
    
    # Rule D: Students with low spare change shouldn't invest today (conserving cash)
    student_mask = (df['user_type'] == 'student') & (df['spare_change_total'] < 1.5)
    invest_today[student_mask] = 0
    
    # Add a small amount of noise (5%) to simulate human unpredictability/edge cases
    noise_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    invest_today[noise_indices] = 1 - invest_today[noise_indices] # Flip labels
    
    df['invest_today'] = invest_today
    return df

if __name__ == "__main__":
    print("Generating synthetic dataset...")
    df = generate_financial_data(10000)
    df.to_csv("autovest_data.csv", index=False)
    print(f"Data saved to autovest_data.csv. Shape: {df.shape}")
    print(df['invest_today'].value_counts(normalize=True))