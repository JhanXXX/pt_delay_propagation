
from src.poi_joint import prepare_stop_pairs
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import os
from sklearn.preprocessing import PolynomialFeatures

def compare_methods(db_path, gap, output_dir,ob_window = 1200):
    # step 1: prepare data
    # gap is how many stations of the delay propagation is considered
    prepare_stop_pairs(db_path,gap,ob_window)

    output_dir=Path(f"{output_dir}/after_{gap}_stop")
    os.makedirs(output_dir, exist_ok=True)

    # step 2: load dataset (only bus)
    with sqlite3.connect(db_path) as conn:
        load_query = """
            SELECT * FROM stop_pairs
            WHERE route_type = '700'
            AND arr_delay_i IS NOT NULL 
            AND arr_delay_j IS NOT NULL
            AND travel_time > 0
        """

        df = pd.read_sql_query(load_query, conn)
        print(f"Loaded {len(df):,} observations")
        # Add transport mode label
        df['transport_mode'] = df['route_type'].map({
            700: 'Bus'
        })

    # Treatment: Continuous delay at stop i (in seconds)
    # Outcome: Continuous delay at stop j ({gap} stations after i) (in seconds)
    df['treatment'] = df['arr_delay_i'].copy()
    df['outcome'] = df['arr_delay_j'].copy()

    # remove 1% of outliers
    # low_i, high_i = np.percentile(df['delay_i'].dropna(), [0.5, 99.5])
    # low_j, high_j = np.percentile(df['delay_j'].dropna(), [0.5, 99.5])
    # df = df[
    #     (df['delay_i'].between(low_i, high_i)) &
    #     (df['delay_j'].between(low_j, high_j))
    #]
    # Keep the "outliers" which are actually extreme delays
    print(f"Remained {len(df):,} observations after outlier filtering")
    print(f"Treatment: delay_i (continuous, in seconds)")
    print(f"  Mean: {df['treatment'].mean():.2f} seconds")
    print(f"  Std: {df['treatment'].std():.2f} seconds")
    print(f"  Range: [{df['treatment'].min():.2f}, {df['treatment'].max():.2f}]")

    print(f"\nOutcome: delay_j (continuous, in seconds)")
    print(f"  Mean: {df['outcome'].mean():.2f} seconds")
    print(f"  Std: {df['outcome'].std():.2f} seconds")

    # Visualize treatment distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df['treatment'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='On-time')
    axes[0].set_xlabel('Treatment: delay_i (seconds)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Distribution of Treatment (Delay at Stop i)', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(df['treatment'].sample(5000), df['outcome'].sample(5000), 
                    alpha=0.3, s=10, color='purple')
    axes[1].plot([df['treatment'].min(), df['treatment'].max()], 
                [df['treatment'].min(), df['treatment'].max()], 
                'r--', linewidth=2, label='y=x')
    axes[1].set_xlabel('Treatment: delay_i (seconds)', fontsize=11)
    axes[1].set_ylabel('Outcome: delay_j (seconds)', fontsize=11)
    axes[1].set_title('Treatment-Outcome Relationship (Raw)', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/treatment_outcome_distribution.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir}/treatment_outcome_distribution.png")
    plt.close()

    # step 3: identify confounders

    print("\n" + "=" * 60)
    print("IDENTIFYING CONFOUNDERS")
    print("=" * 60)

    # Define confounders
    confounders = [
        'hour_of_day',      # Time of day (peak hours)
        'his_dwell_count_i', # how many vehicles have passed by stop i during the last observation window
        'his_avg_delay_i',    # historical average delay at stop i during the last observation window
        'his_avg_dwell_i',  # historical average dwell time at stop i during the last observation window
    ]
    # Whether it's working day was not considered because the data are all weekdays (Mon. - Thurs. morning)
    # Add dummy variables for categorical confounders
    # df['is_bus'] = (df['transport_mode'] == 'Bus').astype(int) # removed because we only look at the buses
    df['is_morning_peak'] = ((df['hour_of_day'] <= 10)).astype(int)
    df['is_evening_peak'] = ((df['hour_of_day'] >= 15) ).astype(int)
    # df['is_weekday'] = (df['day_of_week'].between(1, 5)).astype(int) # removed because only have selected weekdays

    confounders_extended = confounders + ['is_morning_peak', 'is_evening_peak']
    #                                      , 'is_weekday', 'is_bus']

    print("\nConfounders included:")
    for conf in confounders_extended:
        print(f"  - {conf}")

    # Drop rows with missing values
    df_clean = df.dropna(subset=confounders_extended + ['treatment', 'outcome']).copy()
    print(f"\nClean dataset: {len(df_clean):,} observations")


    # ============================================
    # Method 1 - Naive Comparison
    # ============================================
    print("\n" + "=" * 60)
    print("METHOD 1: NAIVE COMPARISON (BASELINE)")
    print("=" * 60)

    # Simple linear regression: Y ~ D (no confounders)
    D = df_clean['treatment'].values.reshape(-1, 1)
    Y = df_clean['outcome'].values

    naive_model = LinearRegression()
    naive_model.fit(D, Y)
    naive_beta = naive_model.coef_[0]
    naive_intercept = naive_model.intercept_

    print(f"Naive estimate: β = {naive_beta:.4f}")
    print(f"Interpretation: Each 1-second increase in delay_i is associated")
    print(f"                with {naive_beta:.4f} seconds increase in delay_j")

    # ============================================
    # Method 2 - Regression Adjustment
    # ============================================
    print("\n" + "=" * 60)
    print("METHOD 2: REGRESSION ADJUSTMENT")
    print("=" * 60)

    # Prepare features: Y ~ D + X
    X_confounders = df_clean[confounders_extended].values
    D_with_X = np.column_stack([D, X_confounders])

    # Standardize for better interpretation
    scaler_X = StandardScaler()
    X_confounders_scaled = scaler_X.fit_transform(X_confounders)
    D_with_X_scaled = np.column_stack([D, X_confounders_scaled])

    # Fit regression
    reg_model = LinearRegression()
    reg_model.fit(D_with_X_scaled, Y)

    regression_beta = reg_model.coef_[0]
    print(f"Regression-adjusted estimate: β = {regression_beta:.4f}")
    print(f"Interpretation: Controlling for confounders, each 1-second increase")
    print(f"                in delay_i causes {regression_beta:.4f} seconds increase in delay_j")
    print("\nRegression coefficients:")
    for i, conf in enumerate(confounders_extended):
        print(f"  {conf}: {reg_model.coef_[i+1]:.4f}")

    # ============================================
    # Method 3 - Generalized Propensity Score (GPS)
    # ============================================
    print("\n" + "=" * 60)
    print("METHOD 3: GENERALIZED PROPENSITY SCORE (GPS)")
    print("=" * 60)
    print("Reference: Hirano & Imbens (2004)")

    # Step 1: Estimate treatment model D|X ~ Normal(μ(X), σ²)
    print("\nStep 1: Estimating treatment model E[D|X]...")
    
    # PolynomialFeatures 
    poly = PolynomialFeatures(degree=1, include_bias=False)
    X_poly = poly.fit_transform(X_confounders_scaled)

    # treatment model
    treatment_model = LinearRegression()
    treatment_model.fit(X_poly, D.ravel())
    # Predicted treatment given confounders
    mu_D = treatment_model.predict(X_poly)

    residuals = D.ravel() - mu_D
    sigma_D = np.std(residuals)

    print(f"Treatment model R²: {treatment_model.score(X_poly, D.ravel()):.4f}")
    print(f"Residual std (σ): {sigma_D:.2f}")

    # Step 2: Calculate GPS (conditional density)
    print("\nStep 2: Calculating Generalized Propensity Score...")
    GPS = norm.pdf(D.ravel(), loc=mu_D, scale=sigma_D)
    df_clean['GPS'] = GPS

    print(f"GPS range: [{GPS.min():.6f}, {GPS.max():.6f}]")
    print(f"Mean GPS: {GPS.mean():.6f}")

    # Visualize GPS distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # GPS distribution
    axes[0].hist(GPS, bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[0].set_xlabel('Generalized Propensity Score', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Distribution of GPS', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # GPS vs Treatment
    scatter = axes[1].scatter(D.ravel(), GPS, alpha=0.3, s=10, c=Y, cmap='coolwarm')
    axes[1].set_xlabel('Treatment: delay_i (seconds)', fontsize=11)
    axes[1].set_ylabel('GPS', fontsize=11)
    axes[1].set_title('GPS vs Treatment (colored by outcome)', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=axes[1], label='Outcome: delay_j')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/gps_distribution.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/gps_distribution.png")
    plt.close()

    # Step 3: Trim for common support
    print("\nStep 3: Trimming for common support...")
    gps_min = np.percentile(GPS, 2.5)
    gps_max = np.percentile(GPS, 97.5)
    common_support = (GPS >= gps_min) & (GPS <= gps_max)

    print(f"Common support: GPS in [{gps_min:.6f}, {gps_max:.6f}]")
    print(f"Observations retained: {common_support.sum():,} ({common_support.mean():.1%})")
    trimmed = ~common_support



    # Apply common support
    df_cs = df_clean[common_support].copy()
    D_cs = D[common_support]
    Y_cs = Y[common_support]
    GPS_cs = GPS[common_support]
    X_cs = X_confounders_scaled[common_support]
    mu_D_cs = mu_D[common_support]

    print(f"\n=== Analysis of Trimmed Observations ===")
    print(f"Trimmed: {trimmed.sum():,} ({trimmed.mean():.1%})")
    print(f"Trimmed mean delay_i: {D[trimmed].mean():.2f}")
    print(f"Trimmed mean delay_j: {Y[trimmed].mean():.2f}")
    print(f"Retained mean delay_i: {D_cs.mean():.2f}")
    print(f"Retained mean delay_j: {Y_cs.mean():.2f}")

    # Step 4: Estimate dose-response function using GPS
    print("\nStep 4: Estimating dose-response function...")

    # Method: Regression of Y on D and GPS
    # E[Y | D, r(D,X)]
    D_GPS_matrix = np.column_stack([D_cs, GPS_cs])
    dose_response_model = LinearRegression()
    dose_response_model.fit(D_GPS_matrix, Y_cs.ravel())

    gps_beta = dose_response_model.coef_[0]
    print(f"GPS-adjusted estimate: β = {gps_beta:.4f}")
    print(f"Interpretation: Balancing treatment assignment, each 1-second increase")
    print(f"                in delay_i causes {gps_beta:.4f} seconds increase in delay_j")

    # Step 5: Estimate dose-response curve
    print("\nStep 5: Estimating dose-response curve...")

    # Create grid of treatment values
    D_grid = np.linspace(D_cs.min(), D_cs.max(), 50)
    dose_response_curve = []

    for d in D_grid:
        # For each treatment level, compute E[Y | D=d, GPS]
        # Use weighted regression where weights come from GPS evaluated at d
        weights = norm.pdf(d, loc=mu_D_cs, scale=sigma_D)
        
        # Weighted average outcome
        weighted_outcome = np.average(Y_cs, weights=weights)
        dose_response_curve.append(weighted_outcome)

    # Plot dose-response curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(D_grid, dose_response_curve, linewidth=3, color='blue', label='GPS dose-response')
    ax.scatter(D_cs[::50], Y_cs[::50], alpha=0.2, s=10, color='gray', label='Observed data (sample)')
    ax.set_xlabel('Treatment: delay_i (seconds)', fontsize=12)
    ax.set_ylabel('Expected Outcome: E[delay_j | delay_i]', fontsize=12)
    ax.set_title('Dose-Response Function (GPS-adjusted)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dose_response_curve.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/dose_response_curve.png")
    plt.close()


    # ============================================
    # Method 4 - Doubly Robust Estimation
    # ============================================
    print("\n" + "=" * 60)
    print("METHOD 4: DOUBLY ROBUST ESTIMATION (GPS + OUTCOME MODEL)")
    print("=" * 60)

    # Step 1: Fit outcome model E[Y|D,X]
    print("\nStep 1: Fitting outcome regression model E[Y|D,X]...")
    D_X_matrix = np.column_stack([D_cs, X_cs])
    outcome_model = LinearRegression()
    outcome_model.fit(D_X_matrix, Y_cs.ravel())

    # Predicted outcomes
    mu_Y = outcome_model.predict(D_X_matrix)

    # Step 2: Compute doubly robust estimator
    # DR = E[mu(D,X)] + E[(Y - mu(D,X)) · h(D,X)]
    # where h(D,X) is a weight function based on GPS

    print("\nStep 2: Computing doubly robust estimate...")

    # Simple DR: regress residuals on treatment with GPS weights
    residuals_Y = Y_cs.ravel() - mu_Y
    weights_dr = 1 / GPS_cs  # Inverse GPS weighting

    # Weighted regression of residuals on D
    dr_model = LinearRegression()
    sample_weights = weights_dr / weights_dr.sum() * len(weights_dr)  # Normalize
    dr_model.fit(D_cs, residuals_Y, sample_weight=sample_weights)

    dr_beta = outcome_model.coef_[0] + dr_model.coef_[0]

    print(f"Doubly Robust estimate: β = {dr_beta:.4f}")

    # ============================================
    # Covariate Balance Check
    # ============================================
    print("\n" + "=" * 60)
    print("COVARIATE BALANCE CHECK")
    print("=" * 60)

    # For continuous treatment, check correlation between D and X
    # before and after GPS adjustment

    balance_results = []

    for i, conf in enumerate(confounders_extended):
        X_conf = df_clean[conf].values
        
        # Before adjustment: correlation between D and X
        corr_before = np.corrcoef(D.ravel(), X_conf)[0, 1]
        
        # After GPS adjustment: partial correlation controlling for GPS
        # Use residuals from regressing X on GPS
        X_conf_cs = X_cs[:, i]
        


        # using polynomial GPS for resid
        poly_gps = PolynomialFeatures(degree=2, include_bias=False)
        GPS_poly = poly_gps.fit_transform(GPS_cs.reshape(-1, 1))

        # X

        gps_x_model = LinearRegression()
        gps_x_model.fit(GPS_poly, X_conf_cs)
        X_resid = X_conf_cs - gps_x_model.predict(GPS_poly)

        # D
        gps_d_model = LinearRegression()
        gps_d_model.fit(GPS_poly, D_cs.ravel())
        D_resid = D_cs.ravel() - gps_d_model.predict(GPS_poly)
        
        # Correlation of residuals
        corr_after = np.corrcoef(D_resid, X_resid)[0, 1]
        
        balance_results.append({
            'Confounder': conf,
            'Correlation_before': corr_before,
            'Correlation_after': corr_after,
            'Reduction': abs(corr_before) - abs(corr_after)
        })

    df_balance = pd.DataFrame(balance_results)
    print("\nCorrelation between Treatment and Confounders:")
    print("(Good balance: correlation close to 0 after GPS adjustment)")
    print(df_balance.to_string(index=False))

    # Visualize balance
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df_balance))
    width = 0.35
    ax.barh(x - width/2, np.abs(df_balance['Correlation_before']), width, 
            label='Before GPS adjustment', alpha=0.7, color='red')
    ax.barh(x + width/2, np.abs(df_balance['Correlation_after']), width, 
            label='After GPS adjustment', alpha=0.7, color='green')
    ax.set_yticks(x)
    ax.set_yticklabels(df_balance['Confounder'])
    ax.set_xlabel('Absolute Correlation with Treatment')
    ax.set_title('Covariate Balance: Before and After GPS Adjustment')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/covariate_balance_gps.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir}/covariate_balance_gps.png")
    plt.close()

    # ============================================
    # Summary of All Methods
    # ============================================
    print("\n" + "=" * 60)
    print("SUMMARY: COMPARISON OF ALL METHODS")
    print("=" * 60)

    results_summary = pd.DataFrame({
        'Method': [
            '1. Naive (No adjustment)',
            '2. Regression (Linear model)',
            '3. GPS (Nonparametric balance)',
            '4. Doubly Robust (GPS + Outcome model)'
        ],
        'Coefficient β': [
            naive_beta,
            regression_beta,
            gps_beta,
            dr_beta
        ],
        'Controls Confounding': [
            'No',
            'Yes (parametric)',
            'Yes (balancing)',
            'Yes (doubly robust)'
        ],
        'Assumes linearity': [
            'Yes',
            'Yes',
            'No (flexible)',
            'Partially'
        ]
    })

    print("\n" + results_summary.to_string(index=False))
    print(f"\nInterpretation of coefficient β:")
    print(f"Each 1-second increase in delay at stop i causes β seconds")
    print(f"increase in delay at stop j (on average)")

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red', 'orange', 'green', 'blue']
    bars = ax.barh(results_summary['Method'], results_summary['Coefficient β'], 
                color=colors, alpha=0.7)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Estimated Causal Effect β (seconds per second)', fontsize=11)
    ax.set_title('Comparison of Causal Effect Estimates\n(Effect of delay_i on delay_j)', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, results_summary['Coefficient β'])):
        ax.text(val, bar.get_y() + bar.get_height()/2, f' {val:.4f}', 
                va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/causal_estimates_comparison.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir}/causal_estimates_comparison.png")
    plt.close()

