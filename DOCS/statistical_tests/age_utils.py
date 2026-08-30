import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import config


def load_subject_ages(csv_path=None):
    """
    Loads subject ages from the CSV file defined in config.
    Uses 'Repseudonym' as the subject ID.
    Returns a dictionary: {subject_id (str): age (float)}
    """
    if csv_path is None:
        csv_path = config.DELCODE_PATIENT_DATA

    try:
        df = pd.read_csv(csv_path)

        df['Repseudonym'] = df['Repseudonym'].astype(str).str.strip()

        age_map = pd.Series(df.age.values, index=df.Repseudonym).to_dict()

        print(f"[age_utils] Example IDs from CSV: {list(age_map.keys())[:5]}")

        return age_map
    except Exception as e:
        print(f"Error loading ages: {e}")
        return {}


def get_aligned_ages(subject_ids, age_map):
    """
    Returns a list of ages corresponding to the provided subject_ids list.
    Entries not found in age_map will be NaN.
    """
    aligned_ages = []
    missing_count = 0

    for subj_id in subject_ids:
        age = age_map.get(subj_id, np.nan)

        if np.isnan(age):
            missing_count += 1
        aligned_ages.append(age)

    aligned_ages = np.array(aligned_ages).reshape(-1, 1)

    if missing_count > 0:
        print(f"Warning: {missing_count} subjects are missing age data.")

    return aligned_ages


def regress_out_covariate(data_matrix, covariates, fit_on_data=True):
    """
    Regresses out covariates (e.g., Age) from the data_matrix (Connectivity).

    Parameters:
    - data_matrix: (N_subjects x N_edges) numpy array of connectivity values.
    - covariates: (N_subjects x N_covariates) numpy array of nuisance vars (e.g. Age).

    Returns:
    - residuals: The data with the linear effect of the covariate removed.
    - valid_indices: Boolean array of subjects kept (those who had valid age data).
    """

    valid_mask = ~np.isnan(covariates).any(axis=1)

    if np.sum(valid_mask) < len(covariates):
        print(f"Dropping {len(covariates) - np.sum(valid_mask)} subjects due to missing covariates.")

    X = covariates[valid_mask]
    Y = data_matrix[valid_mask]

    model = LinearRegression()
    model.fit(X, Y)

    predicted_effect = model.predict(X)

    # Subtract predicted effect to get Residuals
    # Here we add the intercept (mean) back so the values look like connectivity values.
    residuals = Y - predicted_effect + model.intercept_

    return residuals, valid_mask

def regress_out_covariate_mean_projections(data_matrix, covariates, fit_on_data=True):
    """
    Regresses out covariates (e.g., Age) from the data_matrix (Connectivity).

    Parameters:
    - data_matrix: (N_subjects x N_edges) numpy array of connectivity values.
    - covariates: (N_subjects x N_covariates) numpy array of nuisance vars (e.g. Age).

    Returns:
    - residuals: The data with the linear effect of the covariate removed.
    - valid_indices: Boolean array of subjects kept (those who had valid age data).
    """

    valid_mask = ~np.isnan(covariates).any(axis=1)

    if np.sum(valid_mask) < len(covariates):
        print(f"Dropping {len(covariates) - np.sum(valid_mask)} subjects due to missing covariates.")

    X = covariates[valid_mask]
    Y = data_matrix[valid_mask]


    model = LinearRegression()
    model.fit(X, Y)

    # Calculate the slope effect: Beta * Age
    # We only want to remove the VARIATION due to age, not the absolute level
    # Formula: Corrected = Original - (Slope * (Age - Mean_Age))

    # Get the slope (coefficients)
    beta = model.coef_

    # Center the ages around the global mean
    X_centered = X - X.mean(axis=0)

    # Calculate the adjustment
    adjustment = X_centered @ beta.T

    # Subtract adjustment
    residuals = Y - adjustment

    return residuals, valid_mask