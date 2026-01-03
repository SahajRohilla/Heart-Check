from ucimlrepo import fetch_ucirepo
import pandas as pd

# Fetch dataset from UCI repository
heart_disease = fetch_ucirepo(id=45)

# Get features and target
X = heart_disease.data.features
y = heart_disease.data.targets

# Combine into single dataframe
df = pd.concat([X, y], axis=1)

# Rename target column to 'target' and convert to binary (0 = no disease, 1 = disease)
df = df.rename(columns={'num': 'target'})
df['target'] = (df['target'] > 0).astype(int)

# Save to CSV
df.to_csv('data/heart.csv', index=False)

print(f"✓ Downloaded {len(df)} rows to data/heart.csv")
print(f"✓ Shape: {df.shape}")
print(f"✓ Columns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())
