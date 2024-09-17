import pandas as pd
import numpy as np

# Generate random data with no more than 3 consecutive 'Big' or 'Small'
np.random.seed(42)  # For reproducibility

# Parameters
max_consecutive = 3  # Maximum number of consecutive 'Big' or 'Small'
num_records = 1000

# Initialize the first outcome randomly
outcome = [np.random.choice([0, 1])]  # 0: Small, 1: Big

# Generate the remaining outcomes with constraints
for _ in range(1, num_records):
    if len(outcome) >= max_consecutive and all(x == outcome[-1] for x in outcome[-max_consecutive:]):
        # Flip the outcome if the max_consecutive limit is reached
        outcome.append(1 - outcome[-1])
    else:
        # Randomly choose the next outcome
        outcome.append(np.random.choice([0, 1]))

# Create the DataFrame
data = {
    'period_number': np.arange(1, num_records + 1),
    'outcome': outcome
}

df = pd.DataFrame(data)

# Save to CSV
df.to_csv('data.csv', index=False)

print("Sample data.csv file created!")
