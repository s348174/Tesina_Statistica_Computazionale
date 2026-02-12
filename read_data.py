import cdflib
import spacepy
import pandas as pd
import numpy as np
import glob

def drop_fill_value_rows(df, fill_value=-1e31):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return df
    mask = np.isclose(df[numeric_cols], fill_value)
    df.loc[:, numeric_cols] = df.loc[:, numeric_cols].mask(mask)
    return df.loc[~df[numeric_cols].isna().any(axis=1)]

# Load CDF file
# Lista di tutti i file CDF nella cartella data/
cdf_files = glob.glob("data/*.cdf")

dfs = []  # qui accumuliamo i DataFrame

for file in cdf_files:
    print(f"Leggo {file}")
    cdf = cdflib.CDF(file)

    # Time saved as CDF epoch, convert to datetime
    time = cdflib.cdfepoch.to_datetime(cdf.varget("Epoch"))

    # Extract relevant data and create a DataFrame
    data = {
        "time": time,
        "nHe2": cdf.varget("nHe2"),
        "vHe2": cdf.varget("vHe2"),
        "vthHe2": cdf.varget("vthHe2"),
        "vC5": cdf.varget("vC5"),
        "vthC5": cdf.varget("vthC5"),
        "vO6": cdf.varget("vO6"),
        "vthO6": cdf.varget("vthO6"),
        "vFe10": cdf.varget("vFe10"),
        "vthFe10": cdf.varget("vthFe10"),
        "HetoO": cdf.varget("HetoO"),
        "CtoO": cdf.varget("CtoO"),
        "FetoO": cdf.varget("FetoO"),
    }

    df = pd.DataFrame(data)
    df = drop_fill_value_rows(df, fill_value=-1e31)
    dfs.append(df)

# Concatenate all DataFrames
data = pd.concat(dfs, ignore_index=True)
data = data.drop_duplicates(subset="time")
data = data.sort_values("time")

# Export DataFrame to CSV for visualization
data.to_csv("ace_swics_ions_unified.csv", index=False)
print("CSV creato: ace_swics_ions_unified.csv")
print(data.shape)

# Visualize the data
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(data["time"], data["nHe2"], label="nHe2")
plt.xlabel("Time")
plt.ylabel("Density (cm^-3)")
plt.title("Helium Density Over Time")
plt.legend()
plt.tight_layout()
plt.show()

# Visualize ion speeds
plt.figure(figsize=(10,5))
plt.plot(data["time"], data["vHe2"], label="vHe2", color="blue")
plt.plot(data["time"], data["vC5"], label="vC5", color="green")
plt.plot(data["time"], data["vO6"], label="vO6", color="red")
plt.plot(data["time"], data["vFe10"], label="vFe10", color="purple")
plt.xlabel("Time")
plt.ylabel("Speed (km/s)")
plt.title("Ion Speeds Over Time")
plt.legend()
plt.tight_layout()
plt.show()

# Visualize first rows of the DataFrame
print(data.head())