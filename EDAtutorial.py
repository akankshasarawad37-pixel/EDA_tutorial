import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Set a consistent visualization style
sns.set_style("whitegrid")
# Display all columns when viewing a DataFrame
pd.set_option("display.max_columns", None)

url = (
"https://raw.githubusercontent.com/allisonhorst/palmerpenguins/"
"main/inst/extdata/penguins.csv"
)
df = pd.read_csv(url)
print("Dataset loaded successfully!")

# # Example: keep penguins.csv in the same folder as your notebook
# df = pd.read_csv("penguins.csv")  #only if i have .csv file in the same folder as my notebook. 

# First five rows
df.head()
# Last five rows
df.tail()
# Five randomly selected rows
df.sample(5, random_state=42)

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])
df.info()
# Display only the data types
df.dtypes

# Summary of numerical columns
Des = df.describe()
df.describe()
# Summary of categorical columns
df.describe(include="object") 
print(Des)

print(df.head())
print(df.columns)

# 1. Number of observations for each species
species_count = df["species"].value_counts()
print(species_count)

# 2. Percentage of observations for each species
species_percentage = df["species"].value_counts(normalize=True) * 100
print(species_percentage)

# 3. Number of observations from each island
island_count = df["island"].value_counts()
print(island_count)

# 4. Cross-tabulation of species by island
species_island = pd.crosstab(df["species"], df["island"])
print(species_island)

# Number of observations for each species
df["species"].value_counts()
# Percentage of observations for each species
df["species"].value_counts(normalize=True) * 100
# Number of observations from each island
df["island"].value_counts()
# Cross-tabulation of species by island
pd.crosstab(df["species"], df["island"])  

species_count
species_percentage
island_count
species_island

# Count missing values in each column
df.isnull().sum()
# Calculate the percentage of missing values in each column
missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage)

plt.figure(figsize=(10, 5))
sns.heatmap(
df.isnull(),
cbar=False,
yticklabels=False
)
plt.title("Missing Values in the Penguin Dataset")
plt.xlabel("Variables")
plt.ylabel("Observations")
plt.tight_layout()
plt.show()

# 1. Keep the original data unchanged
df_clean = df.copy()

# 2. Identify numerical columns
numerical_cols = df_clean.select_dtypes(include=np.number).columns

# 3. Fill missing numerical values with each column's median
for col in numerical_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# 4. Identify categorical columns
categorical_cols = df_clean.select_dtypes(include="object").columns

# 5. Fill missing categorical values with each column's mode
for col in categorical_cols:
    mode_values = df_clean[col].mode()
    if not mode_values.empty:
        df_clean[col] = df_clean[col].fillna(mode_values[0])

# 6. Verify the result
print(df_clean.isnull().sum())


# Count exact duplicate rows
print("Duplicate rows:", df_clean.duplicated().sum())
# Remove exact duplicates only if they are confirmed to be accidental
df_clean = df_clean.drop_duplicates()
# Check again
print("Duplicates after cleaning:", df_clean.duplicated().sum())

plt.figure(figsize=(8, 5))
plt.hist(
df_clean["body_mass_g"].dropna(),
bins=20,
edgecolor="black"
)
plt.title("Distribution of Penguin Body Mass")
plt.xlabel("Body Mass (grams)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(x=df_clean["body_mass_g"])
plt.title("Boxplot of Penguin Body Mass")
plt.xlabel("Body Mass (grams)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.kdeplot(
data=df_clean,
x="body_mass_g",
fill=True
)
plt.title("Density Distribution of Penguin Body Mass")
plt.xlabel("Body Mass (grams)")
plt.tight_layout()
plt.show()

Q1 = df_clean["body_mass_g"].quantile(0.25)
Q3 = df_clean["body_mass_g"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df_clean[
(df_clean["body_mass_g"] < lower_bound) |
(df_clean["body_mass_g"] > upper_bound)
]
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower bound:", lower_bound)
print("Upper bound:", upper_bound)
print("Number of potential outliers:", len(outliers))
outliers

plt.figure(figsize=(8, 5))
sns.scatterplot(
data=df_clean,
x="flipper_length_mm",
y="body_mass_g",
hue="species",
s=70
)
plt.title("Flipper Length vs Body Mass")
plt.xlabel("Flipper Length (mm)")
plt.ylabel("Body Mass (g)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(
data=df_clean,
x="species",
y="body_mass_g"
)
plt.title("Body Mass Across Penguin Species")
plt.xlabel("Species")
plt.ylabel("Body Mass (g)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.violinplot(
data=df_clean,
x="species",
y="flipper_length_mm"
)
plt.title("Flipper Length Distribution Across Species")
plt.xlabel("Species")
plt.ylabel("Flipper Length (mm)")
plt.tight_layout()
plt.show()


# Select numerical columns
numeric_df = df_clean.select_dtypes(include=np.number)
# Calculate Pearson correlation coefficients
correlation = numeric_df.corr()
print(correlation)
plt.figure(figsize=(9, 6))
sns.heatmap(
correlation,
annot=True,
cmap="coolwarm",
fmt=".2f"
)
plt.title("Correlation Matrix of Penguin Measurements")
plt.tight_layout()
plt.show()

sns.pairplot(
df_clean,
vars=[
"bill_length_mm",
"bill_depth_mm",
"flipper_length_mm",
"body_mass_g"
],
hue="species"
)
plt.show()

species_summary = df_clean.groupby("species")[
[
"bill_length_mm",
"bill_depth_mm",
"flipper_length_mm",
"body_mass_g"
]
].mean()
print(species_summary)

# Calculate several summary statistics for body mass:
body_mass_summary = df_clean.groupby("species")["body_mass_g"].agg(
["count", "mean", "median", "std", "min", "max"]
)
print(body_mass_summary)

sex_comparison = df_clean.dropna(subset=["sex"])
sns.boxplot(
data=sex_comparison,
x="sex",
y="body_mass_g"
)
plt.title("Body Mass by Recorded Sex")
plt.xlabel("Recorded sex")
plt.ylabel("Body mass (g)")
plt.tight_layout()
plt.show()
# Group-wise summary
sex_comparison.groupby("sex")["body_mass_g"].agg(
["count", "mean", "median", "std"]
)

print("EDA completed successfully!")
print("penguins do not care about your feelings, they just want to be understood.")
print("penguins do not care about their body mass or flipper length or anything, they just wanna eat and sleep(aahhh i want to be a penguin)")