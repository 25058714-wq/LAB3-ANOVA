import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ========== Font Settings ==========
plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False

# ========== Read Data ==========
df = pd.read_csv("WorldEnergy.csv")

# ========== ASEAN-6 Countries ==========
asean = [
    "Indonesia", "Thailand", "Malaysia",
    "Philippines", "Singapore", "Vietnam"
]

# ========== Select Columns ==========
cols = [
    "country", "year",
    "renewables_consumption",
    "primary_energy_consumption"
]

df = df[cols].copy()

# ========== Data Filtering ==========
df = df[df["country"].isin(asean)]
df = df[(df["year"] >= 2000) & (df["year"] <= 2023)]

print("=" * 60)
print("Before Cleaning:", df.shape)
print(df.isnull().sum())

# ========== Data Cleaning ==========
df = df.dropna()
df = df[df["primary_energy_consumption"] > 0]

# ========== Feature Engineering ==========
df["renew_ratio"] = df["renewables_consumption"] / df["primary_energy_consumption"]

print("=" * 60)
print("After Cleaning:", df.shape)
print(df.isnull().sum())

# ========== Descriptive Statistics ==========
print("\nDescriptive Statistics by Country:")
desc = df.groupby("country")["renew_ratio"].describe()
print(desc)

# ========== Boxplot ==========
plt.figure(figsize=(10, 5))
sns.boxplot(x="country", y="renew_ratio", data=df)
plt.title("Clean Energy Ratio by ASEAN-6 Country (2000-2023)")
plt.xlabel("Country")
plt.ylabel("Clean Energy Ratio")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# ========== One-way ANOVA ==========
groups = [
    df[df["country"] == country]["renew_ratio"]
    for country in asean
]

f_stat, p_value = stats.f_oneway(*groups)

print("\nOne-way ANOVA Result:")
print("F-statistic:", f_stat)
print("p-value:", p_value)

alpha = 0.05
if p_value < alpha:
    print("Conclusion: Reject H0. There is a significant difference among countries.")
else:
    print("Conclusion: Fail to reject H0. No significant difference is found among countries.")

# ========== ANOVA Table using statsmodels ==========
model = ols("renew_ratio ~ C(country)", data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print("\nANOVA Table:")
print(anova_table)

# ========== Assumption Check ==========
# 1. Normality of residuals
shapiro_stat, shapiro_p = stats.shapiro(model.resid)

print("\nShapiro-Wilk Test for Normality:")
print("Statistic:", shapiro_stat)
print("p-value:", shapiro_p)

# 2. Homogeneity of variance
levene_stat, levene_p = stats.levene(*groups)

print("\nLevene's Test for Homogeneity of Variance:")
print("Statistic:", levene_stat)
print("p-value:", levene_p)

# ========== Tukey HSD Post-hoc Test ==========
if p_value < alpha:
    tukey = pairwise_tukeyhsd(
        endog=df["renew_ratio"],
        groups=df["country"],
        alpha=0.05
    )

    print("\nTukey HSD Post-hoc Test:")
    print(tukey)

    tukey.plot_simultaneous(figsize=(10, 6))
    plt.title("Tukey HSD Post-hoc Comparison")
    plt.xlabel("Clean Energy Ratio Difference")
    plt.tight_layout()
    plt.show()
