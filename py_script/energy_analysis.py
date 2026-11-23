import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "owid_energy.csv"
OUTPUT_DIR = BASE_DIR / "output"
CHART_DIR = OUTPUT_DIR / "charts"

OUTPUT_DIR.mkdir(exist_ok=True)
CHART_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

data = pd.read_csv(DATA_PATH, low_memory = False)
data.columns = data.columns.str.strip().str.lower()

print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())

START_YEAR = 2000
END_YEAR = 2024

if "country" not in data.columns:
    raise ValueError("Country columns not found")

if "year" not in data.columns:
    raise ValueError("Year columns not found")

if "iso_code" in data.columns:
    indonesia = data[(data["iso_code"] == "IDN") & (data["country"].str.lower() == "indonesia")].copy()
else:
    indonesia = data[data["country"].str.lower() == "indonesia"].copy()

if indonesia.empty:
    raise ValueError("Indonesia data not found in the dataset")

indonesia["year"] = pd.to_numeric(indonesia["year"], errors="coerce")
indonesia = indonesia.dropna(subset=["year"])
indonesia["year"] = indonesia["year"].astype(int)

indonesia = indonesia[
    (indonesia["year"] >= START_YEAR) &
    (indonesia["year"] <= END_YEAR)
].copy()

indonesia = (indonesia.sort_values("year").drop_duplicates(subset = ["year"]).reset_index(drop = True))

for col in indonesia.columns:
    if col not in ["country", "iso_code"]:
        indonesia[col] = pd.to_numeric(indonesia[col], errors="coerce")

print("\nFiltered Data :")      
print("Year range :", indonesia["year"].min(), "-", indonesia["year"].max())

# Data quality check
print("Duplicate records :", indonesia.duplicated().sum())
print("\nMissing values : ")
print(indonesia.isnull().sum().sort_values(ascending=False).head(5))

indonesia.to_csv(OUTPUT_DIR / "01_indonesia_energy_data.csv", index = False)


# Helper Functions

def save_chart(filename):
    plt.tight_layout()
    plt.savefig(CHART_DIR / filename, dpi = 300, bbox_inches = "tight")
    plt.close()

def calculate_cagr(series):
    """
    Calculate the Compound Annual Growth Rate (CAGR).
    Uses first and last valid positive observations.
    """
    valid = series.dropna()
    if len(valid) < 2:
        return np.nan
    
    first = valid.iloc[0]
    last = valid.iloc[-1]

    years = valid.index[-1] - valid.index[0]

    if first <= 0 or last < 0 or years <= 0:
        return np.nan
    return ((last / first) ** (1 / years) - 1) * 100

def latest_value(column):
    if column not in indonesia.columns:
        return np.nan

    valid = indonesia[["year", column]].dropna()

    if valid.empty:
        return np.nan

    return valid.iloc[-1][column]


# ANALYSIS
# Total  Energy Consumption

energy = indonesia[["year", "total_energy_twh"]].dropna().copy()
energy = energy.sort_values("year")

energy["yoy_growth_pct"] = (energy["total_energy_twh"].pct_change(fill_method = None) * 100)
energy["annual_chage_pct"] = (energy["total_energy_twh"].diff())

energy.to_csv(OUTPUT_DIR / "02_energy_consumption_analysis.csv", index = False)


plt.figsize = (14, 7)
sns.lineplot(
    data = energy,
    x = "year",
    y = "total_energy_twh",
    marker = "o",
    markersize = 7.0,
    linewidth = 2.0,
    color = "red"
)

plt.title("Indonesia Total Energy Consumption", fontsize = 14 , fontweight = "bold")
plt.xlabel("Year", fontsize = 12)
plt.ylabel("Energy (TWh)", fontsize = 12)

save_chart("01_total_energy_consumption.png")



# Fossil Fuel Dependency Analysis

fossil_columns = [
    "coal_energy_twh",
    "oil_energy_twh",
    "gas_energy_twh"
]
fossil = indonesia[["year"] + fossil_columns].copy()
fossil["fossil_fuel_total_twh"] = (fossil[fossil_columns].sum(axis = 1, min_count = 3))
fossil["fossil_share_pct"] = (fossil["fossil_fuel_total_twh"] / indonesia["total_energy_twh"] * 100)

fossil.to_csv(OUTPUT_DIR / "03_fossil_fuel_analysis.csv", index = False)


# Fossil Fuel Trend
plt.figure(figsize = (14, 7))
for col in fossil_columns:
    plt.plot(
        fossil["year"],
        fossil[col],
        marker = "o",
        markersize = 7.0,
        linewidth = 2.0,
        label = col.replace("_energy_twh","").title()
    )

plt.title("Indonesia Fossil Fuel Consumption", fontsize = 14 , fontweight = "bold")
plt.xlabel("Year", fontsize = 12)
plt.ylabel("Energy (TWh)", fontsize = 12)
plt.legend()

save_chart("02_fossil_fuel_trend.png")


# Fossil Fuel Dependency
plt.figure(figsize = (14, 7))
plt.plot(
        fossil["year"],
        fossil["fossil_share_pct"],
        marker = "o",
        markersize = 7.0,
        linewidth = 2.0,
        color = "navy"
    )

plt.title("Indonesia Fossil Fuel Dependency", fontsize = 14 , fontweight = "bold")
plt.xlabel("Year", fontsize = 12)
plt.ylabel("Fossil Fuel Share (%)", fontsize = 12)

save_chart("03_fossil_fuel_dependency.png")



# Renewable Energy Analysis

renewable = indonesia[
    [
        "year",
        "renewables_energy_twh",
        "renewables_energy_share_pct"
    ]
].copy()

renewable = renewable.dropna(subset = ["renewables_energy_twh"])

renewable["renewable_growth_pct"] = (renewable["renewables_energy_twh"].pct_change(fill_method = None) * 100)

renewable.to_csv(OUTPUT_DIR / "04_renewable_energy_analysis.csv", index = False)


# Renewable Energy Trend
plt.figure(figsize = (14, 7))
plt.plot(
    renewable["year"],
    renewable["renewables_energy_twh"],
    marker = "o",
    markersize = 7.0,
    linewidth = 2.0,
    color = "Brown"
)

plt.title("Indonesia Renewable Energy Consumption", fontsize = 14 , fontweight = "bold")
plt.xlabel("Year", fontsize = 12)   
plt.ylabel("Energy (TWh)", fontsize = 12)

save_chart("04_renewable_energy_trend.png")


# Renwable Energy Share
plt.figure(figsize = (14, 7))
plt.plot(
    renewable["year"],
    renewable["renewables_energy_share_pct"],
    marker = "o",
    markersize = 7.0,
    linewidth = 2.0,
    color = "orange"
)

plt.title("Indonesia Renewable Energy Share in Total Energy", fontsize = 14 , fontweight = "bold")
plt.xlabel("Year", fontsize = 12)
plt.ylabel("Renewable Share (%)", fontsize = 12)

save_chart("05_renewable_energy_share.png")



#Electricity Generation Analysis

electricity_columns = [
    "coal_electricity_twh",
    "gas_electricity_twh",
    "oil_electricity_twh",
    "nuclear_electricity_twh",
    "renewables_electricity_twh"
]

electricity = indonesia[["year"] + electricity_columns].copy()

electricity.to_csv(OUTPUT_DIR / "05_electricity_generation_analysis.csv", index = False)

plt.figure(figsize = (14, 7))
for col in electricity_columns:
    plt.plot(
        electricity["year"],
        electricity[col],
        marker = "o",
        markersize = 7.0,
        linewidth = 2.0,
        label = col.replace("_electricity_twh","").title()
    )

plt.title("Indonesia Electricity Generation by Source", fontsize = 14 , fontweight = "bold")
plt.xlabel("Year", fontsize = 12)  
plt.ylabel("Electricity Generation (TWh)", fontsize = 12)
plt.legend()

save_chart("06_electricity_generation.png")


# Electtricity Renewable Share
plt.figure(figsize = (14, 7))
plt.plot(
    indonesia["year"],
    indonesia["renewables_electricity_share_pct"],
    marker = "o",
    markersize = 7.0,
    linewidth = 2.0,
    color = "darkgreen"
)
plt.title("Indonesia Renewable Energy Share", fontsize = 14 , fontweight = "bold")
plt.xlabel("Year", fontsize = 12)
plt.ylabel("Renewable Electricity Share (%)", fontsize = 12)

save_chart("07_renewable_electricity_share.png")



# Carbon Emissions Analysis

emissions = indonesia[
    [
        "year",
        "electricity_emissions_mtco2eq",
        "electricity_carbon_intensity_gco2eq_per_kwh"
    ]
].copy()

emissions.to_csv(OUTPUT_DIR / "06_carbon_emissions_analysis.csv", index = False)

plt.figure(figsize = (14, 7))
plt.plot(
    emissions["year"],
    emissions["electricity_emissions_mtco2eq"],
    marker = "o",
    markersize = 7.0,
    linewidth = 2.0,
    color = "navy"
)

plt.title("Indonesia Electricity Carbon Emissions", fontsize = 14 , fontweight = "bold")
plt.xlabel("Year", fontsize = 12)   
plt.ylabel("Emissions (MtCO2eq)", fontsize = 12)

save_chart("08_electricity_emissions.png")



# Energy Efficiency and GDP

efficiency = indonesia[
    [
        "year",
        "gdp",
        "total_energy_per_gdp_kwh_per_dollar",
        "total_energy_per_capita_kwh"
    ]
].copy()

efficiency.to_csv(OUTPUT_DIR / "07_energy_efficiency_analysis.csv", index = False)

plt.figure(figsize = (14, 7))
plt.plot(
    efficiency["year"],
    efficiency["total_energy_per_gdp_kwh_per_dollar"],
    marker = "o",
    markersize = 7.0,
    linewidth = 2.0,
    color = "purple"
)

plt.title("Indonesia Energy Efficiency", fontsize = 14 , fontweight = "bold")
plt.xlabel("Year", fontsize = 12)
plt.ylabel("Energy per GDP (kWh per Dollar)", fontsize = 12)

save_chart("09_energy_efficiency.png")



# Production Analysis

production_columns = [
    "coal_production_twh",
    "oil_production_twh",
    "gas_production_twh"
]

production = indonesia[["year"] + production_columns].copy()

production.to_csv(OUTPUT_DIR / "08_energy_production_analysis.csv", index = False)

plt.figure(figsize = (14, 7))
for col in production_columns:
    plt.plot(
        production["year"],
        production[col],
        marker = "o",
        markersize = 7.0,
        linewidth = 2.0,
        label = col.replace("_production_twh","").title()
    )

plt.title("Indonesia Energy Production by Source", fontsize = 14 , fontweight = "bold")
plt.xlabel("Year", fontsize = 12)  
plt.ylabel("Energy Production (TWh)", fontsize = 12)
plt.legend()

save_chart("10_energy_production.png")

# Energy by Sources
latest_year = int(indonesia["year"].max())
year_data = indonesia.loc[indonesia["year"] == latest_year]

sources = ["biofuels", "coal", "fossil_fuels", "gas", "hydro", "oil", "solar", "wind"]

energy_twh = [
    round(year_data["biofuels_energy_twh"].iloc[0], 4),
    round(year_data["coal_energy_twh"].iloc[0], 4),
    round(year_data["fossil_fuels_energy_twh"].iloc[0], 4),
    round(year_data["gas_energy_twh"].iloc[0], 4),
    round(year_data["hydro_energy_twh"].iloc[0], 4),
    round(year_data["oil_energy_twh"].iloc[0], 4),
    round(year_data["solar_energy_twh"].iloc[0], 4),
    round(year_data["wind_energy_twh"].iloc[0], 4)
]

report = pd.DataFrame({
    "sources": sources,
    "energy_twh": energy_twh
})

report.to_csv(OUTPUT_DIR / "09_indonesia_energy_by_sources.csv", index = False)


# Key Performance Indicators

kpi = {
    "latest_year": latest_year,
    "total_energy_twh": round(latest_value("total_energy_twh"), 4),
    "coal_energy_twh": round(latest_value("coal_energy_twh"), 4),
    "oil_energy_twh": round(latest_value("oil_energy_twh"), 4),
    "gas_energy_twh": round(latest_value("gas_energy_twh"), 4),
    "renewables_energy_twh": round(latest_value("renewables_energy_twh"), 4),
    "renewables_energy_share_pct": round(latest_value("renewables_energy_share_pct"), 4),
    "electricity_generation": round(latest_value("electricity_generation_twh"), 4),
    "renewables_electricity_share_pct": round(latest_value("renewables_electricity_share_pct"), 4),
    "electricity_emissions_mtco2eq": round(latest_value("electricity_emissions_mtco2eq"), 4),
    "carbon_intensity": round(latest_value("electricity_carbon_intensity_gco2eq_per_kwh"), 4),
    "energy_per_capita": round(latest_value("total_energy_per_capita_kwh"), 4),
    "energy_cagr_pct": round(calculate_cagr(renewable.set_index("year")["renewables_energy_twh"]), 4)
}

kpi_data = pd.DataFrame([kpi])

kpi_data.to_csv(OUTPUT_DIR / "10_indonesia_energy_kpis.csv", index = False)

print("\nINDONESIA ENERGY KPIs")
for key, value in kpi.items():
    print(f"{key} : {value}")


# FINAL REPORT
print("\nINDONESIA ENERGY ANALYSIS REPORT")

print("Analysis period : ", indonesia["year"].min(), "-", indonesia["year"].max())
print("Total observations : ", len(indonesia))
print("Total charts generated : 10")

