# ==============================================================================
# Project: Age-Friendly Urban Infrastructure & Spatial Accessibility Analysis
# Method: 400m Walkability Buffer & Demographic Spatial Join
# Dependencies: geopandas, pandas, numpy, shapely, matplotlib
# ==============================================================================

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box, Point
import matplotlib.pyplot as plt

np.random.seed(42)

# ------------------------------------------------------------------------------
# STEP 1: Simulate Urban Census Tracts (Demographics) & Public Parks
# ------------------------------------------------------------------------------
# 1. Create a 5x4 grid simulating urban neighborhood tracts
minx, miny, maxx, maxy = 0, 0, 5000, 4000  # 5km x 4km study area (meters)
dx, dy = 1000, 1000
tract_polygons = [box(x, y, x + dx, y + dy) for x in np.arange(minx, maxx, dx) for y in np.arange(miny, maxy, dy)]

census_tracts = gpd.GeoDataFrame({'geometry': tract_polygons}, crs="EPSG:32620") # UTM Projected CRS
census_tracts['TRACT_ID'] = [f"CT_{i+1:02d}" for i in range(len(census_tracts))]

# Generate demographic data (% Senior Citizens age 65+)
census_tracts['total_pop'] = np.random.randint(1500, 5000, len(census_tracts))
census_tracts['pct_seniors'] = np.random.uniform(10.0, 40.0, len(census_tracts))
census_tracts['senior_pop'] = (census_tracts['total_pop'] * (census_tracts['pct_seniors'] / 100)).astype(int)

# 2. Simulate Point Locations of Public Parks & Recreational Spaces
park_coords = [
    (500, 500), (2200, 1500), (3800, 800), (1200, 3200), (4200, 3500)
]
parks = gpd.GeoDataFrame(
    {'PARK_ID': [f"Park_{i+1}" for i in range(len(park_coords))]},
    geometry=[Point(x, y) for x, y in park_coords],
    crs="EPSG:32620"
)

# ------------------------------------------------------------------------------
# STEP 2: Walkability Buffer Analysis (400m Senior Walking Threshold)
# ------------------------------------------------------------------------------
# 400m represents ~5-minute walk, standard for senior mobility planning
WALK_BUFFER_M = 400
parks_buffer = parks.copy()
parks_buffer['geometry'] = parks.buffer(WALK_BUFFER_M)

# Calculate area coverage of parks within each census tract
accessible_areas = gpd.overlay(census_tracts, parks_buffer, how='intersection')
accessible_areas['accessible_sqm'] = accessible_areas.geometry.area

# Aggregate accessible park area back to census tracts
spatial_coverage = accessible_areas.groupby('TRACT_ID')['accessible_sqm'].sum().reset_index()
census_tracts = census_tracts.merge(spatial_coverage, on='TRACT_ID', how='left')
census_tracts['accessible_sqm'] = census_tracts['accessible_sqm'].fillna(0)

# ------------------------------------------------------------------------------
# STEP 3: Compute Age-Friendly Park Deficit Score
# ------------------------------------------------------------------------------
# Park Coverage Ratio = Accessible Park Area / Total Tract Area
census_tracts['tract_area'] = census_tracts.geometry.area
census_tracts['park_coverage_pct'] = (census_tracts['accessible_sqm'] / census_tracts['tract_area']) * 100

# Min-Max Normalization Helper
def min_max(series):
    return (series - series.min()) / (series.max() - series.min())

# High Deficit = High Senior Density + Low Park Coverage
census_tracts['norm_seniors'] = min_max(census_tracts['pct_seniors'])
census_tracts['norm_park_deficit'] = 1 - min_max(census_tracts['park_coverage_pct'])

# Priority Score for Age-Friendly Park Investments (0 to 1 scale)
census_tracts['Priority_Score'] = (census_tracts['norm_seniors'] + census_tracts['norm_park_deficit']) / 2

labels = ['Adequate Access', 'Moderate Deficit', 'High Deficit', 'Critical Priority Zone']
census_tracts['Access_Category'] = pd.qcut(census_tracts['Priority_Score'], q=4, labels=labels)

# ------------------------------------------------------------------------------
# STEP 4: Spatial Visualization
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
census_tracts.plot(
    column='Priority_Score',
    cmap='OrRd',
    legend=True,
    legend_kwds={'label': "Age-Friendly Park Priority Score (High = Infrastructure Deficit)"},
    ax=ax,
    edgecolor='black',
    linewidth=0.5
)

# Overlay Parks & Buffers
parks_buffer.plot(ax=ax, color='green', alpha=0.3, label='400m Walkability Buffer')
parks.plot(ax=ax, color='darkgreen', marker='^', markersize=80, label='Public Parks')

plt.title("Spatial Accessibility & Age-Friendly Park Infrastructure Assessment", fontsize=12)
plt.xlabel("UTM Easting (m)")
plt.ylabel("UTM Northing (m)")
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("age_friendly_park_accessibility.png", dpi=300)

print("--- Senior Accessibility Priority Summary ---")
summary = census_tracts[['TRACT_ID', 'pct_seniors', 'park_coverage_pct', 'Priority_Score', 'Access_Category']]
print(summary.sort_values(by='Priority_Score', ascending=False).head(8))