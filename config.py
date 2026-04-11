# DATect Forecasting Configuration
# Settings for data processing, modeling, and web interface
import os

# Data Sources and Paths

# Historical DA toxin measurement files
ORIGINAL_DA_FILES = {
    "twin-harbors": "./data/raw/da-input/twin-harbors-da.csv",
    "long-beach": "./data/raw/da-input/long-beach-da.csv",
    "quinault": "./data/raw/da-input/quinault-da.csv",
    "kalaloch": "./data/raw/da-input/kalaloch-da.csv",
    "copalis": "./data/raw/da-input/copalis-da.csv",
    "newport": "./data/raw/da-input/newport-da.csv",
    "gold-beach": "./data/raw/da-input/gold-beach-da.csv",
    "coos-bay": "./data/raw/da-input/coos-bay-da.csv",
    "clatsop-beach": "./data/raw/da-input/clatsop-beach-da.csv",
    "cannon-beach": "./data/raw/da-input/cannon-beach-da.csv"
}

# Pseudo-nitzschia cell count data files
ORIGINAL_PN_FILES = {
    "gold-beach-pn": "./data/raw/pn-input/gold-beach-pn.csv",
    "coos-bay-pn": "./data/raw/pn-input/coos-bay-pn.csv",
    "newport-pn": "./data/raw/pn-input/newport-pn.csv",
    "clatsop-beach-pn": "./data/raw/pn-input/clatsop-beach-pn.csv",
    "cannon-beach-pn": "./data/raw/pn-input/cannon-beach-pn.csv",
    "kalaloch-pn": "./data/raw/pn-input/kalaloch-pn.csv",
    "copalis-pn": "./data/raw/pn-input/copalis-pn.csv",
    "long-beach-pn": "./data/raw/pn-input/long-beach-pn.csv",
    "twin-harbors-pn": "./data/raw/pn-input/twin-harbors-pn.csv",
    "quinault-pn": "./data/raw/pn-input/quinault-pn.csv"
}

# Environmental Data URLs

# Climate indices from NOAA ERDDAP
PDO_URL = "https://oceanview.pfeg.noaa.gov/erddap/tabledap/cciea_OC_PDO.nc?time%2CPDO&time%3E=2002-05-01&time%3C=2025-01-01T00%3A00%3A00Z"
ONI_URL = "https://oceanview.pfeg.noaa.gov/erddap/tabledap/cciea_OC_ONI.nc?time%2CONI&time%3E=2002-05-01&time%3C=2024-12-01T00%3A00%3A00Z"

# BEUTI (Biologically Effective Upwelling Transport Index)
BEUTI_URL = "https://oceanview.pfeg.noaa.gov/erddap/griddap/erdBEUTIdaily.nc?BEUTI%5B(2002-05-01):1:(2024-11-28T00:00:00Z)%5D%5B(42):1:(47.0)%5D"

# Columbia River streamflow data
STREAMFLOW_URL = "https://waterservices.usgs.gov/nwis/dv?format=json&siteStatus=all&site=14246900&agencyCd=USGS&statCd=00003&parameterCd=00060&startDT=2002-06-01&endDT=2025-02-22"

# Monitoring Sites

# Pacific Coast monitoring site coordinates [lat, lon]
SITES = {
    "Kalaloch": [47.58597, -124.37914],
    "Quinault": [47.28439, -124.23612],
    "Copalis": [47.10565, -124.1805],
    "Twin Harbors": [46.79202, -124.09969],
    "Long Beach": [46.55835, -124.06088],
    "Clatsop Beach": [46.028889, -123.917222],
    "Cannon Beach": [45.881944, -123.959444],
    "Newport": [44.6, -124.05],
    "Coos Bay": [43.376389, -124.237222],
    "Gold Beach": [42.377222, -124.414167]
}

# Date Ranges
START_DATE = "2003-01-01"
END_DATE = "2023-12-31"

FINAL_OUTPUT_PATH = "./data/processed/final_output.parquet"

SATELLITE_CACHE_PATH = "./data/intermediate/satellite_data_intermediate.parquet"

# Satellite Data Configuration

# MODIS oceanographic data URLs with date placeholders

SATELLITE_DATA = {
    # Chlorophyll-a (8-day composite)
    "modis-chla": {
        "Kalaloch": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWchla8day_LonPM180.nc?chlorophyll%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.4875):1:(47.6875)%5D%5B(-124.575):1:(-124.375)%5D",
        "Quinault": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWchla8day_LonPM180.nc?chlorophyll%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.1875):1:(47.3875)%5D%5B(-124.4375):1:(-124.2375)%5D",
        "Copalis": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWchla8day_LonPM180.nc?chlorophyll%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.0):1:(47.2)%5D%5B(-124.375):1:(-124.175)%5D",
        "Twin Harbors": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWchla8day_LonPM180.nc?chlorophyll%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(46.6875):1:(46.8875)%5D%5B(-124.3):1:(-124.1)%5D",
        "Long Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWchla8day_LonPM180.nc?chlorophyll%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(46.4625):1:(46.6625)%5D%5B(-124.2625):1:(-124.0625)%5D",
        "Clatsop Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWchla8day_LonPM180.nc?chlorophyll%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(45.925):1:(46.125)%5D%5B(-124.1125):1:(-123.9125)%5D",
        "Cannon Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWchla8day_LonPM180.nc?chlorophyll%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(45.7875):1:(45.9875)%5D%5B(-124.1625):1:(-123.9625)%5D",
        "Newport": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWchla8day_LonPM180.nc?chlorophyll%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(44.4):1:(44.8)%5D%5B(-124.45):1:(-124.05)%5D",
        "Coos Bay": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWchla8day_LonPM180.nc?chlorophyll%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(43.175):1:(43.575)%5D%5B(-124.6375):1:(-124.2375)%5D",
        "Gold Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWchla8day_LonPM180.nc?chlorophyll%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(42.175):1:(42.575)%5D%5B(-124.8125):1:(-124.4125)%5D"
    },
    
    # Sea Surface Temperature (8-day composite)
    "modis-sst": {
        "Kalaloch": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWsstd8day_LonPM180.nc?sst%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.4875):1:(47.6875)%5D%5B(-124.575):1:(-124.375)%5D",
        "Quinault": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWsstd8day_LonPM180.nc?sst%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.1875):1:(47.3875)%5D%5B(-124.4375):1:(-124.2375)%5D",
        "Copalis": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWsstd8day_LonPM180.nc?sst%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.0):1:(47.2)%5D%5B(-124.375):1:(-124.175)%5D",
        "Twin Harbors": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWsstd8day_LonPM180.nc?sst%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(46.6875):1:(46.8875)%5D%5B(-124.3):1:(-124.1)%5D",
        "Long Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWsstd8day_LonPM180.nc?sst%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(46.4625):1:(46.6625)%5D%5B(-124.2625):1:(-124.0625)%5D",
        "Clatsop Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWsstd8day_LonPM180.nc?sst%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(45.925):1:(46.125)%5D%5B(-124.1125):1:(-123.9125)%5D",
        "Cannon Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWsstd8day_LonPM180.nc?sst%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(45.7875):1:(45.9875)%5D%5B(-124.1625):1:(-123.9625)%5D",
        "Newport": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWsstd8day_LonPM180.nc?sst%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(44.4):1:(44.8)%5D%5B(-124.45):1:(-124.05)%5D",
        "Coos Bay": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWsstd8day_LonPM180.nc?sst%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(43.175):1:(43.575)%5D%5B(-124.6375):1:(-124.2375)%5D",
        "Gold Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWsstd8day_LonPM180.nc?sst%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(42.175):1:(42.575)%5D%5B(-124.8125):1:(-124.4125)%5D"
    },
    
    # Photosynthetically Available Radiation (8-day composite)
    "modis-par": {
        "Kalaloch": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWpar08day_LonPM180.nc?par%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.4875):1:(47.6875)%5D%5B(-124.575):1:(-124.375)%5D",
        "Quinault": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWpar08day_LonPM180.nc?par%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.1875):1:(47.3875)%5D%5B(-124.4375):1:(-124.2375)%5D",
        "Copalis": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWpar08day_LonPM180.nc?par%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.0):1:(47.2)%5D%5B(-124.375):1:(-124.175)%5D",
        "Twin Harbors": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWpar08day_LonPM180.nc?par%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(46.6875):1:(46.8875)%5D%5B(-124.3):1:(-124.1)%5D",
        "Long Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWpar08day_LonPM180.nc?par%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(46.4625):1:(46.6625)%5D%5B(-124.2625):1:(-124.0625)%5D",
        "Clatsop Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWpar08day_LonPM180.nc?par%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(45.925):1:(46.125)%5D%5B(-124.1125):1:(-123.9125)%5D",
        "Cannon Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWpar08day_LonPM180.nc?par%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(45.7875):1:(45.9875)%5D%5B(-124.1625):1:(-123.9625)%5D",
        "Newport": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWpar08day_LonPM180.nc?par%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(44.4):1:(44.8)%5D%5B(-124.45):1:(-124.05)%5D",
        "Coos Bay": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWpar08day_LonPM180.nc?par%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(43.175):1:(43.575)%5D%5B(-124.6375):1:(-124.2375)%5D",
        "Gold Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWpar08day_LonPM180.nc?par%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(42.175):1:(42.575)%5D%5B(-124.8125):1:(-124.4125)%5D"
    },
    
    # Fluorescence Line Height (8-day composite)
    "modis-flur": {
        "Kalaloch": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWcflh8day_LonPM180.nc?fluorescence%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.4875):1:(47.6875)%5D%5B(-124.575):1:(-124.375)%5D",
        "Quinault": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWcflh8day_LonPM180.nc?fluorescence%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.1875):1:(47.3875)%5D%5B(-124.4375):1:(-124.2375)%5D",
        "Copalis": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWcflh8day_LonPM180.nc?fluorescence%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.0):1:(47.2)%5D%5B(-124.375):1:(-124.175)%5D",
        "Twin Harbors": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWcflh8day_LonPM180.nc?fluorescence%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(46.6875):1:(46.8875)%5D%5B(-124.3):1:(-124.1)%5D",
        "Long Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWcflh8day_LonPM180.nc?fluorescence%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(46.4625):1:(46.6625)%5D%5B(-124.2625):1:(-124.0625)%5D",
        "Clatsop Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWcflh8day_LonPM180.nc?fluorescence%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(45.925):1:(46.125)%5D%5B(-124.1125):1:(-123.9125)%5D",
        "Cannon Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWcflh8day_LonPM180.nc?fluorescence%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(45.7875):1:(45.9875)%5D%5B(-124.1625):1:(-123.9625)%5D",
        "Newport": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWcflh8day_LonPM180.nc?fluorescence%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(44.4):1:(44.8)%5D%5B(-124.45):1:(-124.05)%5D",
        "Coos Bay": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWcflh8day_LonPM180.nc?fluorescence%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(43.175):1:(43.575)%5D%5B(-124.6375):1:(-124.2375)%5D",
        "Gold Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWcflh8day_LonPM180.nc?fluorescence%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(42.175):1:(42.575)%5D%5B(-124.8125):1:(-124.4125)%5D"
    },
    
    # Diffuse Attenuation Coefficient K490 (8-day composite)
    "modis-k490": {
        "Kalaloch": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWk4908day_LonPM180.nc?k490%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.4875):1:(47.6875)%5D%5B(-124.575):1:(-124.375)%5D",
        "Quinault": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWk4908day_LonPM180.nc?k490%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.1875):1:(47.3875)%5D%5B(-124.4375):1:(-124.2375)%5D",
        "Copalis": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWk4908day_LonPM180.nc?k490%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.0):1:(47.2)%5D%5B(-124.375):1:(-124.175)%5D",
        "Twin Harbors": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWk4908day_LonPM180.nc?k490%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(46.6875):1:(46.8875)%5D%5B(-124.3):1:(-124.1)%5D",
        "Long Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWk4908day_LonPM180.nc?k490%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(46.4625):1:(46.6625)%5D%5B(-124.2625):1:(-124.0625)%5D",
        "Clatsop Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWk4908day_LonPM180.nc?k490%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(45.925):1:(46.125)%5D%5B(-124.1125):1:(-123.9125)%5D",
        "Cannon Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWk4908day_LonPM180.nc?k490%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(45.7875):1:(45.9875)%5D%5B(-124.1625):1:(-123.9625)%5D",
        "Newport": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWk4908day_LonPM180.nc?k490%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(44.4):1:(44.8)%5D%5B(-124.45):1:(-124.05)%5D",
        "Coos Bay": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWk4908day_LonPM180.nc?k490%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(43.175):1:(43.575)%5D%5B(-124.6375):1:(-124.2375)%5D",
        "Gold Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWk4908day_LonPM180.nc?k490%5B({start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(42.175):1:(42.575)%5D%5B(-124.8125):1:(-124.4125)%5D"
    },
    
    # Chlorophyll-a Anomaly (monthly, 0-360° longitude)
    "chla-anom": {
        "Kalaloch": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2ChlaAnom.nc?chla_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.4875):1:(47.6875)%5D%5B(235.425):1:(235.625)%5D",
        "Quinault": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2ChlaAnom.nc?chla_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.1875):1:(47.3875)%5D%5B(235.5625):1:(235.7625)%5D",
        "Copalis": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2ChlaAnom.nc?chla_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.0):1:(47.2)%5D%5B(235.625):1:(235.825)%5D",
        "Twin Harbors": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2ChlaAnom.nc?chla_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(46.6875):1:(46.8875)%5D%5B(235.7):1:(235.9)%5D",
        "Long Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2ChlaAnom.nc?chla_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(46.4625):1:(46.6625)%5D%5B(235.7375):1:(235.9375)%5D",
        "Clatsop Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2ChlaAnom.nc?chla_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(45.925):1:(46.125)%5D%5B(235.8875):1:(236.0875)%5D",
        "Cannon Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2ChlaAnom.nc?chla_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(45.7875):1:(45.9875)%5D%5B(235.8375):1:(236.0375)%5D",
        "Newport": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2ChlaAnom.nc?chla_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(44.4):1:(44.8)%5D%5B(235.55):1:(235.95)%5D",
        "Coos Bay": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2ChlaAnom.nc?chla_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(43.175):1:(43.575)%5D%5B(235.3625):1:(235.7625)%5D",
        "Gold Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2ChlaAnom.nc?chla_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(42.175):1:(42.575)%5D%5B(235.1875):1:(235.5875)%5D"
    },
    
    # Sea Surface Temperature Anomaly (monthly)
    "sst-anom": {
        "Kalaloch": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2SstAnom.nc?sst_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.4875):1:(47.6875)%5D%5B(235.425):1:(235.625)%5D",
        "Quinault": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2SstAnom.nc?sst_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.1875):1:(47.3875)%5D%5B(235.5625):1:(235.7625)%5D",
        "Copalis": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2SstAnom.nc?sst_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(47.0):1:(47.2)%5D%5B(235.625):1:(235.825)%5D",
        "Twin Harbors": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2SstAnom.nc?sst_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(46.6875):1:(46.8875)%5D%5B(235.7):1:(235.9)%5D",
        "Long Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2SstAnom.nc?sst_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(46.4625):1:(46.6625)%5D%5B(235.7375):1:(235.9375)%5D",
        "Clatsop Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2SstAnom.nc?sst_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(45.925):1:(46.125)%5D%5B(235.8875):1:(236.0875)%5D",
        "Cannon Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2SstAnom.nc?sst_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(45.7875):1:(45.9875)%5D%5B(235.8375):1:(236.0375)%5D",
        "Newport": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2SstAnom.nc?sst_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(44.4):1:(44.8)%5D%5B(235.55):1:(235.95)%5D",
        "Coos Bay": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2SstAnom.nc?sst_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(43.175):1:(43.575)%5D%5B(235.3625):1:(235.7625)%5D",
        "Gold Beach": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/osu2SstAnom.nc?sst_anomaly%5B({anom_start_date}):1:({end_date})%5D%5B(0.0):1:(0.0)%5D%5B(42.175):1:(42.575)%5D%5B(235.1875):1:(235.5875)%5D"
    },
    
    # Date ranges for satellite data
    "satellite_start_date": "2002-07-16T12:00:00Z",
    "satellite_anom_start_date": "2003-01-16T12:00:00Z",
    "satellite_end_date": "2025-01-16T12:00:00Z"
}

# Forecast Configuration

# Operation mode: "retrospective" (historical validation) or "realtime" (dashboard)
FORECAST_MODE = "retrospective"

# Task type: "regression" (continuous DA levels) or "classification" (risk categories)
FORECAST_TASK = "regression"

# ML algorithm: "ensemble", "naive", or "linear"
FORECAST_MODEL = "ensemble"

# Forecast Horizon Configuration
# How many weeks ahead to forecast from the data cutoff point
FORECAST_HORIZON_WEEKS = 1
FORECAST_HORIZON_DAYS = FORECAST_HORIZON_WEEKS * 7  # Derived days value for internal calculations

# XGBoost Hyperparameters (configurable)
# These override defaults in ModelFactory for reproducible tuning and easy experimentation.
# Regression parameters
XGB_REGRESSION_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "colsample_bylevel": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "gamma": 0.1,
    "min_child_weight": 3,
    "tree_method": "hist",
}

# Classification parameters
XGB_CLASSIFICATION_PARAMS = {
    "n_estimators": 500,
    "max_depth": 7,
    "learning_rate": 0.03,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "colsample_bylevel": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "gamma": 0.2,
    "min_child_weight": 5,
    "tree_method": "hist",
    "eval_metric": "logloss",
}


# Temporal Validation - prevents data leakage (handled by forecast horizon)

# Model Performance
MIN_TRAINING_SAMPLES = 10
RANDOM_SEED = 42

# Bootstrap confidence intervals
ENABLE_BOOTSTRAP_INTERVALS = True  # Disable to skip bootstrap uncertainty
N_BOOTSTRAP_ITERATIONS = 100  # Number of bootstrap iterations for confidence intervals

# Lag Feature Configuration

# Time series lags for raw observation-order lag features (env override: comma-separated or "none")
# Set DATECT_LAG_FEATURES=none to disable all lag features for ablation studies.
_lag_env = os.environ.get("DATECT_LAG_FEATURES", "")
LAG_FEATURES = [] if _lag_env.lower() == "none" else [int(x) for x in _lag_env.split(",") if x.strip()] if _lag_env else [1, 2, 3, 4]

# DA Category Configuration

# Risk thresholds for classification: Low (0-5), Moderate (5-20), High (20-40), Extreme (>40 μg/g)
DA_CATEGORY_BINS = [-float("inf"), 5, 20, 40, float("inf")]
DA_CATEGORY_LABELS = [0, 1, 2, 3]

# Spike Detection Configuration
SPIKE_THRESHOLD = 20.0  # DA > 20 μg/g considered a spike event
SPIKE_ALERT_PROB_THRESHOLD = 0.10  # Probability threshold for spike alert (optimized for transition recall)
SPIKE_CLASSIFIER_ENABLED = True    # Toggle spike binary classifier on/off
SPIKE_REGRESSION_ALERT_THRESHOLD = 12.0  # Fire spike_alert when ensemble prediction >= this (µg/g)

# Spike binary classifier hyperparameters (tuned for per-test-point training
# with small safe-baseline datasets — shallower/simpler than 4-category classifier)
SPIKE_CLASSIFIER_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "gamma": 0.1,
    "min_child_weight": 3,
    "eval_metric": "logloss",
}


# Bootstrap subsample fraction for uncertainty estimation
BOOTSTRAP_SUBSAMPLE_FRACTION = 1.0  # Use full resample for each iteration

# Scientific Methodology Configuration

# Ridge (linear competitor) regularization
LINEAR_REGRESSION_ALPHA = 1.0

# Linear/logistic models use the full feature set (no whitelist)

# Persistence baseline configuration
PERSISTENCE_MAX_DAYS = None  # Set to an int (e.g., 28) to cap lookback

# Confidence interval percentiles for bootstrap predictions
CONFIDENCE_PERCENTILES = [5, 50, 95]  # 5th percentile, median, 95th percentile

# Data Quality Configuration

# Feature engineering toggles
USE_ROLLING_FEATURES = True  # Enable rolling statistics features for raw pipeline

# Biological Decay Interpolation Parameters
# Used for filling gaps in DA/PN measurements with exponential decay

# DA (Domoic Acid) parameters - conservative due to frequent zeros in data
DA_MAX_GAP_WEEKS = 2  # Maximum gap to interpolate (larger gaps filled with 0)
DA_DECAY_RATE = 0.2   # Per week decay rate (half-life ~3.5 weeks)

# PN (Pseudo-nitzschia) parameters - more aggressive due to sparser data
PN_MAX_GAP_WEEKS = 4  # Maximum gap to interpolate
PN_DECAY_RATE = 0.3   # Per week decay rate (half-life ~2.3 weeks)


# =============================================================================
# RAW DATA PIPELINE CONFIGURATION
# Parameters for the raw-data ensemble forecasting pipeline.
# =============================================================================

# Random Forest Regression parameters
# RF uses the same feature pipeline as XGBoost. No per-anchor tuning (RF is robust to hyperparams).
RF_REGRESSION_PARAMS = {
    "n_estimators": 400,
    "max_depth": 12,
    "min_samples_split": 5,
    "min_samples_leaf": 3,
    "max_features": 0.85,
}
# Allow stability study to override RF params via env var (loky workers re-import this)
_rf_params_json = os.environ.get("DATECT_RF_PARAMS_JSON", "")
if _rf_params_json:
    import json as _json
    RF_REGRESSION_PARAMS = {**RF_REGRESSION_PARAMS, **_json.loads(_rf_params_json)}

# Target and model toggles (overridable via env vars for ablation studies)
USE_PER_SITE_MODELS = os.environ.get("DATECT_USE_PER_SITE_MODELS", "true").lower() == "true"
USE_INTERPOLATED_TRAINING = os.environ.get("DATECT_USE_INTERPOLATED_TRAINING", "true").lower() == "true"
USE_GPU = False                  # CPU inference (set True for CUDA-enabled systems)

# Monotonic constraints on persistence features (XGBoost only)
# Prevents learning physically implausible relationships (e.g. higher recent DA → lower forecast)
# Especially valuable at small-N Oregon sites where overfitting is likely.
# +1 = strictly non-decreasing, -1 = strictly non-increasing, 0 = unconstrained
USE_MONOTONIC_CONSTRAINTS = os.environ.get("DATECT_USE_MONOTONIC_CONSTRAINTS", "true").lower() == "true"
MONOTONIC_FEATURE_CONSTRAINTS: dict = {
    "last_observed_da_raw": 1,       # More recent DA → higher or equal forecast
    "raw_obs_roll_max_4": 1,         # Higher 4-week max → higher or equal forecast
    "raw_obs_roll_mean_4": 1,        # Higher 4-week mean → higher or equal forecast
}

# Minimum training rows required before per-anchor XGB tuning is attempted.
# Oregon sites with ~50-80 rows lose 30% of training data to tuning between
# nearly-identical param sets. Skip tuning below this threshold.
MIN_TRAINING_FOR_TUNING = int(os.environ.get("DATECT_MIN_TRAINING_FOR_TUNING", "80"))

# Prediction clipping
PREDICTION_CLIP_Q = 0.99         # Clip predictions to this quantile of training targets
_clip_override = os.environ.get("DATECT_CLIP_Q_OVERRIDE", "")
if _clip_override == "none":
    PREDICTION_CLIP_Q = None
elif _clip_override:
    PREDICTION_CLIP_Q = float(_clip_override)

# Parallelization
ENABLE_PARALLEL = True
N_JOBS = -1                      # Use all cores (-1)

# Per-anchor tuning / calibration
CALIBRATION_FRACTION = 0.3       # Fraction of pre-anchor history used for tuning
MAX_CALIBRATION_ROWS = 20        # Hard cap on calibration rows per anchor
MIN_TUNING_SAMPLES = 10          # Skip tuning if fewer calibration rows available

# Default XGB search grid (used when site has no custom param_grid)
PARAM_GRID = [
    {"max_depth": 4, "n_estimators": 500, "learning_rate": 0.05, "min_child_weight": 5},
    {"max_depth": 6, "n_estimators": 400, "learning_rate": 0.05, "min_child_weight": 3},
]

# Quantile prediction intervals
ENABLE_QUANTILE_INTERVALS = True

# Fraction of per-site raw measurements sampled as test points for retrospective
TEST_SAMPLE_FRACTION = 0.20

# History requirement: anchor must have >= this fraction of site's total history
HISTORY_REQUIREMENT_FRACTION = 0.33

# Features to drop before model training.
# Includes columns from the parquet that have negligible importance,
# plus rolling-mean variants that are computed but not useful.
# Many previously-listed derived features (mhw_flag, beuti_squared, etc.)
# are no longer computed at all — see paper_feature_ablation_results.json.
ZERO_IMPORTANCE_FEATURES = [
    # Parquet columns with negligible importance
    'lat', 'lon',
    'modis-par',
    'modis-k490',       # ΔR² = +0.001 (removing improves performance)
    'chla-anom',        # ΔR² = -0.001
    'modis-chla',       # ΔR² = -0.004
]
# Env override: append extra features to drop (comma-separated)
_extra_drop = os.environ.get("DATECT_EXTRA_DROP_FEATURES", "")
if _extra_drop:
    ZERO_IMPORTANCE_FEATURES += [f.strip() for f in _extra_drop.split(",") if f.strip()]

# Minimum test date (early lower bound; per-site history fraction is the real filter)
MIN_TEST_DATE = "2003-01-01"

# Temporal hold-out evaluation mode
# When scripts/eval/eval_paper_metrics.py --temporal-holdout is used, only dates >= TEMPORAL_HOLDOUT_CUTOFF
# are used as test points. Training still uses all data before each anchor date.
# Provides an uncontaminated generalization estimate separate from hyperparameter tuning.
TEMPORAL_HOLDOUT_CUTOFF = os.environ.get("DATECT_TEMPORAL_HOLDOUT_CUTOFF", "2019-01-01")
TEMPORAL_HOLDOUT_FRACTION = float(os.environ.get("DATECT_TEMPORAL_HOLDOUT_FRACTION", "1.0"))

# Parallelization backend (used by joblib in retrospective evaluation)
PARALLEL_BACKEND = os.environ.get("DATECT_PARALLEL_BACKEND", "loky")
