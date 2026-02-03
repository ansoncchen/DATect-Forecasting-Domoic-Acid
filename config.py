# DATect Forecasting Configuration
# Settings for data processing, modeling, and web interface

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

# ML algorithm: "xgboost" (primary) or "linear" (interpretable)
FORECAST_MODEL = "xgboost"

# Forecast Horizon Configuration
# How many weeks ahead to forecast from the data cutoff point
FORECAST_HORIZON_WEEKS = 1
FORECAST_HORIZON_DAYS = FORECAST_HORIZON_WEEKS * 7  # Derived days value for internal calculations

# XGBoost Hyperparameters (best from hyperparam_search; R² ~0.50 at 500 anchors)
# To re-tune: python3 -m forecasting.hyperparam_search --task regression --trials 100 --anchors 500
# ModelFactory optionally loads cache/hyperparams/regression_xgboost_best.json when present
XGB_REGRESSION_PARAMS = {
    "n_estimators": 1403,
    "max_depth": 6,
    "learning_rate": 0.01830309688651229,
    "subsample": 0.6499694717222572,
    "colsample_bytree": 0.8018123960771946,
    "colsample_bylevel": 0.8980752513315378,
    "min_child_weight": 1,
    "reg_alpha": 0.6300118445763987,
    "reg_lambda": 2.6278277494048377,
    "gamma": 0.06208519697643636,
    "tree_method": "hist",
}

# Classification: mirror regression best params (eval_metric added)
XGB_CLASSIFICATION_PARAMS = {
    "n_estimators": 1403,
    "max_depth": 6,
    "learning_rate": 0.01830309688651229,
    "subsample": 0.6499694717222572,
    "colsample_bytree": 0.8018123960771946,
    "colsample_bylevel": 0.8980752513315378,
    "min_child_weight": 1,
    "reg_alpha": 0.6300118445763987,
    "reg_lambda": 2.6278277494048377,
    "gamma": 0.06208519697643636,
    "tree_method": "hist",
    "eval_metric": "logloss",
}


# Temporal Validation - prevents data leakage (handled by forecast horizon)

# Model Performance
MIN_TRAINING_SAMPLES = 3
RANDOM_SEED = 42

# Retrospective evaluation anchor points (higher = more thorough)
N_RANDOM_ANCHORS = 500
RETROSPECTIVE_N_JOBS = -1

# Bootstrap confidence intervals
N_BOOTSTRAP_ITERATIONS = 20  # Number of bootstrap iterations for confidence intervals

# Lag Feature Configuration (main baseline: disabled)

USE_LAG_FEATURES = False
LAG_FEATURES = [] if not USE_LAG_FEATURES else [1, 2, 3, 52]

PN_LAG_FEATURES = [] if not USE_LAG_FEATURES else [1, 2, 3, 4]
ENVIRONMENTAL_LAG_FEATURES = {} if not USE_LAG_FEATURES else {
    "modis-chla": [1, 2, 3, 4],
    "modis-sst": [1, 2, 3, 4],
    "beuti": [1, 2, 3, 4],
}

# DA Category Configuration

# Risk thresholds for classification: Low (0-5), Moderate (5-20), High (20-40), Extreme (>40 μg/g)
DA_CATEGORY_BINS = [-float("inf"), 5, 20, 40, float("inf")]
DA_CATEGORY_LABELS = [0, 1, 2, 3]

# Spike Detection Configuration
SPIKE_THRESHOLD = 20.0  # DA > 20 μg/g considered a spike event
SPIKE_FALSE_NEGATIVE_WEIGHT = 500.0  # Heavy penalty for missing actual spikes
SPIKE_TRUE_NEGATIVE_WEIGHT = 0.1  # Very low weight for correct non-spike predictions

# Bootstrap subsample fraction for speed optimization
BOOTSTRAP_SUBSAMPLE_FRACTION = 0.75  # Use 75% of data for each bootstrap iteration

# Scientific Methodology Configuration

# Sample weighting strategy for regression models
USE_REGRESSION_SAMPLE_WEIGHTS = True  # False = fair baseline comparison, True = handle imbalance

# Confidence interval percentiles for bootstrap predictions
CONFIDENCE_PERCENTILES = [5, 50, 95]  # 5th percentile, median, 95th percentile

# Data Quality Configuration

# Feature engineering toggles (main baseline)
USE_ROLLING_FEATURES = False
USE_ENHANCED_TEMPORAL_FEATURES = True
USE_SITE_ENCODING = False
USE_LOG_TARGET_TRANSFORM = False

# Polynomial trend analysis minimum periods
MIN_TREND_PERIODS = 2  # Minimum data points required for trend calculation
