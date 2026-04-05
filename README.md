# 🚗 Chicago Ride Demand Analysis & Prediction

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

> A data-driven analysis and prediction for **taxi and ride trip demand in Chicago**, leveraging the City of Chicago's open taxi trips dataset to analyze demand patterns and forecast trip volumes.

## 🔍 Overview

This project analyzes and predicts **taxi and ride trip demand** across **Chicago**, using historical trip-level records from the City of Chicago's open data portal. The project focuses on understanding **when** and **where** demand is highest, building machine learning models to forecast trip volumes, and visualizing demand patterns across time dimensions.

The project covers:

1. **Exploratory Data Analysis (EDA)** of Chicago taxi trip records
2. **Temporal demand pattern analysis** — hourly, daily, and monthly trends
3. **Machine learning models** to predict trip demand volume
4. **Visualization** of demand patterns through plots and charts

---

## 🗺️ About the Dataset

The analysis uses the **City of Chicago Taxi Trips Dataset**, which contains millions of anonymized taxi trip records made available through the City of Chicago's Open Data Portal.

| Feature | Description |
|--------|-------------|
| **Trip Start Timestamp** | Date and time the trip began |
| **Trip End Timestamp** | Date and time the trip ended |
| **Trip Seconds** | Duration of the trip in seconds |
| **Trip Miles** | Distance traveled in miles |
| **Pickup Community Area** | Chicago community area where the trip started |
| **Dropoff Community Area** | Chicago community area where the trip ended |
| **Fare** | Base fare amount charged |
| **Tips** | Tip amount provided by passenger |
| **Tolls** | Toll charges incurred |
| **Extras** | Additional charges |
| **Trip Total** | Total amount charged for the trip |
| **Payment Type** | Method of payment (cash, credit card, etc.) |
| **Company** | Taxi company operating the vehicle |

---

## 🎯 Objectives

- **Analyze** temporal demand patterns — by hour, day of week, and month
- **Identify** peak and off-peak demand windows across the dataset
- **Engineer** time-based features to improve model performance
- **Build and compare** multiple machine learning models for demand forecasting
- **Visualize** demand trends through clear and interpretable plots

---

## 🔑 Key Factors Analyzed

### 🕐 Temporal Patterns

- **Hour of Day** — Identifying peak hours vs. low-demand windows
- **Day of Week** — Weekday vs. weekend demand differences
- **Month** — Seasonal fluctuations in trip volume across the year
- **Year-over-Year Trends** — Changes in taxi demand over time

### 🚕 Trip Characteristics

- **Trip Duration Distribution** — Spread and outliers in trip seconds/minutes
- **Trip Distance Distribution** — Short vs. long trip frequency analysis
- **Fare Analysis** — Fare distribution, average fares, and relationship to distance
- **Payment Type Breakdown** — Cash vs. card usage patterns
- **Company-Level Volume** — Trip volume comparison across taxi companies

### 📍 Geographic Patterns

- **Pickup Community Area Frequency** — Which Chicago community areas generate the most trips
- **Dropoff Community Area Frequency** — Most common trip destinations
- **Origin–Destination Pairs** — Most frequently traveled routes in the city

---

## 📊 Data Sources

| Source | Data Collected |
|--------|----------------|
| [City of Chicago Open Data Portal — Taxi Trips](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew) | Taxi trip records including timestamps, community areas, fares, distance, and payment type |
| [Chicago Transit Authority (CTA) — Bus & Rail Ridership](https://www.transitchicago.com/data/) | Daily CTA boarding totals used to correlate public transit ridership with taxi demand patterns |
| [OpenWeatherMap API — Historical Weather Data](https://openweathermap.org/) | Historical weather data including temperature, precipitation, and weather conditions used to analyze the impact of weather on ride-hailing demand |
