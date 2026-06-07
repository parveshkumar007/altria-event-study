# Altria Event Study — Media & Regulatory Impact on Stock Price

**ECON 435 — Financial Economics and Quantitative Methods**
University of Northern British Columbia | 2024
Authors: Parvesh Kumar & Dean Reid

## Overview
This project examines how media articles and regulatory disclosures
impacted Altria Group's (MO) stock price using event study methodology.
We analyze three landmark events in tobacco industry history and measure
cumulative abnormal returns (CAR) around each event date.

## Events Analyzed
| Event | Date | Type |
|---|---|---|
| Surgeon General's Report on Smoking & Health | Jan 13, 1964 | Media / Health |
| EPA designates passive smoking a carcinogen | Jan 7, 1993 | Regulatory |
| Judge Kessler's Big Tobacco ruling | Aug 17, 2006 | Legal / Regulatory |

## Methodology
- Market model (OLS regression): R_stock = a + B x R_market
- Estimation window: -153 to -6 days before event
- Event window: -5 to +15 days around event
- Data source: Yahoo Finance (via yfinance)
- Statistical significance tested via t-test (p < 0.05)

## Key Finding
Both media coverage and regulatory disclosures produced statistically
significant abnormal returns, with the 1964 and 2006 events generating
notably positive CARs, suggesting the market viewed these negative
industry events as less damaging than expected.

## How to Run
Install dependencies:
    pip3 install pandas numpy yfinance matplotlib scipy

Run the script:
    python3 altria_event_study_v2.py
