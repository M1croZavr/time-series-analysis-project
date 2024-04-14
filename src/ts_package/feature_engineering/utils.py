import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_float(text):
    numerical_part = ""
    for character in text:
        if character in "0123456789.,":
            numerical_part += character
    return float(numerical_part)

def salaries(currentDate):
    return  (currentDate.day % (15 + 1)) / (15)

def getData(df, currentDate, currentType, currentEvent):
    return df[df.event.str.contains(currentEvent)][df.year==currentDate.year][df.month==currentDate.month].iloc[0][currentType]

def create_lag_features(df, max_lag, target_column):
    for i in range(1, max_lag):
        df[f'{target_column} lag {i}'] = df[target_column].shift(i)
    return df

def generate_data_frame():
    file_handle = open('data/financial_data', "r")
    lines = file_handle.readlines()
    event_data = pd.DataFrame(columns=['event', "date", "year", 'month', "actual", 'forecastCorrection'])
    active_date = ''
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    month_codes = [
         'APR', 'AUG', 'DEC', 'FEB', 'JAN', 'JUL', 'JUN', 'MAR', 'MAY', 'NOV', 'OCT', 'SEP'
    ]

    quarters = ["Q1", "Q2", "Q3", "Q4"]

    exclusion_terms = [
        "Interest Rate Decision", "Current Account", "EU", "Prel", "European Council",
        "Monetary Policy Report", "Bond Auction", "Bill Auction", "EcoFin", "Conference",
        "Commission", "Full Year GDP Growth", "Summit", "Election", "Meeting", "Debate",
        "Flash", "Car", "Vehicle", "PPI YoY", "GDP YoY", "Inflation Rate YoY",
        "M2 Money Supply YoY", "Markit Composite PMI"
    ]

    record_index = 0

    for line in lines:
        skip_line = False

        if "timeFlag" in line or line == "RU\n":
            continue

        for day in weekdays:
            if day in line:
                active_date = line
                skip_line = True
                break
        if skip_line:
            continue

        for term in exclusion_terms:
            if term in line:
                skip_line = True
                break

        if skip_line:
            continue

        for month in month_codes:
            if month in line:
                skip_line = True
                parsed_date = pd.to_datetime(active_date)
                data_parts = line[line.index(month)+3:].strip().split("\t")
                event_name = line[:line.index(month)].strip()
                actual_value = parse_float(data_parts[0])
                forecast_value = parse_float(data_parts[-1])
                corrected_year = parsed_date.year
                if parsed_date.month < 4 and month in ["DEC", "NOV", "OCT", "SEP"]:
                    corrected_year -= 1

                forecast_delta = actual_value - forecast_value

                event_data.loc[record_index] = [event_name, parsed_date, corrected_year, month, actual_value, forecast_delta]
                record_index += 1
                break

        for quarter in quarters:
            if quarter in line:
                skip_line = True
                break

        if skip_line:
            continue

    return event_data


def make_time_based_features(df: pd.DataFrame):
    df = df.copy()
    df['weekday'] = df.index.weekday
    df['quarter'] = df.index.quarter
    df['is_weekend'] = df.weekday.isin([5, 6]).astype(int)
    df['Праздники РФ'] = df['Праздники РФ'].astype(int)
    return df
