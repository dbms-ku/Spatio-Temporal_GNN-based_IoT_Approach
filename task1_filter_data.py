import pandas as pd
import json
import os
import matplotlib.pyplot as plt

def load_list(json_file):
    """
    Load list from a JSON file.
    """
    with open(json_file, 'r') as f:
        data_list = json.load(f)
    print(f"Loaded {len(data_list)} items from {json_file}")
    return data_list



def plot_and_save_parameter_chart(df, station_id, parameter, start_date, end_date,
                                   datetime_col='date', station_col='stationID',
                                   parameter_col='parameter', value_col='value',
                                   output_dir='charts'):
    """
    Filters data and generates a time series plot, saving it to file.

    Args:
        df (pd.DataFrame): Input dataframe.
        station_id (str): Target stationID.
        parameter (str): Target parameter.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
        datetime_col (str): Column name for datetime.
        station_col (str): Column name for station ID.
        parameter_col (str): Column name for parameter.
        value_col (str): Column name for values.
        output_dir (str): Directory to save the plot.
    """

    # Ensure datetime is correct type
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    # Filter data
    filtered_df = df[
        (df[datetime_col] >= start_date) &
        (df[datetime_col] <= end_date) &
        (df[station_col] == station_id) &
        (df[parameter_col] == parameter)
    ].sort_values(by=datetime_col)

    if filtered_df.empty:
        print(f"No data found for {station_id}, {parameter} between {start_date} and {end_date}")
        return

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(filtered_df[datetime_col], filtered_df[value_col], marker='o', linestyle='-')
    plt.title(f'{parameter} at {station_id}\n{start_date} to {end_date}')
    plt.xlabel('Date')
    plt.ylabel(parameter)
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{station_id}_{parameter}_{start_date}_to_{end_date}.png".replace(':', '-')
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()

    print(f"Chart saved to {filepath}")




def filter_n_plot_n_save_data(input_csv, station_json, parameter_json, start_date, end_date,
                         output_dir='filtered_data',
                         datetime_col='date', station_col='stationID', parameter_col='parameter'):
    """
    Filters water quality data by station list (from JSON), parameter(s), and date range, saves per parameter.

    Args:
        input_csv (str): Path to input CSV file.
        station_json (str): Path to JSON file containing station list.
        parameter_list (list): List of parameters to filter.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
        output_dir (str): Directory where filtered CSVs are saved.
        datetime_col (str): Datetime column name.
        station_col (str): Station ID column name.
        parameter_col (str): Parameter column name.
    """
    # Load station list
    station_list = load_list(station_json)
    parameter_list = (load_list(parameter_json)).keys()

    # Load CSV
    df = pd.read_csv(input_csv)
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    # Filter by date and station list
    base_filtered_df = df[
        (df[datetime_col] >= start_date) &
        (df[datetime_col] <= end_date) &
        (df[station_col].isin(station_list))
    ]

    print(f"Base filtered data shape: {base_filtered_df.shape}")

    os.makedirs(output_dir, exist_ok=True)

    for param in parameter_list:
        param_filtered_df = base_filtered_df[base_filtered_df[parameter_col] == param]
        output_filename = f"{param}_{start_date}_to_{end_date}.csv".replace(":", "-")
        output_path = os.path.join(output_dir, output_filename)

        param_filtered_df.to_csv(output_path, index=False)
        print(f"Saved {len(param_filtered_df)} rows for {param} → {output_path}")

        for station_id in station_list:
            df = param_filtered_df[param_filtered_df[station_col] == station_id]
            if not df.empty:  
                # Plot and save chart for each station and parameter   
                plot_and_save_parameter_chart(df, station_id, param, start_date, end_date,
                                   datetime_col='date', station_col='stationID',
                                   parameter_col='parameter', value_col='value',
                                   output_dir='charts')


# ========== USAGE EXAMPLE ==========
if __name__ == "__main__":
    input_csv = "data/water_quality_data.csv"
    station_json = "json/stations.json"  # contains e.g., ["BGD00001", "BGD00002"]
    parameter_json = "json/parameters.json"  # contains e.g., ["pH", "DO", "Turbidity"]
    start_date = "2012-03-01"
    end_date = "2022-12-31"

    filter_n_plot_n_save_data(input_csv, station_json, parameter_json, start_date, end_date)
