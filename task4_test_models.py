import os
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import ReLU, Dropout, Linear, MSELoss
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# === Configuration ===
WINDOW_SIZE = 4
DATETIME_FEATURES = ["year", "week"]
BASE_FEATURES = [f"lag_{i+1}" for i in range(WINDOW_SIZE)]
FEATURES = BASE_FEATURES + DATETIME_FEATURES

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, fc_hidden=32):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.relu = ReLU()
        self.dropout = Dropout(0.2)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc1 = Linear(hidden_channels, fc_hidden)
        self.fc_out = Linear(fc_hidden, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc_out(x)
        return x

# === Evaluation Metrics ===
def evaluate_predictions(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }

# === Visualization ===
def plot_actual_vs_predicted(df, param, output_dir):
    """Plot actual vs predicted values with proper date x-axis"""
    os.makedirs(os.path.join(output_dir, "station_plots"), exist_ok=True)
    stations = df['stationID'].unique()
    
    
    
    
    quality_standards =  {
        # Define water quality standards for specific parameters
        'pH': {'min': 5.8, 'max': 8.6, 'description': 'Drinking Water Standard'},
        'MN-DIS': {'max': 0.05, 'description': 'Mn-Dis (<0.05 mg/L)'},
        'CL-DIS': {'max': 200, 'description': 'Cl-Dis (<200 mg/L)'},
        'CA-DIS': {'max': 300, 'description': 'Ca-Dis (<300 mg/L)'},
        
    }
    
     # Convert parameter name to uppercase for comparison
    param_upper = param.upper()
    for station in stations:
        sub_df = df[df['stationID'] == station]
        if len(sub_df) < 3:  # Skip if not enough data
            continue
            
        # Extract timestamps - they should be the latest timestamp from the window
        timestamps = []
        for ts_array in sub_df['timestamp']:
            if isinstance(ts_array, np.ndarray) and len(ts_array) > 0:
                timestamps.append(ts_array[-1])  # Use the latest timestamp in the window
            else:
                timestamps.append(pd.NaT)  # Not a valid timestamp
                
        if all(pd.isna(timestamps)):  # Skip if all timestamps are invalid
            print(f"Warning: No valid timestamps for station {station}, parameter {param}")
            continue
            
        # Create a copy with timestamps
        plot_df = sub_df.copy()
        plot_df['plot_timestamp'] = timestamps
        
        # Sort by timestamp
        plot_df = plot_df.sort_values('plot_timestamp')
        
        plt.figure(figsize=(12, 6))
        plt.plot(plot_df['plot_timestamp'], plot_df['actual'], 'b.-', label='Actual', alpha=0.7, markersize=5)
        plt.plot(plot_df['plot_timestamp'], plot_df['predicted'], 'r.-', label='Predicted', alpha=0.7, markersize=5)

        # Format date x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45, fontsize=15)
        
        
        # Check both original case and uppercase for parameter matching
        matching_param = None
        if param in quality_standards:
            matching_param = param
        elif param_upper in quality_standards:
            matching_param = param_upper
        
         # Add water quality standard ranges for specific parameters
        if matching_param:
            std = quality_standards[matching_param]
            y_min, y_max = plt.ylim()
            
            # Add min line if defined
            if 'min' in std:
                plt.axhline(y=std['min'], color='green', linestyle='--', alpha=0.7, 
                          label=f"Min {std['description']}")
                
            # Add max line if defined
            if 'max' in std:
                plt.axhline(y=std['max'], color='red', linestyle='--', alpha=0.7,
                          label=f"Max {std['description']}")
                
            # Set y-axis limits to show a bit more context
            range_padding = 0.2 * (std.get('max', y_max) - std.get('min', y_min))
            plt.ylim(
                min(y_min, std.get('min', y_min) - range_padding),
                max(y_max, std.get('max', y_max) + range_padding)
            )
                
            # Add colored background for safe/unsafe regions
            if 'min' in std and 'max' in std:
                plt.axhspan(std['min'], std['max'], alpha=0.1, color='green', 
                          label='Safe Range')
            elif 'min' in std:
                plt.axhspan(std['min'], y_max + range_padding, alpha=0.1, color='green',
                          label='Safe Range (Above Min)')
            elif 'max' in std:
                plt.axhspan(y_min - range_padding, std['max'], alpha=0.1, color='green',
                          label='Safe Range (Below Max)')
        
        plt.title(f"{param} - Station {station}", fontsize=20)
        plt.xlabel("Date", fontsize=20)
        plt.ylabel(f"{param} value", fontsize=20)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "station_plots", f"{param}_station_{station}_timeseries.png"))
        plt.close()
        
        # Create scatter plot of actual vs predicted
        plt.figure(figsize=(8, 8))
        plt.scatter(plot_df['actual'], plot_df['predicted'], alpha=0.7)
        
        # Add diagonal line (perfect predictions)
        min_val = min(plot_df['actual'].min(), plot_df['predicted'].min())
        max_val = max(plot_df['actual'].max(), plot_df['predicted'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Add water quality standard ranges for specific parameters in scatter plot too
        if param in quality_standards:
            std = quality_standards[param]
            
            # Draw vertical and horizontal lines for the standards on scatter plot
            if 'min' in std:
                plt.axvline(x=std['min'], color='green', linestyle='--', alpha=0.7)
                plt.axhline(y=std['min'], color='green', linestyle='--', alpha=0.7)
                
            if 'max' in std:
                plt.axvline(x=std['max'], color='red', linestyle='--', alpha=0.7)
                plt.axhline(y=std['max'], color='red', linestyle='--', alpha=0.7)
                
           # Add shaded regions for safe/unsafe areas
            if 'min' in std and 'max' in std:
                plt.axvspan(std['min'], std['max'], alpha=0.05, color='green')
                plt.axhspan(std['min'], std['max'], alpha=0.05, color='green')
                
                # Add annotations
                plt.text(
                    (std['min'] + std['max'])/2, 
                    min_val, 
                    'Safe Range', 
                    ha='center', 
                    va='bottom',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
                )
            elif 'min' in std:
                plt.axvspan(std['min'], max_val, alpha=0.05, color='green')
                plt.axhspan(std['min'], max_val, alpha=0.05, color='green')
                
                # Add annotations
                plt.text(
                    (std['min'] + max_val)/2, 
                    min_val, 
                    'Safe Range (Above Min)', 
                    ha='center', 
                    va='bottom',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
                )
            elif 'max' in std:
                plt.axvspan(min_val, std['max'], alpha=0.05, color='green')
                plt.axhspan(min_val, std['max'], alpha=0.05, color='green')
                
                # Add annotations
                plt.text(
                    (min_val + std['max'])/2, 
                    min_val, 
                    'Safe Range (Below Max)', 
                    ha='center', 
                    va='bottom',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
                )
        
        
        
        plt.title(f"{param} - Station {station} - Scatter Plot")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "station_plots", f"{param}_station_{station}_scatter.png"))
        plt.close()

def plot_parameter_summary(df, param, output_dir):
    """Create summary plots for a parameter across all stations"""
    os.makedirs(os.path.join(output_dir, "parameter_summaries"), exist_ok=True)
    
    
    # Define water quality standards for specific parameters
    quality_standards = {
        'pH': {'min': 5.8, 'max': 8.6, 'description': 'Drinking Water Standard'},
        'MN-DIS': {'max': 0.05, 'description': 'Drinking Water Standard (<0.05 mg/L)'},
        'CL-DIS': {'max': 200, 'description': 'Drinking Water Standard (<200 mg/L)'},
        'CA-DIS': {'max': 300, 'description': 'Drinking Water Standard (<300 mg/L)'},
        # Add other parameters as needed
    }
    
      # Convert parameter name to uppercase for comparison
    param_upper = param.upper()
    
    # Extract the valid timestamps
    timestamps = []
    for ts_array in df['timestamp']:
        if isinstance(ts_array, np.ndarray) and len(ts_array) > 0:
            timestamps.append(ts_array[-1])
        else:
            timestamps.append(pd.NaT)
    
    df = df.copy()
    df['plot_timestamp'] = timestamps
    df = df.dropna(subset=['plot_timestamp'])
    
    if len(df) == 0:
        print(f"Warning: No valid data for parameter {param} summary")
        return
    
    # Sort by time
    df = df.sort_values('plot_timestamp')
    
    # 1. Error over time plot
    plt.figure(figsize=(14, 6))
    df['abs_error'] = np.abs(df['actual'] - df['predicted'])
    plt.scatter(df['plot_timestamp'], df['abs_error'], alpha=0.5)
    
    # Add trend line
    z = np.polyfit(np.arange(len(df)), df['abs_error'], 1)
    p = np.poly1d(z)
    plt.plot(df['plot_timestamp'], p(np.arange(len(df))), "r--", alpha=0.8)
    
    plt.title(f"{param} - Absolute Error Over Time")
    plt.xlabel("Date")
    plt.ylabel("Absolute Error")
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "parameter_summaries", f"{param}_error_over_time.png"))
    plt.close()
    
    # 2. Combined scatter plot for all stations
    plt.figure(figsize=(10, 10))
    for station in df['stationID'].unique():
        station_df = df[df['stationID'] == station]
        plt.scatter(station_df['actual'], station_df['predicted'], label=f'Station {station}', alpha=0.7)
        
    # Add diagonal line
    all_min = min(df['actual'].min(), df['predicted'].min())
    all_max = max(df['actual'].max(), df['predicted'].max())
    plt.plot([all_min, all_max], [all_min, all_max], 'r--')
    
    # Check both original case and uppercase for parameter matching
    matching_param = None
    if param in quality_standards:
        matching_param = param
    elif param_upper in quality_standards:
        matching_param = param_upper
    
    # Add water quality standards if available for this parameter
    if matching_param:
        std = quality_standards[matching_param]

        if 'min' in std:
            plt.axvline(x=std['min'], color='green', linestyle='--', alpha=0.7, 
                      label=f"Min {std['description']}")
            plt.axhline(y=std['min'], color='green', linestyle='--', alpha=0.7)
            
        if 'max' in std:
            plt.axvline(x=std['max'], color='red', linestyle='--', alpha=0.7,
                      label=f"Max {std['description']}")
            plt.axhline(y=std['max'], color='red', linestyle='--', alpha=0.7)
            
      # Add shaded regions for safe/unsafe areas
        if 'min' in std and 'max' in std:
            plt.axvspan(std['min'], std['max'], alpha=0.05, color='green')
            plt.axhspan(std['min'], std['max'], alpha=0.05, color='green')
        elif 'min' in std:
            plt.axvspan(std['min'], all_max, alpha=0.05, color='green')
            plt.axhspan(std['min'], all_max, alpha=0.05, color='green')
        elif 'max' in std:
            plt.axvspan(all_min, std['max'], alpha=0.05, color='green')
            plt.axhspan(all_min, std['max'], alpha=0.05, color='green')
    
    
    
    plt.title(f"{param} - All Stations")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "parameter_summaries", f"{param}_all_stations_scatter.png"))
    plt.close()

    # Create distribution plot of actual vs predicted values
    if param in quality_standards:
        plt.figure(figsize=(12, 6))
        
        # Setup for split violin plot
        def split_violin(x, y, **kwargs):
            d = {'actual': x, 'predicted': y}
            df_violin = pd.DataFrame(data=d)
            df_violin = pd.melt(df_violin, value_vars=['actual', 'predicted'])
            ax = sns.violinplot(x='variable', y='value', data=df_violin, **kwargs)
            return ax
            
        ax = split_violin(df['actual'], df['predicted'], palette="Set2")
        
        # Add water quality standard ranges
        std = quality_standards[param]
        if 'min' in std:
            plt.axhline(y=std['min'], color='green', linestyle='--', alpha=0.7, 
                       label=f"Min {std['description']}")
            
        if 'max' in std:
            plt.axhline(y=std['max'], color='red', linestyle='--', alpha=0.7,
                       label=f"Max {std['description']}")
            
        # Add shaded region for safe range
        if 'min' in std and 'max' in std:
            plt.axhspan(std['min'], std['max'], alpha=0.1, color='green', label='Safe Range')
        elif 'min' in std:
            y_min, y_max = plt.ylim()
            plt.axhspan(std['min'], y_max, alpha=0.1, color='green', label='Safe Range (Above Min)')
        elif 'max' in std:
            y_min, y_max = plt.ylim()
            plt.axhspan(y_min, std['max'], alpha=0.1, color='green', label='Safe Range (Below Max)')
        
        plt.title(f"{param} - Distribution of Actual vs Predicted Values")
        plt.ylabel(param)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "parameter_summaries", f"{param}_distribution.png"))
        plt.close()

def dot_plot_summary(df_all, output_dir):
    """Create dot plot of normalized errors by parameter"""
    plt.figure(figsize=(12, 8))
    sns.pointplot(data=df_all, x="parameter", y="normalized_error", join=False, capsize=0.2)
    plt.xticks(rotation=45)
    plt.title("Normalized Error per Parameter")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dot_plot_normalized_errors.png"))
    plt.close()

def plot_metrics_summary(metrics_df, output_dir):
    """Create summary visualizations of model metrics"""
    # Sort by R2 score
    sorted_df = metrics_df.sort_values('R2', ascending=False)
    
    # Plot R2 scores
    plt.figure(figsize=(12, 6))
    bars = plt.bar(sorted_df['parameter'], sorted_df['R2'], color='skyblue')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', va='bottom', rotation=0)
                
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)  # Line at R2=0
    plt.axhline(y=0.5, color='g', linestyle='--', alpha=0.3)  # Line at R2=0.5
    plt.axhline(y=0.8, color='g', linestyle='-', alpha=0.3)  # Line at R2=0.8
    
    plt.xlabel('Parameter')
    plt.ylabel('R² Score')
    plt.title('Model Performance by Parameter (R² Score)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_parameters_r2_scores.png"), dpi=300)
    plt.close()
    
    # Plot RMSE values
    plt.figure(figsize=(12, 6))
    bars = plt.bar(sorted_df['parameter'], sorted_df['RMSE'], color='salmon')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', va='bottom', rotation=90 if height > 100 else 0)
                
    plt.xlabel('Parameter')
    plt.ylabel('RMSE')
    plt.title('Model Performance by Parameter (RMSE)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_parameters_rmse.png"), dpi=300)
    plt.close()

# === Main Evaluation Pipeline ===
def run_evaluation(data_csv, config_json, model_dir, edge_index_dir, output_dir, start_date, end_date):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_csv)
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

    with open(config_json) as f:
        config = json.load(f)

    # Set device for model computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    metrics_list = []
    all_predictions = []

    for param in df["parameter"].unique():
        if param not in config:
            print(f"Skipping {param}: not in config")
            continue

        print(f"Evaluating {param}")
        df_param = df[df["parameter"] == param].copy()
        split_idx = int(0.8 * len(df_param))
        test_df = df_param.iloc[split_idx:].reset_index(drop=True)

        edge_file = os.path.join(edge_index_dir, f"edge_index_{param}_{start_date}_to_{end_date}.pt")
        if not os.path.exists(edge_file):
            print(f"Skipping {param}: missing edge index file")
            continue
        edge_index = torch.load(edge_file)

        # Feature engineering
        test_df["timestamp"] = pd.to_datetime(test_df["date"] + " " + test_df["time"])
        test_df["year"] = test_df["timestamp"].dt.year.astype(int)
        test_df["week"] = test_df["timestamp"].dt.isocalendar().week.astype(int)

        feature_dfs = []
        timestamps_list = []  # Store timestamps for each window
        
        for station, group in test_df.groupby("stationID"):
            group = group.sort_values("timestamp").reset_index(drop=True)
            if len(group) < WINDOW_SIZE + 1:
                continue
                
            for i in range(WINDOW_SIZE, len(group)):
                window = group.iloc[i - WINDOW_SIZE:i]["value"].values
                year = group.iloc[i]["year"]
                week = group.iloc[i]["week"]
                target = group.iloc[i]["value"]
                timestamps = group.iloc[i - WINDOW_SIZE:i+1]["timestamp"].values  # Include current timestamp
                
                feature_dfs.append({
                    "features": list(window) + [year, week], 
                    "target": target, 
                    "stationID": station
                })
                timestamps_list.append(timestamps)
        
        if not feature_dfs:
            print(f"Skipping {param}: no feature data generated")
            continue

        feat_df = pd.DataFrame(feature_dfs)
        X = np.vstack(feat_df["features"])
        y = feat_df["target"].values.astype(np.float32)

        # Load scaler and model
        try:
            scaler_path = os.path.join(model_dir, f"scaler_{param}.pkl")
            if not os.path.exists(scaler_path):
                print(f"Skipping {param}: missing scaler file")
                continue
                
            scaler = joblib.load(scaler_path)
            X_scaled = scaler.transform(X)
            
            x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

            # Create PyG data object
            data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
            
            # Load model
            model_path = os.path.join(model_dir, f"gnn_{param}.pt")
            if not os.path.exists(model_path):
                print(f"Skipping {param}: missing model file")
                continue
                
            model = GNN(in_channels=len(FEATURES), hidden_channels=config[param]["hidden_channels"])
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            
            # Move data to device
            data = data.to(device)

            # Make predictions
            with torch.no_grad():
                predictions = model(data).cpu().numpy().flatten()

            # Compute metrics
            metrics = evaluate_predictions(y, predictions)
            metrics["parameter"] = param
            metrics["num_samples"] = len(y)
            metrics_list.append(metrics)

            # Create DataFrame with results
            temp_df = pd.DataFrame({
                "stationID": feat_df["stationID"].values,
                "actual": y,
                "predicted": predictions,
                "parameter": param,
                "timestamp": timestamps_list  # Store all timestamps from window
            })
            all_predictions.append(temp_df)

            # Generate plots for this parameter
            plot_actual_vs_predicted(temp_df, param, output_dir)
            plot_parameter_summary(temp_df, param, output_dir)
            
        except Exception as e:
            print(f"Error processing {param}: {str(e)}")
            continue

    # Save metrics
    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.to_csv(os.path.join(output_dir, "evaluation_metrics.csv"), index=False)
        plot_metrics_summary(metrics_df, output_dir)
    else:
        print("No metrics to save - all parameters were skipped")
        return

    # Save all predictions
    if all_predictions:
        all_predictions_df = pd.concat(all_predictions, ignore_index=True)
        all_predictions_df.to_csv(os.path.join(output_dir, "all_predictions.csv"), index=False)

        # Normalized error for dot plot
        all_predictions_df["normalized_error"] = np.abs(all_predictions_df["actual"] - all_predictions_df["predicted"]) / (all_predictions_df["actual"].max() - all_predictions_df["actual"].min() + 1e-8)
        dot_plot_summary(all_predictions_df, output_dir)
    else:
        print("No predictions to save")

if __name__ == "__main__":
    run_evaluation(
        data_csv="/nas.dbms/lia/graph4waterQualityMonitoring/data/water_quality_data.csv",
        config_json="/nas.dbms/lia/graph4waterQualityMonitoring/json/parameters.json",
        model_dir="/nas.dbms/lia/graph4waterQualityMonitoring/models/",
        edge_index_dir="/nas.dbms/lia/graph4waterQualityMonitoring/edge_indices/",
        output_dir="/nas.dbms/lia/graph4waterQualityMonitoring/eval_fix/",
        start_date='2012-03-01',
        end_date='2022-12-31'
    )