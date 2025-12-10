import os
import json
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import ReLU, Dropout, Linear, MSELoss
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
import joblib

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

def add_datetime_features(df):
    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"])
    df["year"] = df["timestamp"].dt.year.astype(int)
    df["week"] = df["timestamp"].dt.isocalendar().week.astype(int)
    return df

def generate_windowed_features(df, station_col="stationID", value_col="value"):
    feature_dfs = []
    for station, group in df.groupby(station_col):
        group = group.sort_values("timestamp").reset_index(drop=True)
        if len(group) < WINDOW_SIZE + 1:
            continue
        for i in range(WINDOW_SIZE, len(group)):
            window = group.loc[i - WINDOW_SIZE:i - 1, value_col].values
            year = group.loc[i, "year"]
            week = group.loc[i, "week"]
            target = group.loc[i, value_col]
            temp = list(window) + [year, week]
            feature_dfs.append({"features": temp, "target": target, "stationID": group.loc[i, station_col]})
    df_feat = pd.DataFrame(feature_dfs)
    return df_feat

def prepare_data(df_feat, edge_index, node_mapping, scaler=None):
    df_feat = df_feat[df_feat["stationID"].isin(node_mapping)]
    df_feat["node_idx"] = df_feat["stationID"].map(node_mapping)
    X = np.vstack(df_feat["features"])
    y = df_feat["target"].values.astype(np.float32)
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
    return data, scaler

def train_model(model, data, epochs=500, lr=1e-3):
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = MSELoss()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        if torch.isnan(loss):
            print(f"NaN detected in loss at epoch {epoch}")
            break
        loss.backward()
        optimizer.step()
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")
    return model

def run_training(data_csv, config_json, edge_index_dir, output_dir, start_date, end_date):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_csv)
    df = add_datetime_features(df)
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    with open(config_json) as f:
        config = json.load(f)
    node_mapping = torch.load("data/node_mapping_gh_t1.pt")
    for param in df["parameter"].unique():
        if param not in config:
            continue
        df_param = df[df["parameter"] == param]
        edge_index = torch.load(os.path.join(edge_index_dir, f"edge_index_{param}_{start_date}_to_{end_date}.pt"))
        df_feat = generate_windowed_features(df_param)
        if df_feat.empty:
            continue
        data, scaler = prepare_data(df_feat, edge_index, node_mapping)
        model = GNN(in_channels=len(FEATURES), hidden_channels=config[param]["hidden_channels"])
        model = train_model(model, data, epochs=config[param]["epochs"], lr=config[param]["lr"])
        torch.save(model.state_dict(), os.path.join(output_dir, f"gnn_{param}.pt"))
        joblib.dump(scaler, os.path.join(output_dir, f"scaler_{param}.pkl"))
        print(f"Finished training for {param}")

if __name__ == "__main__":
    run_training(
        data_csv="/nas.dbms/lia/graph4waterQualityMonitoring/data/water_quality_data.csv",
        config_json="/nas.dbms/lia/graph4waterQualityMonitoring/json/param_ph.json",
        edge_index_dir="/nas.dbms/lia/graph4waterQualityMonitoring/edge_indices/",
        output_dir="/nas.dbms/lia/graph4waterQualityMonitoring/models/param_ph",
        start_date='2012-03-01',
        end_date='2022-12-31'
    )
