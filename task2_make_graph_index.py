import pandas as pd
import torch
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

def load_parameter_config(json_file):
    with open(json_file, 'r') as f:
        config = json.load(f)
    print(f"Loaded {len(config)} parameters from {json_file}")
    return config

def create_knn_edge_index(coords, k, metric='euclidean'):
    nbrs = NearestNeighbors(n_neighbors=k, metric=metric).fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    edge_index = []
    for i, neighbors in enumerate(indices):
        for j in neighbors:
            if i != j:
                edge_index.append([i, j])
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

def create_radius_edge_index(coords, radius, metric='euclidean'):
    adj = radius_neighbors_graph(coords, radius=radius, metric=metric, mode='connectivity', include_self=False)
    coo = adj.tocoo()
    return torch.tensor([coo.row, coo.col], dtype=torch.long)

def create_mst_edge_index(coords, metric='euclidean'):
    distances = squareform(pdist(coords, metric=metric))
    mst = minimum_spanning_tree(distances).tocoo()
    edge_index = torch.tensor([mst.row, mst.col], dtype=torch.long)
    print(f"MST created with {edge_index.shape[1]} edges (should be N-1)")
    return edge_index

def combine_mst_with_method(coords, method, method_data, metric='euclidean'):
    """
    Combines MST with additional KNN or Radius edges.
    """
    edge_index_mst = create_mst_edge_index(coords, metric=metric)

    if method == 'knn':
        k = method_data.get('k', 5)
        edge_index_extra = create_knn_edge_index(coords, k=k, metric=metric)
    elif method == 'radius':
        radius = method_data.get('radius', 0.5)
        edge_index_extra = create_radius_edge_index(coords, radius=radius, metric=metric)
    else:
        print(f"Unknown method '{method}', only MST used.")
        return edge_index_mst

    # Combine edges and remove duplicates
    combined_edges = torch.cat([edge_index_mst, edge_index_extra], dim=1)
    # Convert to numpy, sort, deduplicate
    combined_edges_np = combined_edges.numpy()
    combined_edges_np = np.unique(combined_edges_np.T, axis=0).T
    edge_index_combined = torch.tensor(combined_edges_np, dtype=torch.long)

    print(f"Combined MST + {method.upper()} edges: {edge_index_combined.shape[1]} total edges")
    return edge_index_combined

def plot_and_save_graph(coords, edge_index, save_path):
    plt.figure(figsize=(8, 6))
    x, y = coords[:, 1], coords[:, 0]
    plt.scatter(x, y, c='blue', label='Stations')
    for src, dst in edge_index.t().tolist():
        plt.plot([x[src], x[dst]], [y[src], y[dst]], color='gray', alpha=0.5)
    plt.title("Graph Structure (MST + Neighbors)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Graph plot saved to {save_path}")

def generate_and_save_edge_indices(df, station_coords_df, config, start_date, end_date,
                                   datetime_col='date', station_col='stationID',
                                   lat_col='Latitude', lon_col='Longitude',
                                   output_dir='edge_indices'):
    os.makedirs(output_dir, exist_ok=True)
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    

    for param, param_conf in config.items():
        
        filtered_df = df[(df[datetime_col] >= start_date) & (df[datetime_col] <= end_date)]
        filtered_df = filtered_df[filtered_df['parameter'] == param]
        valid_stations = filtered_df[station_col].unique()
        station_coords = station_coords_df[station_coords_df[station_col].isin(valid_stations)].reset_index(drop=True)
        coords = station_coords[[lat_col, lon_col]].to_numpy()

        print(f"Using {len(station_coords)} stations with coordinates.")
    
    
        method = param_conf['edge_index']['method']
        method_data = param_conf['edge_index'].get('data', {})
        metric = method_data.get('metric', 'euclidean')

        print(f"\nGenerating MST + {method.upper()} edges for {param}")
        edge_index = combine_mst_with_method(coords, method, method_data, metric=metric)

        filename_base = f"edge_index_{param}_{start_date}_to_{end_date}".replace(":", "-")
        pt_path = os.path.join(output_dir, filename_base + ".pt")
        png_path = os.path.join(output_dir, filename_base + ".png")

        torch.save(edge_index, pt_path)
        print(f"Saved edge_index to {pt_path}")
        plot_and_save_graph(coords, edge_index, png_path)

# ====== USAGE EXAMPLE ======
if __name__ == "__main__":
    input_csv = "data/water_quality_data.csv"
    coord_csv = "data/main_data_with_coordinates.csv"
    config_json = "json/parameters.json"
    start_date = '2012-03-01'
    end_date = '2022-12-31'

    df = pd.read_csv(input_csv)
    station_coords_df = pd.read_csv(coord_csv)
    config = load_parameter_config(config_json)

    generate_and_save_edge_indices(df, station_coords_df, config, start_date, end_date)
