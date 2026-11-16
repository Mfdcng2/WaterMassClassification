import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
import plotly.graph_objs as go
from collections import defaultdict
import plotly.express as px
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Geod
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import gsw
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
import tempfile, os
from joblib import Parallel, delayed, cpu_count
import tqdm
from sklearn.mixture import GaussianMixture
import random

# Sample bins of bins (geographical and depth bins) in projection space
def sample_bins_of_bins_proj(df, n_x_bins=100, n_y_bins=100, n_depth_bins=5, projection='npstere', min_flag=True):
    """
    Separates the dataset by creating bins in projection space (default: Polar Stereographic Projection) and depth, and randomly samples from each (x,y,depth) bin.

    Args:
        - df (pd.DataFrame): The dataset to sample.
        - n_x_bins (int): Number of bins in the x axis. Defaults to 100.
        - n_y_bins (int): Number of bins in the y axis. Defaults to 100.
        - n_depth_bins (int): Number of bins in the depth axis. Defaults to 5.
        - projection (str): Basemap projection string. Defaults to 'npstere' (Polar Stereographic Projection).
        - min_flag (bool): If True, sample the minimum number of points from each bin. If False, sample the average number of points from each bin. Defaults to True.
    
    Returns:
        - df_sampled (pd.DataFrame): The sampled dataset in pandas dataframe.
    """
    # Define minimum latitude for projection
    lat_min = df['Latitude_[deg_N]'].min()

    # Convert lat/lon → x/y (projection coordinates in meters)
    lon_vals = df['Longitude_[deg_E]'].values
    lat_vals = df['Latitude_[deg_N]'].values
    x_vals, y_vals = Basemap(projection=projection, boundinglat=lat_min, lon_0=0, resolution='l')(lon_vals, lat_vals)

    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()

    # xy bins in projection space
    x_bins = np.linspace(x_min, x_max, n_x_bins + 1)
    y_bins = np.linspace(y_min, y_max, n_y_bins + 1)

    # depth bins in original depth space
    depth_bins = np.linspace(df['Depth_[m]'].min(), df['Depth_[m]'].max(), n_depth_bins + 1)

    # Assign each point to a bin
    df['x_bin'] = pd.cut(x_vals, bins=x_bins, labels=False, include_lowest=True)
    df['y_bin'] = pd.cut(y_vals, bins=y_bins, labels=False, include_lowest=True)  
    df['depth_bin'] = pd.cut(df['Depth_[m]'], bins=depth_bins, labels=False, include_lowest=True)
    
    # Find bin with least data points and sample that many points from each bin
    if min_flag:
        min_count = df.groupby(['x_bin', 'y_bin', 'depth_bin']).size().min()
        df_sampled = df.groupby(['x_bin', 'y_bin', 'depth_bin']).apply(lambda x: x.sample(min_count, random_state=22)).reset_index(drop=True)
    else:
        avg_count = df.groupby(['x_bin', 'y_bin', 'depth_bin']).size().mean().round().astype(int)
        df_sampled = (df.groupby(['x_bin', 'y_bin', 'depth_bin'], group_keys=False).apply(lambda g: g.sample(n=min(len(g), avg_count), random_state=22)).reset_index(drop=True))
    return df_sampled

def profileNo_vs_xy_bins(
    df: pd.DataFrame,
    n_x_bins: int = 100,
    n_y_bins: int = 100,
    show_threshold: bool = False,
    threshold_quantile: float = 0.95,
    projection: str = 'npstere',
    showfig: bool = True
) -> None:
    """
    Plots the number of profiles in each bin using 'npstere' (Polar Stereographic Projection).

    Args:
        - df (pd.DataFrame): The dataset to plot.
        - n_x_bins (int): Number of bins in the x axis. Defaults to 100.
        - n_y_bins (int): Number of bins in the y axis. Defaults to 100.
        - show_threshold (bool, optional): Whether to show the threshold marker. Defaults to False.
        - threshold_quantile (float, optional): The quantile value for threshold calculation. Defaults to 0.95.
        - projection (str, optional): Basemap projection name. Defaults to 'npstere'.
        - showfig (bool, optional): Whether to display the figure. Defaults to True.

    Returns:
        - g_len_sorted[transition_point] (float | None): The number of profiles in the bin at the transition point, or None if no transition point is found.
        - fig (plotly.graph_objs._figure.Figure): The Plotly figure object containing the scatter plot of number of profiles per bin.
    """
    # Set up the polar stereographic projection
    lat_min = df['Latitude_[deg_N]'].min()
    m = Basemap(projection=projection, boundinglat=lat_min, lon_0=0, resolution='l')

    # Reproject lat/lon to x/y (projection coordinates in metres)
    lon_vals = df['Longitude_[deg_E]'].values
    lat_vals = df['Latitude_[deg_N]'].values
    x_vals, y_vals = m(lon_vals, lat_vals)

    # Define XY bin edges in projected space
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()
    x_bins = np.linspace(x_min, x_max, n_x_bins + 1)
    y_bins = np.linspace(y_min, y_max, n_y_bins + 1)

    # Digitize points into bins
    x_idx = np.digitize(x_vals, x_bins) - 1
    y_idx = np.digitize(y_vals, y_bins) - 1

    # Count number of profiles per bin
    H = np.zeros((n_x_bins, n_y_bins), dtype=int)
    for i in range(len(x_vals)):
        xi, yi = x_idx[i], y_idx[i]
        if 0 <= xi < n_x_bins and 0 <= yi < n_y_bins:
            H[xi, yi] += 1

    # Flatten to get non-zero bin counts (matching original behaviour of only counting occupied bins)
    g_len = H[H > 0].flatten().tolist()
    g_len_sorted = sorted(g_len)

    # Threshold / transition point detection
    transition_point = None
    if len(g_len_sorted) > 1:
        g_len_diff = np.diff(g_len_sorted)
        threshold = np.quantile(g_len_diff, threshold_quantile)
        for i in range(1, len(g_len_sorted)):
            if g_len_sorted[i] - g_len_sorted[i - 1] > threshold:
                transition_point = i - 1
                break

    # Build Plotly traces
    trace = go.Scatter(
        x=list(range(len(g_len_sorted))),
        y=g_len_sorted
    )

    if show_threshold and transition_point is not None:
        red_dot = go.Scatter(
            x=[transition_point],
            y=[g_len_sorted[transition_point]],
            mode='markers',
            marker=dict(color='red', size=10)
        )
        fig_data = [trace, red_dot]
    else:
        fig_data = [trace]

    layout = go.Layout(
        xaxis=dict(title='Bin No.'),
        yaxis=dict(title='Number of points per bin')
    )
    fig = go.Figure(data=fig_data, layout=layout)
    if showfig:
        fig.show()

    return g_len_sorted[transition_point]

def plot_geo(df, n_clusters=4, color='gmm_label', browser=False):
    """
    Plots the geographic distribution of GMM clusters on a map. Given a dataset of points with latitude, longitude and the hard clustering label. 
    This function visualises the clusters using a scatter plot on a geographic map. Note this is an interactive plot so it takes a while to render 
    in a notebook and takes up a lot of memory. If you want to save memory, set browser=True and it will open in a new tab of your prefered browser.

    Args:
        - df (pd.DataFrame): DataFrame containing the data points with 'Latitude_[deg_N]', 'Longitude_[deg_E]', and the clustering label column.
        - n_clusters (int): Number of clusters. Default is 4.
        - color (str): Column name in df that contains the clustering labels. Default is 'gmm_label'.
        - browser (bool): If True, the plot will be opened in a web browser, this saves memory when running in a notebook. Default is False.

    Returns:
        - fig (plotly.graph_objs._figure.Figure): The Plotly figure object containing the geographic scatter plot of clusters.
    """
    # Compute data centroid (mean location)
    center_lat = df['Latitude_[deg_N]'].mean()
    center_lon = df['Longitude_[deg_E]'].mean()

    colors = px.colors.sample_colorscale(
        px.colors.sequential.Viridis_r,
        np.linspace(0, 1, n_clusters)
    )

    # Stepped colorscale
    discrete_colorscale = []
    for i, c in enumerate(colors):
        discrete_colorscale.append([i / n_clusters, c])
        discrete_colorscale.append([(i + 1) / n_clusters, c])

    fig = px.scatter_geo(
        df,
        lat='Latitude_[deg_N]',
        lon='Longitude_[deg_E]',
        color=color,
        color_continuous_scale=discrete_colorscale,
        range_color=(0, n_clusters),
        title="GMM clusters using sampled TS on Geographic Map"
    )

    fig.update_traces(
        marker=dict(size=5, opacity=0.9)
    )
    fig.update_coloraxes(showscale=False)

    for i, c in enumerate(colors):
        fig.add_trace(
            go.Scattergeo(
                lat=[None], lon=[None],
                mode="markers",
                marker=dict(size=8, color=c),
                name=f"Cluster {i}",
                showlegend=True
            )
        )

    fig.update_geos(
        projection_type="orthographic",
        projection_rotation=dict(lat=center_lat, lon=center_lon),
        showcoastlines=True,
        showcountries=True
    )

    fig.update_layout(
        height=800,
        width=800,
        legend_title_text="GMM Cluster",
        legend_traceorder="normal"
    )

    if browser:
        fig.show(renderer='browser')
    else:
        fig.show()
    return fig

def plot_heatmap_npts_xy(df, n_x_bins=100, n_y_bins=100, bin_max=None, projection='npstere', projection_name='Polar Stereographic'):
    """
    Plots a heatmap of the number of points per bin in projection space (default: Polar Stereographic Projection).

    Args:
        - ds_cleaned (pd.DataFrame): The dataset to plot.
        - n_x_bins (int): Number of bins in the x axis. Defaults to 100.
        - n_y_bins (int): Number of bins in the y axis. Defaults to 100.
        - bin_max (int, optional): Maximum value for color scale. If None, it will be set to the maximum count in the bins. Defaults to None.
        - projection (str): Basemap projection string. Defaults to 'npstere' (Polar Stereographic Projection).
        - projection_name (str): Name of the projection for the title. Defaults to 'Polar Stereographic'.
    
    Returns:
        - fig (matplotlib.figure.Figure): The matplotlib figure object containing the heatmap of number of points per bin.
    """
    lat_min = df['Latitude_[deg_N]'].min()
    resolution = 'l'

    fig = plt.figure(figsize=(8, 8), dpi=300)
    m = Basemap(projection=projection, boundinglat=lat_min, lon_0=0, resolution=resolution)

    # Convert lat/lon → x/y (projection coordinates in meters)
    lon_vals = df['Longitude_[deg_E]'].values
    lat_vals = df['Latitude_[deg_N]'].values

    # Reproject lat/lon to x/y
    x_vals, y_vals = m(lon_vals, lat_vals)

    # Define XY bin edges
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()

    x_bins = np.linspace(x_min, x_max, n_x_bins + 1)
    y_bins = np.linspace(y_min, y_max, n_y_bins + 1)

    # Grid to store max depth
    H = np.zeros((n_x_bins, n_y_bins))

    # Digitize into bins
    x_idx = np.digitize(x_vals, x_bins) - 1
    y_idx = np.digitize(y_vals, y_bins) - 1

    # Count number of points in each bin
    for i in range(len(x_vals)):
        xi, yi = x_idx[i], y_idx[i]
        if 0 <= xi < n_x_bins and 0 <= yi < n_y_bins:
            H[xi, yi] += 1

    # Create meshgrid for plotting
    X, Y = np.meshgrid(x_bins, y_bins)

    # Colormap
    cmap = plt.get_cmap('Wistia').copy()
    cmap.set_under(color='white', alpha=1.0)
    cmap.set_over(color='white', alpha=1.0)

    if bin_max is None:
        bin_max = H.max()

    # Plot heatmap
    m.pcolormesh(X, Y, H.T, cmap=cmap, vmin=1, vmax=bin_max)

    # Draw bin grid (in xy space)
    for xb in x_bins:
        plt.plot([xb]*len(y_bins), y_bins, color='grey', linewidth=0.05, zorder=3)
    for yb in y_bins:
        plt.plot(x_bins, [yb]*len(x_bins), color='grey', linewidth=0.05, zorder=3)

    # Plot coastlines on top
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey', lake_color='lightblue')

    cbar = m.colorbar(location='bottom', pad="5%")
    cbar.set_label('Number of points per bin')

    plt.xticks([])
    plt.yticks([])
    plt.title('Number of Points per Bin in '+ f'({projection_name} Projection)')

    plt.show()

def profileNo_vs_latlon_bins(
    df: pd.DataFrame,
    lat_step: float,
    lon_step: float,
    show_threshold: bool = False,
    threshold_quantile: float = 0.95
) -> float | None:
    """
    Plots the number of profiles in each bin defined by latitude and longitude steps.

    Args:
        - df (pd.DataFrame): The dataset to plot.
        - lat_step (float): Latitude step size in degrees.
        - lon_step (float): Longitude step size in degrees.
        - show_threshold (bool, optional): Whether to show the threshold marker. Defaults to False.
        - threshold_quantile (float, optional): The quantile value for threshold calculation. Defaults to 0.95.

    Returns:
        - g_len_sorted[transition_point] (float | None): The number of profiles in the bin at the transition point, or None if no transition point is found.
        - fig (plotly.graph_objs._figure.Figure): The Plotly figure object containing the scatter plot of number of profiles per bin.
    """
    
    # === Identical bin construction to sample_cap_bins / plot_heatmap_npts ===
    lon_min, lon_max = -180, 180
    lat_min, lat_max = 55, 90

    lon_bins = np.arange(lon_min, lon_max + lon_step, lon_step)

    sin_step = np.sin(np.radians(lat_step)) - np.sin(np.radians(0))
    sin_min = np.sin(np.radians(lat_min))
    sin_max = np.sin(np.radians(lat_max))
    sin_bins = np.arange(sin_min, sin_max + sin_step, sin_step)
    lat_bins = np.degrees(np.arcsin(np.clip(sin_bins, -1, 1)))

    n_lon_bins = len(lon_bins) - 1
    n_lat_bins = len(lat_bins) - 1
    # =========================================================================

    lon_vals = df['Longitude_[deg_E]'].values
    lat_vals = df['Latitude_[deg_N]'].values

    lon_idx = np.clip(np.digitize(lon_vals, lon_bins) - 1, 0, n_lon_bins - 1)
    lat_idx = np.clip(np.digitize(lat_vals, lat_bins) - 1, 0, n_lat_bins - 1)

    # Count per bin
    from collections import defaultdict
    bin_counts = defaultdict(int)
    for i in range(len(lon_vals)):
        bin_counts[(lon_idx[i], lat_idx[i])] += 1

    g_len = list(bin_counts.values())
    g_len_sorted = sorted(g_len)

    # Threshold detection
    transition_point = None
    if len(g_len_sorted) > 1:
        g_len_diff = np.diff(g_len_sorted)
        threshold = np.quantile(g_len_diff, threshold_quantile)
        for i in range(1, len(g_len_sorted)):
            if g_len_sorted[i] - g_len_sorted[i - 1] > threshold:
                transition_point = i - 1
                break

    trace = go.Scatter(
        x=list(range(len(g_len_sorted))),
        y=g_len_sorted
    )

    if show_threshold and transition_point is not None:
        red_dot = go.Scatter(
            x=[transition_point],
            y=[g_len_sorted[transition_point]],
            mode='markers',
            marker=dict(color='red', size=10)
        )
        fig_data = [trace, red_dot]
    else:
        fig_data = [trace]

    layout = go.Layout(
        xaxis=dict(title='Bin No.'),
        yaxis=dict(title='Number of points per bin')
    )
    fig = go.Figure(data=fig_data, layout=layout)
    fig.show()

    return g_len_sorted[transition_point]


def sample_cap_bins(df, thresh, lon_step=5, lat_step=5):
    """
    Samples the dataset by creating bins in latitude and longitude, and randomly samples from each bin.

    Args:
        - df (pd.DataFrame): The dataset to sample.
        - thresh (int): Maximum number of samples to take from each bin.
        - lon_step (float): Longitude step size in degrees. Defaults to 5.
        - lat_step (float): Latitude step size in degrees. Defaults to 5.

    Returns:
        - df_sampled (pd.DataFrame): The sampled dataset in pandas dataframe.
    """
    lon_min, lon_max = -180, 180
    lat_min, lat_max = 55, 90

    lon_bins = np.arange(lon_min, lon_max + lon_step, lon_step)

    sin_step = np.sin(np.radians(lat_step)) - np.sin(np.radians(0))
    sin_min = np.sin(np.radians(lat_min))
    sin_max = np.sin(np.radians(lat_max))
    sin_bins = np.arange(sin_min, sin_max + sin_step, sin_step)
    lat_bins = np.degrees(np.arcsin(np.clip(sin_bins, -1, 1)))

    n_lon_bins = len(lon_bins) - 1
    n_lat_bins = len(lat_bins) - 1

    lon_vals = df['Longitude_[deg_E]'].values
    lat_vals = df['Latitude_[deg_N]'].values

    lon_idx = np.clip(np.digitize(lon_vals, lon_bins) - 1, 0, n_lon_bins - 1)
    lat_idx = np.clip(np.digitize(lat_vals, lat_bins) - 1, 0, n_lat_bins - 1)

    bin_members = defaultdict(list)
    for i in range(len(lon_vals)):
        bin_members[(lon_idx[i], lat_idx[i])].append(i)

    sampled_indices = []
    for (xi, yi), members in bin_members.items():
        if len(members) > thresh:
            sampled_indices.extend(
                np.random.choice(members, size=thresh, replace=False).tolist()
            )
        else:
            sampled_indices.extend(members)
    
    df_sampled = df.iloc[sampled_indices].copy()

    return df_sampled


def plot_heatmap_npts(df, lon_step=5, lat_step=5):
    """
    Plots a heatmap of the number of points per bin defined by latitude and longitude steps.

    Args:
        - df (pd.DataFrame): The dataset to plot.
        - lon_step (float): Longitude step size in degrees. Defaults to 5.
        - lat_step (float): Latitude step size in degrees. Defaults to 5.

    Returns:
        - fig (matplotlib.figure.Figure): The matplotlib figure object containing the heatmap of number of points per bin.
    """
    lon_min, lon_max = -180, 180
    lat_min, lat_max = 55, 90
    resolution = 'l'

    # Longitude bins: uniform steps in degrees
    lon_bins = np.arange(lon_min, lon_max + lon_step, lon_step)

    # Equal-area latitude bins: uniform steps in sin(latitude)
    # Convert lat_step in degrees to an equivalent step in sin-space
    sin_step = np.sin(np.radians(lat_step)) - np.sin(np.radians(0))
    sin_min = np.sin(np.radians(lat_min))
    sin_max = np.sin(np.radians(lat_max))
    sin_bins = np.arange(sin_min, sin_max + sin_step, sin_step)
    lat_bins = np.degrees(np.arcsin(np.clip(sin_bins, -1, 1)))

    n_lon_bins = len(lon_bins) - 1
    n_lat_bins = len(lat_bins) - 1

    lon_vals = df['Longitude_[deg_E]'].values
    lat_vals = df['Latitude_[deg_N]'].values

    lon_idx = np.clip(np.digitize(lon_vals, lon_bins) - 1, 0, n_lon_bins - 1)
    lat_idx = np.clip(np.digitize(lat_vals, lat_bins) - 1, 0, n_lat_bins - 1)

    count = np.zeros((n_lon_bins, n_lat_bins), dtype=int)
    for i in range(len(lon_vals)):
        count[lon_idx[i], lat_idx[i]] += 1

    fig = plt.figure(figsize=(8, 8), dpi=300)
    m = Basemap(projection='npstere', boundinglat=lat_min, lon_0=0, resolution=resolution)
    m.drawcoastlines()
    m.fillcontinents(color='grey', lake_color='lightblue')

    for lat in lat_bins:
        lons = np.linspace(lon_min, lon_max, 500)
        x, y = m(lons, np.full_like(lons, lat))
        m.plot(x, y, color='grey', linewidth=0.05, zorder=3)

    for lon in lon_bins:
        lats = np.linspace(lat_min, lat_max, 500)
        x, y = m(np.full_like(lats, lon), lats)
        m.plot(x, y, color='grey', linewidth=0.05, zorder=3)

    X, Y = np.meshgrid(lon_bins, lat_bins)
    X_map, Y_map = m(X, Y)
    cmap = plt.get_cmap('Wistia').copy()
    cmap.set_under(color='white', alpha=1.0)
    cmap.set_over(color='white', alpha=1.0)
    m.pcolormesh(X_map, Y_map, count.T, cmap=cmap, vmin=1, vmax=None)

    m.drawcoastlines()
    m.fillcontinents(color='lightgrey', lake_color='lightblue')
    cbar = m.colorbar(location='bottom', pad="5%")
    cbar.set_label('Number of points in bin')
    plt.xticks([])
    plt.yticks([])
    plt.title(f'Number of Points per Bin ({lon_step}° lon x ~{lat_step}° lat equal-area bins)')
    plt.show()


def sample_cap_bins_xy(df, thresh, n_x_bins, n_y_bins):
    """
    Separate dataset into bins in projected (Polar Stereographic Projection) space and randomly sample from each bin.

    Args:
        - df (pd.DataFrame): The dataset to sample.
        - thresh (int): Maximum number of samples to take from each bin.
        - n_x_bins (int): Number of bins in the x axis. Defaults to 100.
        - n_y_bins (int): Number of bins in the y axis. Defaults to 100.

    Returns:
        - df_sampled (pd.DataFrame): The sampled dataset in pandas dataframe.
    """
    # Define projection and reproject lat/lon to x/y
    lat_min = df['Latitude_[deg_N]'].min()
    m = Basemap(projection='npstere', boundinglat=lat_min, lon_0=0, resolution='l')
    lon_vals = df['Longitude_[deg_E]'].values
    lat_vals = df['Latitude_[deg_N]'].values
    x_vals, y_vals = m(lon_vals, lat_vals)

    # Define XY bin edges in projected space
    x_bins = np.linspace(x_vals.min(), x_vals.max(), n_x_bins + 1)
    y_bins = np.linspace(y_vals.min(), y_vals.max(), n_y_bins + 1)

    # Assign points to bins and clip edge cases
    x_idx = np.clip(np.digitize(x_vals, x_bins) - 1, 0, n_x_bins - 1)
    y_idx = np.clip(np.digitize(y_vals, y_bins) - 1, 0, n_y_bins - 1)

    # Build a dict mapping (xi, yi) -> list of original df indices
    # This is the single source of truth for bin membership
    from collections import defaultdict
    bin_members = defaultdict(list)
    for i in range(len(x_vals)):
        bin_members[(x_idx[i], y_idx[i])].append(i)

    # Sample each bin: cap at thresh if over, keep all if under
    sampled_indices = []
    for (xi, yi), members in bin_members.items():
        if len(members) > thresh:
            sampled_indices.extend(
                np.random.choice(members, size=thresh, replace=False)
            )
        else:
            sampled_indices.extend(members)

    df_sampled = df.iloc[sampled_indices].copy()
    return df_sampled

def plot_3d(df, n_clusters, color='gmm_label', browser=False):
    """
    Plots a 3D scatter plot of the sampled dataset with latitude, longitude, and depth. This is a very rudimentary way of plotting this data in 3D,
    but it is useful for visualising the clusters in 3D space. This plot is a 3D cube so it distorts the longitude term significantly (in this case -180≠0).
    Note this is an interactive plot so it takes a while to render in a notebook and takes up a lot of memory. If you want to save memory, 
    set browser=True and it will open in a new tab of your prefered browser.

    Args:
        - df (pd.DataFrame): DataFrame containing the data points with 'Latitude_[deg_N]', 'Longitude_[deg_E]', 'Depth_[m]', and the clustering label column.
        - n_clusters (int): Number of clusters.
        - color (str): Column name in df that contains the clustering labels. Default is 'gmm_label'.
        - browser (bool): If True, the plot will be opened in a web browser, this saves memory when running in a notebook. Default is False.
    
    Returns:
        - fig_3d (plotly.graph_objs._figure.Figure): The Plotly figure object containing the 3D scatter plot of clusters.
    """
    #Plot in 3D
    colors = px.colors.sample_colorscale(
        px.colors.sequential.Viridis_r,
        np.linspace(0, 1, n_clusters)
    )

    # Stepped colorscale
    discrete_colorscale = []
    for i, c in enumerate(colors):
        discrete_colorscale.append([i / n_clusters, c])
        discrete_colorscale.append([(i + 1) / n_clusters, c])

    fig_3d = px.scatter_3d(
        df,
        x='Latitude_[deg_N]',
        y='Longitude_[deg_E]',
        z='Depth_[m]',
        color=color,
        color_continuous_scale=discrete_colorscale,
        range_color=(0, n_clusters),
        title="3D Scatter Lat-Lon-Depth plot of GMM clusters"
    )

    fig_3d.update_traces(
        marker=dict(
            size=5
        )
    )

    fig_3d.update_coloraxes(showscale=False)  # hide colorbar

    for i, c in enumerate(colors):
        fig_3d.add_trace(
            go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode="markers",
                marker=dict(size=8, color=c),
                name=f"Cluster {i}",
                showlegend=True
            )
        )

    fig_3d.update_layout(
        scene=dict(
            xaxis_title='Latitude',
            yaxis_title='Longitude',
            zaxis_title='Depth (m)',
            zaxis=dict(autorange='reversed')
        ),
        height=800,
        width=800,
        legend_title_text="GMM Cluster",
        legend_traceorder="normal"
    )

    if browser:
        fig_3d.show(renderer='browser')
    else:
        fig_3d.show()


# Cross section like Kate's
def compute_great_circle_distance_km(lat_ref, lon_ref, lats, lons):
    """
    Auxiliary function to compute distance in km from a reference point along a great circle. This function is used for the cross section plots to compute 
    the distance from the start point to each point in the dataset. 

    Args:
        - lat_ref (float): Latitude of the reference point in degrees.
        - lon_ref (float): Longitude of the reference point in degrees.
        - lats (np.ndarray): Array of latitudes of points to compute distance to.
        - lons (np.ndarray): Array of longitudes of points to compute distance to.

    Returns:
        - dist_km (np.ndarray): Array of distances in kilometers from the reference point to each point in the input arrays.
    """
    geod = Geod(ellps='WGS84')
    _, _, dist_m = geod.inv(
        np.full(len(lats), lon_ref),
        np.full(len(lats), lat_ref),
        lons,
        lats
    )
    return dist_m / 1000.0


def cross_sec_lon(df, n_clusters, lon_value, tol=0.5, start_lat=70.0, start_lon=-140.0, save_fig=False, fig_name='cross_section_plot.png'):
    """
    Plots a cross-section of the dataset along a specified longitude, including the great-circle distance from a starting point. 
    This function is useful for visualizing the distribution of points and their clustering along a specific longitudinal slice. 
    This code is inspired by Oglethorpe et al (2026) to create similar transsect plots to their water mass fraction plots.
    
    Args:
    - df (pd.DataFrame): The sampled dataset containing latitude, longitude, depth, and clustering labels.
    - n_clusters (int): Number of clusters in the GMM model.
    - lon_value (float): The longitude value for the cross-section slice (in degrees East).
    - tol (float): Tolerance for selecting points around the specified longitude (in degrees). Default is 0.5.
    - start_lat (float): Latitude of the starting point for distance calculation (in degrees North). Default is 70.0.
    - start_lon (float): Longitude of the starting point for distance calculation (in degrees East). Default is -140.0.
    - save_fig (bool): Whether to save the figure as a PNG file. Default is False.
    - fig_name (str): Filename for saving the figure if save_fig is True. Default is 'cross_section_plot.png'.
    
    Returns:
    - fig (matplotlib.figure.Figure): The matplotlib figure object containing the cross-section scatter plot and polar map, 
    with the option of saving it as a PNG.
    """

    # Primary slice from start_lon and secondary for lon_value intersecting at the pole
    df_primary = df[
        (df['Longitude_[deg_E]'] >= start_lon - tol) &
        (df['Longitude_[deg_E]'] <= start_lon + tol)
    ].copy()

    df_antipodal = df[
        (df['Longitude_[deg_E]'] >= lon_value - tol) &
        (df['Longitude_[deg_E]'] <= lon_value + tol)
    ].copy()

    df_lon_slice = pd.concat([df_primary, df_antipodal], ignore_index=True)

    # Compute great-circle distance from start point
    df_lon_slice['dist_km'] = compute_great_circle_distance_km(
        start_lat, start_lon,
        df_lon_slice['Latitude_[deg_N]'].values,
        df_lon_slice['Longitude_[deg_E]'].values
    )

    # ── Colormap (GMM labels) ─────────────────────────────────────────────────
    n = n_clusters
    base_cmap = plt.cm.viridis_r
    colors = base_cmap(np.linspace(0, 1, n))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, n + 0.5, 1), cmap.N)

    # ── Side-by-side layout ───────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)

    ax = fig.add_subplot(gs[0])
    ax_map = fig.add_subplot(gs[1], projection=ccrs.NorthPolarStereo())

    # ── Cross-section scatter plot ────────────────────────────────────────────
    sc = ax.scatter(
        df_lon_slice['dist_km'],
        df_lon_slice['Depth_[m]'],
        c=df_lon_slice['gmm_label_'+str(n)],
        cmap=cmap,
        norm=norm,
        s=8,
        alpha=0.3
    )

    ax.invert_yaxis()
    ax.set_xlabel(f'Distance (km) from {start_lat}°N, {abs(start_lon)}°W')
    ax.set_ylabel('Depth (m)')
    ax.set_title('GMM Clusters — Cross Section through North Pole')

    # Discrete colorbar
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_ticks(range(n))
    cbar.set_ticklabels([f'Cluster {i}' for i in range(n)])
    cbar.set_label('GMM Cluster')

    # ── Polar map ─────────────────────────────────────────────────────────────
    ax_map.set_extent([-180, 180, 55, 90], crs=ccrs.PlateCarree())
    ax_map.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax_map.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)

    # Section track: start → pole → antipodal end
    end_lat = start_lat
    end_lon = lon_value
    lats_track = [start_lat, 90.0, end_lat]
    lons_track = [start_lon, lon_value, end_lon]

    ax_map.plot(lons_track, lats_track,
                color='black', linewidth=1.5,
                transform=ccrs.Geodetic(), zorder=3)

    ax_map.text(start_lon, start_lat - 3, 'Start',
                transform=ccrs.PlateCarree(),
                fontsize=7, ha='center', color='black')
    ax_map.text(end_lon, end_lat - 3, 'End',
                transform=ccrs.PlateCarree(),
                fontsize=7, ha='center', color='black')

    ax_map.set_title('Section Location', fontsize=9)

    if save_fig:
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()




def cross_sec_lon_prob(df, n_clusters, lon_value, tol=0.5, start_lat=70.0, start_lon=-140.0, browser=False, save_fig=False, fig_name='cross_section_prob_plot.png'):
    """
    Plots a cross-section of the dataset along a specified longitude, including the great-circle distance from a starting point, 
    and visualizes the GMM cluster probabilities. This is point data and not a gridded field, so the plot is a scatter plot of 
    points colored by their cluster probabilities.

    Args:
    - df (pd.DataFrame): The sampled dataset containing latitude, longitude, depth, and clustering probabilities.
    - n_clusters (int): Number of clusters in the GMM model.
    - lon_value (float): The longitude value for the cross-section slice (in degrees East).
    - tol (float): Tolerance for selecting points around the specified longitude (in degrees). Default is 0.5.
    - start_lat (float): Latitude of the starting point for distance calculation (in degrees North). Default is 70.0.
    - start_lon (float): Longitude of the starting point for distance calculation (in degrees East). Default is -140.0.
    - browser (bool): If True, the plot will be opened in a web browser, this saves memory when running in a notebook. Default is False.
    - save_fig (bool): Whether to save the figure as a PNG file. Default is False.
    - fig_name (str): Filename for saving the figure if save_fig is True. Default is 'cross_section_prob_plot.png'.

    Returns:
        - fig (matplotlib.figure.Figure): The matplotlib figure object containing the cross-section scatter plot with cluster 
        probabilities, with the option of saving it as a PNG.
    """
    # Antipodal longitude for the over-pole extension
    #lon_value = lon_value if lon_value <= 180 & lon_value >=0 else lon_value + 180
    if lon_value < 0 and lon_value >= -180:
        lon_value = lon_value + 180

    # Primary slice from start_lon and secondary for lon_value intersecting at the pole
    df_primary = df[
        (df['Longitude_[deg_E]'] >= start_lon - tol) &
        (df['Longitude_[deg_E]'] <= start_lon + tol)
    ].copy()

    df_antipodal = df[
        (df['Longitude_[deg_E]'] >= lon_value - tol) &
        (df['Longitude_[deg_E]'] <= lon_value + tol)
    ].copy()

    df_lon_slice = pd.concat([df_primary, df_antipodal], ignore_index=True)

    # Compute great-circle distance from start point
    df_lon_slice['dist_km'] = compute_great_circle_distance_km(
        start_lat, start_lon,
        df_lon_slice['Latitude_[deg_N]'].values,
        df_lon_slice['Longitude_[deg_E]'].values
    )

    # Layout with n_cluster scatter panels and 1 polar map
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(int(np.floor(n_clusters/2)), int(np.floor(n_clusters/2)), width_ratios=[1, 1, 0.5], wspace=0.3, hspace=0.35)

    axes = []
    for i in range(int(np.floor(n_clusters/2))):
        for j in range(int(np.floor(n_clusters/3))):
            axes.append(fig.add_subplot(gs[i, j]))
    
    ax_map = fig.add_subplot(gs[:, 2], projection=ccrs.NorthPolarStereo())

    # ── Plot each cluster probability ─────────────────────────────────────────
    for i, ax in enumerate(axes):
        col = f'prob_cluster_{i}'

        sc = ax.scatter(
            df_lon_slice['dist_km'],
            df_lon_slice['Depth_[m]'],
            c=df_lon_slice[col],
            cmap='viridis',
            vmin=0,
            vmax=1,
            s=n_clusters,
            alpha=0.5
        )

        ax.invert_yaxis()
        ax.set_title(f'Cluster {i} Probability')
        ax.set_xlabel(f'Distance (km) from {start_lat}°N, {abs(start_lon)}°W')
        ax.set_ylabel('Depth (m)')

        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label('Probability')

    # ── Polar map (spans both rows on the right) ──────────────────────────────
    end_lat = start_lat
    end_lon = lon_value
    lats_track = [start_lat, 90.0, end_lat]
    lons_track = [start_lon, lon_value, end_lon]

    ax_map.set_extent([-180, 180, 55, 90], crs=ccrs.PlateCarree())
    ax_map.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax_map.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)

    ax_map.plot(lons_track, lats_track,
                color='black', linewidth=1.5,
                transform=ccrs.Geodetic(), zorder=3)

    ax_map.text(start_lon, start_lat - 3, 'Start',
                transform=ccrs.PlateCarree(),
                fontsize=7, ha='center', color='black')
    ax_map.text(end_lon, end_lat - 3, 'End',
                transform=ccrs.PlateCarree(),
                fontsize=7, ha='center', color='black')

    ax_map.set_title('Section Location', fontsize=9)

    fig.suptitle(
        f'GMM Cluster Probabilities — Cross Section through North Pole\n',
        fontsize=13
    )

    if browser:
        plt.show(renderer='browser')
    elif save_fig:
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def cross_sec_lon_prob2(df,
    n_clusters,
    lon_value,
    tol=0.5,
    start_lat=70.0,
    start_lon=-140.0,
    max_depth=850,
    depth_bin_size=10,
    dist_bin_size=2.2,
    browser=False,
    save_fig=False,
    fig_name='cross_section_avg_prob_plot.png'):

    """
    Plots a cross-section of the dataset along a specified longitude, including the great-circle distance from a starting point, 
    and visualizes the GMM cluster probabilities. This function bins the data in depth and distance, averages the probabilities within each bin
    and then interpolates onto a regular grid for plotting. This code is inspired by Oglethorpe et al (2026).

    Args:
        - df (pd.Dataframe): The sampled dataset containing latitude, longitude, depth, and clustering probabilities.
        - n_clusters (int): Number of clusters in the GMM model.
        - lon_value (float): The longitude value for the cross-section slice (in degrees East).
        - tol (float): Tolerance for selecting points around the specified longitude (in degrees). Default is 0.5.
        - start_lat (float): Latitude of the starting point for distance calculation (in degrees North). Default is 70.0.
        - start_lon (float): Longitude of the starting point for distance calculation (in degrees East). Default is -140.0.
        - max_depth (float): Maximum depth to plot (in meters). Default is 850.
        - depth_bin_size (float): Size of depth bins for averaging probabilities (in meters). Default is 10.
        - dist_bin_size (float): Size of distance bins for averaging probabilities (in kilometers). Default is 2.2.
        - browser (bool): If True, the plot will be opened in a web browser, this saves memory when running in a notebook. Default is False.
        - save_fig (bool): Whether to save the figure as a PNG file. Default is False.
        - fig_name (str): Filename for saving the figure if save_fig is True. Default is 'cross_section_avg_prob_plot.png'.

    Returns:
        - fig (matplotlib.figure.Figure): The matplotlib figure object containing the cross-section scatter plot with averaged cluster probabilities, 
        with the option of saving it as a PNG.
    """
    # Antipodal longitude for the over-pole extension
    if lon_value < 0 and lon_value >= -180:
        lon_value = lon_value + 180
        
    # Primary slice from start_lon and secondary for lon_value intersecting at the pole
    df_primary = df[
        (df['Longitude_[deg_E]'] >= start_lon - tol) &
        (df['Longitude_[deg_E]'] <= start_lon + tol)
    ].copy()

    df_antipodal = df[
        (df['Longitude_[deg_E]'] >= lon_value - tol) &
        (df['Longitude_[deg_E]'] <= lon_value + tol)
    ].copy()

    df_lon_slice = pd.concat([df_primary, df_antipodal], ignore_index=True)

    # Compute great-circle distance from start point
    df_lon_slice['dist_km'] = compute_great_circle_distance_km(
        start_lat, start_lon,
        df_lon_slice['Latitude_[deg_N]'].values,
        df_lon_slice['Longitude_[deg_E]'].values
    )

    # Bin coordinates
    df_lon_slice['depth_bin'] = (
        np.floor(df_lon_slice['Depth_[m]'] / depth_bin_size)
        * depth_bin_size
    )

    df_lon_slice['dist_bin'] = (
        np.floor(df_lon_slice['dist_km'] / dist_bin_size)
        * dist_bin_size
    )

    # Average probabilities within bins
    prob_cols = [f'prob_cluster_{i}' for i in range(n_clusters)]

    df_binned = (
        df_lon_slice
        .groupby(['dist_bin', 'depth_bin'])[prob_cols]
        .mean()
        .reset_index()
    )

    # Regular plotting grid
    xi = np.arange(
        df_binned['dist_bin'].min(),
        df_binned['dist_bin'].max(),
        dist_bin_size
    )

    yi = np.arange(
        0,
        max_depth + depth_bin_size,
        depth_bin_size
    )

    XI, YI = np.meshgrid(xi, yi)

    # Figure
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(int(np.floor(n_clusters/2)), int(np.floor(n_clusters/2)), 
                          width_ratios=[1, 1, 0.1], wspace=0.4, hspace=0.5)

    axes = []
    for i in range(int(np.floor(n_clusters/2))):
        for j in range(int(np.floor(n_clusters/3))):
            axes.append(fig.add_subplot(gs[i, j]))
    
    cax = fig.add_subplot(gs[:, 2])
    # Plot each field
    for i, ax in enumerate(axes):

        col = f'prob_cluster_{i}'
        # Interpolate onto regular grid
        ZI = griddata(
            (
                df_binned['dist_bin'],
                df_binned['depth_bin']
            ),
            df_binned[col] * 100,
            (XI, YI),
            method='linear'
        )

        # Smooth field
        ZI = gaussian_filter(ZI, sigma=1.0)

        # Plot
        pcm = ax.pcolormesh(
            XI,
            YI,
            ZI,
            shading='auto',
            cmap='viridis',
            vmin=0,
            vmax=100)

        # Pole marker
        ax.axvline(
            2200,
            color='white',
            linestyle='--',
            linewidth=1)

        # Formatting
        ax.set_ylim(max_depth, 0)
        ax.set_title(f'Cluster {i} Probability')
        ax.set_xlabel(f'Distance (km) from 70°N, 140°W')
        ax.set_ylabel('Depth (m)')

    # Shared colorbar
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.set_label('Probability (%)', fontsize=12)

    # Title
    fig.suptitle(
        'GMM Cluster Probabilities — Cross Section through North Pole',
        fontsize=18,
        y=0.98
    )

    if browser:
        plt.show(renderer='browser')
    elif save_fig:
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def water_mass_fraction_plot_kate(
    df,
    n_WM,
    lon_value,
    tol=0.5,
    start_lat=70.0,
    start_lon=-140.0,
    max_depth=850,
    depth_bin_size=10,
    dist_bin_size=2.2,
    browser=False,
    save_fig=False,
    fig_name='cross_section_water_mass_fraction_plot.png'):
    """
    Plots water mass fractions in a transsect accross the Arctic based on data from Ogelthorpe et al 2026 (must be downloaded). 
    This code was inspired by their code to create similar transsect plots to their water mass fraction plots. 
    Averages the water mass fractions in bins of depth and distance from a starting point, then interpolates onto a regular grid for plotting.
    The Oglethorpe et al (2026) column names for each water mass fraction are hardcoded in this function, so this is not built for general use.

    Args:
        - df (pd.DataFrame): The sampled dataset containing latitude, longitude, depth, and water mass fractions.
        - n_WM (int): Number of water masses
        - lon_value (float): The longitude value for the cross-section slice (in degrees East).
        - tol (float): Tolerance for selecting points around the specified longitude (in degrees). Default is 0.5.
        - start_lat (float): Latitude of the starting point for distance calculation (in degrees North). Default is 70.0.
        - start_lon (float): Longitude of the starting point for distance calculation (in degrees East). Default is -140.0.
        - max_depth (float): Maximum depth to plot (in meters). Default is 850.
        - depth_bin_size (float): Size of depth bins for averaging water mass fractions (in meters). Default is 10.
        - dist_bin_size (float): Size of distance bins for averaging water mass fractions (in kilometers). Default is 2.2.
        - browser (bool): If True, the plot will be opened in a web browser, this saves memory when running in a notebook. Default is False.
        - save_fig (bool): Whether to save the figure as a PNG file. Default is False.
        - fig_name (str): Filename for saving the figure if save_fig is True. Default is 'cross_section_water_mass_fraction_plot.png'.
    
    Returns:
        - fig (matplotlib.figure.Figure): The matplotlib figure object containing the cross-section scatter plot with averaged water mass fractions, 
        with the option of saving it as a PNG.
    """
    # Antipodal longitude for the over-pole extension
    if lon_value < 0 and lon_value >= -180:
        lon_value = lon_value + 180
        
    # Primary slice from start_lon and secondary for lon_value intersecting at the pole
    df_primary = df[
        (df['Longitude_[deg_E]'] >= start_lon - tol) &
        (df['Longitude_[deg_E]'] <= start_lon + tol)
    ].copy()

    df_antipodal = df[
        (df['Longitude_[deg_E]'] >= lon_value - tol) &
        (df['Longitude_[deg_E]'] <= lon_value + tol)
    ].copy()

    df_lon_slice = pd.concat([df_primary, df_antipodal], ignore_index=True)

    # Compute great-circle distance from start point
    df_lon_slice['dist_km'] = compute_great_circle_distance_km(
        start_lat, start_lon,
        df_lon_slice['Latitude_[deg_N]'].values,
        df_lon_slice['Longitude_[deg_E]'].values
    )

    # Bin coordinates
    df_lon_slice['depth_bin'] = (
        np.floor(df_lon_slice['Depth_[m]'] / depth_bin_size)
        * depth_bin_size
    )

    df_lon_slice['dist_bin'] = (
        np.floor(df_lon_slice['dist_km'] / dist_bin_size)
        * dist_bin_size
    )

    # Average water mass fractions within bins
    WMfrac_cols = ['Arctic_Surface_Water_[fraction]',
       'Modified_summer_Pacific_Water_[fraction]',
       'Summer_Pacific_Water_[fraction]', 'Winter_Pacific_Water_[fraction]',
       'Norwegian_Current_Water_[fraction]', 'Atlantic_Water_[fraction]',
       'Brine-enriched_Water_[fraction]']

    df_binned = (
        df_lon_slice
        .groupby(['dist_bin', 'depth_bin'])[WMfrac_cols]
        .mean()
        .reset_index()
    )

    # Regular plotting grid
    xi = np.arange(
        df_binned['dist_bin'].min(),
        df_binned['dist_bin'].max(),
        dist_bin_size
    )

    yi = np.arange(
        0,
        max_depth + depth_bin_size,
        depth_bin_size
    )

    XI, YI = np.meshgrid(xi, yi)

    # Figure
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(int(np.floor(n_WM/2)), int(np.floor(n_WM/2)), 
                          width_ratios=[1, 1, 0.1], wspace=0.4, hspace=0.5)

    axes = []
    for i in range(int(np.floor(n_WM/2))):
        for j in range(int(np.floor(n_WM/3))):
            axes.append(fig.add_subplot(gs[i, j]))
    
    cax = fig.add_subplot(gs[:, 2])
    # Plot each field
    for i, ax in enumerate(axes):

        col = WMfrac_cols[i]
        # Interpolate onto regular grid
        ZI = griddata(
            (
                df_binned['dist_bin'],
                df_binned['depth_bin']
            ),
            df_binned[col] * 100,
            (XI, YI),
            method='linear'
        )

        # Smooth field
        ZI = gaussian_filter(ZI, sigma=1.0)

        # Plot
        pcm = ax.pcolormesh(
            XI,
            YI,
            ZI,
            shading='auto',
            cmap='viridis',
            vmin=0,
            vmax=100)

        # Pole marker
        ax.axvline(
            2200,
            color='white',
            linestyle='--',
            linewidth=1)

        # Formatting
        ax.set_ylim(max_depth, 0)
        ax.set_title(col.replace('_', ' ').replace('[fraction]', '').title())
        ax.set_xlabel(f'Distance (km) from 70°N, 140°W')
        ax.set_ylabel('Depth (m)')

    # Shared colorbar
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.set_label('Fraction (%)', fontsize=12)

    # Title
    fig.suptitle(
        'Water Mass Fractions — Cross Section through North Pole',
        fontsize=18,
        y=0.98
    )

    if browser:
        plt.show(renderer='browser')
    elif save_fig:
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def calc_PV_basic(df):
    """
    Calculate potential vorticity (PV) for each profile in the dataframe.
    PV is calculated using the formula:
        PV = -f * N^2 / g
    where:
        - f is the Coriolis parameter (f = 2 * omega * sin(latitude))
        - N^2 is the buoyancy frequency squared, calculated from the vertical profiles of temperature and salinity
        - g is the acceleration due to gravity  
    The function adds a new column 'Pv' to the dataframe with the calculated potential vorticity values.
    
    Args:
    - df: pandas DataFrame containing the oceanographic profile data with columns for temperature, salinity, depth, and latitude.
    
    Returns:
    - df: the input DataFrame with an additional column 'Pv' containing the calculated potential vorticity values.
    """

    # Calculate potential vorticity
    omega = 7.292115e-5 # Earth's rotation rate [rad/s]
    g = 9.81 # gravity [m/s^2] 

    # Create empty column for potential vorticity 
    df['Pv'] = np.nan 

    # Loop through each profile and calculate potential vorticity 
    for profile_id, prof in df.groupby('Profile_Number'): 
        prof = ( prof .sort_values('Depth_[m]') 
                .dropna(subset=[ 'Conservative_Temperature_[deg_C]', 'Absolute_Salinity_[PSU]', 'Depth_[m]', 'Latitude_[deg_N]', ]) 
                .drop_duplicates(subset='Depth_[m]') ) 
        
        # Need at least 3 levels 
        if len(prof) < 3: continue 
        
        # Extract the necessary variables for potential vorticity calculation 
        CT = prof['Conservative_Temperature_[deg_C]'].values 
        SA = prof['Absolute_Salinity_[PSU]'].values 
        z = prof['Depth_[m]'].values 
        lat = prof['Latitude_[deg_N]'].iloc[0] 
        
        # Convert depth to pressure 
        p = gsw.p_from_z(-z, lat) 
        
        # Calculate N squared 
        N2, p_mid = gsw.Nsquared(SA, CT, p, lat) 
        
        # Coriolis parameter 
        f = 2 * omega * np.sin(np.deg2rad(lat)) 
        
        # Calculate potential vorticity
        Pv = -f * N2 / g 
        
        # Assign Pv values back to the dataframe 
        upper_level_indices = prof.index[:-1] 
        df.loc[upper_level_indices, 'Pv'] = Pv

    df.dropna(subset=['Pv'], inplace=True)
    return df

def normalize(arr, range_min, range_max):
    """
    Auxiliary function for calc_PV_e_fit to normalize an array to a specified range [range_min, range_max].
    
    Args:
        - arr: numpy array of values to be normalized.
        - range_min: minimum value of the desired range.
        - range_max: maximum value of the desired range.

    Returns:
        - norm_arr: numpy array of normalized values within the specified range.
    """
    norm_arr = []
    diff = range_max - range_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        if diff_arr == 0:
            norm_arr.append(np.nan)  # Handle division by zero
        else:
            temp = (((i - min(arr)) * diff) / diff_arr) + range_min
            norm_arr.append(temp)
    return np.array(norm_arr)

def calculate_h_and_PV_from_rho(profile):
    """
    Auxiliary function for calc_PV_e_fit. This function calculates the density layer thickness (h_d) and potential vorticity (PV) for a given oceanographic profile, 
    using a rolling window exponential fit method. The function first removes the mixed layer from the profile,
    normalizes the potential density. Original code given by the authors of Oglethorpe et al (2026), modified by 
    the repo author.
    
    Args:
        - profile: pandas DataFrame containing the oceanographic profile data with columns for potential density (sigma0), depth, and latitude.
    
    Returns:
        - profile: the input DataFrame with two additional columns 'h_d' and 'PV' containing the calculated density layer thickness and potential vorticity values.
    """

    # Remove mixed layer
    # Check the profile starts shallower than 10m
    if profile['Depth_[m]'].iloc[0]< 10:
        # Find the surface density
        surface_density = profile['sigma0'].iloc[0]
        # Find the maximum depth at which potential density is within 0.1kg/m3 of surface density
        mld = profile.loc[(profile['sigma0'] <= surface_density + 0.1), 'Depth_[m]'].max()
        # Remove depths in profile shallower than the mixed layer depth
        profile = profile[profile['Depth_[m]'] >= mld]

    # Check if the difference in sigma0 is zero
    if np.diff(profile['sigma0']).any() == 0:
        return None  # Return None to indicate skipping this nprof

    # Normalize profile
    rho_norm = normalize(profile['sigma0'], 10, 100)
    
    # Calculate density layer thickness
    rho_0 = np.nanmin(rho_norm)
    rholog = np.log(rho_norm / rho_0)
    profile['h_d'] = profile['Depth_[m]'] / rholog
    
    # Calculate Coriolis parameter
    profile['f'] = gsw.f(profile['Latitude_[deg_N]'])
    
    # Calculate Potential Vorticity
    profile['PV'] = profile['f'] / profile['h_d']

    return profile

def compute_gradient(group):
    """
    Auxiliary function for calc_PV_e_fit. This function computes the gradient of potential vorticity (PV) with respect to depth for a given group of oceanographic profile data.
    
    Args:
        - group (pd.Dataframe): pandas DataFrame containing the oceanographic profile data for a specific group (e.g., a specific profile), with columns for depth and potential vorticity (PV).
        
    Returns:
        - gradient_PV: pandas Series containing the computed gradient of PV with respect to depth for the given group.
    """
    delta_depth = group['Depth_[m]'].diff()  # Compute difference in depth between adjacent rows
    delta_PV = group['PV'].diff()  # Compute difference in PV between adjacent rows
    gradient_PV = delta_PV / delta_depth  # Compute gradient as change in PV divided by change in depth

    # Handle boundary conditions within each group
    gradient_PV.iloc[0] = (group['PV'].iloc[1] - group['PV'].iloc[0]) / (group['Depth_[m]'].iloc[1] - group['Depth_[m]'].iloc[0])
    gradient_PV.iloc[-1] = (group['PV'].iloc[-1] - group['PV'].iloc[-2]) / (group['Depth_[m]'].iloc[-1] - group['Depth_[m]'].iloc[-2])

    return gradient_PV

def calc_PV(df, S_col='Absolute_Salinity_[PSU]', T_col='Conservative_Temperature_[deg_C]', lat_col='Latitude_[deg_N]', depth_col='Depth_[m]'):
    """
    Calculate Potential Vorticity (PV) using a rolling exponential fit to profiles of potential density (sigma0) and depth. 
    This function first calculates the potential density from salinity and temperature, 
    then applies the calculate_h_and_PV_from_rho function to each profile to compute the density layer thickness and PV. 
    Finally, it computes the gradient of PV with respect to depth for each profile.
    
    Args:
        - df: pandas DataFrame containing the oceanographic profile data with columns for temperature, salinity, depth, and latitude.
    
    Returns:
        - df: the input DataFrame with two additional columns 'PV' and 'dPVdz containing the calculated potential vorticity values and their gradients.
    """
    df['sigma0'] = gsw.sigma0(df[S_col],df[T_col])

    # Apply the function to each 'nprof' of each 'source'
    df = df.groupby(['Source','Profile_Number']).apply(calculate_h_and_PV_from_rho)
    df = df.drop(['Source','Profile_Number'], axis=1)
    df.reset_index(inplace=True)

    # Drop sigma0, h_d and f columns
    df.drop(columns=['sigma0','h_d','f'], inplace=True)

    # Group the DataFrame by profile
    grouped = df.groupby(['Source', 'Profile_Number'])

    # Compute the gradient of PV with respect to depth for each profile
    gradients = grouped.apply(compute_gradient)
    gradients = gradients.reset_index()
    gradients = gradients.rename(columns={0: 'gradient_PV'})
    df['dPVdz'] = gradients['gradient_PV']

    # Replace infinite values with NaN
    df['dPVdz'].replace([np.inf, -np.inf], np.nan, inplace=True)

    return df
    
def method34(df, seed=22, year_rng_0=[1980,2000], year_rng_1=[2000,2023], threshold_qtl_0=0.99, threshold_qtl_1=0.935):
    """
    Sampling method 34: Bin data into polar stereographic projection grid, analyze bin occupancy distribution (histogram method from Zheng (2023)), 
    and sample bins over the threshold. Sample also monthly distribution with cap on average bin size for uniformity.
    Returns the sampled dataframe.

    Args:
    - df: pandas DataFrame containing the oceanographic profile data with columns for temperature, salinity, depth, latitude, longitude, and date.
    - seed: random seed for reproducibility
    - year_rng_0: list with start and end year for the first time period (e.g., [1980, 2000])
    - year_rng_1: list with start and end year for the second time period (e.g., [2000, 2023])
    - threshold_qtl_0: quantile threshold for bin occupancy in the first time period (e.g., 0.99)
    - threshold_qtl_1: quantile threshold for bin occupancy in the second time period (e.g., 0.935)

    Returns:
    - df_sampled: pandas DataFrame containing the sampled oceanographic profile data after applying the binning and thresholding method.
    """

    # Break into two parts (1980-2000 and 2000-2023)
    df_80_00 = df[(df['year'] >= year_rng_0[0]) & (df['year'] < year_rng_0[1])]
    df_00_23 = df[(df['year'] >= year_rng_1[0]) & (df['year'] < year_rng_1[1])]

    # Find thresholds for each period using Peisong's histogram method in polar stereographic projection
    threshold_0 = profileNo_vs_xy_bins(df_80_00, n_x_bins=100, n_y_bins=100, show_threshold=True, threshold_quantile=0.99, projection='npstere', showfig=False)
    threshold_1 = profileNo_vs_xy_bins(df_00_23, n_x_bins=100, n_y_bins=100, show_threshold=True, threshold_quantile=0.935, projection='npstere', showfig=False)

    # Sample each period using the respective thresholds
    df_sampled_0 = sample_cap_bins_xy(df_80_00, thresh=threshold_0, n_x_bins=100, n_y_bins=100)
    df_sampled_1 = sample_cap_bins_xy(df_00_23, thresh=threshold_1, n_x_bins=100, n_y_bins=100)
    df_sampled = pd.concat([df_sampled_0, df_sampled_1], ignore_index=True)

    # Sample monthly distribution with cap on average bin size for uniformity
    monthly_counts = df_sampled.groupby('month').size()
    average_monthly_count = monthly_counts.mean()
    df_sampled = df_sampled.groupby('month').apply(lambda x: x.sample(n=int(average_monthly_count), random_state=seed) if len(x) > average_monthly_count else x).reset_index(drop=False)

    return df_sampled

def feature_histograms(df, feat_col='Conservative_Temperature_[deg_C]', label_col='gmm_label_6', feature='Temperature [°C]', x_min=None, x_max=None, save_fig=False, fig_name='feature_histograms.png'):
    """
    Plots histograms of a specified feature for each GMM cluster in the dataframe.
    
    Args:
        - df (pd.DataFrame): The dataframe containing the data.
        - feat_col (str): The column name of the feature to plot. Default is 'Conservative_Temperature_[deg_C]'.
        - label_col (str): The column name of the GMM cluster labels. Default is 'gmm_label_6'.
        - feature (str): The label for the x-axis of the plot. Default is 'Temperature [°C]'.
        - x_min (float): Minimum value for the x-axis. If None, it will be set to the minimum of the feature column.
        - x_max (float): Maximum value for the x-axis. If None, it will be set to the maximum of the feature column.

    Returns:
        - fig (matplotlib.figure.Figure): The matplotlib figure object containing the histograms for each GMM cluster, with the option of saving it as a PNG.
        
    """
    n_clusters = df[label_col].nunique()
    colors = plt.cm.viridis_r(np.linspace(0, 1, n_clusters))

    fig, axes = plt.subplots(n_clusters, 1, figsize=(22, 15), sharex=True)

    if n_clusters == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        cluster_data = df[df[label_col] == i][feat_col]
        ax.hist(cluster_data, bins=50, alpha=0.5, color=colors[i], label=f'Cluster {i}')
        ax.legend()
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    fig.supxlabel(feature)
    fig.supylabel('Frequency', x=0.01)
    fig.suptitle(feature + ' Histograms by GMM Cluster')

    if x_min is None:
        x_min = df[feat_col].min()
    if x_max is None:
        x_max = df[feat_col].max()
    axes[0].set_xlim(x_min, x_max)

    if save_fig:
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()

def autocorrelation_matrix(df: pd.DataFrame, max_lag: int = 20) -> pd.DataFrame:
    """
    Computes the lag-k autocorrelation for each numeric column at lags 0…max_lag in a pandas
    DataFrame.

    Args:
        - df (pd.Dataframe): DataFrame whose numeric columns will be analysed.
        - max_lag (int): Maximum lag to compute (inclusive). Defaults to 20.

    Returns:
        - acf_df : DataFrame of shape (n_numeric_features, max_lag+1).
                Index   = column names from df.
                Columns = lag values [0, 1, 2, …, max_lag].
                Values  = Pearson autocorrelation at that lag (always 1.0 at lag-0).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("DataFrame contains no numeric columns.")

    lags = range(max_lag + 1)
    acf_values = {}

    for col in numeric_cols:
        series = df[col].dropna()
        acf_values[col] = [series.autocorr(lag=lag) for lag in lags]

    acf_df = pd.DataFrame(acf_values, index=list(lags)).T
    acf_df.columns.name = "lag"
    return acf_df

def plot_autocorrelation_matrix(
    acf_df: pd.DataFrame,
    title: str = "Autocorrelation Matrix",
    figsize: tuple = (14, 6),
    cmap: str = "coolwarm",
    annot: bool = True,
    fmt: str = ".2f",
) -> plt.Figure:
    """
    Auxiliary function to autocorrelation_matrix. Plots the autocorrelation matrix as a heatmap.

    Args:
    - acf_df (pd.DataFrame): Output of autocorrelation_matrix().
    - title (str): Plot title.
    - figsize (tuple): (width, height) in inches.
    - cmap (str): Matplotlib colormap name.
    - annot (bool): Whether to annotate cells with numeric values.
    - fmt (str): Format string for annotations.

    Returns:
        - fig : Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Diverging normalisation centred on 0
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    sns.heatmap(
        acf_df,
        ax=ax,
        cmap=cmap,
        norm=norm,
        annot=annot,
        fmt=fmt,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Autocorrelation", "shrink": 0.8},
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Lag", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    ax.tick_params(axis="x", labelrotation=0)
    ax.tick_params(axis="y", labelrotation=0)

    plt.tight_layout()
    return fig

def cross_sec_lon_prob3(
    df,
    n_clusters,
    lon_value,
    tol=0.5,
    start_lat=70.0,
    start_lon=-140.0,
    max_depth=850,
    point_size=8,
    browser=False,
    save_fig=False,
    fig_name='cross_section_prob_plot_raw_points.png'):
    """
    Plots the point cloud of GMM cluster probabilities along a cross-section through the North Pole, colored by probability values, without binning points.
    
    Args:
        - df (pd.DataFrame): The sampled dataset containing latitude, longitude, depth, and clustering probabilities.
        - n_clusters (int): Number of clusters in the GMM model.
        - lon_value (float): The longitude value for the cross-section slice (in degrees East).
        - tol (float): Tolerance for selecting points around the specified longitude (in degrees). Default is 0.5.
        - start_lat (float): Latitude of the starting point for distance calculation (in degrees North). Default is 70.0.
        - start_lon (float): Longitude of the starting point for distance calculation (in degrees East). Default is -140.0.
        - max_depth (float): Maximum depth to plot (in meters). Default is 850.
        - point_size (int): Size of the scatter plot points. Default is 8.
        - browser (bool): If True, the plot will be opened in a web browser, this saves memory when running in a notebook. Default is False.
    
    Returns:
        - fig (matplotlib.figure.Figure): The matplotlib figure object containing the cross-section scatter plot with raw GMM cluster probabilities, 
        with the option of displaying it in a browser or saving it as a PNG.
    """

    # Antipodal longitude for the over-pole extension
    if lon_value < 0 and lon_value >= -180:
        lon_value = lon_value + 180

    # Primary slice from start_lon and secondary for lon_value intersecting at the pole
    df_primary = df[
        (df['Longitude_[deg_E]'] >= start_lon - tol) &
        (df['Longitude_[deg_E]'] <= start_lon + tol)
    ].copy()

    df_antipodal = df[
        (df['Longitude_[deg_E]'] >= lon_value - tol) &
        (df['Longitude_[deg_E]'] <= lon_value + tol)
    ].copy()

    df_lon_slice = pd.concat([df_primary, df_antipodal], ignore_index=True)

    # Compute great-circle distance from start point
    df_lon_slice['dist_km'] = compute_great_circle_distance_km(
        start_lat, start_lon,
        df_lon_slice['Latitude_[deg_N]'].values,
        df_lon_slice['Longitude_[deg_E]'].values
    )

    # No binning, no averaging, no griddata, no smoothing — plot raw points.

    prob_cols = [f'prob_cluster_{i}' for i in range(n_clusters)]

    # Figure
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(
        int(np.floor(n_clusters / 2)), int(np.floor(n_clusters / 2)),
        width_ratios=[1, 1, 0.1], wspace=0.4, hspace=0.5
    )

    axes = []
    for i in range(int(np.floor(n_clusters / 2))):
        for j in range(int(np.floor(n_clusters / 3))):
            axes.append(fig.add_subplot(gs[i, j]))
    cax = fig.add_subplot(gs[:, 2])

    # Plot each field
    sc = None
    for i, ax in enumerate(axes):
        col = f'prob_cluster_{i}'

        sc = ax.scatter(
            df_lon_slice['dist_km'],
            df_lon_slice['Depth_[m]'],
            c=df_lon_slice[col] * 100,
            cmap='viridis',
            vmin=0,
            vmax=100,
            s=point_size,
            edgecolors='none'
        )

        # Pole marker
        ax.axvline(
            2200,
            color='white',
            linestyle='--',
            linewidth=1
        )

        # Formatting
        ax.set_ylim(max_depth, 0)
        ax.set_title(f'Cluster {i} Probability')
        ax.set_xlabel('Distance (km) from 70°N, 140°W')
        ax.set_ylabel('Depth (m)')

    # Shared colorbar (built from the last scatter's mappable)
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label('Probability (%)', fontsize=12)

    # Title
    fig.suptitle(
        'GMM Cluster Probabilities — Cross Section through North Pole (raw points)',
        fontsize=18,
        y=0.98
    )

    if browser:
        plt.show(renderer='browser')
    elif save_fig:
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def cross_sec_lon_var(
    df,
    n_clusters,
    lon_value,
    tol=0.5,
    start_lat=70.0,
    start_lon=-140.0,
    max_depth=850,
    depth_bin_size=10,
    dist_bin_size=2.2,
    browser=False,
    save_fig=False,
    fig_name='cross_section_var_plot.png'):
    """
    Plots the variance per point accross ensemble members, binned in the same way as function cross_sec_lon_prob2. 
    Also uses the same transsect plotting method.

    Args:
        - df (pd.DataFrame): The sampled dataset containing latitude, longitude, depth, and clustering variance.
        - n_clusters (int): Number of clusters in the GMM model.
        - lon_value (float): The longitude value for the cross-section slice (in degrees East).
        - tol (float): Tolerance for selecting points around the specified longitude (in degrees). Default is 0.5.
        - start_lat (float): Latitude of the starting point for distance calculation (in degrees North). Default is 70.0.
        - start_lon (float): Longitude of the starting point for distance calculation (in degrees East). Default is -140.0.
        - max_depth (float): Maximum depth to plot (in meters). Default is 850.
        - depth_bin_size (float): Size of depth bins for averaging variances (in meters). Default is 10.
        - dist_bin_size (float): Size of distance bins for averaging variances (in kilometers). Default is 2.2.
        - browser (bool): If True, the plot will be opened in a web browser, this saves memory when running in a notebook. Default is False.
        - save_fig (bool): Whether to save the figure as a PNG file. Default is False.
        - fig_name (str): Filename for saving the figure if save_fig is True. Default is 'cross_section_var_plot.png'.

    Returns:
        - fig (matplotlib.figure.Figure): The matplotlib figure object containing the cross-section scatter plot with averaged cluster variances, 
        with the option of saving as a PNG.
    
    """
    # Antipodal longitude for the over-pole extension
    if lon_value < 0 and lon_value >= -180:
        lon_value = lon_value + 180
        
    # Primary slice from start_lon and secondary for lon_value intersecting at the pole
    df_primary = df[
        (df['Longitude_[deg_E]'] >= start_lon - tol) &
        (df['Longitude_[deg_E]'] <= start_lon + tol)
    ].copy()

    df_antipodal = df[
        (df['Longitude_[deg_E]'] >= lon_value - tol) &
        (df['Longitude_[deg_E]'] <= lon_value + tol)
    ].copy()

    df_lon_slice = pd.concat([df_primary, df_antipodal], ignore_index=True)

    # Compute great-circle distance from start point
    df_lon_slice['dist_km'] = compute_great_circle_distance_km(
        start_lat, start_lon,
        df_lon_slice['Latitude_[deg_N]'].values,
        df_lon_slice['Longitude_[deg_E]'].values
    )

    # Bin coordinates
    df_lon_slice['depth_bin'] = (
        np.floor(df_lon_slice['Depth_[m]'] / depth_bin_size)
        * depth_bin_size
    )

    df_lon_slice['dist_bin'] = (
        np.floor(df_lon_slice['dist_km'] / dist_bin_size)
        * dist_bin_size
    )

    # Average probabilities within bins
    prob_cols = [f'var_per_sample_{i}' for i in range(n_clusters)]

    df_binned = (
        df_lon_slice
        .groupby(['dist_bin', 'depth_bin'])[prob_cols]
        .mean()
        .reset_index()
    )

    # Regular plotting grid
    xi = np.arange(
        df_binned['dist_bin'].min(),
        df_binned['dist_bin'].max(),
        dist_bin_size
    )

    yi = np.arange(
        0,
        max_depth + depth_bin_size,
        depth_bin_size
    )

    XI, YI = np.meshgrid(xi, yi)

    # Figure
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(int(np.floor(n_clusters/2)), int(np.floor(n_clusters/2)), 
                          width_ratios=[1, 1, 0.1], wspace=0.4, hspace=0.5)

    axes = []
    for i in range(int(np.floor(n_clusters/2))):
        for j in range(int(np.floor(n_clusters/3))):
            axes.append(fig.add_subplot(gs[i, j]))
    
    cax = fig.add_subplot(gs[:, 2])
    # Plot each field
    for i, ax in enumerate(axes):

        col = f'var_per_sample_{i}'
        # Interpolate onto regular grid
        ZI = griddata(
            (
                df_binned['dist_bin'],
                df_binned['depth_bin']
            ),
            df_binned[col] * 100,
            (XI, YI),
            method='linear'
        )

        # Smooth field
        ZI = gaussian_filter(ZI, sigma=1.0)

        # Plot
        pcm = ax.pcolormesh(
            XI,
            YI,
            ZI,
            shading='auto',
            cmap='viridis',
            vmin=0,
            vmax=np.nanmax(ZI))

        # Pole marker
        ax.axvline(
            2200,
            color='white',
            linestyle='--',
            linewidth=1)

        # Formatting
        ax.set_ylim(max_depth, 0)
        ax.set_title(f'Cluster {i} Variance')
        ax.set_xlabel(f'Distance (km) from 70°N, 140°W')
        ax.set_ylabel('Depth (m)')

    # Shared colorbar
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.set_label('Variance', fontsize=12)

    # Title
    fig.suptitle(
        'GMM Cluster Variance — Cross Section through North Pole',
        fontsize=18,
        y=0.98
    )

    if browser:
        plt.show(renderer='browser')
    elif save_fig:
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def cross_sec_lon_var2(
    df_sampled,
    n_clusters,
    lon_value,
    tol=0.5,
    start_lat=70.0,
    start_lon=-140.0,
    max_depth=850,
    point_size=8,
    browser=False,
    save_fig=False,
    fig_name='cross_section_var_plot_raw_points.png'):
    """
    Similar to the cross_sec_lon_var function, but instead of binning and averaging the variance values, this function plots each sample point 
    directly as a scatter point colored by its own variance value. This allows for a more detailed view of the variance distribution across the 
    cross-section without any smoothing or interpolation.

    Args:
        - df_sampled (pd.DataFrame): The sampled dataset containing latitude, longitude, depth, and clustering variance.
        - n_clusters (int): Number of clusters in the GMM model.
        - lon_value (float): The longitude value for the cross-section slice (in degrees East).
        - tol (float): Tolerance for selecting points around the specified longitude (in degrees). Default is 0.5.
        - start_lat (float): Latitude of the starting point for distance calculation (in degrees North). Default is 70.0.
        - start_lon (float): Longitude of the starting point for distance calculation (in degrees East). Default is -140.0.
        - max_depth (float): Maximum depth to plot (in meters). Default is 850.
        - point_size (int): Size of the scatter plot points. Default is 8.
        - browser (bool): If True, the plot will be opened in a web browser, this saves memory when running in a notebook. Default is False.
        - save_fig (bool): Whether to save the figure as a PNG file. Default is False.
        - fig_name (str): Filename for saving the figure if save_fig is True. Default is 'cross_section_var_plot.png'.
    
    Returns:
        - fig (matplotlib.figure.Figure): The matplotlib figure object containing the cross-section scatter plot with raw cluster variances,
        with the option of displaying it in a browser or saving it as a PNG.
    
    """

    # Antipodal longitude for the over-pole extension
    if lon_value < 0 and lon_value >= -180:
        lon_value = lon_value + 180

    # Primary slice from start_lon and secondary for lon_value intersecting at the pole
    df_primary = df_sampled[
        (df_sampled['Longitude_[deg_E]'] >= start_lon - tol) &
        (df_sampled['Longitude_[deg_E]'] <= start_lon + tol)
    ].copy()

    df_antipodal = df_sampled[
        (df_sampled['Longitude_[deg_E]'] >= lon_value - tol) &
        (df_sampled['Longitude_[deg_E]'] <= lon_value + tol)
    ].copy()

    df_lon_slice = pd.concat([df_primary, df_antipodal], ignore_index=True)

    # Compute great-circle distance from start point
    df_lon_slice['dist_km'] = compute_great_circle_distance_km(
        start_lat, start_lon,
        df_lon_slice['Latitude_[deg_N]'].values,
        df_lon_slice['Longitude_[deg_E]'].values
    )

    # Figure
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(
        int(np.floor(n_clusters / 2)), int(np.floor(n_clusters / 2)),
        width_ratios=[1, 1, 0.1], wspace=0.4, hspace=0.5
    )

    axes = []
    for i in range(int(np.floor(n_clusters / 2))):
        for j in range(int(np.floor(n_clusters / 3))):
            axes.append(fig.add_subplot(gs[i, j]))
    cax = fig.add_subplot(gs[:, 2])

    # Plot each field
    sc = None
    for i, ax in enumerate(axes):
        col = f'var_per_sample_{i}'

        sc = ax.scatter(
            df_lon_slice['dist_km'],
            df_lon_slice['Depth_[m]'],
            c=df_lon_slice[col] * 100,
            cmap='viridis',
            vmin=0,
            vmax=1,
            s=point_size,
            edgecolors='none'
        )

        # Pole marker
        ax.axvline(
            2200,
            color='white',
            linestyle='--',
            linewidth=1
        )

        # Formatting
        ax.set_ylim(max_depth, 0)
        ax.set_title(f'Cluster {i} Variance')
        ax.set_xlabel('Distance (km) from 70°N, 140°W')
        ax.set_ylabel('Depth (m)')

    # Shared colorbar (built from the last scatter's mappable)
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label('Variance', fontsize=12)

    # Title
    fig.suptitle(
        'GMM Cluster Variances — Cross Section through North Pole (raw points)',
        fontsize=18,
        y=0.98
    )

    if browser:
        plt.show(renderer='browser')
    elif save_fig:
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def fit_member(seed, n_clusters, X):
    """
    Auxiliary function for gmm_ensemble. Fits a Gaussian Mixture Model (GMM) to the input data X 
    using a specified random seed and number of clusters.

    Args:
        - seed (int): Random seed for reproducibility.
        - n_clusters (int): Number of clusters for the GMM.
        - X (np.ndarray): Input data array to fit the GMM.
    
    Returns:
        - gmm (GaussianMixture): The fitted GMM object.
        - labels (np.ndarray): Predicted cluster labels for the input data X, as int8.
    """
    gmm = GaussianMixture(
        n_components=n_clusters,   # number of clusters
        covariance_type='full',    # full cov matrix has more flexibility
        random_state=seed,
    )
    gmm.fit(X)
    return gmm, gmm.predict(X).astype(np.int8)

def gmm_ensemble(df, n_clusters, features, ensemble_size=100, random_state=None):
    """
    This is the main function of the project. Parallely, for each random seed initialisation, 
    this function fits a GMM to the data given as a pandas DataFrame. An ensemble_size GMMs are fitted, 
    and the resulting cluster labels and probabilities are stored in the DataFrame. Each worker also predicts 
    labels on the full data and returns them as int8 (K <= 127), keeping the transfer back small.
    
    Args:
        - df (pd.DataFrame): The input DataFrame containing the data to be clustered (should go through method_34 before input here).
        - n_clusters (int): The number of clusters for the GMM.
        - features (list of str): List of column names in the DataFrame to be used as input features for clustering.
        - ensemble_size (int): The number of GMMs to fit in the ensemble. Default is 100.
        - random_state (int or None): Random seed for reproducibility. Default is None.
        
    Returns:
        - gmms (list of GaussianMixture): List of fitted GMM objects for each ensemble member.
        - labels (list of np.ndarray): List of predicted cluster labels for each ensemble member, as int8 arrays.
    """

    # create list of seeds
    seeds = random.sample(range(1, 10000), ensemble_size)

    X = df[features].to_numpy()
    tmp = tempfile.mktemp(suffix='.npy')
    np.save(tmp, X)
    X_memmap = np.load(tmp, mmap_mode='r')
    del X  # free the in-memory copy

    print(f"Fitting {len(seeds)} ensemble members on full data {X_memmap.shape} at K={n_clusters}")
    print(f"CPU cores available: {cpu_count()}")

    try:
        fitted = Parallel(n_jobs=-1, backend="loky")(
            delayed(fit_member)(seed, n_clusters, X_memmap)
            for seed in tqdm.tqdm(seeds, desc="Fitting members", unit="member")
        )
    finally:
        os.remove(tmp)  # always clean up temp file

    # A list of fitted GMMs and per-member labels.
    gmms   = [g for g, _ in fitted]
    labels = [lab for _, lab in fitted]

    print(f"Done. {len(gmms)} members fitted at K={n_clusters}.")

    return gmms, labels

def match_labels(df, gmms, labels, ensemble_size, n_clusters):
    """
    Auxiliary function for gmm_ensemble. This function matches the labels of each ensemble member to a reference member (the first one) 
    using the Hungarian algorithm. It returns a DataFrame containing the matched labels for each ensemble member.

    Args:
        - df (pd.DataFrame): The input DataFrame containing the data to be clustered 
        (that has now been clustered but needs postprocessing to add label columns).
        - gmms (list): List of fitted GMM objects for each ensemble member.
        - labels (list): List of predicted cluster labels for each ensemble member.
        - ensemble_size (int): The number of GMMs in the ensemble.
        - n_clusters (int): The number of clusters for the GMM.

    Returns:
        - df (pd.DataFrame): The input DataFrame with additional columns for matched labels of each ensemble member,
    """
    # Clusters with similar mean T, S and D (depth) should be matched across different seeds
    cluster_means = []
    for i in range(ensemble_size):
        means = gmms[i].means_
        cluster_means.append(means)

    # Match clusters across different seeds based on closest mean T, S and D (depth)
    matched_labels = []
    for i in range(ensemble_size):
        means = cluster_means[i]
        matched = np.zeros(n_clusters, dtype=int)
        for j in range(n_clusters):
            mean_T = means[j, 0]
            mean_S = means[j, 1]
            min_dist = np.inf
            match = -1
            for k in range(n_clusters):
                dist = np.sqrt((mean_T - cluster_means[0][k, 0])**2 + (mean_S - cluster_means[0][k, 1])**2)
                if dist < min_dist:
                    min_dist = dist
                    match = k
            matched[j] = match
        matched_labels.append(matched)

    for i in range(ensemble_size):
        df['gmm_label_'+str(i)] = matched_labels[i][labels[i]]
    
    return df, matched_labels

def gmm_probs(df, gmms, ensemble_size, n_clusters, matched_labels, features):
    """
    Auxiliary function for gmm_ensemble. This function computes the probabilities of each data point belonging to each cluster 
    for each ensemble member and adds these probabilities as new columns in the DataFrame.

    Args:
        - df (pd.DataFrame): The input DataFrame containing the data to be clustered.
        - gmms (list): List of fitted GMM objects for each ensemble member.
        - ensemble_size (int): The number of GMMs in the ensemble.
        - n_clusters (int): The number of clusters for the GMM.

    Returns:
        - df (pd.DataFrame): The input DataFrame with additional columns for probabilities of each data point belonging to each cluster 
        for each ensemble member.
    """
    # Feature matrix
    X_feat = df[features].to_numpy()


    def _gmm_proba(gmm, matched, X, n_clusters):
        """Return predict_proba aligned to reference clusters: (n_samples, n_clusters)."""
        proba = gmm.predict_proba(X)
        aligned = np.zeros_like(proba)
        for j in range(n_clusters):
            aligned[:, matched[j]] += proba[:, j]  # place member cluster j into its reference column
        return aligned


    proba_sum = np.zeros((len(df), n_clusters))
    batch_size = 4  # tune this

    batches = [range(ensemble_size)[i:i + batch_size]
            for i in range(0, ensemble_size, batch_size)]

    for batch in tqdm.tqdm(batches, desc="GMM batches", unit="batch"):
        batch_results = Parallel(n_jobs=len(batch), backend="loky")(
            delayed(_gmm_proba)(gmms[i], matched_labels[i], X_feat, n_clusters)
            for i in batch
        )
        proba_sum += np.sum(batch_results, axis=0)

    proba_mean = proba_sum / ensemble_size  # (n_samples, n_clusters), rows sum to 1

    for j in tqdm.tqdm(range(n_clusters), desc="Writing clusters", unit="cluster"):
        df[f'prob_mean_cluster_{j}'] = proba_mean[:, j]