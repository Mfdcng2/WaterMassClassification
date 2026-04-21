import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
import plotly.graph_objs as go
from collections import defaultdict
import plotly.express as px

# Sample bins of bins (geographical and depth bins) in projection space
def sample_bins_of_bins_proj(df, n_x_bins=100, n_y_bins=100, n_depth_bins=5, projection='npstere', min_flag=True):
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
        sampled_df = df.groupby(['x_bin', 'y_bin', 'depth_bin']).apply(lambda x: x.sample(min_count, random_state=22)).reset_index(drop=True)
    else:
        avg_count = df.groupby(['x_bin', 'y_bin', 'depth_bin']).size().mean().round().astype(int)
        sampled_df = (df.groupby(['x_bin', 'y_bin', 'depth_bin'], group_keys=False).apply(lambda g: g.sample(n=min(len(g), avg_count), random_state=22)).reset_index(drop=True))
    return sampled_df

def profileNo_vs_xy_bins(
    df: pd.DataFrame,
    n_x_bins: int = 100,
    n_y_bins: int = 100,
    show_threshold: bool = False,
    threshold_quantile: float = 0.95,
    projection: str = 'npstere'
) -> None:
    """
    Plot the number of profiles in each bin using polar stereographic projection.
    Args:
        df (pd.DataFrame): The dataset to plot.
        n_x_bins (int): Number of bins in the x direction. Defaults to 100.
        n_y_bins (int): Number of bins in the y direction. Defaults to 100.
        show_threshold (bool, optional): Whether to show the threshold marker. Defaults to False.
        threshold_quantile (float, optional): The quantile value for threshold calculation. Defaults to 0.95.
        projection (str, optional): Basemap projection string. Defaults to 'npstere'.
    Returns:
        None
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
    fig.show()

    return g_len_sorted[transition_point]

def plot_geo(df, n_clusters=4):
    # Compute data centroid (mean location)
    center_lat = df['Latitude_[deg_N]'].mean()
    center_lon = df['Longitude_[deg_E]'].mean()

    colors = px.colors.sample_colorscale(
        px.colors.sequential.Plasma,
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
        color='gmm_label_4',
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

    return fig

def plot_heatmap_npts_xy(ds_cleaned, n_x_bins=100, n_y_bins=100, bin_max=None, projection='npstere', projection_name='Polar Stereographic'):
    lat_min = ds_cleaned['Latitude_[deg_N]'].min()
    resolution = 'l'

    fig = plt.figure(figsize=(8, 8), dpi=300)
    m = Basemap(projection=projection, boundinglat=lat_min, lon_0=0, resolution=resolution)

    # Convert lat/lon → x/y (projection coordinates in meters)
    lon_vals = ds_cleaned['Longitude_[deg_E]'].values
    lat_vals = ds_cleaned['Latitude_[deg_N]'].values

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

    return df.iloc[sampled_indices].copy()


def plot_heatmap_npts(ds_cleaned, lon_step=5, lat_step=5):
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

    lon_vals = ds_cleaned['Longitude_[deg_E]'].values
    lat_vals = ds_cleaned['Latitude_[deg_N]'].values

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