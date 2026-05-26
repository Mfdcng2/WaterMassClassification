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
    projection: str = 'npstere',
    showfig: bool = True
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
        showfig (bool, optional): Whether to display the figure. Defaults to True.
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
    if showfig:
        fig.show()

    return g_len_sorted[transition_point]

def plot_geo(df, n_clusters=4):
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
        color='gmm_label_'+str(n_clusters),
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

def plot_3d(n_clusters, df_sampled):
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
        df_sampled,
        x='Latitude_[deg_N]',
        y='Longitude_[deg_E]',
        z='Depth_[m]',
        color='gmm_label_'+str(n_clusters),
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

    fig_3d.show()


# Cross section like Kate's
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Geod
from matplotlib.colors import ListedColormap, BoundaryNorm

def compute_great_circle_distance_km(lat_ref, lon_ref, lats, lons):
    """Compute distance in km from a reference point along a great circle."""
    geod = Geod(ellps='WGS84')
    _, _, dist_m = geod.inv(
        np.full(len(lats), lon_ref),
        np.full(len(lats), lat_ref),
        lons,
        lats
    )
    return dist_m / 1000.0


def cross_sec_lon(df_sampled, n_clusters, lon_value, tol=0.5, start_lat=70.0, start_lon=-140.0):

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
        alpha=0.6
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

    plt.tight_layout()
    plt.show()


def cross_sec_lon_prob(df_sampled, n_clusters, lon_value, tol=0.5, start_lat=70.0, start_lon=-140.0):
    # Antipodal longitude for the over-pole extension
    #lon_value = lon_value if lon_value <= 180 & lon_value >=0 else lon_value + 180
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

    plt.show()


from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

def cross_sec_lon_prob2(
    df_sampled,
    n_clusters,
    lon_value,
    tol=0.5,
    start_lat=70.0,
    start_lon=-140.0,
    max_depth=850,
    depth_bin_size=10,
    dist_bin_size=2.2
):
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

    plt.tight_layout()
    plt.show()

import gsw
def calc_pv(df):
    """
    Calculate potential vorticity (PV) for each profile in the dataframe.
    PV is calculated using the formula:
        PV = -f * N^2 / g
    where:
        - f is the Coriolis parameter (f = 2 * omega * sin(latitude))
        - N^2 is the buoyancy frequency squared, calculated from the vertical profiles of temperature and salinity
        - g is the acceleration due to gravity  
    The function adds a new column 'Pv' to the dataframe with the calculated potential vorticity values.
    
    Inputs:
    - df: pandas DataFrame containing the oceanographic profile data with columns for temperature, salinity, depth, and latitude.
    
    Outputs:
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

def method34(df, seed=22, year_rng_0=[1980,2000], year_rng_1=[2000,2023], threshold_qtl_0=0.99, threshold_qtl_1=0.935):
    """
    Sampling method 34: Bin data into polar stereographic projection grid, analyze bin occupancy distribution (Peisong's histogram method), 
    and sample bins over the threshold. Sample also monthly distribution with cap on average bin size for uniformity.
    Returns the sampled dataframe.

    Inputs:
    - df: pandas DataFrame containing the oceanographic profile data with columns for temperature, salinity, depth, latitude, longitude, and date.
    - seed: random seed for reproducibility
    - year_rng_0: list with start and end year for the first time period (e.g., [1980, 2000])
    - year_rng_1: list with start and end year for the second time period (e.g., [2000, 2023])
    - threshold_qtl_0: quantile threshold for bin occupancy in the first time period (e.g., 0.99)
    - threshold_qtl_1: quantile threshold for bin occupancy in the second time period (e.g., 0.935)

    Outputs:
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
    df_sampled = df_sampled.groupby('month').apply(lambda x: x.sample(n=int(average_monthly_count), random_state=seed) if len(x) > average_monthly_count else x).reset_index(drop=True)

    return df_sampled

from matplotlib.ticker import ScalarFormatter

def feature_histograms(df, feat_col='Conservative_Temperature_[deg_C]', label_col='gmm_label_6', feature='Temperature [°C]'):
    n_clusters = df[label_col].nunique()
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))

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

    x_min = df[feat_col].min()
    x_max = df[feat_col].max()
    axes[0].set_xlim(x_min, x_max)

    plt.tight_layout()
    plt.show()