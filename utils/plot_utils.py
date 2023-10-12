# UTILS/PLOT_UTILS.PY
#
# DESCRIPTION: Contains useful functions for plotting results, and subfunctions to handle result datasets
#
# @MDC, 2023

# IMPORTS
from __future__ import annotations
from typing import Any, Iterable, Optional
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from utils import utils


def extract_crossval_results(results:dict, var:str, new_var:str, to_cpu=True)->dict:
    """Formats result dictionary from training for single-variable plotting functions

    Args:
        results (dict): training results dictionary
        var (str): variable to extract, default to 'test_r2'
        new_var (str): variable name in new dictionary
        to_cpu (bool, optional): Flag to send data back to cpu if it was not done earlier, frees GPU memory. Defaults to True.

    Returns:
        extracted_results (dict): for a single var, per model, per epoch.
    """
    extracted_results = dict()
    extracted_results[new_var]=[]
    extracted_results['fold']=[]
    extracted_results['epoch']=[]
    cpt_epoch = 0
    for fold in results:
        if to_cpu:
            extracted_list = [t.cpu().numpy()[()] for t in results[fold][var]]
        else : extracted_list = [t[()] for t in results[fold][var]]
        extracted_fold = [fold for i in range(len(extracted_list))]
        extracted_results[new_var] += extracted_list
        extracted_results['fold'] += extracted_fold
        extracted_results['epoch'] += [i-cpt_epoch for i in range(len(extracted_list))]
    return extracted_results


def split_regplot(
        data: pd.DataFrame,
        col_to_split: str,
        newcol_name: str,
        split_val: str,
        labelsup: str,
        labelinf: str,
        pal,
        scatter_kws=None,
        line_kws=None,
    ):
    """Displays a duo of regression plots from a single dataset, splitting each one given a single variable threshold value.

    Args:
        data (pd.DataFrame): test dataset
        col_to_split (str): threshold variable 
        newcol_name (str): variable name displayed with plot
        split_val (str): threshold value
        labelsup (str): label if above threshold
        labelinf (str): label if below threshold
        scatter_kws (_type_, optional): seaborn kwargs. Defaults to None.
        line_kws (_type_, optional): seaborn kwargs. Defaults to None.
    """
    plt.figure(figsize=(10, 8))
    data[newcol_name] = data.apply(lambda x: labelsup if x[col_to_split] >= split_val else labelinf, axis=1)
    
    r2_sup = r2_score(data[data[newcol_name] == labelsup]["True Wealth"], data[data[newcol_name] == labelsup]["Predicted Wealth"])
    r2_inf = r2_score(data[data[newcol_name] == labelinf]["True Wealth"], data[data[newcol_name] == labelinf]["Predicted Wealth"])
    
    ax = sns.regplot(x="True Wealth", y="Predicted Wealth", data=data[data[newcol_name] == labelsup], x_ci='sd', marker='o', scatter_kws={'alpha': 0.5, 'color': pal.as_hex()[3], 's': 10}, line_kws={'color': pal.as_hex()[4]})
    ax = sns.regplot(x="True Wealth", y="Predicted Wealth", data=data[data[newcol_name] == labelinf], x_ci='sd', marker='o', scatter_kws={'alpha': 0.5, 'color': pal.as_hex()[1], 's': 10}, line_kws={'color': pal.as_hex()[0]})

    # Adjust the y-coordinate of the labels to avoid overlap
    max_y = max(data["Predicted Wealth"].max(), data["True Wealth"].max())
    plt.text(-1.5, max_y - 3.9, 'Urban R2 = ' + str(round(r2_sup, 4)), fontsize='18', weight='bold', color=pal.as_hex()[3])
    plt.text(-1.5, max_y - 1.1, 'Rural R2 = ' + str(round(r2_inf, 4)), fontsize='18', weight='bold', color=pal.as_hex()[1])
    
    plt.plot()

    print(f'r2 urban = {r2_sup}')
    print(f'r2 rural = {r2_inf}')

    return



def country_plot(
        bg_map:gpd.GeoDataFrame,
        data:pd.DataFrame,
        cmap:str,
        bg_color:str,
        edgecolor:str
    ):
    """Maps R2 score per country

    Args:
        bg_map (gpd.GeoDataFrame): Africa shapefile
        data (pd.DataFrame): test dataset
        cmap (str): colormap (matplotlib format)
        bg_color (str): shapefile background color
        edgecolor (str): country boundaries color
    """
    data['country'] = data['country'].apply(utils.standardize_countryname)
    data = data.rename({'country':'ADM0_NAME'},axis='columns')
    # Compute Average R2 per Country
    country_wise = pd.DataFrame()
    country_wise['ADM0_NAME'] = data['ADM0_NAME'].unique()
    for country in country_wise.ADM0_NAME.unique():
        r2 = r2_score(data[data.ADM0_NAME==country]['True Wealth'], data[data.ADM0_NAME==country]['Predicted Wealth'])
        country_wise.loc[country_wise['ADM0_NAME'] == country, ["R2"]] = r2
    base = bg_map.plot(color=bg_color, edgecolor=edgecolor)
    base.set_axis_off()
    countryplot = bg_map.merge(country_wise, on='ADM0_NAME', how='left')
    plt.figure(figsize=(8,10))
    countryplot.plot(
        column='R2', 
        scheme="quantiles",
        k=12,
        edgecolor="k",
        ax=base, 
        legend=True, 
        cmap=cmap,
        figsize=(50,50),
        legend_kwds={'bbox_to_anchor': (0.35, 0.55), "fontsize":"6"}
        )
    return


def setup_ax(fig: matplotlib.figure.Figure,
             pos: tuple[int, int, int] = (1, 1, 1),
             ) -> matplotlib.axes.Axes:
    '''
    Args
    - fig: matplotlib.figure.Figure
    - pos: 3-tuple of int, axes position (nrows, ncols, index)

    Returns: matplotlib.axes.Axes
    '''
    ax = fig.add_subplot(*pos, projection=ccrs.PlateCarree())

    # draw land (better version of cfeature.LAND)
    # land = cfeature.NaturalEarthFeature(
    #     category='physical', name='land', scale='10m',
    #     edgecolor='face', facecolor=cfeature.COLORS['land'], zorder=-1)
    ax.add_feature(cfeature.LAND)

    # draw borders of countries (better version of cfeature.BORDERS)
    countries = cfeature.NaturalEarthFeature(
        category='cultural', name='admin_0_boundary_lines_land', scale='10m',
        edgecolor='black', facecolor='none')
    ax.add_feature(countries)

    # draw coastline (better version of cfeature.COASTLINE)
    coastline = cfeature.NaturalEarthFeature(
        category='physical', name='coastline', scale='10m',
        edgecolor='black', facecolor='none')
    ax.add_feature(coastline)

    # draw lakes (better version of cfeature.LAKES)
    lakes = cfeature.NaturalEarthFeature(
        category='physical', name='lakes', scale='10m',
        edgecolor='face', facecolor=cfeature.COLORS['water'])
    ax.add_feature(lakes)

    # draw ocean (better version of cfeature.OCEAN)
    ocean = cfeature.NaturalEarthFeature(
        category='physical', name='ocean', scale='50m',
        edgecolor='face', facecolor=cfeature.COLORS['water'], zorder=-1)
    ax.add_feature(ocean)

    # draw rivers (better version of cfeature.RIVERS)
    rivers = cfeature.NaturalEarthFeature(
        category='physical', name='rivers_lake_centerlines', scale='10m',
        edgecolor=cfeature.COLORS['water'], facecolor='none')
    ax.add_feature(rivers)

    # draw borders of states/provinces internal to a country
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural', name='admin_1_states_provinces_lines', scale='50m',
        edgecolor='gray', facecolor='none')
    ax.add_feature(states_provinces)

    ax.set_aspect('equal')
    gridliner = ax.gridlines(draw_labels=True)
    gridliner.top_labels = False
    gridliner.right_labels = False
    return ax



def plot_locs(locs: np.ndarray,
              fig: Optional[matplotlib.figure.Figure] = None,
              pos: tuple[int, int, int] = (1, 1, 1),
              figsize: tuple[int, int] = (15, 15),
              title: Optional[str] = None,
              colors: Optional[Iterable[int]] = None,
              cbar_label: Optional[str] = None,
              show_cbar: bool = True,
              **scatter_kws: Any
              ) -> matplotlib.axes.Axes:
    '''
    Args
    - locs: np.array, shape [N, 2], each row is [lat, lon]
    - fig: matplotlib.figure.Figure
    - pos: 3-tuple of int, axes position (nrows, ncols, index)
    - figsize: list, [width, height] in inches, only used if fig is None
    - title: str, title of axes
    - colors: list of int, length N
    - cbar_label: str, label for the colorbar
    - show_cbar: bool, whether to show the colorbar
    - scatter_kws: other arguments for ax.scatter

    Returns: matplotlib.axes.Axes
    '''
    if fig is None:
        fig = plt.figure(figsize=figsize)
    ax = setup_ax(fig, pos)
    if title is not None:
        ax.set_title(title)

    if 's' not in scatter_kws:
        scatter_kws['s'] = 2
    pc = ax.scatter(locs[:, 1], locs[:, 0], c=colors, **scatter_kws)
    if colors is not None and show_cbar:
        cbar = fig.colorbar(pc, ax=ax, fraction=0.03)
        if cbar_label is not None:
            cbar.set_label(cbar_label)
    return ax