import seaborn as sns
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from utils import utils

def extract_crossval_results(results:dict, var:str, new_var:str)->dict:
    extracted_results = dict()
    extracted_results[new_var]=[]
    extracted_results['fold']=[]
    extracted_results['epoch']=[]
    cpt_epoch = 0
    for fold in results:
        extracted_list = [t.cpu().numpy()[()] for t in results[fold][var]]
        extracted_fold = [fold for i in range(len(extracted_list))]
        extracted_results[new_var] += extracted_list
        extracted_results['fold'] += extracted_fold
        extracted_results['epoch'] += [i-cpt_epoch for i in range(len(extracted_list))]
    return extracted_results

def split_regplot(
        data:pd.DataFrame,
        col_to_split:str, 
        newcol_name:str, 
        split_val:str, 
        labelsup:str,
        labelinf:str,
        scatter_kws=None,
        line_kws=None
        ):
    data[newcol_name]=data.apply(lambda x: labelsup if x.col_to_split>=split_val else labelinf, axis=1)
    r2_sup = r2_score(data[data.newcol_name==labelsup].wealthpooled, data[data.newcol_name==labelsup].predicted_wealth)
    r2_inf = r2_score(data[data.frenewcol_nameshness==labelinf].wealthpooled, data[data.newcol_name==labelinf].predicted_wealth)
    sns.lmplot(x="wealthpooled", y="predicted_wealth", hue=newcol_name, line_kws=line_kws, scatter_kws=scatter_kws, data=data)
    plt.text(-1.0,2.3, 'R2 = ' + str(round(r2_sup,4)), fontsize='large', weight='bold', color=sns.color_palette().as_hex()[1])
    plt.text(-1.0,2.0, 'R2 = ' + str(round(r2_inf,4)), fontsize='large', weight='bold', color=sns.color_palette().as_hex()[0])
    plt.plot()
    return


def country_plot(
        bg_map:gpd.GeoDataFrame,
        data:pd.DataFrame,
        cmap:str,
        bg_color:str,
        edgecolor:str
):
    data['country'] = data['country'].apply(utils.standardize_countryname)
    data = data.rename({'country':'ADM0_NAME'},axis='columns')
    # Compute Average R2 per Country
    country_wise = pd.DataFrame()
    country_wise['ADM0_NAME'] = data['ADM0_NAME'].unique()
    for country in country_wise.ADM0_NAME.unique():
        r2 = r2_score(data[data.ADM0_NAME==country]['wealthpooled'], data[data.ADM0_NAME==country]['predicted_wealth'])
        country_wise.loc[country_wise['ADM0_NAME'] == country, ["R2"]] = r2
    base = bg_map.plot(color=bg_color, edgecolor=edgecolor)
    base.set_axis_off()
    countryplot = bg_map.merge(country_wise, on='ADM0_NAME', how='left')
    countryplot.plot(
        column='R2', 
        scheme="quantiles",
        k=12,
        edgecolor="k",
        ax=base, 
        legend=True, 
        cmap=cmap,
        figsize=(50,50),
        legend_kwds={'loc': 'lower left', "fontsize":"8"}
        )
    return