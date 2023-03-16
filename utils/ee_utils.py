import requests
import os
import ee
import json
from datetime import date, timedelta


def getImage(collection):
    '''
    Query single Image from GEE
    :param string collection: Image GEE-dataset name
    :return tiff: selected Image
    '''
    data = ee.Image(collection)
    return data
      
def getYearlyData(collection, year):
    '''Returns a list of monthly cropped images
    Args:
        year (string): use 4 digits typo ex : '2021'
    '''
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    
    data_list = []
    for m in months:
        data_list.append( getMonthlyData(collection, year, m) )
    return data_list


def download(city, variable, year=None):
    bounding_box = json.load(open('../bounding_box.json'))
    box = bounding_box[city]
    [(ymax, xmin), (ymin, xmax)] = box['box']
    margin = 0.05
    region = ee.Geometry.BBox(xmin-margin, ymin-margin, xmax+margin, ymax+margin)
    variables = json.load(open('config.json'))
    collection = variables['datasets'][variable]['name']
    bands = variables['datasets'][variable]['bands']
    if year is None:
        data = getImage(collection)
        downloadAsLink(data, city, region, bands, variable)
        return
    data_list = getYearlyData(collection, year)
    for data in data_list:
        downloadAsLink(data, city, region, bands, variable, year=year)
        

def downloadAsLink( data, city, region, bands, variable, year=None):
    for i in range(12):
        if year is None:
            name = str(city)+'.tif'
        else: name = str(city)+'_'+year+'.tif'
        url=data.getDownloadUrl({
            'name': name,
            'bands': bands,
            'region':region,
            'format':'GEO_TIFF',
            'scale':1000
        })
        path = os.path.join("..", "data", variable, "")
        with open(path+name, 'wb') as out_file:
            content = requests.get(url, stream=True).content
            out_file.write(content)
        if year is None: return
    return
