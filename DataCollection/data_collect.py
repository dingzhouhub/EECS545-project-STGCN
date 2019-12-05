import pandas as pd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from datetime import datetime
from geopy.distance import distance

stationFile = 'station/d08_text_meta_2019_09_07.txt'
LATITUDE_MIN = 34
LATITUDE_MAX = 34.17
LONGTITUDE_MIN = -117.6
LONGTITUDE_MAX = -117.25
#weekdays in one month
days1 = [1,4,5,6,7,8] #<10
days2 = [11,12,13,14,15,18,19,20,21,22,25,26,27,28,29] #>=10

stationGeo = pd.read_csv(stationFile,sep='\t')

# creating a geometry column
geometry = [Point(xy) for xy in zip(stationGeo['Longitude'], stationGeo['Latitude'])]
# Coordinate reference system : WGS84
crs = {'init': 'epsg:4326'}
# Creating a Geographic data frame
gdf = gpd.GeoDataFrame(stationGeo, crs=crs, geometry=geometry)

#set plot parameters
plt.rcParams['figure.figsize'] = (12.0, 12.0)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 72

#plot original station
fig1, ax1 = plt.subplots(1, 1)
gdf.plot(ax=ax1, marker='o', color='b', markersize=0.5)
ax1.set_title('Original District 8 Station')

geovalue = [xy for xy in zip(stationGeo['Latitude'], stationGeo['Longitude'])]
gdf['geovalue'] = geovalue

df = pd.DataFrame()
for day in days1:
    df_temp = pd.read_csv('timestep/d08_text_station_5min_2019_11_0%a.txt.gz'%day,sep=',',header=None)
    df = pd.concat([df,df_temp])
for day in days2:
    df_temp = pd.read_csv('timestep/d08_text_station_5min_2019_11_%a.txt.gz'%day,sep=',',header=None)
    df = pd.concat([df,df_temp])
df = df.iloc[:,:17].rename(columns={0:'Timestamp', 1:'Station', 2:'District', 3:'Freeway', 4:'Direction', 5:'Lane', 6:'Length', 7:'Samples_1', 8:'Samples_2', 9:'TotalFlow', 10:'AvgOccu', 11:'AvgSpeed', 12:'LNSamples', 13:'LNFlow', 14:'LNAvgOccu', 15:'LNAvgSpeed', 16:'LNobserved'})
df_clean = df.dropna(subset=['AvgSpeed'])
dftosave = df_clean[['Timestamp','Station','TotalFlow','AvgOccu','AvgSpeed']]

#time convert
dftosave['convert_to_date'] = pd.to_datetime(dftosave['Timestamp'])
dftosave['Hour'] = dftosave['convert_to_date'].dt.hour
dftosave['Minute'] = dftosave['convert_to_date'].dt.minute
dftosave['Second'] = dftosave['convert_to_date'].dt.second
dftosave['Day'] = dftosave['convert_to_date'].dt.day
dftosave['Month'] = dftosave['convert_to_date'].dt.month
dftosave['Year'] = dftosave['convert_to_date'].dt.year
dftosave.to_csv('data.csv')

newgdf = gdf.set_index(gdf['ID'])
newgdf = newgdf.loc[(i for i in dftosave['Station'].unique()),:]
newgdf = newgdf.drop(columns='ID')
newgdf = newgdf.reset_index()

fig2, ax2 = plt.subplots(1, 1)
newgdf.plot(ax=ax2, marker='o', color='b', markersize=0.5)
ax2.set_title('District 8 Available Station')
ax2.set_xlabel('Longtitude')
ax2.set_ylabel('Latitude')

subStationGeo = gdf[(stationGeo['Latitude']>LATITUDE_MIN)
                   & (stationGeo['Latitude']<LATITUDE_MAX)
                   & (stationGeo['Longitude']>LONGTITUDE_MIN)
                   & (stationGeo['Longitude']<LONGTITUDE_MAX)]
subStationGeo = subStationGeo[['ID','geometry','geovalue']]

#part of data is missing in some station
time = len(dftosave.Timestamp.unique())
for i in subStationGeo['ID']:
    if (len(dftosave[dftosave['Station']==i]) != time):
        subStationGeo = subStationGeo.drop(subStationGeo[subStationGeo['ID']==i].index,axis=0)

#sample 200, to reduce computation cost
num_station = len(subStationGeo['ID'])
subStationGeo = subStationGeo.sample(frac = 200/num_station)

fig3, ax3 = plt.subplots(1, 1)
subStationGeo.plot(ax=ax3, marker='s', color='r', markersize=12)
ax3.set_title('District 8 Station')
ax3.set_xlabel('Longtitude')
ax3.set_ylabel('Latitude')

num_station = len(subStationGeo['ID'])
point = subStationGeo['geovalue'].tolist()
D_initial = []
for i in range(num_station):
    for j in range(num_station):
        d = distance(point[i],point[j]).m
        D_initial.append(d)
sig_square = 10000000
D = np.array(D_initial)
D = np.exp(-np.array(D)**2/sig_square)
D[D<1e-5] = 0
D = D.reshape(num_station,num_station)
D = D.astype(np.float32)

AvgSpeed = dftosave[dftosave['Station']==subStationGeo['ID'].iloc[0]].AvgSpeed[:,np.newaxis].T
for i in subStationGeo['ID']:
    AvgSpeed_temp = dftosave[dftosave['Station']==i].AvgSpeed[:,np.newaxis].T
    AvgSpeed = np.concatenate((AvgSpeed,AvgSpeed_temp),axis=0)
AvgSpeed = AvgSpeed[1:,:]

numtimestep = len(dftosave[dftosave['Station']==i].AvgSpeed)

TotalFlow = dftosave[dftosave['Station']==subStationGeo['ID'].iloc[0]].TotalFlow[:,np.newaxis].T
for i in subStationGeo['ID']:
    TotalFlow_temp = dftosave[dftosave['Station']==i].TotalFlow[:,np.newaxis].T
    TotalFlow = np.concatenate((TotalFlow,TotalFlow_temp),axis=0)
TotalFlow = TotalFlow[1:,:]

node_values = np.zeros((numtimestep,num_station,2))

node_values[:,:,0] = AvgSpeed.T
node_values[:,:,1] = TotalFlow.T
node_values = node_values.astype(np.float32)

np.save('node_values', node_values)
np.save('adj_mat', D)
np.save('station_one',node_values[:,0,0])
fig1.savefig('D8.png')
fig2.savefig('D8StationAvailable.png')
fig3.savefig('D8Select.png')
subStationGeo.to_csv('location_200station.csv')
