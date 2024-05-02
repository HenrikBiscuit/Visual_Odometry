import pandas as pd
import utm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

file = '/home/henrik/Documents/Large_Scale_Drone/Submission2/Data/DJIFlightRecord_2021-03-18_[13-04-51]-TxtLogToCsv.csv'

with open(file, encoding='latin1') as file:
    df = pd.read_csv(file, low_memory=False)

latitude = df['OSD.latitude'].values.tolist()
longitude = df['OSD.longitude'].values.tolist()
altitude = df['OSD.altitude [m]'].values.tolist()

utm_coords = [utm.from_latlon(lat, lon) for lat, lon in zip(latitude, longitude)]

utm_easting = [coord[0] for coord in utm_coords]
utm_northing = [coord[1] for coord in utm_coords]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(utm_easting, utm_northing, altitude, marker='o', linestyle='-')
ax.set_xlabel('UTM Easting')
ax.set_ylabel('UTM Northing')
ax.set_zlabel('Altitude (m)')
ax.set_title('Flight Path')
plt.show()
