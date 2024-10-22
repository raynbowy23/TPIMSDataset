import math
from os import path as osp
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
import argparse

from logging import getLogger, INFO
# logging.basicConfig(filename='./logs/{}.txt'.format(datetime.datetime.now().strftime("%y-%m-%d_%H-%M")))
logger = getLogger(__name__)
logger.setLevel(INFO)

def getDistanceFromLatLonInKm(lat1, lon1, lat2, lon2):
    '''
    Calculate distance with lat and lon
    Use this codes (Haversine formula): https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula
    '''
    R = 6371 # radius of the earth in km
    dLat = deg2rad(lat2-lat1) # deg2rad below

    dLon = deg2rad(lon2-lon1)
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(deg2rad(lat1)) * math.cos(deg2rad(lat2)) * math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c # Distance in km
    return d

def deg2rad(deg):
    return deg * (math.pi/180)

def unique(siteId):
    unique_id = []
    unique_idx = []

    for i, site in enumerate(siteId):
        if site not in unique_id:
            unique_id.append(site)
            unique_idx.append(i)

    return unique_id, unique_idx


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="small", help="small / medium / large")
    args = parser.parse_args()

    site_dict = {}

    dataset_root = osp.join('TPIMS', 'raw_data')

    if args.dataset == "small":
        time_range = 14
    elif args.dataset == "medium":
        time_range = 92
    elif args.dataset == "large":
        time_range = 365
    dfLOC = pd.read_csv(osp.join(dataset_root, 'tpims_data_{}.csv').format(args.dataset))
    dataset_processed_dir = osp.join('TPIMS', 'processed', 'nodes', args.dataset)


    logger.info('Overall shape: {}'.format(dfLOC.shape))
    logger.info('Unique siteId: {}'.format(len(dfLOC['site_id'].unique())))

    # site_id, site_idx = unique(dfLOC['siteId'])
    # for site in site_id:
    #     site_dict[site] = None # initialize dictionary

    # dfSTATS = pd.read_csv(osp.join(dataset_processed_root, 'links', 'link_data.csv')) # Location data
    dfSTATS = pd.read_csv(osp.join(dataset_root, 'tpims_location.csv')) # Location data
    site_id, site_idx = unique(dfSTATS['site_id'])
    capacity_stats = dfSTATS['capacity']

    ame_len = [] # amenity number
    if not isinstance(dfSTATS['z_amenities'][0], int):
        for i in range(len(dfSTATS['z_amenities'])):
            if (dfSTATS['z_amenities'][i] != '') or (dfSTATS['z_amenities'][i] != None):
                ame_len.append(len(str(dfSTATS['z_amenities'][i]).replace(' ', '').split(',')))
            else:
                ame_len.append(0)

        # Replace amenity by number
        dfSTATS['amenity'] = ame_len 
    amenity_stats = dfSTATS['amenity']

    # Ownership
    owner_list = []
    if not isinstance(dfSTATS['ownership'][0], int):
        for i in range(len(dfSTATS['ownership'])):
            if (dfSTATS['ownership'][i] != '') or (dfSTATS['ownership'][i] != None):
                if dfSTATS['ownership'][i] == 'PU':
                    owner_list.append(0)
                else:
                    owner_list.append(1)
            else:
                owner_list.append(-1)

    # Get travel_time, travel_mile -> Deprecated
    """
    travel_time_list = []
    travel_mile_list = []
    for i in range(len(dfSTATS['latitude'])):
        city_name = dfSTATS['city'][i]
        state_name = dfSTATS['state'][i]
        lat = dfSTATS['latitude'][i]
        lng = dfSTATS['longitude'][i]
        if city_name != '' or city_name != None or city_name != None:
            travel_time, travel_mile = distance(lat, long, city_name=str(city_name) + ", " + str(state_name))

            travel_time_list.append(travel_time)
            travel_mile_list.append(travel_mile)
        else:
            continue
        travel_time_list.append(travel_time)
        travel_mile_list.append(travel_mile)
    """

    # Mile Marker
    mile_marker_list = []
    for i in range(len(dfSTATS['mile_marker'])):
        mile_marker = dfSTATS['mile_marker'][i]
        mile_marker_list.append(mile_marker)

    # TODO: Traffic Volume

    print(f'Number of site: {len(site_idx)}')


    # Replacement
    # NaN -> 0
    dfLOC = dfLOC.replace({np.nan: 0})

    # For loop time range by 10 mins
    t_prev = datetime.datetime.strptime('2022-03-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    logger.info('Start analysis time: {}'.format(t_prev))
    available = [0 for i in range(len(dfLOC))]

    for i in tqdm(range(6*24*time_range)):
        idx = 0

        t = (t_prev + datetime.timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S')
        t_prev = t_prev.strftime('%Y-%m-%d %H:%M:%S')
        siteId = dfLOC.loc[dfLOC['time_stamp'].between(str(t_prev), str(t)), 'site_id'].values
        timestamp = dfLOC.loc[dfLOC['time_stamp'].between(str(t_prev), str(t)), 'time_stamp'].values
        capacity = dfLOC.loc[dfLOC['time_stamp'].between(str(t_prev), str(t)), 'capacity'].values
        tmp_available = dfLOC.loc[dfLOC['time_stamp'].between(str(t_prev), str(t)), 'available'].values


        # Get unique siteId within the time range
        siteId_uni, siteId_idx = unique(siteId)

        ### Creating/preparing csv files first
        # Get associated feature values with siteId index
        # Then put info into csv
        t_file = t_prev.replace(':', '-') # change notation for file name

        # Initialize dataframe
        adj_week = 0
        adj_day = 0
        adj_hour = 0
        temp_s_idx = len(site_id)
        ## IMPORTANT: All node data must be in the location data.
        for j, site in enumerate(site_id):

            # Filling up the unknown values
            if 'IN' not in site and 'MI' not in site and 'MIN' not in site:
            # if 'WI' in site:
                idx += 1
                if site in siteId_uni:
                    # s_idx = np.ndarray.tolist(siteId).index(site)
                    s_idx = siteId_uni.index(site)
                    ts = datetime.datetime.strptime(timestamp[s_idx], '%Y-%m-%d %H:%M:%S')
                    # week = timestamp[s_idx].split('-')[1]
                    # week = int(int(timestamp[s_idx].split('T')[0].split('-')[2]) / 7)
                    week = int(int(timestamp[s_idx].split(' ')[0].split('-')[2]) / 7)
                    day = ts.weekday() # Mon - Sun
                    # hour = timestamp[s_idx].split('T')[1].split(':')[0]
                    hour = timestamp[s_idx].split(' ')[1].split(':')[0]
                    adj_week = int(week)
                    adj_day = int(day)
                    adj_hour = int(hour)

                    dfNew = pd.DataFrame({'SITE_IDX': [s_idx], 'SITE_ID': [site], 'TIMESTAMP': [timestamp[s_idx]], 'WEEKID': [week], 'DAYID': [day], 'HOURID': [hour], 'MILE_MARKER': [mile_marker_list[j]], 'OWNER': [owner_list[j]], 'AMENITY': [amenity_stats[j]], 'CAPACITY': [capacity_stats[j]], 'AVAILABLE': [tmp_available[s_idx]], 'OCCRATE': [available[s_idx]/capacity_stats[j]]})
                    available[s_idx] = tmp_available[s_idx]
                else:
                    # Change available/occrate when the site is not found
                    dfNew = pd.DataFrame({'SITE_IDX': [temp_s_idx], 'SITE_ID': [site], 'TIMESTAMP': ['2021-03-01 00:00:00'], 'WEEKID': [adj_week], 'DAYID': [adj_day], 'HOURID': [adj_hour], 'MILE_MARKER': [mile_marker_list[j]], 'OWNER': [owner_list[j]], 'AMENITY': [amenity_stats[j]], 'CAPACITY': [capacity_stats[j]], 'AVAILABLE': [int(available[temp_s_idx])], 'OCCRATE': [available[temp_s_idx]/capacity_stats[j]]})
                    temp_s_idx += 1

                # Upload to csv file
                # dfNew.to_csv(osp.join('dataset', 'nodes', '0322', 'node_data_{}.csv'.format(t_file)), mode='a', header=False, index=False, encoding='utf-8')
                dfNew.to_csv(osp.join(dataset_processed_dir, 'node_data_{}.csv'.format(t_file)), mode='a', header=False, index=False, encoding='utf-8')


        # Update time
        t_prev = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')