"""
TPIMS format:
    1. File is separated by timestamp
    2. Each file has all truck parking data (equally, number of nodes)
    e.g.
    File n: Time from 00:00:00 to 00:05:00
        site id, timestamp, latitude, longitude, ...

METR-LA format:
    1. Timestamp is in the first column
    2. At every row, it has all location information for each column
    e.g.

        timestamp          , node_1, node_2, node_3, node_n, ...
        2018/01/01 00:00:00, 60.0  , 65.0  , 70.0  ,         ...
"""

import os
import os.path as osp
import pandas as pd
import datetime
from tqdm import tqdm


def unique(siteId):
    unique_id = []
    unique_idx = []

    for i, site in enumerate(siteId):
        if site not in unique_id:
            unique_id.append(site)
            unique_idx.append(i)

    return unique_id, unique_idx

def align_with_metr_format(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Sort files by timestamp
    dataset_root = osp.join('dataset', 'data')
    dfSTATS = pd.read_csv(osp.join(dataset_root, 'TruckParkingQuery_Location2.csv')) # Location data
    dfNODE1 = pd.read_csv(osp.join(dataset_root, 'TruckParkingQuery_20220301_20220307.csv'))
    dfNODE2 = pd.read_csv(osp.join(dataset_root, 'TruckParkingQuery_20220308_20220314.csv'))
    dfNODE = pd.concat([dfNODE1, dfNODE2])
    capacity_stats = dfSTATS['CAPACITY']

    site_id, site_idx = unique(dfSTATS['SITE_ID'])
    # site_id, site_idx = unique(dfNODE['siteId'])

    columns_list = ['timestamp']
    columns_list.extend([s for s in site_id])
    dfNew = pd.DataFrame(columns=columns_list)

    t_prev = datetime.datetime.strptime('2022-03-01T00:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
    available = [0 for i in range(len(dfNODE))]

    for i in tqdm(range(6*24*14)):
        idx = 0

        t = (t_prev + datetime.timedelta(minutes=10)).strftime('%Y-%m-%dT%H:%M:%SZ')
        t_write = (t_prev + datetime.timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S')
        t_prev = t_prev.strftime('%Y-%m-%dT%H:%M:%SZ')
        siteId = dfNODE.loc[dfNODE['timestamp'].between(str(t_prev), str(t)), 'siteId'].values
        tmp_available = dfNODE.loc[dfNODE['timestamp'].between(str(t_prev), str(t)), 'available'].values

        # Get unique siteId within the time range
        siteId_uni, siteId_idx = unique(siteId)
        data_list = [t_write]

        temp_s_idx = len(site_id)
        for j, site in enumerate(site_id):
            # dfNew.insert(i+1, 'timestamp', t_prev) 
            if 'IN' not in site[:2] and 'MI' not in site[:2] and 'MIN' not in site[:2]:
                idx += 1
                if site in siteId_uni:
                    s_idx = siteId_uni.index(site)

                    # dfNew.insert(i, site, tmp_available[s_idx]) 
                    data_list.append((capacity_stats[j] - available[s_idx])/capacity_stats[j])

                    available[s_idx] = tmp_available[s_idx]
                else:
                    # Change available/occrate when the site is not found
                    data_list.append((capacity_stats[j] - available[temp_s_idx])/capacity_stats[j])
                    temp_s_idx += 1


        dataDf = pd.DataFrame([data_list], columns=dfNew.columns)
        dfNew = pd.concat([dfNew, dataDf], ignore_index=True)

        # Update time
        t_prev = datetime.datetime.strptime(t, '%Y-%m-%dT%H:%M:%SZ')
    dfNew.set_index('timestamp', inplace=True)
    dfNew.to_csv(osp.join(output_dir, 'tpims.csv'))


if __name__ == '__main__':
    data_dir = 'dataset/data'
    output_dir = 'dataset/TPIMS'
    align_with_metr_format(data_dir, output_dir)