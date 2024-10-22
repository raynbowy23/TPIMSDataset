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

import os.path as osp
import pandas as pd
import numpy as np
import datetime
import argparse
from tqdm import tqdm


def unique(siteId):
    unique_id = []
    unique_idx = []

    for i, site in enumerate(siteId):
        if 'IN' not in site and 'MI' not in site and 'MIN' not in site:
            unique_id.append(site)
            unique_idx.append(i)

    return unique_id, unique_idx

def align_with_metr_format(args, data_dir, output_dir):
    # if not osp.exists(output_dir):
    #     os.makedirs(output_dir)

    # # Get all files in the data directory
    # files = os.listdir(data_dir)
    # files = [f for f in files if f.endswith('.csv')]

    # Sort files by timestamp
    # files = sorted(files)
    dfSTATS = pd.read_csv(osp.join(data_dir, 'tpims_location.csv')) # Location data
    dfNODE = pd.read_csv(osp.join(data_dir, 'tpims_data_{}.csv'.format(args.dataset)))
    capacity_stats = dfSTATS['capacity']

    site_id, site_idx = unique(dfSTATS['site_id'])

    if args.dataset == 'small':
        time_range = 14
    elif args.dataset == 'medium':
        time_range = 92
    elif args.dataset == 'large':
        time_range = 365

    site_id_dict = {site: idx for idx, site in enumerate(site_id)}
    available_dict = {site: 0 for site in site_id}
    columns_list = ['time_stamp']
    columns_list.extend([s for s in site_id])
    dfNew = pd.DataFrame(columns=columns_list)

    t_prev = datetime.datetime.strptime('2022-03-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    # available = [0 for i in range(len(dfNODE))]

    for i in tqdm(range(6*24*time_range)):
        idx = 0

        t = (t_prev + datetime.timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S')
        t_write = (t_prev + datetime.timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S')
        t_prev = t_prev.strftime('%Y-%m-%d %H:%M:%S')
        mask = dfNODE['time_stamp'].between(str(t_prev), str(t))
        filtered_df = dfNODE[mask]
        siteId = filtered_df['site_id'].values
        tmp_available = filtered_df['available'].values

        # Get unique siteId within the time range
        siteId_uni, siteId_idx = unique(siteId)
        site_to_available = dict(zip(siteId, tmp_available))

        data_list = [t_write]

        temp_s_idx = len(site_id)
        for j, site in enumerate(site_id):
            if 'IN' not in site[:2] and 'MI' not in site[:2] and 'MIN' not in site[:2]:
                idx += 1
                if site in site_to_available:
                    s_idx = site_id_dict[site]
                    available_value = site_to_available[site]

                    if capacity_stats[j] == 0:
                        capacity_stats[j] = np.finfo(np.float32).eps
                    data_list.append(available_value / capacity_stats[j])

                    available_dict[site] = available_value
                else:
                    # Change available/occrate when the site is not found
                    if capacity_stats[j] == 0:
                        capacity_stats[j] = np.finfo(np.float32).eps
                    data_list.append(available_dict[site] / capacity_stats[j])
                    temp_s_idx += 1


        dataDf = pd.DataFrame([data_list], columns=dfNew.columns)
        dfNew = pd.concat([dfNew, dataDf], ignore_index=True)

        # Update time
        t_prev = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
    dfNew.set_index('time_stamp', inplace=True)
    dfNew.to_csv(osp.join(output_dir, 'tpims_{}.csv'.format(args.dataset)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="small", help="small / medium / large")
    args = parser.parse_args()

    data_dir = osp.join('TPIMS', 'raw_data')
    output_dir = osp.join('TPIMS', 'processed')
    align_with_metr_format(args, data_dir, output_dir)