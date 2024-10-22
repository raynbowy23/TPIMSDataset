from os import path as osp
import pandas as pd
import numpy as np
import torch
import datetime
from tqdm import tqdm
import argparse
from sklearn.preprocessing import MinMaxScaler

from encoders import IdentityEncoder

# def convert_csv_period_to_comma(input_file, output_file):
#     # Read the CSV file with period (.) as the delimiter
#     df = pd.read_csv(input_file, delimiter='.')
    
#     # Save the DataFrame as a new CSV file with comma (,) as the delimiter
#     columns_list = ["site_id", "time_stamp", "time_stamp_static", "available", "trend", "open", "trustdata", "reported_available", "manual_reset", "low_threshold", "lastverificationcheck", "verificationcheckamplitude", "capacity"]
#     df.columns = columns_list
#     df.to_csv(output_file, sep=',', index=False)

# convert_csv_period_to_comma("TPIMS/raw_data/tpims_data_large.csv", "TPIMS/raw_data/tpims_data_large2.csv")

def unique(siteId):
    unique_id = []
    unique_idx = []

    for i, site in enumerate(siteId):
        if site not in unique_id:
            unique_id.append(site)
            unique_idx.append(i)

    return unique_id, unique_idx

def load_node_csv(path, idx_col, names, encoders=None, **kwargs):
    '''
    Load node/link csv files
    '''
    df = pd.read_csv(path, index_col=False, names=names, **kwargs)
    # mapping = {index: i for i, index in enumerate(df.index.unique())}
    mapping = {index: i for i, index in enumerate(df[df.columns[0]].unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)
    
    return x, mapping

## TODO: If there is no mapping (no corresponding nodes or edges due to the update), the mapping should also be updated accordingly.
def load_edge_csv(path, mapping=None, src_index_col=None, dst_index_col=None, names=None, encoders=None, edge_cut=None, visualize_adj=False, **kwargs):
    df = pd.read_csv(path, names=names, **kwargs)

    # Should be 0 to index
    src = []
    dst = []
    for index in df[src_index_col]:
        try:
            src.append(mapping[index])
        except KeyError:
            mapping[index] = len(mapping)
            src.append(mapping[index])
    for index in df[dst_index_col]:
        try:
            dst.append(mapping[index])
        except:
            mapping[index] = len(mapping)
            dst.append(mapping[index])
    # src = [mapping[index] for index in df[src_index_col]]
    # dst = [mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])
    batch = torch.tensor([0, 0, 1, 1])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    # if visualize_adj:
    #     adj_matrix = to_dense_adj(edge_index)
    #     # G = nx.from_torch_sparse(adj_matrix)
    #     fig = plt.figure(figsize=(5,5))
    #     plt.imshow(adj_matrix.squeeze(0), cmap="Greys", interpolation="none")
    
    if edge_cut == 'random':
        edge_index, edge_mask = random_edge_sampler(edge_index, 0.8)
    elif edge_cut == 'neural':
        pass
    
    return edge_index, edge_attr

def random_edge_sampler(edge_index, percent, normalization=None):
    '''
    Can be replaced by Random Temporal GNN.
    percent: The percent of the preserved edges
    '''

    def stub_sampler(normalizatin, cuda):
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index

    if percent >= 1.0:
        return stub_sampler(normalization, edge_index.device)

    row, column = edge_index
    
    edge_mask = torch.rand(row.size(0), device=edge_index.device) >= percent

    edge_index = edge_index[:, edge_mask] 

    return edge_index, edge_mask


def load_node_data(path, index_col, encoders=None):
    data, mapping = load_node_csv(path, index_col, names=['SITE_IDX', 'SITE_ID', 'TIMESTAMP', 'WEEKID', 'DAYID', 'HOURID', 'TRAVEL_TIME', 'TRAVEL_MILE', 'OWNER', 'AMENITY', 'CAPACITY', 'AVAILABLE', 'OCCRATE'], encoders=encoders)
    return data, mapping

def load_edge_data(path, mapping, src_index_col, dst_index_col, encoders=None):
    edge_idx, edge_attr = load_edge_csv(path, mapping, src_index_col, dst_index_col, names=['SRC_IDX', 'SRC', 'DST_IDX', 'DST', 'DIST'], encoders=encoders, visualize_adj=False)
    return edge_idx, edge_attr

# Read data from small dataset and save it to pkl directly (not transitting at csv)
def process(args, capacity_stats, amenity_stats, owner_list, mile_marker_list):
    node_root = osp.join('TPIMS', 'procesesd', 'nodes', args.dataset)
    link_root = osp.join('TPIMS', 'processed', 'links')
    data_dir = osp.join('TPIMS', 'raw_data')
    processed_dir = osp.join('TPIMS', 'processed')

    link_path = osp.join(link_root, 'link_data.csv')
    link_IA = osp.join(link_root, 'link_IA_data.csv')
    link_KS = osp.join(link_root, 'link_KS_data.csv')
    link_KY = osp.join(link_root, 'link_KY_data.csv')
    link_OH = osp.join(link_root, 'link_OH_data.csv')
    link_WI = osp.join(link_root, 'link_WI_data.csv')

    node_data_list = []
    sc = MinMaxScaler(feature_range=(0,1))

    # Make mapping for edge data
    dfLOC = pd.read_csv(osp.join(data_dir, 'tpims_location.csv'))
    # Replacement
    # NaN -> 0
    dfLOC = dfLOC[~dfLOC['site_id'].str.startswith(('IL', 'MI', 'MN', 'IN'), na=False)]
    dfLOC = dfLOC.replace({np.nan: 0})

    dfNODE = pd.read_csv(osp.join(data_dir, 'tpims_data_{}.csv'.format(args.dataset)))
    mapping = {index-1: i for i, index in enumerate(range(1, len(dfLOC)))} # site_idx to idx

    if args.dataset == 'small':
        time_range = 14
    elif args.dataset == 'medium':
        time_range = 92
    elif args.dataset == 'large':
        time_range = 365

    # One edge data for all states
    edge_index, edge_attr = load_edge_data(link_path, mapping=mapping, src_index_col='SRC_IDX', dst_index_col='DST_IDX',
                                                encoders={'DIST': IdentityEncoder(dtype=torch.float)})
    edge_IA_index, edge_IA_attr = load_edge_data(link_IA, mapping=mapping, src_index_col='SRC_IDX', dst_index_col='DST_IDX',
                                                encoders={'DIST': IdentityEncoder(dtype=torch.float)})
    edge_KS_index, edge_KS_attr = load_edge_data(link_KS, mapping=mapping, src_index_col='SRC_IDX', dst_index_col='DST_IDX',
                                                encoders={'DIST': IdentityEncoder(dtype=torch.float)})
    edge_KY_index, edge_KY_attr = load_edge_data(link_KY, mapping=mapping, src_index_col='SRC_IDX', dst_index_col='DST_IDX',
                                                encoders={'DIST': IdentityEncoder(dtype=torch.float)})
    edge_OH_index, edge_OH_attr = load_edge_data(link_OH, mapping=mapping, src_index_col='SRC_IDX', dst_index_col='DST_IDX',
                                                encoders={'DIST': IdentityEncoder(dtype=torch.float)})
    edge_WI_index, edge_WI_attr = load_edge_data(link_WI, mapping=mapping, src_index_col='SRC_IDX', dst_index_col='DST_IDX',
                                                encoders={'DIST': IdentityEncoder(dtype=torch.float)})
    edge_attr = edge_attr.squeeze().clone().detach()
    edge_IA_attr = edge_IA_attr.squeeze().clone().detach()
    edge_KS_attr = edge_KS_attr.squeeze().clone().detach()
    edge_KY_attr = edge_KY_attr.squeeze().clone().detach()
    edge_OH_attr = edge_OH_attr.squeeze().clone().detach()
    edge_WI_attr = edge_WI_attr.squeeze().clone().detach()
    
    t_prev = datetime.datetime.strptime('2022-03-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    
    site_id, site_idx = unique(dfLOC['site_id'])
    # available = [0 for i in range(len(dfLOC))]
    site_id_dict = {site: idx for idx, site in enumerate(site_id)}
    available_dict = {site: 0 for site in site_id}
    available = [0 for i in range(len(dfLOC))]

    for i in tqdm(range(6*24*time_range)):
        idx = 0

        t = (t_prev + datetime.timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S')
        t_prev = t_prev.strftime('%Y-%m-%d %H:%M:%S')
        mask = dfNODE['time_stamp'].between(str(t_prev), str(t))
        filtered_df = dfNODE[mask]
        siteId = filtered_df['site_id'].values
        timestamp = filtered_df['time_stamp'].values
        tmp_available = filtered_df['available'].values

        week = int(int(t.split(' ')[0].split('-')[2]) / 7)
        day = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S').weekday() # Mon - Sun
        hour = t.split(' ')[1].split(':')[0]
        adj_week = int(week)
        adj_day = int(day)
        adj_hour = int(hour)

        # Get unique siteId within the time range
        siteId_uni, siteId_idx = unique(siteId)

        ### Creating/preparing csv files first
        # Get associated feature values with siteId index
        # Then put info into csv
        t_file = t_prev.replace(':', '-') # change notation for file name

        site_to_available = dict(zip(siteId, tmp_available))

        # Initialize dataframe
        adj_week = 0
        adj_day = 0
        adj_hour = 0
        temp_s_idx = len(site_id)
        _node_data_list = []
        ## IMPORTANT: All node data must be in the location data.
        for j, site in enumerate(site_id):
            # Filling up the unknown values
            if 'IN' not in site[:2] and 'MI' not in site[:2] and 'MN' not in site[:3] and 'IL' not in site[:2]:
                idx += 1
                if site in siteId:
                    s_idx = site_id_dict[site]
                    available_value = site_to_available[site]
                    # ts = datetime.datetime.strptime(timestamp[j], '%Y-%m-%d %H:%M:%S')
                    # # week = timestamp[s_idx].split('-')[1]
                    # week = int(int(timestamp[s_idx].split(' ')[0].split('-')[2]) / 7)
                    # day = ts.weekday() # Mon - Sun
                    # hour = timestamp[s_idx].split(' ')[1].split(':')[0]

                    if capacity_stats[j] == 0:
                        capacity_stats[j] = np.finfo(np.float32).eps
                    else:
                        node_dict = {'SITE_IDX': [s_idx], 'SITE_ID': [site], 'TIMESTAMP': [t], 'WEEKID': [int(week)], 'DAYID': [int(day)], 'HOURID': [int(hour)], 'MILE_MARKER': [mile_marker_list[j]], 'OWNER': [owner_list[j]], 'AMENITY': [amenity_stats[j]], 'CAPACITY': [capacity_stats[j]], 'AVAILABLE': [available_value], 'OCCRATE': [available_value/capacity_stats[j]]}
                    # _node_data = [s_idx, site, timestamp[s_idx], week, day, hour, mile_marker_list[j], owner_list[j], amenity_stats[j], capacity_stats[j], tmp_available[s_idx], available_value[s_idx]/capacity_stats[j]]
                    available[s_idx] = available_value
                else:
                    try:
                        # Change available/occrate when the site is not found
                        node_dict = {'SITE_IDX': [temp_s_idx], 'SITE_ID': [site], 'TIMESTAMP': [t], 'WEEKID': [int(week)], 'DAYID': [int(day)], 'HOURID': [adj_hour], 'MILE_MARKER': [mile_marker_list[j]], 'OWNER': [owner_list[j]], 'AMENITY': [amenity_stats[j]], 'CAPACITY': [capacity_stats[j]], 'AVAILABLE': [int(available[temp_s_idx])], 'OCCRATE': [available[temp_s_idx]/capacity_stats[j]]}
                        # _node_data = [temp_s_idx, site, '2021-03-01 00:00:00', adj_week, adj_day, adj_hour, mile_marker_list[j], owner_list[j], amenity_stats[j], capacity_stats[j], available_value[temp_s_idx], available_value[temp_s_idx]/capacity_stats[j]]
                    except IndexError:
                        node_dict = {'SITE_IDX': [temp_s_idx], 'SITE_ID': [site], 'TIMESTAMP': [t], 'WEEKID': [int(week)], 'DAYID': [int(day)], 'HOURID': [adj_hour], 'MILE_MARKER': [mile_marker_list[j]], 'OWNER': [owner_list[j]], 'AMENITY': [amenity_stats[j]], 'CAPACITY': [capacity_stats[j]], 'AVAILABLE': [0], 'OCCRATE': [0.0]}
                    temp_s_idx += 1
                        
                encoders = {'WEEKID': IdentityEncoder(dtype=torch.long), 'DAYID': IdentityEncoder(dtype=torch.long), 'HOURID': IdentityEncoder(dtype=torch.long),
                        'MILE_MARKER': IdentityEncoder(dtype=torch.float), 'OWNER': IdentityEncoder(dtype=torch.long),
                        'AMENITY': IdentityEncoder(dtype=torch.long), 'CAPACITY': IdentityEncoder(dtype=torch.long), args.train_feature.upper(): IdentityEncoder(dtype=torch.float)}

                if encoders is not None:
                    xs = [encoder(node_dict[col]) for col, encoder in encoders.items()]
                    x = torch.cat(xs, dim=-1)
                    x = torch.nan_to_num(x)

                # _node_data_list.append(x.transpose(0, 1))
                _node_data_list.append(x)

        # Upload to csv file
        node_data = torch.cat(_node_data_list, dim=0)
        # _node_data = torch.from_numpy(sc.fit_transform(x.transpose(0, 1).numpy()))
        node_data = torch.from_numpy(sc.fit_transform(node_data))
        node_data_list.append(node_data)

        # Update time
        t_prev = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')

    torch.save((edge_index, edge_attr, edge_IA_index, edge_IA_attr, edge_KS_index, edge_KS_attr, edge_KY_index,
                edge_KY_attr, edge_OH_index, edge_OH_attr, edge_WI_index, edge_WI_attr, node_data_list), osp.join(processed_dir, 'tpims_data_{}.pkl'.format(args.dataset)))

def preprocess(dataset_root):
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

    # Mile Marker
    mile_marker_list = []
    for i in range(len(dfSTATS['mile_marker'])):
        mile_marker = dfSTATS['mile_marker'][i]
        mile_marker_list.append(mile_marker)

    print(f'Number of site: {len(site_idx)}')

    return capacity_stats, amenity_stats, owner_list, mile_marker_list



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="small", help="small / medium / large")
    parser.add_argument("-tf", "--train_feature", type=str, default="occrate", help="available / occrate")
    args = parser.parse_args()

    site_dict = {}

    dataset_root = osp.join('TPIMS', 'raw_data')

    if args.dataset == "small":
        time_range = 14
    elif args.dataset == "medium":
        time_range = 92
    elif args.dataset == "large":
        time_range = 365
    # dfLOC = pd.read_csv(osp.join(dataset_root, 'tpims_data_{}.csv').format(args.dataset))

    capacity_stats, amenity_stats, owner_list, mile_marker_list = preprocess(dataset_root)

    process(args, capacity_stats, amenity_stats, owner_list, mile_marker_list)