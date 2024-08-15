#%%
import pandas as pd
import numpy as np

# %%

import src.event_extreme as evext

#%%
import importlib
importlib.reload(evext)

# %%
# %%
# load the data
data = pd.read_csv('/work/mh0033/m300883/Event_extremes/data/NAO/example_nao.csv')

pos_threshold = pd.read_csv('/work/mh0033/m300883/Event_extremes/data/NAO/pos_threshold_first10_allens.csv')
########### Example 1: Single dimension data ############
single_data = data[data['plev'] == 25000][['time', 'pc']]
single_pos_threshold = pos_threshold[pos_threshold['plev'] == 25000][['dayofyear', 'threshold']]
# %%
extremes = evext.EventExtreme_single(single_data)

#%%
extremes.pos_abs_thr = single_pos_threshold

# %%
extremes.extract_positive_extremes
# %%
extremes.positive_events
# %%
extremes.extract_negative_extremes

# %%
extremes.negative_events

# %%
