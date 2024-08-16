#%%
import pandas as pd
import numpy as np

# %%

import eventextreme.eventextreme as evext
import eventextreme.extreme_extract as ee
import eventextreme.extreme_threshold as et
#%%
import importlib
importlib.reload(evext)
importlib.reload(ee)
importlib.reload(et)
# %%
################# load the data #################
## data that contains 'plev' as the independent dimension
data = pd.read_csv('/work/mh0033/m300883/Event_extremes/data/NAO/example_nao.csv')
pos_threshold = pd.read_csv('/work/mh0033/m300883/Event_extremes/data/NAO/pos_threshold_first10_allens.csv')


#%%
## a single time series data
single_data = data[data['plev'] == 25000][['time', 'pc']]
single_pos_threshold = pos_threshold[pos_threshold['plev'] == 25000][['dayofyear', 'threshold']]
# %%
################## extreme events in a single time series ##################

extremes = evext.EventExtreme(single_data, independent_dim=None)

# positive extreme events
positive_events = extremes.extract_positive_extremes

# negative extreme events
negative_events = extremes.extract_negative_extremes

#%%
# test with pre-defined threshold
extremes = evext.EventExtreme(single_data, independent_dim=None)
extremes.pos_thr_dayofyear = single_pos_threshold[['dayofyear', 'threshold']]
positive_events = extremes.extract_positive_extremes
# %%


############# extreme events in multiple time series ##################

extremes = evext.EventExtreme(data, independent_dim='plev')
positive_events = extremes.extract_positive_extremes
negative_events = extremes.extract_negative_extremes
# %%

# test with pre-defined threshold
extremes = evext.EventExtreme(data, independent_dim='plev')
extremes.pos_thr_dayofyear = pos_threshold
positive_events = extremes.extract_positive_extremes
# %%
