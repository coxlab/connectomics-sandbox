requirements
============

The codes will need at least working versions of:

 - numpy
 - scipy
 - PIL
 - the V1 like code of Nicolas (just the directory *v1like/*)
 - a `resample` function (self-contained in `resample.py` here)
 - a `view_as_windows` function (self-contained in `shape` here)

how to use the codes
====================

generate the V1 features for the full connectome dataset
--------------------------------------------------------

Be careful here. The code will generate more than 280GB
of data (if one uses defaults, like 5 scales). So one needs
a large storage capacity, and don't store the result in a
directory on the NFS.

The code `generate_connectome_v1_features.py` is on the
**coxlabdata** Github account.

1. Edit file `parameters.py` and give values to the following
   parameters (DATASET_PATH, V1_FEATURES_FILENAME, MAX_EDGE_L,
   V1_MODEL_CONFIG).
   DATASET_PATH gives the full path to the *connectome* dataset
   on your machine.
   V1_FEATURES_FILENAME will be the name of the (big) file
   containing all the V1 features you asked for in the following
   order in memory: image_index, scale_index, h, w, d.
   MAX_EDGE_L should be a list of integers containing the values
   of the scales at which to compute V1 features (e.g. 1024, 512).
   V1_MODEL_CONFIG is a string that tell Nicolas' V1 like code
   which *configuration* to use. Default is 96 Gabor filters.

2. run the generation code
   `python generate_connectome_v1_features.py`

run a model
-----------

1. Edit file `parameters.py` to include the path to the
   connectomics dataset on your computer (DATASET_PATH).
   Also edit the parameter `V1_FEATURES_FILENAME` which
   indicates where the file containing the V1 features
   is. This will be used when building the memmap array.
2. run the *master* program `compute_tm.py`. To see the
   available options just type
   `python compute_tm.py -h`

analyze the output(s) of a model run
------------------------------------

Use the script `extract_and_plot_predicted_target_maps.py`. To
see the options just type `python extract_and_plot_predicted_target_maps.py -h`.
