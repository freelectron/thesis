""" 
All the preproccesing flags (& variables) are here:

nan_df = how many NaNs in total
dim_df = dimensions of the loaded dataset
plot_dims = dimensios of the plot
cat_unique =if (the features unique val's <cat_uniq_flag)
                            suggest the feature to be one-hot-encoded 
txt_flag = True if data is in the txt format 
"""

# Path Varaibles
script_directroty = '' ; 
path_to_data = '' ;
boxplots_path = '' ;
barplots_path = '' ;


# Variables within a dataset
global num_vars, cat_vars, obj_vars, bad_vars, noplot_vars, id_vars, date_vars,binary_vars
num_vars = None ;
cat_vars = None ;
obj_vars = None; 
binary_vars = None ;
bad_vars = None ;
noplot_vars = None; 
id_vars = None ;
date_vars = None ;

# Other
missing_vals_th = 0.5

# Format Variables
cat_unique_th = 100
plot_dims = (10, 10)
plot_resolution = (400, 400)
bins = 11
txt_flag = False ;

def set_defaults_helper():
    # Variables within a dataset
    global num_vars, cat_vars, obj_vars, bad_vars, noplot_vars, id_vars, date_vars, binary_vars
    num_vars = [] ;
    cat_vars = [] ;
    obj_vars = [] ; 
    bad_vars = [] ;
    noplot_vars = [] ; 
    id_vars = [] ;
    date_vars = [] ;
    binary_vars = [] ;

    return 