# Load the script with functions
import sys
import qgrid
# Custom
from iDora_eda import *
from iDora_FeatImp import *
import settings
# Heleprs
settings.set_defaults_helper()
cwd_fix()
add_styling()

# ------- MAIN  --------
def eda_run(path=None,txt_flag=False,df=None):
    settings.txt_flag = txt_flag
    
    # - Exploration: should always be the first step you do.
    if ( (df is None) and (path is not None) ):
        df = path_load(path)
    elif (df is None):
        raise ValueError('Neither path nor pre-loaded dataframe was given.')
    _ = explore(df)
    summary = summarize(df) 
    
    # - Plotting
    settings.boxplots_path = make_boxplots(summary, df=df)
    settings.barplots_path = make_barplots(summary, df=df)
    
    # - Display HTMl
    df_summary_plots = produce_df_plot(summary)
    html_example = display_df(df_summary_plots,['Count','Unique','Mean'])
    
    # - Transformation
    df_sparse = onehot_encode(settings.cat_vars, df)
    # df_new = label_encode(settings.cat_vars, df)
    
    # - Variables Removal
    df_new = remove_vars(df_sparse) 
    
    # - Saving the output
    save_html(html_example)
    save_df(df_new)
    
    return df_new, html_example, summary

# - Stylers & Misc

# 1 Pivot Table functionality to explore further
# pivot_func()


















