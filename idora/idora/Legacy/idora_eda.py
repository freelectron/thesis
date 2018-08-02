# General
import pandas as pd
import numpy as np
import os
import sys
import qgrid
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display import HTML
from IPython.display import Image, HTML
from IPython.display import Javascript
import sys
import traceback

# # Nice plot-output
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')

# Putting images to df
import glob
import random
import base64
from PIL import Image
from io import BytesIO

# Transfomrations
from sklearn.preprocessing import LabelEncoder

# Custom
import settings

""" 
All the preproccesing flags (& variables) are here:

nan_df = how many NaNs in total
dim_df = dimensions of the loaded dataset
plot_dims = dimensios of the plot
cat_unique =if (the features unique val's <cat_uniq_flag)
                            suggest the feature to be one-hot-encoded 
txt_flag = True if data is in the txt format 
"""
print("All inputs must be without spaces and separated by commas!")

# FUNCTIONS & HELPERS
                                          #  ------ Exploration  ------

def path_helper(path): 
    path_to_data = path.split("\\")[0:-1]
    path_to_data = "\\".join(str(x) for x in path_to_data)
    
    file_name = path.split("\\")[-1:]
    
    return path_to_data, file_name    

def cwd_fix():
    settings.script_directory = os.getcwd()
    
    return

def path_load(path,txt_flag = False): 
    # Load the data
    settings.txt_flag = txt_flag
    settings.path_to_data = path_helper(path)[0]
    if settings.txt_flag:
        df = pd.read_csv(path, sep=' ', encoding='latin-1')
    else:
        df = pd.read_csv(path)

    return df

def explore(df):
    # Start explorings
    ## -- This is an implementation which counts all missing cells ! 
    nan_df = df.isnull().sum().sum()
    dim_df = df.shape
    variables = list(df.columns)
    print('Your data has this dimensionality: {}'.format(dim_df),'\n')
    print('You have {} missing values (cells) in your data.'.format(nan_df),'\n')
    print('These are all the variables you have:', variables,'\n')

    return 1

def summarize(df):
    # ID column and Date columns should not be summarized like other numerical
    # dfc = cleaner dataframe
    all_vars = list(df.columns)   # NEEDS 'settings'
    
    print('Tell me the ID column name, please.')
    settings.id_vars = input()
    
    if (settings.id_vars != ''):
        try:
            dfc = df.drop(columns=settings.id_vars.split(','), inplace=False)
            settings.num_vars = list(dfc.columns)
        except:
            print('Indicated columns are not in the original DF\n')
            dfc = df
    else:
        dfc = df
        settings.num_vars = list(dfc.columns)

    print('Tell me the Date column name, please.')
    settings.date_vars = input()
    
    if (settings.date_vars != ''):
        try:
            dfc.drop(columns=settings.date_vars.split(','), inplace=True)
            settings.num_vars = list(dfc.columns)
        except:
            print('Indicated columns are not in the original DF\n')
    else: 
        pass

    print('You have these variables with numerical values to put in a model:')
    print(settings.num_vars,'\n')
    
    # Produce summary with numerical variables
    df_summary = _produce_summary(df, dfc)
    
    settings.obj_vars = list(set(all_vars) - set(settings.num_vars))
    print('You have these variables with object type (e.g. strings):')
    print(settings.obj_vars,'\n')
    
    print('You might want to (one-hot) encode these variables as they have less than {} unique numerical values'.format(settings.cat_unique_th)) 
    settings.cat_vars = []
    for i,x in df_summary.iterrows():
        if ( (x['Unique'] <= settings.cat_unique_th) and (x['Unique'] >2) ):
            settings.cat_vars.append(i)
        elif ( (x['Unique'] == 2) ):
            settings.binary_vars.append(i)
    print(settings.cat_vars,'\n')
    
    print('You have these varaiables that have more than {}% missing values'.format(settings.missing_vals_th * 100))
    for i, x in df_summary.iterrows():
        if ( ((x['Missing']+0.00001) / x['Count'] >= settings.missing_vals_th) ):
            settings.bad_vars.append(i)
            if (x['Count'] == 0):
                settings.noplot_vars.append(i)
    print(settings.bad_vars,'\n')
        
    return df_summary

def _produce_summary(df, dfc):
    # Do simple stuff: 
    # Summarize the data:
    df_summary = dfc.describe().T
    df_summary.columns = [ "Count", "Mean", "Std", "Min", "25p", "50p", "75p", "Max" ]

    # Get easy additional details: unique values 
    temp_list = []
    for i,x in df_summary.iterrows():
        temp_count = len( df[i].unique() )
        temp_list.append(temp_count)
    df_summary['Unique'] = temp_list

    # Add missing values column
    temp_list = []
    for i,x in df_summary.iterrows():
        temp_count = ( df[i].isnull().sum() )
        temp_list.append(temp_count)
    df_summary['Missing'] = temp_list

    df_summary = df_summary.loc[:,\
     ["Count", "Missing", "Unique", "Mean", "Std", "Min", "Max", "25p", "50p", "75p" ] ]

    # Make the count column look nicer
    df_summary['Count'] = df_summary['Count'].astype(int)
    
    return df_summary

                                          #  ------  PLOTTING  ------
    
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=settings.plot_dims)  

### ---- Bar Plots ----
def _produce_barplot(df, var):
        if var in settings.cat_vars:
            df[var] = df[var].fillna('Missing')
            ax = sns.countplot(data=df, x=var)
            fig = ax.get_figure()
            name = var + '_barplot'
            fig.savefig(name + '.png')
            plt.clf()
        else:
            ax = sns.distplot(a=df[var].dropna(how='all'), bins=settings.bins, color="y")
            fig = ax.get_figure()
            name = var + '_barplot'
            fig.savefig(name + '.png')
            plt.clf()
            
        return 

# Create barplot images and save them 
def make_barplots(df_summary, df):
    # Next command works with ipynb files, if  you work in .py substitute '__file__' by __file__
    script_directory = settings.script_directory    
    barplots_path = script_directory + '\\barplots'
    if not os.path.exists(barplots_path):
        os.makedirs(barplots_path)
    os.chdir(barplots_path)
    for i,x in df_summary.iterrows():
        try: 
            _produce_barplot(df, i)  
        except:
            print('Plot for', i,'was not produced. See pink Traceback')
            traceback.print_exc()
    os.chdir(script_directory)                 

    return barplots_path

### ---- Box Plots ----
def _produce_boxplot(df, var):
        ax = sns.boxplot(x=df[var].dropna(), palette="Set3",orient='v',
                     fliersize=5, linewidth=3, width = 0.9)
        fig = ax.get_figure()
        name = var + '_boxplot'
        fig.savefig(name + '.png')
        plt.clf()
    
        return 

# Create boxplot images and save them.
def make_boxplots(df_summary, df):   
    # Next command works with ipynb files, if  you work in .py substitute '__file__' by __file__
    script_directory = settings.script_directory   
    boxplots_path = script_directory + '\\boxplots'  
    if not os.path.exists(boxplots_path):
        os.makedirs(boxplots_path)
    os.chdir(boxplots_path)
    for i,x in df_summary.iterrows():
        try: 
            _produce_boxplot(df,i)
        except:
            print('Plot for', i,'was not produced')
            traceback.print_exc()   # should stay for testing
    os.chdir(script_directory)
    
    return boxplots_path

def _get_thumbnail(path):
        try:
            i = Image.open(path)
            i.thumbnail(settings.plot_resolution, Image.LANCZOS)
        except:
            i ='nan'
            
        return i
    
def _image_base64(im):
    if isinstance(im, str):
        im = _get_thumbnail(im)
    with BytesIO() as buffer:
        im.save(buffer, 'png')
        return base64.b64encode(buffer.getvalue()).decode()
                  
def _image_formatter(im):
    return f'<img src="data:image/png;base64,{_image_base64(im)}">'

def produce_df_plot(df_summary):      
#     df_summary['file'] = df_summary.index.map(lambda id: f'{settings.boxplots_path}]/id}_boxplot.png')
    df_summary['file'] = df_summary.index.map(lambda id: f'{settings.boxplots_path}/{id}_boxplot.png')
    df_summary['BoxPlot'] = df_summary.file.map(lambda f: _get_thumbnail(f), na_action='ignore')
    
    # Comment out if you want only box plots
#     df_summary['file'] = df_summary.index.map(lambda id: f'{settings.barplots_path}/{id}_barplot.png')
    df_summary['file'] = df_summary.index.map(lambda id: f'{settings.barplots_path}/{id}_barplot.png')
    df_summary['BarPlot'] = df_summary.file.map(lambda f: _get_thumbnail(f), na_action='ignore')
    
    # Not to clutter the main df, drop file column
    df_summary.drop(columns='file',inplace=True)
    df_summary_plots = df_summary
    
    return df_summary_plots

def display_df(df_summary_plots, columns, variables='all'):    
    if ('BoxPlot' not in columns):
        columns.append('BoxPlot')
    if ('BarPlot' not in columns):
        columns.append('BarPlot')
    if variables == 'all':
        variables = list(df_summary_plots.index)
    # df_html = HTML((df_summary_plots.loc[variables,columns]).to_html(formatters={'BoxPlot': _image_formatter}, escape=False))
    # Comment out if you want only box plots
    df_html = HTML((df_summary_plots.loc[variables,columns]).to_html(formatters={'BoxPlot':_image_formatter,'BarPlot': _image_formatter}, escape=False))
    
    return df_html
                                          #  ------  Transformations  ------
# One-hot Encoding (works with numerical, categorical (strings) and mixed variables)
def onehot_encode(var_list, data, exclude=[]):
    le = LabelEncoder()
    for i in var_list:
        data[i] = data[i].apply(str)
    columnsToEncode = list( data.select_dtypes(include=['category','object']) )
    # Exclude dummies 
    columnsToEncode_new = []
    for i in columnsToEncode:
        if ( len(data[i].unique()) > 2 ):
            columnsToEncode_new.append(i)
        elif (  len(data[i].unique()) == 2 ):
            # Label Encode the variables 
            data[i] = data[i].astype('category')
            data[i] = data[i].cat.codes
    columnsToEncode = columnsToEncode_new
    # ^ FIND A BETTER WAY
    print('These  are going to be onehot-encoded. Which do you want to exclude?:\n' , columnsToEncode)   
    temp = input() 
    temp = temp.split(',')
    print(temp)
    for i in temp:
        if  i != '':
            columnsToEncode.remove(i)
    for i in columnsToEncode:
        data[i] = data[i].apply(str)
     # ^    
    print('final columsn to encode', columnsToEncode)
    for feature in columnsToEncode:
        try:
            data[feature] = le.fit_transform(data[feature])
            data = pd.concat([data,pd.get_dummies(data[feature]).rename(columns=lambda x: feature + '_' + str(x))], axis=1)
            data = data.drop(feature, axis=1)
        except:
            print('Error encoding ' + feature)
            data[feature]  = data[feature].apply(pd.to_numeric, errors='coerce')
            
    return data

def label_encode(var_list, data, le_dict = {}, exclude=[]):
    for i in var_list:
        data[i] = data[i].apply(str)
    
    if not le_dict:
        columnsToEncode = list(data.select_dtypes(include=['category','object']))
    else:
        columnsToEncode = le_dict.keys()   
    
    # Exclude dummies 
    columnsToEncode_new = []
    for i in columnsToEncode:
        if ( len(data[i].unique()) > 2 ):
            columnsToEncode_new.append(i)
        elif (  len(data[i].unique()) == 2 ):
            # Label Encode the variables 
            data[i] = data[i].astype('category')
            data[i] = data[i].cat.codes
    columnsToEncode = columnsToEncode_new

    # ^ FIND A BETTER WAY
    print('These  are going to be label-encoded. Which do you want to exclude?:\n' , columnsToEncode)   ##
    temp = input() 
    temp = temp.split(',')
    print(temp)
    for i in temp:
        if  i != '':
            columnsToEncode.remove(i)
    
    for i in columnsToEncode:
        data[i] = data[i].apply(str)
     # ^
        
    for feature in columnsToEncode:
        try:
            data[feature] = data[feature].astype('category')
            data[feature] = data[feature].cat.codes
        except:
            print('Error encoding ' + feature)      
            
    return data

# Dropping Variables
def remove_vars(df, var_list=[]):
    print("You have these variables, tell me which ones you do not need.")
    print( list(df.columns) )
    if var_list == []:
        temp = input()
        var_list = temp.split(',')          
            
    if var_list != ['']:                    # ^ Still gives an error sometimes ! 
        df = df.drop(columns = var_list)
      
    return df

### ---- Creating Fakes datasets ----
def permute_rand(data,target, frac=1,var_lst=None):
    """ Permuted the values of some variables in dataframe """
    if var_lst is None:
        var_lst = []
        print('Using automatically identified discrete variable')
        for i in list(data.columns):
            if ( (len(data[i].unique()) <= settings.cat_unique_th) and (i!=target) ):
                var_lst.append(i)
    res = pd.DataFrame() 
    for feat in var_lst:
        sample_size = int( len(data)* frac )
        temp = data.sample(sample_size)
        modified_indices = list(temp.index)
        df_1st = data.drop(index = modified_indices)
        temp['new'] = np.random.permutation(temp[feat])
        temp.index = range(0,len(temp)) 
        temp[feat] = temp['new']
        temp = temp.drop(columns='new')
        new = pd.concat([df_1st,temp])[feat]
        res[feat] = new
    cont = list(set(list(data.columns)) - set(list(res.columns)) )
    for i in cont:
        res[i] = data[i]
    res[target] = data[target]
        
    return res

def add_noise(data,var_lst=None,frac=1,fact=2.0):
    """ Add noise to continuous variables """
    dist = np.random.rand
    dist_lst = [ np.random.randn, np.random.uniform, np.random.gamma, np.random.normal ] 
    # params for different noises
    shape, scale = 2., 2. # Gamma 
    mu, sigma = 0, 0.1    # Guass
    if var_lst is None:
        var_lst = []
        print('Using automatically identified discrete variable')
        for i in list(data.columns):
            if ( (len(data[i].unique()) > settings.cat_unique_th) and (i!=target) ):
                var_lst.append(i)
    res = pd.DataFrame() 
    for feat in var_lst:
        mult = data[feat].std() / fact
        sample_size = int( len(data) * frac )
        temp = data.sample(sample_size)
        modified_indices = list(temp.index)
        df_1st = data.drop(index = modified_indices)
        noise = mult * dist(sample_size)
        noise = pd.Series(noise)
        temp2 = temp[feat] + noise
        temp.index = range(0,len(temp)) 
        temp[feat] = temp2
        new = pd.concat([df_1st,temp])[feat]
        res[feat] = new
    disc = list(set(list(data.columns)) - set(list(res.columns)) )
    for i in disc:
        res[i] = data[i]
    res[target] = data[target]
        
    return res

def permute_add_noise(data,target,var_lst=None,frac=1,fact=2.0):
    """ Permuted Discrete and add noise to Continuous """
    if var_lst is None:
        var_lst = []
        print('Using automatically identified discrete variable')
        for i in list(data.columns):
            if ( (len(data[i].unique()) <= settings.cat_unique_th) and (i!=target) ):
                var_lst.append((i,1))
            elif (len(data[i].unique()) > settings.cat_unique_th):
                var_lst.append((i,0))
    res = pd.DataFrame()
    dist = np.random.rand
    for feat in var_lst:
        flag = feat[1]
        feat = feat[0]
        if ( flag ==1 ): 
            sample_size = int( len(data)* frac )
            temp = data.sample(sample_size)
            modified_indices = list(temp.index)
            df_1st = data.drop(index = modified_indices)
            temp['new'] = np.random.permutation(temp[feat])
            temp.index = range(0,len(temp)) 
            temp[feat] = temp['new']
            temp = temp.drop(columns='new')
            new = pd.concat([df_1st,temp])[feat]
            res[feat] = new
        else:
            mult = data[feat].std() / fact
            sample_size = int( len(data) * frac )
            temp = data.sample(sample_size)
            modified_indices = list(temp.index)
            df_1st = data.drop(index = modified_indices)
            noise = mult * dist(sample_size)
            noise = pd.Series(noise)
            temp2 = temp[feat] + noise
            temp.index = range(0,len(temp)) 
            temp[feat] = temp2
            new = pd.concat([df_1st,temp])[feat]
            res[feat] = new
    res[target] = data[target]
        
    return res

                                          #  ------  Mics  ------
def pivot_func_on():
    qgrid.enable()
    
    return

def pivot_func_off():
    qgrid.disable()
    
    return

def save_html(html_):
    a = html_
    html = a.data
    with open('Symmary_Plots.html', 'w') as f:
        f.write(html)
    
    return   

def save_df(df):
    name = 'DataFrame_Dora'
    df.to_csv(name + '.csv','w')
    
    return
    
def add_styling():
    # Add some styling and change precision
    pd.set_option('precision', 2)
    pd.set_option('float_format', '{:.2f}'.format)
    pd.set_option('display.max_colwidth', -2)
    pd.set_option('display.height', 10000000)
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.width', 1000000)

    CSS = """
    body {
        margin: 0;
        font-family: Helvetica;
    }
    table.dataframe {
        border-collapse: collapse;
        border: none;
    }
    table.dataframe tr {
        border: none;
    }
    table.dataframe td, table.dataframe th {
        margin: 0;
        border: 1px solid white;
        padding-left: 0.25em;
        padding-right: 0.25em;
    }
    table.dataframe th:not(:empty) {
        background-color: #fec;
        text-align: left;
        font-weight: normal;
    }
    table.dataframe tr:nth-child(2) th:empty {
        border-left: none;
        border-right: 1px dashed #888;
    }
    table.dataframe td {
        border: 2px solid #ccf;
        background-color: #f4f4ff;
    }
    """
    HTML('<style>{}</style>'.format(CSS))
    
    return 





