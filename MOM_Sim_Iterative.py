import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from scipy.stats import skewnorm, norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os
import sys

if(False):
    Group_extras = [-1,1]  # Values of extra weight for each group absolute value each time - not used b/c need variation in each trip
else:
    Group_extras = [0, 1]  # Values of extra weight for each group, how much to adjust each lo

# Function to get random number from distribution defined by mean, std, max, min
def get_random_from_distribution(the_mean, the_STD, the_max, the_min):
    X = np.random.normal(the_mean, the_STD)
    if X > the_max:
        X = the_max
    if X < the_min:
        X = the_min
    return X

# Function to build a list of birds and return as a DataFrame
def build_bird_list(n_birds, group, b_mean, b_std, b_min, b_max, extra_mean, extra_std = 1, Group_extras_Pct=0.2):
    bird_data = []
    group_index = int(group.split('_')[1]) - 1  # Assuming group names are formatted as 'Group_X'
    for i in range(n_birds):
        bird_ID = f"Bird_{group_index}_{i}"
        body_size = get_random_from_distribution(b_mean, b_std, b_max, b_min)
        WL = np.round(body_size * 0.156 + 151.09, 0)
        extra = Group_extras_Pct * Group_extras[group_index]
        bird_data.append({'group_ID': group, 'bird_ID': bird_ID, 'body_size': body_size, 'WL': WL, 'Extra': extra, 'group_index': group_index})

    bird_df = pd.DataFrame(bird_data)
    return bird_df

# Function to adjust the delivery amount by the group percentage
def adjust_delivery(the_load, the_delivery, the_pct, group):
    adj_extra = the_delivery * the_pct * group
    adj_delivery = the_delivery + adj_extra
    if adj_delivery >= the_load:
        adj_delivery = the_load - 0.1
    return adj_delivery

# Function to perform statistical tests and reporting using GLM
def do_Stats(chick_feeds_df):
    chick_feeds_df['MOM_Del_Size'] = pd.to_numeric(chick_feeds_df['MOM_Del_Size'], errors='coerce')
    chick_feeds_df['WL'] = pd.to_numeric(chick_feeds_df['WL'], errors='coerce')  
    
    chick_feeds_df['study_group'] = pd.Categorical(chick_feeds_df['study_group']).codes
    X = sm.add_constant(chick_feeds_df[['study_group', 'WL']])
    y = chick_feeds_df['MOM_Del_Size']
    model = sm.GLM(y, X, family=sm.families.Gaussian())
    results = model.fit()
    
    formula = 'MOM_Del_Size ~ study_group'
    Rand_eff_model = smf.mixedlm(formula, chick_feeds_df, groups=chick_feeds_df['bird_ID'])
    results_bfgs = model.fit(method='bfgs', maxiter=2000, full_output=True, disp=1)
    RE_p_value_study_group = round(results_bfgs.pvalues['study_group'], 3)
    
    group_stats = chick_feeds_df.groupby('study_group')['MOM_Del_Size'].agg(['mean', 'std', 'count'])
    group_stats['std_error'] = (group_stats['std'] / group_stats['count'] ** 0.5).round(2)

    mean_group1 = round(group_stats.loc[0, 'mean'], 2)
    std_error_group1 = round(group_stats.loc[0, 'std_error'], 2)
    mean_group2 = round(group_stats.loc[1, 'mean'], 2)
    std_error_group2 = round(group_stats.loc[1, 'std_error'], 2)
    
    p_value_study_group = results.pvalues['study_group']
    X = sm.add_constant(chick_feeds_df['study_group'])
    y = chick_feeds_df['MOM_Del_Size']
    model = sm.OLS(y, X).fit()
    p_value_study_group2 = model.pvalues['study_group']
    
    return round(p_value_study_group2, 3), mean_group1, std_error_group1, mean_group2, std_error_group2, round(RE_p_value_study_group, 3)

# Main function to simulate the MOM process
def MOM_simulation(N_Birds_in_Group, N_Sims, Group_extras_Pct):
    bird_dfs = []
    chick_feeds_data = []
    
    for group in range(1, 3):  # Assuming N_Groups = 2
        group_name = f'Group_{group}'
        bird_df = build_bird_list(N_Birds_in_Group, group_name, 45, 1, 40, 55, 2.0, 1, Group_extras_Pct)
        bird_dfs.append(bird_df)
        
        for index, bird in bird_df.iterrows():
            for _ in range(8):  # Assuming Trip_per_bird = 8
                load_size = get_random_from_distribution(15, 1, 20, 4)
                MOM_Arr_err = get_random_from_distribution(0, 1.8, 7, -7)
                Bird_Arr_Size = bird['body_size'] + load_size
                MOM_Arr_Size = Bird_Arr_Size + MOM_Arr_err
                Del_size = get_random_from_distribution(7.67, 2.6, 14, 1)
                Del_size = adjust_delivery(load_size, Del_size, Group_extras_Pct, bird['group_index'])
                MOM_del_err = get_random_from_distribution(0, 1.8, 7, -7)
                Bird_Depart_Size = MOM_Arr_Size - (Del_size + bird['Extra'])
                MOM_Dep_Size = Bird_Depart_Size + MOM_del_err
                MOM_Del_Size = MOM_Arr_Size - MOM_Dep_Size

                chick_feeds_data.append({
                    'study_group': bird['group_ID'],
                    'bird_ID': bird['bird_ID'],
                    'WL': bird['WL'],
                    'body_size_of_bird': np.round(bird['body_size'], 1),
                    'load_size': np.round(load_size, 1),
                    'Bird_Arr_Size': np.round(Bird_Arr_Size, 1),
                    'MOM_Arr_Size': np.round(MOM_Arr_Size, 1),
                    'Del_Size': np.round(Del_size, 1),
                    'MOM_Dep_Size': np.round(MOM_Dep_Size, 1),
                    'Bird_Depart_Size': np.round(Bird_Depart_Size, 1),
                    'MOM_Del_Size': np.round(MOM_Del_Size, 1),
                    'MOM_Arr_err': np.round(MOM_Arr_err, 1),
                    'MOM_del_err': np.round(MOM_del_err, 1)
                })
    
    chick_feeds_df = pd.DataFrame(chick_feeds_data)
    myReturn = do_Stats(chick_feeds_df)
    return myReturn[2]  # Returning count_RE_0_05 as part of the result tuple


# Function to run simulations and count 'P' <= 0.05
def run_simulation(N_Birds_in_Group, N_Sims, Group_extras_Pct):
    results_data = []
    
    for i in range(N_Sims):
        sim_results = MOM_simulation(N_Birds_in_Group, N_Sims, Group_extras_Pct)
        results_data.append({'P': sim_results[0], 'mean_group1': sim_results[1], 'std_error_group1': sim_results[2], 
                             'mean_group2': sim_results[3], 'std_error_group2': sim_results[4], 'RndEff_Mod_P': sim_results[5]})

    results_df = pd.DataFrame(results_data)
    
    count_RE_0_05 = (results_df['RndEff_Mod_P'] <= 0.05).sum()

    return count_RE_0_05

# Example usage:
N_Birds_in_Group = 15
N_Sims = 2
Group_extras_Pct = 0.2
count_RE_0_05 = run_simulation(N_Birds_in_Group, N_Sims, Group_extras_Pct)
print(f"Number of simulations with p-value <= 0.05: {count_RE_0_05}")
