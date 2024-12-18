################
#
#   MOM_Sim_06 by RAM, July 24 2024 based on 04 that works
#       simulates measuring chick feeding using MOMs in the field
#       test if we can detect differences between groups that deliver differnt loads given the error inherent in the MOMs
#       change from 04 - save bird ID, then average over bird ID or put into GLM with BirdID as a Random effect
#       many other changes since then, too
###########

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from scipy.stats import skewnorm, norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os
import sys


global N_Birds_in_Group, N_Groups, body_mean, body_STD, Trip_per_bird
global load_mean, load_STD, Err_mean, Err_STD, max_MOM_Err_stds
global Delivery_mean, Delivery_STD, Group_extras, Group_extras_Pct  # Include Group_extras as a global variable
global N_Sims
global alpha

# Define the global variables

# number of simulations
N_Sims = 1000

# significance level
alpha = 0.05

# Sample size issues
N_Birds_in_Group = 15  ## our proposal calls for 3 years at 15 birds/group per year, but will have 80 birds total per year, so...
N_Groups = 2            # test for 15 burrows that "worked", 15 birds in each group

# bird size cd
body_mean = 45
body_STD = 1
body_min = 40
body_max = 55

# foraging parameters
Trip_per_bird = 8  # was 9 days; have day 10 to 50 = 40 days, trip every 5 days = 8
load_mean = 15 # 15 gives RER results: was 15, why?, tried 12 and 10 and not much diff in final result, even down to 7.67 + 1 gram for self = 8.67, p = 0.04
load_STD = 1
load_min = 4
load_max = 20

# Error associated with the MOMs
Err_mean = 0         #W1 N=77, 0.53±1.72SD, W2 0.33±1.71SD ## for only >49g N=36 0.13±1.57g
Err_STD = 1.8 # 2.2  #1.93         # 1.72 - with Err_STD = 2, Err_Max/Min = 6, P = 0.01, 1.93 max/min = 7, P = 0.0 with 15 birds/group
Err_max = 7
Err_min = -7
max_MOM_Err_std = 5     # WHAT IS THE CUTOFF POINT FOR POSSIBLE ERRORS - IN UNITS OF STD. used to calculate MOM_Arr_err and MOM_Del_err

# background food delivery parameters (from RER 87 or a89?)
Delivery_mean = 7.67     # New global - Mean from RER = 8.67, lower so overall mean is same
Delivery_STD = 2.6    # New global - STD from RER = 2.94 
Delivery_max = 14
Delivery_min = 1

# Group effect for testing
Group_extras_Pct = 0.15 # how much less (more) one group will deliver than the other
if(False):
    Group_extras = [-1,1]  # Values of extra weight for each group absolute value each time - not used b/c need variation in each trip
else:
    Group_extras = [0, 1]  # Values of extra weight for each group, how much to adjust each lo

#########################
# Function to get random number from distribution defined by mean, std, max, min
#########################
def get_random_from_distribution(the_mean, the_STD, the_max, the_min):

    X = np.random.normal(the_mean, the_STD)
    # print(f"in the function1, here is arg mean: {the_mean} and here is X {X} and here is max {the_max}")
    if(X > the_max):
        X = the_max

    if(X < the_min):
        X = the_min
    # print(f"in the function2, here is arg mean: {the_mean} and here is X {X}")
    return X

#########
# generate_random_value
#   adjsts for a not quite normal distribuiton with kurtosis and skew specified
#   NOT using this as of July 27th. Using simple normal distribution as source
######
def generate_random_value(mean, std, skewness, kurtosis):
    # Generate a standard normal value
    z = np.random.normal(0, 1)
    
    # Adjust for skewness using skewnorm
    if skewness != 0:
        z = skewnorm.rvs(a=skewness, loc=0, scale=1)
    
    # Adjust for kurtosis
    if kurtosis != 3:
        adjustment_factor = ((kurtosis / 3.0) ** 0.5)  # Adjust tails, less than 3 means less in tails
        z = np.sign(z) * (abs(z) ** adjustment_factor)
    
    # Scale to desired mean and standard deviation
    value = mean + z * std
    
    return value

#########################
# Function to build a list of birds and return as a DataFrame
#########################
def build_bird_list(n_birds, group, b_mean, b_std, b_min, b_max, extra_mean, extra_std = 1):

    # Initialize an empty list to hold the bird data
    bird_data = []

    # Determine index for Group_extras based on group name
    group_index = int(group.split('_')[1]) - 1  # Assuming group names are formatted as 'Group_X'

    # Populate the list with n_birds
    for i in range(n_birds):
        #### generate a bird ID
        bird_ID = "Bird_" + str(group_index) + "_" + str(i) 
        body_size = get_random_from_distribution(b_mean, b_std, b_max, b_min)
        WL = np.round(body_size * 0.156 + 151.09, 0)  # Calculate WL and round to 0 decimal places - turns out it makes no difference so calc gives all same WL values
        if(False):
            # if we want a constant extra amount for each bird
            extra = get_random_from_distribution(extra_mean, extra_std, extra_mean + 2.0, extra_mean - 2.0) * Group_extras[group_index]
        else:
            # we want a pct to be applied to each delivery
            extra = Group_extras_Pct * Group_extras[group_index]
        bird_data.append({'group_ID': group, 'bird_ID': bird_ID, 'body_size': body_size, 'WL': WL, 'Extra': extra, 'group_index': group_index})

    # Convert the list to a DataFrame
    bird_df = pd.DataFrame(bird_data)
    
    return bird_df

#########################
# Function to adjust the delivery amount by the group pctg
#########################
def adjust_delivery(the_load, the_delivery, the_pct, group):
    
    # Calculate the amount extra of the adj delivery - cancel if in wrong group
    adj_extra = the_delivery * the_pct * group

    # add to the delivery planned
    adj_delivery = the_delivery + adj_extra

    # Ensure the adjusted delivery is at least 0.1 less than the load
    if adj_delivery >= the_load:
        adj_delivery = the_load - 0.1

    # print(f"adj_DEL: {round(adj_delivery,1)} load: {round(the_load,1)} delivery {round(the_delivery,1)} pct: {round(the_pct,1)}, group: {group}")
 
    return adj_delivery



#########################
# Function to perform statistical tests and reporting using GLM
#########################
def do_Stats(chick_feeds_df):

    ### flag for printing
    do_print = False

    # Convert necessary columns to numeric type
    chick_feeds_df['MOM_Del_Size'] = pd.to_numeric(chick_feeds_df['MOM_Del_Size'], errors='coerce')  # Handle any conversion errors
    chick_feeds_df['WL'] = pd.to_numeric(chick_feeds_df['WL'], errors='coerce')  # Similarly for other numeric columns
       
    # Convert study_group to numeric (0 and 1) - could just grab the new series in the bird_df? 
    chick_feeds_df['study_group'] = pd.Categorical(chick_feeds_df['study_group']).codes
    
    # Perform GLM with MOM_Del_Size as dependent variable and study_group + WL as independent variables
    X = sm.add_constant(chick_feeds_df[['study_group', 'WL']])
    y = chick_feeds_df['MOM_Del_Size']

    model = sm.GLM(y, X, family=sm.families.Gaussian())
    results = model.fit()

    # Perform GLM with MOM_Del_Size as dependent variable and study_group + WL as independent variables and Bird_ID as a random effect
    # Define the model formula
    formula = 'MOM_Del_Size ~ study_group'  # got rid of WL because caused problems of convergence

    # Fit the mixed-effects model with a random effect for Bird_ID
    Rand_eff_model = smf.mixedlm(formula, chick_feeds_df, groups=chick_feeds_df['bird_ID'])

    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

    # Your LMM modeling code here
    results_bfgs = model.fit(method='bfgs', maxiter=2000, full_output=True, disp=1)
    # print(results_bfgs.summary())
    RE_p_value_study_group = round(results_bfgs.pvalues['study_group'],3)
 

    # Reset stdout and stderr
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__   

    #############
    #  extract adjusted means
    #####
    if(True):

        # Calculate group statistics for MOM_Del_Size
        group_stats = chick_feeds_df.groupby('study_group')['MOM_Del_Size'].agg(['mean', 'std', 'count'])

        # Calculate standard error for each group
        group_stats['std_error'] = (group_stats['std'] / group_stats['count']**0.5).round(2)

        # Access mean and std_error for 'Group_1'
        mean_group1 = round(group_stats.loc[0, 'mean'],2) # assuming 1 corresponds to 'Group_1'
        std_error_group1 = round(group_stats.loc[0, 'std_error'],2)

        # print(f"Mean of MOM_Del_Size for Group_1 (rounded to 1 decimal place): {mean_group1} ± {std_error_group1}")

        mean_group2 = round(group_stats.loc[1, 'mean'],2) # assuming 1 corresponds to 'Group_1'
        std_error_group2 = round(group_stats.loc[1, 'std_error'],2)

        ######## change to get means of the arr and del errors
        mean_group2 = round(group_stats.loc[1, 'mean'],2) # assuming 1 corresponds to 'Group_1'
        std_error_group2 = round(group_stats.loc[1, 'std_error'],2)

        # print(f"Mean of MOM_Del_Size for Group_2 (rounded to 1 decimal place): {mean_group2} ± {std_error_group2}")
       



    if(do_print):
        # Print GLM summary including p-values
        print("\nGLM Summary for MOM_Del_Size as dependent variable and study_group + WL as independent variables:")
        print(results.summary())

    ##### compare MOM delivery to actual delivery
    chick_feeds_df['MOM_ERROR'] = chick_feeds_df['Del_Size'] - chick_feeds_df['MOM_Del_Size']
    # chick_feeds_df['MOM_ERROR'] = chick_feeds_df['MOM_Del_Size'] - chick_feeds_df['Del_Size']
    # Calculate stats of MOM error
    mean_MOM_ERROR = round(chick_feeds_df['MOM_ERROR'].mean(),2)
    std_MOM_ERROR = round(chick_feeds_df['MOM_ERROR'].std(),2)
    min_MOM_ERROR = round(chick_feeds_df['MOM_ERROR'].min(),2)
    max_MOM_ERROR = round(chick_feeds_df['MOM_ERROR'].max(),2)

    mean_Del_Size = round(chick_feeds_df['Del_Size'].mean(),2)
    std_Del_Size = round(chick_feeds_df['Del_Size'].std(),2)

    # save the mom errors
    MOM_Arr_err = round(chick_feeds_df['MOM_Arr_err'].mean(),2)
    MOM_del_err = round(chick_feeds_df['MOM_del_err'].mean(),2)

    # print("MOM errors:")
    # print(chick_feeds_df['MOM_ERROR'])
    # print(f"Mean: {mean_MOM_ERROR} STD: {std_MOM_ERROR} Range: {min_MOM_ERROR} to {max_MOM_ERROR}")

    # Calculate p-value for study_group from the regression
    p_value_study_group = results.pvalues['study_group']

    ### get p-value from using only the group
    # Define the independent variable (predictor) and dependent variable (response)
    X = sm.add_constant(chick_feeds_df['study_group'])  # Add a constant term for the intercept
    y = chick_feeds_df['MOM_Del_Size']
    
    # Fit the linear model
    model = sm.OLS(y, X).fit()
    
    # Get the p-value for the study_group coefficient
    p_value_study_group2 = model.pvalues['study_group']

    return round(p_value_study_group2,3), mean_MOM_ERROR, std_MOM_ERROR, min_MOM_ERROR, max_MOM_ERROR, mean_Del_Size, std_Del_Size, MOM_Arr_err, MOM_del_err, mean_group1, std_error_group1, mean_group2, std_error_group2, round(RE_p_value_study_group,3)

   

#########################
# Main function to simulate the MOM process
#########################
def MOM_simulation():
    
    bird_dfs = []
    chick_feeds_data = []
    
    # Generate a DataFrame for each group
    for group in range(1, N_Groups + 1):
        group_name = f'Group_{group}'
        bird_df = build_bird_list(N_Birds_in_Group, group_name, body_mean, body_STD, body_min, body_max, 2.0, 1)
        bird_dfs.append(bird_df)
        
        # Iterate through each bird in the DataFrame
        for index, bird in bird_df.iterrows():
            for _ in range(Trip_per_bird):
                # Arrival calculations
                # print("Load - random")
                load_size = get_random_from_distribution(load_mean, load_STD, load_max, load_min)
                
                if(True):
                    # draw from strict normal distribution
                    #MOM_Arr_err = get_random_from_distribution(Err_mean, Err_STD, Err_max, Err_min)
                    MOM_Arr_err = get_random_from_distribution(Err_mean, Err_STD, Err_max, Err_min)
                    # print(f"MOM ARR ERR: {round(MOM_Arr_err,2)}")
                else:
                    # draw from distribution adusted for same skew and kurtosis as our data
                    MOM_Arr_err = generate_random_value(Err_mean, Err_STD, 1.44, 3.55)

                Bird_Arr_Size = bird['body_size'] + load_size
                MOM_Arr_Size = Bird_Arr_Size + MOM_Arr_err
                
                # Delivery calculations using Delivery_mean and Delivery_STD
                # print(f"Delivery - random - mean: {Delivery_mean}")
                Del_size = get_random_from_distribution(Delivery_mean, Delivery_STD, (load_size - 1), Delivery_min)
                # print(f"Delivery - returned: {Del_size}")
                # Adjust Delivery Size for the group percentage 
                Del_size = adjust_delivery(load_size, Del_size, Group_extras_Pct, bird['group_index'])
                # print(f"Delivery - Adjusted: {Del_size}")

                # MOM_del_err = get_random_from_distribution(Err_mean, Err_STD, Err_max, Err_min)
                #MOM_del_err = generate_random_value(Err_mean, Err_STD, 1.44, 3.55)
                #print(f"MOM_del_err: {round(MOM_del_err,1)}")
                MOM_del_err = get_random_from_distribution(Err_mean, Err_STD, Err_max, Err_min)
                # print(f"MOM_del_err: {round(MOM_del_err,1)}")
                
                # Ensure Bird_Depart_Size > Bird_Arr_Size
                Bird_Depart_Size = MOM_Arr_Size - (Del_size + bird['Extra'])  # Include 'Extra' here
                if Bird_Depart_Size >= Bird_Arr_Size:
                    Del_size = load_size - 0.1  # Subtract 0.1 from load_size so that bird delivers everything
                    Bird_Depart_Size = Bird_Arr_Size - Del_size
                
                MOM_Dep_Size = Bird_Depart_Size + MOM_del_err
                MOM_Del_Size = MOM_Arr_Size - MOM_Dep_Size

                chick_feeds_data.append({
                    'study_group': bird['group_ID'],
                    'bird_ID': bird['bird_ID'],
                    'WL': bird['WL'],  # Add WL to chick_feeds_data
                    'body_size_of_bird': np.round(bird['body_size'], 1),
                    'load_size': np.round(load_size, 1),
                    'Bird_Arr_Size': np.round(Bird_Arr_Size, 1),
                    'MOM_Arr_Size': np.round(MOM_Arr_Size, 1),
                    'Del_Size': np.round(Del_size, 1),
                    'MOM_Dep_Size': np.round(MOM_Dep_Size, 1),
                    'Bird_Depart_Size': np.round(Bird_Depart_Size, 1),
                    'MOM_Del_Size': np.round(MOM_Del_Size, 1),  # Added MOM_Del_Size
                    'MOM_Arr_err': np.round(MOM_Arr_err, 1),  # Added MOM_Del_Size
                    'MOM_del_err': np.round(MOM_del_err, 1)
                })
    
    # Convert the chick_feeds_data list to a DataFrame
    chick_feeds_df = pd.DataFrame(chick_feeds_data)
    
    # Reorder columns as requested
    chick_feeds_df = chick_feeds_df[['study_group', 'bird_ID', 'WL', 'body_size_of_bird', 'load_size', 'Bird_Arr_Size', 'MOM_Arr_Size', 'Del_Size', 'MOM_Dep_Size', 'Bird_Depart_Size', 'MOM_Del_Size', 'MOM_Arr_err', 'MOM_del_err']]
    
    if(False):
        # Print the values in each DataFrame
        for bird_df in bird_dfs:
            print(bird_df)
            print()
        
        # Print the chick_feeds DataFrame
        print(chick_feeds_df)

    myReturn = do_Stats(chick_feeds_df)

    # print("chickfeeds before return")
    # print(myReturn)
    
    return myReturn


###############
# Run the iterative simulation
##
#### show the user what we are doing while she waits...
print(f"Running: {N_Sims} iterations at alpha {alpha} for {N_Birds_in_Group} birds/group with {Trip_per_bird} trip/bird with difference of {Group_extras_Pct} - Be patient!")

results_data = []   # somewhere to hold all the results for exporting or reporting at the end

# Populate the list with n_birds
for i in range(N_Sims):
    iteration = i
    if(False):      # old way that only returned a single p-value. no longer used
        my_Results_df = round(MOM_simulation(),2)
        results_data.append({'Group Extras': Group_extras, 'P': p_value})
    else:
        sim_results = MOM_simulation()
        results_data.append({
            'P': sim_results[0],
            'mean_MOM_ERROR': sim_results[1],
            'std_MOM_ERROR': sim_results[2],
            'min_MOM_ERROR': sim_results[3],
            'max_MOM_ERROR': sim_results[4],
            'mean_Del_Size': sim_results[5],
            'std_Del_Size': sim_results[6],
            'MOM_Arr_err': sim_results[7],
            'MOM_del_err': sim_results[8],
            'mean_group1': sim_results[9],      ### these hold mean ± SE for treatment groups
            'std_error_group1': sim_results[10],
            'mean_group2': sim_results[11],
            'std_error_group2': sim_results[12],
            'RndEff_Mod_P': sim_results[13]
    })

# print("results_df:")
# print(results_df)

print(f"Results for group difference for {N_Birds_in_Group} birds/group with {Trip_per_bird} trips/bird with difference of {Group_extras_Pct*100}%")

# Count the number of 'P' values <= 0.05
results_df = pd.DataFrame(results_data)
count_p_le_0_05 = (results_df['P'] <= alpha).sum()
print(f"Number iterations below {alpha}: {count_p_le_0_05} out of: {len(results_df['P'])} -- P = {round(1-count_p_le_0_05/len(results_df['P']),2)}")

count_RE_0_05 = (results_df['RndEff_Mod_P'] <= alpha).sum()
# print(f"Number iterations below {alpha}: {count_RE_0_05} out of: {len(results_df['P'])} -- P = {round(1-count_p_le_0_05/len(results_df['P']),2)}")

# print the summary results - mean value for each of them across all iterations
print("Means:")
print(results_df.mean().round(2))




