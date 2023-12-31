#Natalia Czub


# import packages
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro
from scikit_posthocs import posthoc_dunn
from scipy.stats import levene

from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sa
import statsmodels.formula.api as sfa
import scikit_posthocs as sp
from itertools import combinations

# import data
file = "example_of_mordred_descriptors_serotonin_receptors.csv"
df_descriptor = pd.read_csv(file)
grouping_variable = "receptor"
descriptor = "mordred"
file_name = f"ser_rec_{descriptor}"
# Assumption
# α = 0.05
alpha = 0.05

#define groups - 10 groups, as there are in grouping_variable
group_1 = '5-HT1A'
group_2 = '5-HT1B'
group_3 = '5-HT1D'
group_4 = '5-HT2A'
group_5 = '5-HT2B'
group_6 = '5-HT2C'
group_7 = '5-HT3'
group_8 = '5-HT5A'
group_9 = '5-HT6'
group_10 = '5-HT7'



# DO NOT CHANGED
# delete constant columns in file
def delete_constant_columns(df, output_file):
    unique_columns = df.columns[df.nunique() > 1]
    constant_columns = df.columns[df.nunique() == 1]
    df[constant_columns].to_csv(output_file, index=False)
    df = df[unique_columns]
    df.columns = df.columns.str.replace('/', '_')
    return df
df_descriptor = delete_constant_columns(df_descriptor, f'constant_columns_{descriptor}.csv')

df_descriptor_clean = df_descriptor.iloc[:, 2:]
df_descriptor_clean.head(2)

# Shapiro-Wilk test (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html)
# Checking the normal distribution for each column and each category of serotonin receptor
# Hypothesis
# H0 - variable has normal distribution
# H1 - variable has no normal distribution
# normal distribution: p value >= α 


list_normal_distribution_column = []
list_no_normal_distribution_column = []

for column in df_descriptor_clean:
    p_group_1 = shapiro(df_descriptor_clean[df_descriptor[grouping_variable] == group_1][column])[1]
    p_group_2 = shapiro(df_descriptor_clean[df_descriptor[grouping_variable] == group_2][column])[1]
    p_group_3 = shapiro(df_descriptor_clean[df_descriptor[grouping_variable] == group_3][column])[1]
    p_group_4 = shapiro(df_descriptor_clean[df_descriptor[grouping_variable] == group_4][column])[1]
    p_group_5 = shapiro(df_descriptor_clean[df_descriptor[grouping_variable] == group_5][column])[1]
    p_group_6 = shapiro(df_descriptor_clean[df_descriptor[grouping_variable] == group_6][column])[1]
    p_group_7 = shapiro(df_descriptor_clean[df_descriptor[grouping_variable] == group_7][column])[1]
    p_group_8 = shapiro(df_descriptor_clean[df_descriptor[grouping_variable] == group_8][column])[1]
    p_group_9 = shapiro(df_descriptor_clean[df_descriptor[grouping_variable] == group_9][column])[1]
    p_group_10 = shapiro(df_descriptor_clean[df_descriptor[grouping_variable] == group_10][column])[1]
    if ((p_group_1 >= alpha) and (p_group_2 >= alpha) and (p_group_3 >= alpha) and (p_group_4 >= alpha) and (p_group_5 >= alpha)
       and (p_group_6 >= alpha) and (p_group_7 >= alpha) and (p_group_8 >= alpha) and (p_group_9 >= alpha) and (p_group_10 >= alpha)):
        list_normal_distribution_column.append((column, p_group_1, p_group_2, p_group_3, p_group_4, p_group_5, p_group_6, p_group_7, p_group_8, 
                                               p_group_9, p_group_10))
        normal_distribution_column = pd.DataFrame(list_normal_distribution_column, columns = ["descriptor", 
                                                                                              f"p_{grouping_variable}_{group_1}",f"p_{grouping_variable}_{group_2}",
                                                                                              f"p_{grouping_variable}_{group_3}",f"p_{grouping_variable}_{group_4}",
                                                                                              f"p_{grouping_variable}_{group_5}",f"p_{grouping_variable}_{group_6}",
                                                                                              f"p_{grouping_variable}_{group_7}", f"p_{grouping_variable}_{group_8}", 
                                                                                              f"p_{grouping_variable}_{group_9}", f"p_{grouping_variable}_{group_10}"])
        normal_distribution_column.to_csv(f"{file_name}_normal_distribution.csv", index = False, header = True)
    
    else:
        list_no_normal_distribution_column.append((column, p_group_1, p_group_2, p_group_3, p_group_4, p_group_5, p_group_6, p_group_7, p_group_8, 
                                               p_group_9, p_group_10))
        no_normal_distribution_column = pd.DataFrame(list_no_normal_distribution_column, columns = ["descriptor", 
                                                                                              f"p_{grouping_variable}_{group_1}",f"p_{grouping_variable}_{group_2}",
                                                                                              f"p_{grouping_variable}_{group_3}",f"p_{grouping_variable}_{group_4}",
                                                                                              f"p_{grouping_variable}_{group_5}",f"p_{grouping_variable}_{group_6}",
                                                                                              f"p_{grouping_variable}_{group_7}", f"p_{grouping_variable}_{group_8}", 
                                                                                              f"p_{grouping_variable}_{group_9}", f"p_{grouping_variable}_{group_10}"])
        no_normal_distribution_column.to_csv(f"{file_name}_no_normal_distribution.csv", index = False, header = True)
  

# NO NORMAL DISTRIBUTION
# Kruskal-Wallis test - searching for statistically significant differences for groups without normal distribution
# (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html)
# H0 = no differences between groups
# H1 = there is at least one pair of values that are different from each other

# read file with descriptors with normal distribution
data_no_normal_dist = pd.read_csv(f"{file_name}_no_normal_distribution.csv")
data_no_normal_dist_list = data_no_normal_dist.descriptor
df = pd.read_csv(file)
df.columns = df.columns.str.replace('/', '_')
df_no_normal_dist = pd.concat([df["id"], df[grouping_variable], df[data_no_normal_dist_list]], axis = 1)
columns_list = df_no_normal_dist.iloc[:, 2:].columns
results = []
for column in columns_list:
    grouped_test2=df_no_normal_dist[[grouping_variable, column]].groupby([grouping_variable])
    statistic, pvalue = stats.kruskal(grouped_test2.get_group(group_1)[column], grouped_test2.get_group(group_2)[column], grouped_test2.get_group(group_3)[column],
                                  grouped_test2.get_group(group_4)[column], grouped_test2.get_group(group_5)[column],grouped_test2.get_group(group_6)[column],
                                  grouped_test2.get_group(group_7)[column], grouped_test2.get_group(group_8)[column], 
                                  grouped_test2.get_group(group_9)[column], grouped_test2.get_group(group_10)[column])

    results.append((column, statistic, pvalue))
df_stat = pd.DataFrame(results)
df_stat.columns =['descriptor', 'F_value', 'p_value']
df_stat.sort_values(["F_value"], ascending = False)
df_stat.to_csv(f"kruskal_wallis_no_normal_distribution_{file_name}_all_columns.csv", 
               index = False, header = True)

df_stat_to_cut = df_stat[df_stat["p_value"] < alpha]
df_stat_to_cut.to_csv(f"kruskal_wallis_no_normal_distribution_{file_name}_stat_import_{alpha}.csv", 
                      index = False, header = True)
list_of_stat_important_column = df_stat_to_cut["descriptor"]
df = pd.read_csv(file)
df.columns = df.columns.str.replace('/', '_')
data = df[df.columns.intersection(df_stat_to_cut["descriptor"])]
data_all = pd.concat([df["id"], df[grouping_variable], data], axis = 1)


# post-hoc test for no normal distribution - test Dunn (https://scikit-posthocs.readthedocs.io/en/stable/generated/scikit_posthocs.posthoc_dunn/)
os.mkdir(f'posthoc_dunn_{descriptor}')
for descriptor_column in list_of_stat_important_column:
    result = sp.posthoc_dunn(data_all, val_col=descriptor_column, group_col='receptor')
    result_flattened = np.ndarray.flatten(result.values)
    groups = data_all['receptor'].unique()
    comparisons = sorted([f'comparison_{group1}_{group2}' for group1 in groups for group2 in groups if group1 != group2])
    comparisons_unique = []
    for comparison in comparisons:
        group1, group2 = comparison.split('_')[1], comparison.split('_')[2]
        reverse_comparison = f'comparison_{group2}_{group1}'
        if comparison not in comparisons_unique and reverse_comparison not in comparisons_unique:
            comparisons_unique.append(comparison)
    result_dict = {}
    for i, group1 in enumerate(groups):
        for j, group2 in enumerate(groups):
            if group1 != group2:
                comparison = f'comparison_{group1}_{group2}'
                if comparison in comparisons_unique:
                    value = result.loc[group1, group2]
                    result_dict[comparison] = value

    result_df = pd.DataFrame(result_dict, index=[0])
    result_df = result_df[comparisons_unique]
    result_df.insert(0, "descriptor_column", descriptor_column)
    result_df.to_csv(f'posthoc_dunn_{descriptor}/posthoc_results_dunn_{file_name}_{descriptor_column}.csv', index=False)
all_filenames = [i for i in glob.glob(f'posthoc_dunn_{descriptor}/*.csv')]
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
combined_csv.to_csv(f"posthoc_dunn_{file_name}_no_normal_distribution.csv", index = False, header = True)
reduced_df = combined_csv[(combined_csv.iloc[:, 1:] < alpha).all(axis=1)]
reduced_df.to_csv(f"posthoc_dunn_{file_name}_no_normal_distribution_descriptors_differentiate_all_receptors.csv", index = False, header = True)



# For variables with normal distribution        
# ANOVA (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html)
# 'The one-way ANOVA tests the null hypothesis that two or more groups have the same population mean. The test is applied to samples from two or more groups, possibly with differing sizes.'
# Hypothesis
# H0: μ1 = μ2 = μ3 = ... = μk
# (where μ1, μ2, μ3, ..., μk represent the population means of the k groups being compared)
# In other words, under the null hypothesis, all group means are equal.

# H1: At least one μi is different from the others
# (where μi represents one of the population means)
# In other words, under the alternative hypothesis, there is a difference in at least one of the group means.

# p value >= α -> accept null hypothesis (H0)
# read file with descriptors with normal distribution

data_normal_dist = pd.read_csv(f"{file_name}_normal_distribution.csv")
data_normal_dist_list = data_normal_dist.descriptor
# append all results of F and p value
df = pd.read_csv(file)
df_normal_dist = pd.concat([df["id"], df[grouping_variable], df[data_normal_dist_list]], axis = 1)
columns_list = df_normal_dist.iloc[:, 2:].columns
results = []
for column in columns_list:
    grouped_test2=df_normal_dist[[grouping_variable, column]].groupby([grouping_variable])
    f_val, p_val = stats.f_oneway(grouped_test2.get_group(group_1)[column], grouped_test2.get_group(group_2)[column], grouped_test2.get_group(group_3)[column],
                                  grouped_test2.get_group(group_4)[column], grouped_test2.get_group(group_5)[column],grouped_test2.get_group(group_6)[column],
                                  grouped_test2.get_group(group_7)[column], grouped_test2.get_group(group_8)[column], 
                                  grouped_test2.get_group(group_9)[column], grouped_test2.get_group(group_10)[column])

    results.append((column, f_val, p_val))
df_stat = pd.DataFrame(results)
df_stat.columns =['descriptor', 'F_value', 'p_value']
df_stat.sort_values(["F_value"], ascending = False)
df_stat.to_csv(f"anova_statistics_normal_distribution_{file_name}_all_columns.csv", 
               index = False, header = True)    

df_stat_to_cut = pd.read_csv(f"anova_statistics_normal_distribution_{file_name}_all_columns.csv")
df_stat_to_cut = df_stat_to_cut[df_stat_to_cut["p_value"] < alpha]
df_stat_to_cut.to_csv(f"anova_statistics_normal_distribution_{file_name}_stat_importance_{alpha}.csv", index = False, header = True)
list_of_anova_variables = df_stat_to_cut["descriptor"]

# Levene's test
#To know which post-hoc test to choose, you need to check "homoscedasticity" between groups using Levene's test
# (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levene.html)
# Levene's test - a test whose task is to assess whether the variance in our data set is equal across the groups we analyze.
# This method tests the null hypothesis indicating equality of variances.
# Therefore, if the significance value of the Levene test is less than p <  α, then we consider the variances to be heterogeneous,
# i.e. there are differences between the variances in the compared groups.
# However, in the case of a statistically insignificant result (p >  α), we assume homogeneity of variance.
# H0 = homogeneous variances 
# H1 = heterogeneous variances

homoscedasticity_list = []
heteroscedasticity_list = []
for column in list_of_anova_variables:
    grouped_test2=data_all[[grouping_variable, column]].groupby([grouping_variable])
    stat, p = levene(grouped_test2.get_group(group_1)[column], grouped_test2.get_group(group_2)[column], grouped_test2.get_group(group_3)[column],
                                  grouped_test2.get_group(group_4)[column], grouped_test2.get_group(group_5)[column],grouped_test2.get_group(group_6)[column],
                                  grouped_test2.get_group(group_7)[column], grouped_test2.get_group(group_8)[column], 
                                  grouped_test2.get_group(group_9)[column], grouped_test2.get_group(group_10)[column])
    p_value_levene = p
    if p_value_levene > alpha:
        homoscedasticity_list.append((column, p_value_levene, stat))
    else:
        heteroscedasticity_list.append((column, p_value_levene, stat))

homoscedasticity_df = pd.DataFrame(homoscedasticity_list, columns=["descriptor", "p_value_levene", "stat_levene"])
print("homoscedasticity variable:", homoscedasticity_list)
heteroscedasticity_df = pd.DataFrame(heteroscedasticity_list, columns=["descriptor", "p_value_levene", "stat_levene"])
print("heteroscedasticity variable:", heteroscedasticity_list)
data_heteroscedasticity = df[df.columns.intersection(heteroscedasticity_df["descriptor"])]
data_heteroscedasticity = pd.concat([df["id"], df["receptor"], data_heteroscedasticity], axis=1)




# post-hoc for hereroscedasticity Dunnett test (https://scipy.github.io/devdocs/reference/generated/scipy.stats.dunnett.html)
dunnett_results = []
for descriptor_column in heteroscedasticity_list:
    result = sp.posthoc_dunnett(data_all, val_col=descriptor_column, group_col='receptor')
    result_flattened = np.ndarray.flatten(result.values)
    groups = data_all['receptor'].unique()
    comparisons = sorted([f'comparison_{group1}_{group2}' for group1 in groups for group2 in groups if group1 != group2])

    comparisons_unique = []
    for comparison in comparisons:
        group1, group2 = comparison.split('_')[1], comparison.split('_')[2]
        reverse_comparison = f'comparison_{group2}_{group1}'
        if comparison not in comparisons_unique and reverse_comparison not in comparisons_unique:
            comparisons_unique.append(comparison)

    result_dict = {}

    for i, group1 in enumerate(groups):
        for j, group2 in enumerate(groups):
            if group1 != group2:
                comparison = f'comparison_{group1}_{group2}'
                if comparison in comparisons_unique:
                    value = result.loc[group1, group2]
                    result_dict[comparison] = value

    result_df = pd.DataFrame(result_dict, index=[0])
    result_df = result_df[comparisons_unique]

    result_df.insert(0, "descriptor_column", descriptor_column)
    result_df.to_csv(f'posthoc_results_dunnett_{file_name}_{descriptor_column}.csv', index=False)


# homoscedasticity - Tukey test (https://www.statsmodels.org/dev/generated/statsmodels.stats.multicomp.pairwise_tukeyhsd.html)
data_homoscedasticity = df[df.columns.intersection(homoscedasticity_df["descriptor"])]
data_homoscedasticity = pd.concat([df["id"], df[grouping_variable], data_homoscedasticity], axis = 1)
data_homoscedasticity_cut = data_homoscedasticity.iloc[:, 2:]
variable_diffs = {}
all_tukey_results = []
group_combinations = list(combinations(data_homoscedasticity[grouping_variable].unique(), 2))
for variable in data_homoscedasticity_cut:
    p_values = []
    for group_1, group_2 in group_combinations:
        group_1_data = data_homoscedasticity[data_homoscedasticity[grouping_variable] == group_1][variable]
        group_2_data = data_homoscedasticity[data_homoscedasticity[grouping_variable] == group_2][variable]
        tukey = pairwise_tukeyhsd(endog=pd.concat([group_1_data, group_2_data]),
                                  groups=pd.Series([group_1] * len(group_1_data) + [group_2] * len(group_2_data)),
                                  alpha=alpha)
        p_value = tukey.pvalues[1]
        p_values.append(p_value)
        all_tukey_results.append((variable, group_1, group_2, p_value))
    if all(p < alpha for p in p_values):
        variable_diffs[variable] = p_values

all_results_df = pd.DataFrame(all_tukey_results, columns=["descriptor", "group_1", "group_2", "p_value"])
all_results_df.to_csv(f'tukey_results_{file_name}_all_pairs.csv', index=False)
result_df = pd.DataFrame(variable_diffs)
result_df.to_csv(f'tukey_results_{file_name}_variables_differentiate_all_groups.csv', index=False)
