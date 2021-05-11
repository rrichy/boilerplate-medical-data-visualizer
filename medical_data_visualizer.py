import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Import data
df = pd.read_csv(os.path.dirname(os.path.abspath(__file__)).replace('\\', '/') + '/medical_examination.csv')

# Add 'overweight' column
df['overweight'] = [1 if bmi > 25 else 0 for bmi in (df['weight'] / ((df['height'] / 100) ** 2)).values]
# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
for column in ['cholesterol', 'gluc']:
    df[column] = [0 if a == 1 else 1 for a in df[column].values]

df_heat = df[(df['ap_lo'] <= df['ap_hi']) & (df['height'] >= df['height'].quantile(0.025)) & (df['height'] <= df['height'].quantile(0.975)) & (df['weight'] >= df['weight'].quantile(0.025)) & (df['weight'] <= df['weight'].quantile(0.975))]
corr = df_heat.corr(method='pearson')

mask = np.triu(corr)

print(corr)
print(mask)
f, ax = plt.subplots(figsize=(11,9))
sns.heatmap(corr, square=True)

f.savefig('heatmap.png')
print(f)

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df[['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio', 'overweight']], id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    df_cat['total'] = df_cat['value']

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = df_cat.groupby(by=['cardio', 'variable', 'value']).count().reset_index()

    # Draw the catplot with 'sns.catplot()'
    fig = sns.catplot(x='variable', y='total', hue='value', data=df_cat, col='cardio', kind='bar')


    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) & (df['height'] >= df['height'].quantile(0.025)) & (df['height'] <= df['height'].quantile(0.975)) & (df['weight'] >= df['weight'].quantile(0.025)) & (df['weight'] <= df['weight'].quantile(0.975))]

    # Calculate the correlation matrix
    corr = df_heat.corr(method='pearson')

    # Generate a mask for the upper triangle
    mask = corr
    for col, column in enumerate(mask.columns):
        temp = mask[column]
        for row in range(mask.shape[1]):
            if row <= col: temp[row] = 1
            else: temp[row] = 0

        mask[column] = temp

    # Set up the matplotlib figure
    fig, ax = plt.subplots()

    # Draw the heatmap with 'sns.heatmap()'
    


    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
