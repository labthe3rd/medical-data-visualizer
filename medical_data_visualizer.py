import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv("medical_examination.csv")

# Add 'overweight' column
# BMI = weight(kg)/height(m)^2 then BMI greater than 25 is overweight
# Height in medical_examination.csv is in cm, we will convert to m by dividing by 100
# Now calculate the BMI
df["BMI"] = df["weight"] / ((df["height"]/100) ** 2)
# Add 'overweight' column, wet to 1 if BMI > 25 else 0
df["overweight"] = np.where(df["BMI"] > 25, 1, 0)


# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
# Create a normalize function for normalizing the cholestorol or gluc
def normalize(value):
    if value == 1:
        return 0
    if value > 1:
        return 1


# Now normalize the cols
cols_to_normalize = ["cholesterol", "gluc"]
df[cols_to_normalize] = df[cols_to_normalize].apply(lambda col: col.map(normalize))

# df[cols_to_normalize] = df[cols_to_normalize].applymap(normalize)


# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    # Melt the DataFrame to reshape it
    df_cat = pd.melt(
        df,
        id_vars=["cardio"],
        value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"],
    )

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = df_cat.groupby(["cardio", "variable", "value"], as_index=False).size()
    df_cat = df_cat.rename(columns={"size": "total"})

    # Draw the catplot with 'sns.catplot()'
    g = sns.catplot(
        x="variable", y="total", hue="value", col="cardio", data=df_cat, kind="bar"
    )

    # Get the figure for the output
    fig = g.fig

    # Do not modify the next two lines
    fig.savefig("catplot.png")
    return fig


def draw_heat_map():

    df_heat = df.loc[(df["ap_lo"] <= df["ap_hi"]) & 
                     (df['weight'] >= df['weight'].quantile(0.025)) & 
                     (df['weight'] <= df['weight'].quantile(0.975)) & 
                     (df['height'] >= df['height'].quantile(0.025)) & 
                     (df['height'] <= df['height'].quantile(0.975))]

    # Drop BMI from previous calculations
    if "BMI" in df_heat.columns:
        df_heat = df_heat.drop("BMI", axis=1)

    # Calculate the correlation matrix
    corr = df_heat.corr(method='pearson')

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(
        corr,
        mask=mask,
        cmap="icefire",
        vmax=0.3,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        annot=True,
        fmt=".1f",
    )
    # Do not modify the next two lines
    fig.savefig("heatmap.png")
    return fig

draw_heat_map()
