import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def calc_change(df, column_name):
    n = len(df.index)
    changes = np.zeros(n)
    for index in range(1, n):
        changes[index] = df[column_name].iloc[index] - df[column_name].iloc[index-1]
    return changes.astype('int')

def plot_data(df, column_name):
    features = df[column_name]
    fig, ax = plt.subplots()
    ax.hist(features, bins=100)
    plt.show()

cumulative_data = pd.read_csv("cumulative_data.csv")
print(cumulative_data.head(5))

daily_counts = pd.read_csv("daily_counts.csv")
print(daily_counts.head(5))

df = pd.merge(cumulative_data, daily_counts, how="outer")
df.drop(df.columns[[3, 6,7,8]], axis = 1, inplace = True)
df = df.fillna(0)
df["Daily Cases"] = df["Daily Cases"].astype(int)

column_name = "Daily Cases"
print(df[column_name].head(5))

n = len(df.index)

new_columns = [("Daily Cases", "Prior Daily Cases", "Change Daily Cases"), ("Died", "Prior Total Deaths", "Current Deaths"), ("Estimated Active", "Prior Active", "Change Active"), ("Current Deaths", "Prior Deaths", "Change Deaths")]
for new_column in new_columns:
    prior_column = df[new_column[0]].to_numpy(copy=True)
    prior_column = np.insert(prior_column, 0, 0)
    prior_column = np.delete(prior_column, n)
    df[new_column[1]] = prior_column.tolist()
    df[new_column[2]] = df[new_column[0]] - df[new_column[1]]

for column in ["Daily Cases", "Change Daily Cases", "Estimated Active", "Change Active", "Current Deaths", "Change Deaths"]:
    print("{}:  The maximum is {}, the minimum is {}, the mean is {} and the std dev is {}".format(column, df[column].max(), df[column].min(), df[column].mean(), df[column].std()))
#    plot_data(df, column)

def label_for_current_data(x, std_dev):
    if 0 <= x <= 0.5 * std_dev:
        return "L"
    elif 0.5 * std_dev < x <= std_dev:
        return "M"
    elif std_dev < x <= 2 * std_dev:
        return "H"
    else:
        return "E"

def label_for_change_in_data(x, std_dev):
    if 0 <= x <= 0.5 * std_dev:
        return "L+"
    elif 0.5 * std_dev < x <= std_dev:
        return "M+"
    elif std_dev < x:
        return "H+"
    elif -0.5 * std_dev <= x < 0:
        return "L-"
    elif -std_dev <= x < -0.5 * std_dev:
        return "M-"
    else:
        return "H-"

# Create Labels for daily cases and change in daily cases
#for columns in [("Daily Cases", "Change Daily Cases")]:
#    std_dev = df[columns[0]].std()
#    df[columns[0] + " Label"] = df[columns[0]].copy().apply(label_for_current_data, args=(std_dev,))
#    df[columns[1] + " Label"] = df[columns[1]].copy().apply(label_for_change_in_data, args=(std_dev,))


for columns in [("Daily Cases", "Change Daily Cases"), ("Estimated Active", "Change Active"),("Current Deaths", "Change Deaths")]:
    df[columns[0] + " Label"] = df[columns[0]].copy().apply(label_for_current_data, args=(df[columns[0]].std(),))
    df[columns[1] + " Label"] = df[columns[1]].copy().apply(label_for_change_in_data, args=(df[columns[1]].std(),))



print(df)
df.to_csv("covid_data.csv", index=False)
