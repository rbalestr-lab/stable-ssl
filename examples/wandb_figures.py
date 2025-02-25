"""
This script demonstrates how to retrieve data from wandb using the stable-SSL library and create plots from it.
"""

import stable_ssl as ssl
import re  
import matplotlib.pyplot as plt
from tqdm import tqdm

entity= "rbalestr-brown"
project = "LLM-spurious-correlation"

# want to retrieve finished runs from wandb
configs, dfs = ssl.reader.wandb_project(entity=entity,
                                        project=project,
                                        filters={"state": "finished"})

# access all runs with the wanted dataset and model backbone
wanted_dataset = "imdb"
wanted_backbone = "Snowflake/snowflake-arctic-embed-xs"   

""" This section allows for the users to define lists that will contain the information to plot on the graphs"""

# Define separate lists for different cases where spurious correlation injection locations
# full init and training
spurious_proportions_random = []
balanced_accuracies_random = []

spurious_proportions_end = []
balanced_accuracies_end = []

spurious_proportions_beginning = []
balanced_accuracies_beginning = []

# Lora Rank 2
spurious_proportions_random2 = []
balanced_accuracies_random2 = []

spurious_proportions_end2 = []
balanced_accuracies_end2 = []

spurious_proportions_beginning2 = []
balanced_accuracies_beginning2 = []


# Lora Rank 32
spurious_proportions_random32 = []
balanced_accuracies_random32 = []

spurious_proportions_end32 = []
balanced_accuracies_end32 = []

spurious_proportions_beginning32 = []
balanced_accuracies_beginning32 = []


# Iterate through runs and gather information
for run_id, df in tqdm(dfs.items(), desc="Processing runs", unit="run"):
    # Get the dataset, backbone, and run name from the runs
    dataset = df.get("dataset", "")
    backbone = df.get("backbone", "")
    run_name = df.get("run_name", "")

    # make sure the ones we are using met the conditions for what we want to graph
    if wanted_dataset.lower() in dataset.lower() and wanted_backbone.lower() in backbone.lower():
        # Extract spurious correlation proportion, location, and lora_rank used
        spurious_proportion = df.get("spurious_proportion", None)
        spurious_location = df.get("spurious_location", None)
        lora_rank = df.get("lora_rank", None)
        use_spurious = df.get("use_spurious", None)


        # only access if it contains everything wanted
        if spurious_proportion is not None and spurious_location is not None and spurious_proportion > 0 and lora_rank is not None and use_spurious == True:

            # Extract balanced accuracy from the run
            new_df, config = ssl.reader.wandb(entity,project,run_id)
            # drop the ones that are NAN
            balanced_acc = new_df["eval/NonSpurious_balanced_accuracy"].dropna()

            # Add the last one to be plotted
            if not balanced_acc.empty:
                balanced_acc = balanced_acc.iloc[-1]  # Get the last valid accuracy

                # Add to the correct lists depending on how the model was trained
                if lora_rank == 0:
                    if spurious_location == "random":
                        spurious_proportions_random.append(spurious_proportion)
                        balanced_accuracies_random.append(balanced_acc)
                    elif spurious_location == "end":
                        spurious_proportions_end.append(spurious_proportion)
                        balanced_accuracies_end.append(balanced_acc)
                    elif spurious_location == "beginning":
                        spurious_proportions_beginning.append(spurious_proportion)
                        balanced_accuracies_beginning.append(balanced_acc)
                elif lora_rank == 2:
                    if spurious_location == "random":
                        spurious_proportions_random2.append(spurious_proportion)
                        balanced_accuracies_random2.append(balanced_acc)
                    elif spurious_location == "end":
                        spurious_proportions_end2.append(spurious_proportion)
                        balanced_accuracies_end2.append(balanced_acc)
                    elif spurious_location == "beginning":
                        spurious_proportions_beginning2.append(spurious_proportion)
                        balanced_accuracies_beginning2.append(balanced_acc)
                elif lora_rank == 32:
                    if spurious_location == "random":
                        spurious_proportions_random32.append(spurious_proportion)
                        balanced_accuracies_random32.append(balanced_acc)
                    elif spurious_location == "end":
                        spurious_proportions_end32.append(spurious_proportion)
                        balanced_accuracies_end32.append(balanced_acc)
                    elif spurious_location == "beginning":
                        spurious_proportions_beginning32.append(spurious_proportion)
                        balanced_accuracies_beginning32.append(balanced_acc)


""" Functions used to simplify the plotting process, making it more extensible"""
# Sort values for plotting
def sort_and_unpack(proportions, accuracies):
    if proportions:  # Avoid empty lists
        sorted_data = sorted(zip(proportions, accuracies))
        return zip(*sorted_data)
    return [], []

def plot_data(x, y, label, linestyle="-", marker="o"):
    plt.plot(x, y, marker=marker, linestyle=linestyle, label=label)

# Sort values for plotting so that they are graphed in ascending order
spurious_proportions_random, balanced_accuracies_random = sort_and_unpack(spurious_proportions_random, balanced_accuracies_random)
spurious_proportions_end, balanced_accuracies_end = sort_and_unpack(spurious_proportions_end, balanced_accuracies_end)
spurious_proportions_beginning, balanced_accuracies_beginning = sort_and_unpack(spurious_proportions_beginning, balanced_accuracies_beginning)

spurious_proportions_random2, balanced_accuracies_random2 = sort_and_unpack(spurious_proportions_random2, balanced_accuracies_random2)
spurious_proportions_end2, balanced_accuracies_end2 = sort_and_unpack(spurious_proportions_end2, balanced_accuracies_end2)
spurious_proportions_beginning2, balanced_accuracies_beginning2 = sort_and_unpack(spurious_proportions_beginning2, balanced_accuracies_beginning2)

spurious_proportions_random32, balanced_accuracies_random32 = sort_and_unpack(spurious_proportions_random32, balanced_accuracies_random32)
spurious_proportions_end32, balanced_accuracies_end32 = sort_and_unpack(spurious_proportions_end32, balanced_accuracies_end32)
spurious_proportions_beginning32, balanced_accuracies_beginning32 = sort_and_unpack(spurious_proportions_beginning32, balanced_accuracies_beginning32)

# Create the figure
plt.figure(figsize=(10, 7))# Create the figure

# Plot each dataset with different styles
plot_data(spurious_proportions_random, balanced_accuracies_random, "Random", linestyle="--", marker="s")
plot_data(spurious_proportions_end, balanced_accuracies_end, "End", linestyle="-.", marker="d")
plot_data(spurious_proportions_beginning, balanced_accuracies_beginning, "Beginning", linestyle=":", marker="x")

plot_data(spurious_proportions_random2, balanced_accuracies_random2, "Random Lora 2", linestyle="--", marker="s")
plot_data(spurious_proportions_end2, balanced_accuracies_end2, "End Lora 2", linestyle="-.", marker="d")
plot_data(spurious_proportions_beginning2, balanced_accuracies_beginning2, "Beginning Lora 2", linestyle=":", marker="x")

plot_data(spurious_proportions_random32, balanced_accuracies_random32, "Random Lora 32", linestyle="--", marker="s")
plot_data(spurious_proportions_end32, balanced_accuracies_end32, "End Lora 32", linestyle="-.", marker="d")
plot_data(spurious_proportions_beginning32, balanced_accuracies_beginning32, "Beginning Lora 32", linestyle=":", marker="x")

# Label the plot and axis
plt.xlabel("Spurious Correlation Proportion")
plt.ylabel("Balanced Accuracy on Clean Test Set")
plt.title("Balanced Accuracy vs Spurious Correlation")
plt.grid()

# Save the figure locally
plt.savefig("balanced_accuracy_vs_spurious_correlation.png", dpi=300, bbox_inches="tight")

