import os
import shutil
import json
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from image_generation import generate_images
from image_safety import img_safety_check
import argparse


def run_model(directory, model, prompts):
    """
    Prompts the model to generate the images and checks if the images are safe
    @param directory: directory to save the images in
    @param model: model name
    @param prompts: list of prompts for the model to generate images of
    @return tuple: (hard_punt, soft_punt, success) in percentages
    """
    # Generate images
    num_generated, num_rejected = generate_images(prompts, directory, model)
    # Safety check
    cnt_unsafe, cnt_safe = img_safety_check(directory)

    res = [num_rejected, cnt_safe, cnt_unsafe]
    return [x / sum(res) for x in res]


def plot_experiment(result_directory, data):
    """
    Plots the results of the experiment
    @param result_directory: directory to save the plot
    @param models: list of models
    @param hard_punt: list of hard punt percentages
    @param soft_punt: list of soft punt percentages
    @param success: list of success percentages
    """

    # Plot parameters
    colors = ['#e05658', '#f3d065', '#58a04e']  # success, soft punt, hard punt

    # Create figure with two subplots side by side
    plt.figure(figsize=(12, 6), dpi=300)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        for ax in [ax1, ax2]:
            ax.spines[s].set_visible(False)

    # Add main title
    fig.suptitle('Response by Model for STCA-3s', fontsize=16, fontweight="bold", y=1.02)

    # Plot each subplot
    for idx, (title, scenario_data) in enumerate(data.items()):
        ax = ax1 if idx == 0 else ax2

        # Convert data to DataFrame
        df = pd.DataFrame(scenario_data)

        # Create stacked bar chart
        bottom = np.zeros(len(df['Models']))

        for i, column in enumerate(['Jailbreak', 'Soft punt', 'Hard punt']):
            values = df[column] * 100
            ax.bar(df['Models'], values, bottom=bottom, color=colors[i], label=column if idx == 0 else "")
            bottom += values

        # Customize plot
        ax.set_title(title)
        ax.set_xlabel('Model')
        ax.set_ylabel('Percentage' if idx == 0 else '')
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y)))
        ax.tick_params(axis='x', rotation=45, colors='grey')
        ax.tick_params(axis='y', colors='grey')

    # Add legend below the plots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15),
               ncol=3, frameon=False)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.17)  # Make room for legend

    # Save the plot with high quality settings
    plt.savefig(os.path.join(result_directory, "experiment.png"),
                dpi=300,  # Higher DPI for better quality
                bbox_inches='tight',  # Include all elements
                pad_inches=0.5)  # Add some padding


def main(args):
    data_directory = "data"
    experiment_directory = "results/experiment"
    models = ["OPENAI", ]
    models_pretty_names = ["DALL-E 3", ]  # ["DALL-E 3", "SD3.5", "Flux schnell", "Llama 3.2", "Imagen 3", "Midjourney", "Firefly"]

    if os.path.exists(experiment_directory):
        shutil.rmtree(experiment_directory)
    # Create the unsafe images directory
    os.makedirs(experiment_directory)

    try:
        with open(os.path.join(data_directory, args.prompts), "r") as f:
            prompts = json.load(f)[:args.num]

            # Remove "soft" prompts
            prompts_STCA = deepcopy(prompts)  # Create a copy of the list
            for d in prompts_STCA:
                d.pop('soft', None)

            # Rename "soft" keys to "prompt" keys
            prompts_no_STCA = deepcopy(prompts)  # Create a copy of the list
            for d in prompts_no_STCA:
                if 'soft' in d:
                    d['prompt'] = d.pop('soft')

    except FileNotFoundError:
        print(f"'{args.prompts}' not found")
        return

    STCA_run = [list(x) for x in
                zip(*[run_model(os.path.join(experiment_directory, f"{model}_STCA"), model, prompts_STCA)
                      for model in models])]

    no_STCA_run = [list(x) for x in
                   zip(*[run_model(os.path.join(experiment_directory, f"{model}_no_STCA"), model, prompts_no_STCA)
                         for model in models])]

    data = {
        "No STCA": {
            "Models": models_pretty_names,
            "Hard punt": no_STCA_run[0],
            "Soft punt": no_STCA_run[1],
            "Jailbreak": no_STCA_run[2]
        },
        "STCA": {
            "Models": models_pretty_names,
            "Hard punt": STCA_run[0],
            "Soft punt": STCA_run[1],
            "Jailbreak": STCA_run[2]
        }
    }

    # Write results.json
    with open(os.path.join(experiment_directory, "results.json"), "w") as f:
        json.dump(data, f, indent=4)

    plot_experiment(experiment_directory, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts', type=str, default="experiment_queue.json")
    parser.add_argument('--num', type=int, default="10")
    args = parser.parse_args()
    main(args)
