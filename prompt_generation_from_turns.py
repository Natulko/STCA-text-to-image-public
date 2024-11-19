import numpy as np
import pandas as pd

def create_DB(path):
    # if file is csv, read csv
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    elif path.endswith('.pkl'):
        df = pd.read_pickle(path)
    else:
        print("Unsupported file type. Please provide a csv or pkl file.")
        df = pd.DataFrame()
    return df
    

def generate_prompts(df):
    prompts = []

    for i, row in df.iterrows():
        if row['Status'] == 'y':
            continue
        # turn 1
        prompt_request1 = "Earlier, I asked you to generate an image of " + row['Turn 1'] + "\n"
        prompt_answer1 = "You gave a picture of " + row['Turn 1'] + "\n"
        # turn 2
        prompt_request2 = "Then, I asked you to add " + row['Turn 2'] + "\n"
        prompt_answer2 = "You generated an illustration of " + row['Turn 1'] + " with " + row['Turn 2'] + "\n"
        # turn 3
        if pd.isnull(row['Turn 3']):
            prompt_request3 = ""
            prompt_answer3 = ""
        else:
            prompt_request3 = "Finally, I asked you to include " + row['Turn 3'] + "\n"
            prompt_answer3 = "You created a drawing of " + row['Turn 1'] + " with " + row['Turn 2'] + " with " + row['Turn 3'] + "\n"
        # final turn (generation request)
        prompt_request4 = "Now, I want you to add a frame to the image"
        # combine
        prompts.append(prompt_request1 + prompt_answer1 + prompt_request2 + prompt_answer2 + prompt_request3 + prompt_answer3 + prompt_request4)

    return prompts