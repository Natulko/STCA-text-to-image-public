import os
import re
import json
from dotenv import load_dotenv
import requests
from tqdm import tqdm
import replicate
from openai import OpenAI, OpenAIError

load_dotenv()
client = OpenAI()


def existing_image_set(directory, mode):
    """
    @param directory: directory in which to search
    @param mode: "index" for image indexes / "name" for image names
    @return set: set of image indexes/names of specific format in the directory
    """
    # Use a regular expression to match filenames of the form image_{i}.png
    pattern = re.compile(r'^image_(\d+)\.png$')

    if mode == "index":
        result = set(int(pattern.match(f).group(1)) for f in os.listdir(directory) if pattern.match(f))
    elif mode == "name":
        result = set(f for f in os.listdir(directory) if pattern.match(f))
    else:
        raise ValueError("Mode must be either 'index' or 'name'")
    return result


def find_available_image_index(directory):
    """
    @param directory: directory name in which to search
    @return index: the smallest image index not yet in the directory
    """
    i = 0
    while i in existing_image_set(directory, "index"):
        i += 1

    return i


def download_image(url, directory, filename):
    """
    Downloads an image from a URL and saves it to a given directory.
    @param url: URL of the image to download
    @param directory: directory name in which to save the image
    @param filename: file name of the image
    """
    response = requests.get(url)
    if response.status_code == 200:
        # Create the file path
        file_path = os.path.join(directory, filename)

        # Save the image to the file
        with open(file_path, "wb") as file:
            file.write(response.content)


def generate_images(prompts, directory, model):
    """
    Generates images based on a list of prompts and saves them to the specified directory.
    @param prompts: list of prompts to generate images from (prompt is a str or dict with a "prompt" key)
    @param directory: directory name in which to save the images
    @param model: model to use for image generation. Options: "DALL-E", "SD", "BFL"(cheapest, thus best for testing)
    @return tuple: (num_generated, num_not_generated)
    """
    prompts_json_path = os.path.join(directory, "prompts.json")

    # If new directory, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Read prompts.json
    image_info_list = []
    if os.path.isfile(prompts_json_path):
        with open(prompts_json_path, "r") as f:
            image_info_list = json.load(f)

    # Remove json image entries that no longer exist in the directory
    image_info_list = [x for x in image_info_list if x["image_name"] in existing_image_set(directory, "name")]

    num_generated = 0
    pbar = None
    if len(prompts) > 1:
        print("\nGenerating Images:")
        pbar = tqdm(total=len(prompts),
                    desc=f'Prompting {model}',
                    position=0, leave=True)

    for i in range(len(prompts)):
        # Process prompt instance
        prompt = prompts[i]
        try:
            if isinstance(prompt, dict):
                if "prompt" not in prompt:
                    raise TypeError
                image_info = prompt
                prompt = prompt["prompt"]
            elif isinstance(prompt, str):
                image_info = {"prompt": prompt}
            else:
                raise TypeError
        except TypeError as e:
            print("Wrong prompt format: ", e)
            continue

        # Generate an image and save it to directory
        try:
            # DALL-E
            if model == "OPENAI":
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    size="1024x1024",
                    quality="standard",
                    style="vivid",
                )
                image_url = response.data[0].url

            # Stability Diffusion
            elif model == "SD":
                output = replicate.run(
                    "stability-ai/stable-diffusion-3",
                    input={
                        "prompt": prompt,
                        "aspect_ratio": "3:2"
                    },
                )
                image_url = output[0]

            # Black Forest Labs Flux-schell
            elif model == "BFL":
                output = replicate.run(
                    "black-forest-labs/flux-schnell",
                    input={
                        "prompt": prompt
                    },
                )
                image_url = output[0]

            image_info["image_url"] = image_url

            image_name = f"image_{find_available_image_index(directory)}.png"
            image_info["image_name"] = image_name

            download_image(image_url, directory, image_name)
        
        # If image banned, alert about it and continue
        except OpenAIError as e:
            if pbar is not None:
                pbar.set_postfix_str("Rejected")
            else:
                print("Image not generated: ", e)

        else:
            num_generated += 1
            # If generated successfully, update prompts info
            image_info_list.append(image_info)
            if pbar is not None:
                pbar.set_postfix_str("Generated")

        finally:
            if pbar is not None:
                # Update percentage bar
                pbar.update(1)

    if pbar is not None:
        pbar.close()
        print(f"Generated {num_generated}/{len(prompts)}")

    # Write prompts.json
    with open(prompts_json_path, "w") as f:
        json.dump(image_info_list, f, indent=4)

    return num_generated, len(prompts) - num_generated
