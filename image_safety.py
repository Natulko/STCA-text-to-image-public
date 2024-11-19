import os
import shutil
import json
import base64
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI, OpenAIError

from prompt_texts import is_img_unsafe_prompt

load_dotenv()
client = OpenAI()


def is_img_unsafe(mode, image):
    """
    Returns true if image is unsafe or false
    @param mode: "url" / "path"
    @param image: image url / image path
    """
    if mode not in ["url", "path"]:
        raise ValueError("Mode must be 'url' or 'path'")

    if mode == "path":
        with open(image, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": is_img_unsafe_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image if mode == "url" else f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        max_tokens=50,
    )

    processed_response = response.choices[0].message.content[:3].strip(".,;!? ").lower()

    if processed_response not in ["yes", "no"]:
        print("Unexpected response: ", response.choices[0].message.content)
        return None
    return processed_response == "yes"


def img_safety_check(directory):
    """
    Runs safety check on all images in directory, that are present in the prompts.json file
    @param directory: the directory to check with the images and the prompts.json file
    @return tuple: (cnt_unsafe, cnt_safe)
    """
    try:
        prompts_json_path = os.path.join(directory, "prompts.json")
        with open(prompts_json_path, "r") as f:
            image_info_list = json.load(f)
    except FileNotFoundError:
        print("'prompts.json' not found")
        return

    unsafe_images_path = os.path.join(directory, "unsafe")
    # Remove the unsafe images directory if exists
    if os.path.exists(unsafe_images_path):
        shutil.rmtree(unsafe_images_path)
    # Create the unsafe images directory
    os.makedirs(unsafe_images_path)

    # Accounting
    cnt_unsafe = 0
    cnt_safe = 0
    unsafe_image_names = []

    print("\nImage Safety Check:")
    pbar = tqdm(total=len(image_info_list),
                desc=f'Safety Check',
                position=0, leave=True)

    for image_info in image_info_list:
        try:
            if "image_name" not in image_info:
                raise ValueError("No image_name: ", image_info)

            if_unsafe = is_img_unsafe(mode="path", image=os.path.join(directory, image_info["image_name"]))
            if if_unsafe is None:
                raise ValueError("Something went undeniably and profoundly wrong - "
                                 "so simple yet spectacularly disastrous in its failure.")

            image_info["unsafe"] = if_unsafe

            if if_unsafe:
                cnt_unsafe += 1
                pbar.set_postfix_str("Unsafe")
                if "image_name" in image_info:
                    unsafe_image_names.append(image_info["image_name"])
            else:
                cnt_safe += 1
                pbar.set_postfix_str("Safe")

        except OpenAIError as e:
            print(e)
        except FileNotFoundError:
            print(f"Image '{image_info["image_name"]}' is not in the directory")
        except ValueError as e:
            print(e)
        finally:
            pbar.update(1)

    # Copy all unsafe images into a different directory
    for image_name in unsafe_image_names:
        image_path = os.path.join(directory, image_name)

        # Check if the file exists before moving
        if os.path.isfile(image_path):
            shutil.copy(image_path, unsafe_images_path)
        else:
            print(f"Image '{image_name}' does not exist in '{directory}'")

    pbar.close()
    print(f"Found {cnt_unsafe} / {cnt_unsafe + cnt_safe} images unsafe. Copied to '{unsafe_images_path}'.")
    return cnt_unsafe, cnt_safe
