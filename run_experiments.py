import os
import subprocess
import shutil

name = "flux-schnell"
height = 256
width = 256
seed = 11267150287381669669

prompts = [
    "a person walking in a field of yellow flowers with a red dress",
    "the masked wrestler hits the unmasked wrestler",
    "the person closer to the camera weightlifts and the person farther from the camera runs",
    "a bird eats a snake",
    "a tree smashed into a car",
    "a brown dog is on a white couch",
    "there are 50 stars on the right flag and 5 stars on the left flag",
    "the person without earrings pays the person with earrings",
    "the dog's leg is on the person's torso",
    "the person with green legs is running quite slowly and the red legged one runs faster",
    "the dog wears as a hat what someone would normally bite",
    "the uncaged bird has an opened cage door"
]

num_prompts = len(prompts)

for i in range(num_prompts):
    prompt = prompts[i]

    cmd = [
        "python", "-m", "flux",
        "--name", name,
        "--height", str(height),
        "--width", str(width),
        "--prompt", prompt,
        "--seed", str(seed)
    ]
    
    print(f"Generating image {i+1}/{num_prompts}: {prompt}")
    
    try:
        result = subprocess.run(cmd, check=True, stdout=None, stderr=None)
        print(f"Image {i+1} generated successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"Error generating image {i+1}: {e.stderr}")
        continue 
