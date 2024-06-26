import json
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import threestudio
from PIL import Image
import torch.nn.functional as F

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# change this line for new prompt
prompt_new = "a rock in a river"
config = {
    "max_iters": 1400,
    "seed": 1,
    "scheduler": None,
    "mode": "latent",
    "prompt_processor_type": "sd3-prompt-processor",
    "prompt_processor": {
        "pretrained_model_name_or_path": "stabilityai/stable-diffusion-3-medium-diffusers",
        # change this line for previous prompt
        "prompt": "a boat in a river",
        "spawn": False,
    },
    "guidance_type": "iRFDS-sd3",
    "guidance": {
        "half_precision_weights": True,
        "view_dependent_prompting": True,
        "guidance_scale": 1,
        "pretrained_model_name_or_path": "stabilityai/stable-diffusion-3-medium-diffusers",
        "min_step_percent": 0.02,
        "max_step_percent": 0.98,
        "camera_condition_type": "extrinsics",
    },
    "image": {
        "width": 512,
        "height": 512,
    },
    "n_particle": 1,
    "batch_size": 1,
    "n_accumulation_steps": 2,
    "save_interval": 200,
    "clip": False,
    "tanh": False,
    "lr": {
        "image": 2e-3,
    },
}

from diffusers import StableDiffusion3Pipeline

pipe_sd3 = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers",
                                                torch_dtype=torch.float16, requires_safety_checker=False)

pipe_sd3 = pipe_sd3.to('cuda')
pipe_sd3.set_progress_bar_config(disable=True)
prompt = config['prompt_processor']['prompt']

# path of the original image
image = Image.open("data_assets/a_boat_in_a_river.png")


import torchvision.transforms as transforms
transform = transforms.Compose([
    transforms.PILToTensor()
])

image_tensor = transform(image).unsqueeze(0).to("cuda")/255

rgb_BCHW = image_tensor
rgb_BCHW_512 = F.interpolate(
    rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
)
seed_everything(config["seed"])

guidance = threestudio.find(config["guidance_type"])(config["guidance"]).cuda()
guidance.camera_embedding = guidance.camera_embedding.cuda()
prompt_processor = threestudio.find(config["prompt_processor_type"])(
    config["prompt_processor"]
)

n_images = config["n_particle"]
batch_size = config["batch_size"]

w, h = config["image"]["width"], config["image"]["height"]
mode = config["mode"]

target = torch.randn((1,16,64,64)).to(device=guidance.device)
target.requires_grad = True
with torch.no_grad():
    target_image = guidance.encode_images(rgb_BCHW_512).permute(0, 2, 3, 1).repeat(n_images,1,1,1).detach().to(device=guidance.device)
    # initialize the targe noise with latents of original image
    target.data = guidance.encode_images(rgb_BCHW_512).data


optimizer = torch.optim.AdamW(
    [
        {"params": [target], "lr": config["lr"]["image"]},
    ],
    weight_decay=0,
)
num_steps = config["max_iters"]
scheduler = None

# add time to out_dir
timestamp = datetime.now().strftime("@%Y%m%d-%H%M%S")
out_dir = os.path.join(
    "outputs", "iRFDS_edit", f"{config['prompt_processor']['prompt']}{timestamp}"
)
os.makedirs(out_dir, exist_ok=True)
image.save(os.path.join(out_dir, f"original.png"))

with open(os.path.join(out_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=4)

plt.axis("off")

elevation = torch.zeros([batch_size], device=guidance.device)
azimuth = torch.zeros([batch_size], device=guidance.device)
distance = torch.zeros([batch_size], device=guidance.device)
prompt_utils = prompt_processor()
save_interval = config["save_interval"]

mvp_mtx = torch.zeros([batch_size, 4, 4], device=guidance.device)
n_accumulation_steps = config["n_accumulation_steps"]

for step in tqdm(range(num_steps * n_accumulation_steps + 1)):
    loss_dict = guidance(
        noise_to_optimize=target,
        rgb = target_image,
        prompt_utils=prompt_utils,
        mvp_mtx=mvp_mtx,
        elevation=elevation,
        azimuth=azimuth,
        camera_distances=distance,
        c2w=mvp_mtx.clone(),
        rgb_as_latents=(mode != "rgb"),
    )

    loss = (loss_dict["loss_iRFDS"] + loss_dict["loss_regularize"]) / n_accumulation_steps
    loss.backward()

    if (step + 1) % n_accumulation_steps == 0:
        actual_step = (step + 1) // n_accumulation_steps
        guidance.update_step(epoch=0, global_step=actual_step)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


        if actual_step % save_interval == 0:
            images = pipe_sd3(prompt=prompt,
                             num_inference_steps=15,
                             latents=target,
                             guidance_scale=2).images
            images[0].save(os.path.join(out_dir, f"{actual_step:05d}_inversion.png"))

            images = pipe_sd3(prompt=prompt_new,
                             num_inference_steps=15,
                             latents=target,
                             guidance_scale=2).images
            images[0].save(os.path.join(out_dir, f"{actual_step:05d}_new.png"))


