import json
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import threestudio
from PIL import Image


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

config = {
    "max_iters": 800,
    "seed": 1,
    "scheduler": None,
    "mode": "latent",
    "prompt_processor_type": "sd3-prompt-processor",
    "prompt_processor": {
        "pretrained_model_name_or_path": "stabilityai/stable-diffusion-3-medium-diffusers",
        "prompt": "A DSLR image of a hamburger",
        "spawn": False,
    },
    "guidance_type": "RFDS-Rev-sd3",
    "guidance": {
        "half_precision_weights": True,
        "view_dependent_prompting": True,
        "guidance_scale": 50,
        "pretrained_model_name_or_path": "stabilityai/stable-diffusion-3-medium-diffusers",
        "min_step_percent": 0.02,
        "max_step_percent": 0.98,
        "camera_condition_type": "extrinsics",
    },
    "image": {
        "width": 512,
        "height": 512,
    },
    "n_particle": 6,
    "batch_size": 3,
    "n_accumulation_steps": 2,
    "save_interval": 100,
    "clip": False,
    "tanh": False,
    "lr": {
        "image": 3e-2,
    },
}

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
if mode == "rgb":
    target = nn.Parameter(torch.rand(n_images, h, w, 3, device=guidance.device))
else:
    target = nn.Parameter(2 * torch.rand(n_images, h, w, 16, device=guidance.device) - 1)


optimizer = torch.optim.AdamW(
    [
        {"params": [target], "lr": config["lr"]["image"]},
    ],
    # lr=3e-2,
    weight_decay=0,
)
num_steps = config["max_iters"]
scheduler = None

# add time to out_dir
timestamp = datetime.now().strftime("@%Y%m%d-%H%M%S")
out_dir = os.path.join(
    "outputs", "2d_RFDS_Rev_sd3", f"{config['prompt_processor']['prompt']}{timestamp}"
)
os.makedirs(out_dir, exist_ok=True)
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
    # random select batch_size images from target with replacement
    particles = target[torch.randint(0, n_images, [batch_size])]
    if mode == "latent" and config["tanh"]:
        particles = torch.tanh(particles)

    loss_dict = guidance(
        rgb=particles,
        prompt_utils=prompt_utils,
        mvp_mtx=mvp_mtx,
        elevation=elevation,
        azimuth=azimuth,
        camera_distances=distance,
        c2w=mvp_mtx.clone(),
        rgb_as_latents=(mode != "rgb"),
    )

    loss = (loss_dict["loss_rfds"].mean() ) / n_accumulation_steps
    loss.backward()
    # wandb.log({"loss": loss_dict["loss_vsd"]})
    if (step + 1) % n_accumulation_steps == 0:
        actual_step = (step + 1) // n_accumulation_steps
        guidance.update_step(epoch=0, global_step=actual_step)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if mode == "latent" and config["clip"]:
            with torch.no_grad():
                particles.data = particles.data.clip(-1, 1)

        if actual_step % save_interval == 0:
            if mode == "rgb":
                rgb = target
            else:
                del loss
                torch.cuda.empty_cache()
                with torch.no_grad():
                    rgb = guidance.decode_latents(target.permute(0, 3, 1, 2)).permute(
                        0, 2, 3, 1
                    )

            img_rgb = rgb.clamp(0, 1).detach().squeeze(0).cpu().numpy()* 255

            for col in range(n_images):
                rgb_tosave = Image.fromarray(np.uint8(img_rgb[col]))
                rgb_tosave.save(os.path.join(out_dir, f"{actual_step:05d}_{col:02d}.png"))
            plt.close('all')
