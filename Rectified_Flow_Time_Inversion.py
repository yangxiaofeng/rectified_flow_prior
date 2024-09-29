from pipeline_rf_inverse import RectifiedFlowPipeline as RectifiedFlowPipeline_inverse
from pipeline_rf import RectifiedFlowPipeline

import torch
import threestudio
def merge_dW_to_unet(pipe, dW_dict, alpha=1.0):
    _tmp_sd = pipe.unet.state_dict()
    for key in dW_dict.keys():
        _tmp_sd[key] += dW_dict[key] * alpha
    pipe.unet.load_state_dict(_tmp_sd, strict=False)
    return pipe

def load_hf_hub_lora(pipe_rf, lora_path='Lykon/absolute-reality-1.81', save_dW=False, base_sd='runwayml/stable-diffusion-v1-5',
                     alpha=1.0):
    # get weights of base sd models
    from diffusers import DiffusionPipeline
    _pipe = DiffusionPipeline.from_pretrained(
        base_sd,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    sd_state_dict = _pipe.unet.state_dict()

    # get weights of the customized sd models, e.g., the aniverse downloaded from civitai.com
    _pipe = DiffusionPipeline.from_pretrained(
        lora_path,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    lora_unet_checkpoint = _pipe.unet.state_dict()

    # get the dW
    dW_dict = {}
    for key in lora_unet_checkpoint.keys():
        dW_dict[key] = lora_unet_checkpoint[key] - sd_state_dict[key]

    # return and save dW dict
    if save_dW:
        save_name = lora_path.split('/')[-1] + '_dW.pt'
        torch.save(dW_dict, save_name)

    pipe_rf = merge_dW_to_unet(pipe_rf, dW_dict=dW_dict, alpha=alpha)
    pipe_rf.vae = _pipe.vae
    pipe_rf.text_encoder = _pipe.text_encoder

    return dW_dict
pipe = RectifiedFlowPipeline_inverse.from_pretrained("XCLIU/2_rectified_flow_from_sd_1_5", torch_dtype=torch.float32)
pipe2 = RectifiedFlowPipeline.from_pretrained("XCLIU/2_rectified_flow_from_sd_1_5", torch_dtype=torch.float32)
### switch to torch.float32 for higher quality
# _ = load_hf_hub_lora(pipe, save_dW=False, alpha=1.0)

pipe.to("cuda")  ### if GPU is not available, comment this line

from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms

image = Image.open("data_assets/A_castle_next_to_a_river.png")
transform = transforms.Compose([
    transforms.PILToTensor()
])

image_tensor = transform(image).unsqueeze(0).to("cuda")/255
rgb_BCHW = image_tensor
rgb_BCHW_512 = F.interpolate(
    rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
)
guidance_config = {
        "half_precision_weights": True,
        "view_dependent_prompting": True,
        "guidance_scale": 1,
        "pretrained_model_name_or_path": "XCLIU/2_rectified_flow_from_sd_1_5",
        "min_step_percent": 0.1,
        "max_step_percent": 0.8,
        "camera_condition_type": "extrinsics",
    }
guidance = threestudio.find('rf-sds-guidance-test-inversion')(guidance_config).cuda()
target_image = guidance.encode_images(rgb_BCHW_512).detach().to(device="cuda")


prompt = "A castle next to a river."
prompt_new = "A castle next to a river, winter scene"
### 2-rectified flow is a multi-step text-to-image generative model.
### It can generate with extremely few steps, e.g, 2-8
### For guidance scale, the optimal range is [1.0, 2.0], which is smaller than normal Stable Diffusion.
### You may set negative_prompts like normal Stable Diffusion
inverted_noise = pipe(prompt=prompt,
              latents=target_image,
            negative_prompt="painting, unreal, twisted",
              num_inference_steps=50,
              guidance_scale=1.5)

images = pipe2(prompt=prompt,
              negative_prompt="painting, unreal, twisted",
              latents=inverted_noise,
              num_inference_steps=5,
              guidance_scale=1.5).images
images[0].save("image_inverted.png")
images = pipe2(prompt=prompt_new,
              negative_prompt="painting, unreal, twisted",
              latents=inverted_noise,
              num_inference_steps=5,
              guidance_scale=1.5).images
images[0].save("image_inverted_new.png")
