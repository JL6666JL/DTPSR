import os
import sys
sys.path.append(os.getcwd())
import cv2
import glob
import argparse
import numpy as np
from PIL import Image

import torch
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

from pipelines.pipeline_dtpsr import StableDiffusionControlNetPipeline
from utils.misc import load_dreambooth_lora
from utils.wavelet_color_fix import wavelet_color_fix, adain_color_fix

from ram.models.ram_lora import ram
from ram import inference_ram as inference
from ram import get_transform


from typing import Mapping, Any
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

############## import Mask2Former model ##############
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config


from tqdm.auto import tqdm
from transformers import AutoProcessor, Blip2ForConditionalGeneration

import pickle
import json

######################################################

logger = get_logger(__name__, log_level="INFO")


tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def load_state_dict_diffbirSwinIR(model: nn.Module, state_dict: Mapping[str, Any], strict: bool=False) -> None:
    state_dict = state_dict.get("state_dict", state_dict)
    
    is_model_key_starts_with_module = list(model.state_dict().keys())[0].startswith("module.")
    is_state_dict_key_starts_with_module = list(state_dict.keys())[0].startswith("module.")
    
    if (
        is_model_key_starts_with_module and
        (not is_state_dict_key_starts_with_module)
    ):
        state_dict = {f"module.{key}": value for key, value in state_dict.items()}
    if (
        (not is_model_key_starts_with_module) and
        is_state_dict_key_starts_with_module
    ):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=strict)

def load_pipeline(args, accelerator, enable_xformers_memory_efficient_attention):
    
    from models.controlnet import ControlNetModel
    from models.unet_2d_condition import UNet2DConditionModel

    # Load scheduler, tokenizer and models.
    scheduler = DDPMScheduler.from_pretrained(args.sd_model_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.sd_model_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.sd_model_path, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(args.sd_model_path, subfolder="vae")
    feature_extractor = CLIPImageProcessor.from_pretrained(f"{args.sd_model_path}/feature_extractor")
    unet = UNet2DConditionModel.from_pretrained(args.model_path, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(args.model_path, subfolder="controlnet")
    
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Get the validation pipeline
    validation_pipeline = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, feature_extractor=feature_extractor, 
        unet=unet, controlnet=controlnet, scheduler=scheduler, safety_checker=None, requires_safety_checker=False,
    )
    
    validation_pipeline._init_tiled_vae(encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)

    return validation_pipeline

def load_tag_model(args, device='cuda'):
    
    model = ram(pretrained='preset/models/ram_swin_large_14m.pth',
                pretrained_condition=args.ram_ft_path,
                image_size=384,
                vit='swin_l')
    model.eval()
    model.to(device)
    
    return model
    
def get_validation_prompt(args, image, model, device='cuda'):
    validation_prompt = ""
 
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    lq = ram_transforms(lq)
    res = inference(lq, model)
    ram_encoder_hidden_states = model.generate_image_embeds(lq)

    validation_prompt = f"{res[0]}, {args.prompt},"

    return validation_prompt, ram_encoder_hidden_states

def load_seg_description_emb(args,image_path):
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    parent_dir = os.path.dirname(image_path)
    folder_name = os.path.basename(parent_dir) 

    dataset_name = os.path.basename(os.path.dirname(parent_dir))

    file_name = f"{dataset_name}_{folder_name}_{image_name}_descriptions.pkl"

    with open(os.path.join(args.local_des_pkl_path,file_name), 'rb') as f:
        local_descriptions = pickle.load(f)
    return local_descriptions["panoptic_seg"], local_descriptions["seg_emb_dict"]

def get_joint_desemb(panoptic_seg, seg_emb_dict, max_tensors):
    arr_flat = panoptic_seg.flatten()
    unique, counts = np.unique(arr_flat, return_counts=True)
    sorted_indices = np.argsort(-counts)  # 负号表示降序
    sorted_numbers = unique[sorted_indices]

    selected_indices = sorted_numbers[:max_tensors]

    selected_hf_emb = [seg_emb_dict[i]["hf_emb"].view(-1,1024) for i in selected_indices ]
    selected_lf_emb = [seg_emb_dict[i]["lf_emb"].view(-1,1024) for i in selected_indices ]

    # 计算需要填充的零 tensor 数量
    num_padding = max_tensors - len(selected_hf_emb)
    # 补零
    if num_padding > 0:
        device = seg_emb_dict[selected_indices[0]]["hf_emb"].device  # 确保和输入 tensor 同设备（CPU/GPU）
        dtype = seg_emb_dict[selected_indices[0]]["hf_emb"].dtype    # 确保数据类型一致
        zero_tensor_hf = torch.zeros(77, 1024, device=device, dtype=dtype)
        zero_tensor_lf = torch.zeros(77, 1024, device=device, dtype=dtype)

        selected_hf_emb.extend([zero_tensor_hf] * num_padding)
        selected_lf_emb.extend([zero_tensor_lf] * num_padding)

    cat_hf_emb = torch.cat(selected_hf_emb, dim=0).unsqueeze(0)
    cat_lf_emb = torch.cat(selected_lf_emb, dim=0).unsqueeze(0)

    # print(len(selected_indices), num_padding, valid_mask, cat_hf_emb.shape, cat_lf_emb.shape, hf_lf_attention_mask.shape)
    return cat_hf_emb, cat_lf_emb

def main(args, enable_xformers_memory_efficient_attention=True,):
    txt_path = os.path.join(args.output_dir, 'prompts')
    os.makedirs(txt_path, exist_ok=True)


    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    text_encoder = CLIPTextModel.from_pretrained(args.sd_model_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.sd_model_path, subfolder="tokenizer")
    text_encoder.to(accelerator.device, dtype=torch.float16)

    negative_prompt = args.negative_prompt
    uncond_tokens = [negative_prompt]
    uncond_input = tokenizer(
        uncond_tokens,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    negative_prompt_embeds = text_encoder(
        uncond_input.input_ids.to(accelerator.device),
        attention_mask=None,
    )[0]

    hf_negative_prompt = args.hf_negative_prompt
    hf_uncond_tokens = [hf_negative_prompt]
    hf_uncond_input = tokenizer(
        hf_uncond_tokens,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    hf_negative_prompt_embeds = text_encoder(
        hf_uncond_input.input_ids.to(accelerator.device),
        attention_mask=None,
    )[0]

    lf_negative_prompt = args.lf_negative_prompt
    lf_uncond_tokens = [lf_negative_prompt]
    lf_uncond_input = tokenizer(
        lf_uncond_tokens,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    lf_negative_prompt_embeds = text_encoder(
        lf_uncond_input.input_ids.to(accelerator.device),
        attention_mask=None,
    )[0]

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(f'{args.output_dir}/masks/', exist_ok=True)
        os.makedirs(f'{args.output_dir}/masks_meta/', exist_ok=True)
        os.makedirs(f'{args.output_dir}/samples/', exist_ok=True)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("DTPSR")

    pipeline = load_pipeline(args, accelerator, enable_xformers_memory_efficient_attention)
    model = load_tag_model(args, accelerator.device)

    if accelerator.is_main_process:
        generator = torch.Generator(device=accelerator.device)
        if args.seed is not None:
            generator.manual_seed(args.seed)

        if os.path.isdir(args.image_path):
            image_names = sorted(glob.glob(f'{args.image_path}/*.*'))
        else:
            image_names = [args.image_path]
        
        if args.start_index !=-1 and args.end_index != -1:
            image_names = image_names[args.start_index:args.end_index]

        progress_bar = tqdm(
            range(0, len(image_names)),
            initial=0,
            desc="Processing images",
        )
        with open(args.des_path, 'r') as f:
            all_des = json.load(f)

        for image_idx, image_name in enumerate(image_names[:]):
            print(f'================== process {image_idx} imgs... ===================')
            validation_image = Image.open(image_name).convert("RGB")

            panoptic_seg, seg_emb_dict = load_seg_description_emb(args,image_name)
            cat_hf_emb, cat_lf_emb = get_joint_desemb(panoptic_seg, seg_emb_dict, 5)

            ph, pw = panoptic_seg.shape
            panoptic_seg = cv2.resize(panoptic_seg, (ph*4,pw*4), interpolation=cv2.INTER_NEAREST)

            description = all_des[image_name]

            _, ram_encoder_hidden_states = get_validation_prompt(args, validation_image, model)

            ori_width, ori_height = validation_image.size
            resize_flag = False
            rscale = args.upscale
            if ori_width < args.process_size//rscale or ori_height < args.process_size//rscale:
                scale = (args.process_size//rscale)/min(ori_width, ori_height)
                tmp_image = validation_image.resize((int(scale*ori_width), int(scale*ori_height)))

                validation_image = tmp_image
                resize_flag = True

            validation_image = validation_image.resize((validation_image.size[0]*rscale, validation_image.size[1]*rscale))
            validation_image = validation_image.resize((validation_image.size[0]//8*8, validation_image.size[1]//8*8))
            width, height = validation_image.size
            resize_flag = True #

            print(f'input size: {height}x{width}')

            # load image for Mask2Former, which should be BGR format
            validation_image_cv2 = cv2.imread(image_name)
            # resize to 512x512
            validation_image_cv2 = cv2.resize(validation_image_cv2, (args.process_size, args.process_size))
            validation_prompt = ""

            
            # if args.added_prompt == "":
            #     validation_prompt = validation_prompt[:-2] # remove the last comma and space
            # else:
            #     validation_prompt = validation_prompt

            validation_prompt = description     #使用全局description代替原本的label组成的prompt
            validation_prompt += args.added_prompt # clean, extremely detailed, best quality, sharp, clean
            negative_prompt = args.negative_prompt #dirty, messy, low quality, frames, deformed, 

            if args.save_prompts:
                txt_save_path = f"{txt_path}/{os.path.basename(image_name).split('.')[0]}.txt"
                file = open(txt_save_path, "w")
                file.write(validation_prompt)
                file.close()
            print(f'{validation_prompt}')

            with torch.autocast("cuda"):
                image = pipeline(
                        validation_prompt, validation_image,  cat_hf_emb=cat_hf_emb, cat_lf_emb=cat_lf_emb, num_inference_steps=args.num_inference_steps, generator=generator, height=height, width=width,
                        guidance_scale=args.guidance_scale, negative_prompt=negative_prompt, negative_prompt_embeds=negative_prompt_embeds, hf_negative_prompt_embeds=hf_negative_prompt_embeds, lf_negative_prompt_embeds=lf_negative_prompt_embeds, conditioning_scale=args.conditioning_scale,
                        start_point=args.start_point, ram_encoder_hidden_states=ram_encoder_hidden_states,
                        latent_tiled_size=args.latent_tiled_size, latent_tiled_overlap=args.latent_tiled_overlap,
                        args=args,
                    ).images[0]
            
            if args.align_method == 'nofix':
                image = image
            else:
                if args.align_method == 'wavelet':
                    image = wavelet_color_fix(image, validation_image)
                elif args.align_method == 'adain':
                    image = adain_color_fix(image, validation_image)

            if resize_flag: 
                image = image.resize((ori_width*rscale, ori_height*rscale))
                
            name, ext = os.path.splitext(os.path.basename(image_name))
            
            image.save(f'{args.output_dir}/samples/{name}.png')
            progress_bar.update(1)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--ram_ft_path", type=str, default='preset/models/DAPE.pth')
    parser.add_argument("--sd_model_path", type=str, default='preset/models/stable-diffusion-2-base')
    parser.add_argument("--prompt", type=str, default="") # user can add self-prompt to improve the results
    parser.add_argument("--added_prompt", type=str, default="")
    parser.add_argument("--negative_prompt", type=str, default="blurry, dotted, noise, raster lines, unclear, lowres, over-smoothed")
    parser.add_argument("--hf_negative_prompt", type=str, default="fake texture, excessive details, harsh edges, ringing artifacts, noise, aliasing, sharpening artifacts, hallucinated texture, jaggies")
    parser.add_argument("--lf_negative_prompt", type=str, default="blurry, misshaped object, bad anatomy, inconsistent lighting, unnatural shading, distorted global structure, low frequency banding, uneven illumination")
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--mixed_precision", type=str, default="fp16") # no/fp16/bf16
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--conditioning_scale", type=float, default=1.0)
    parser.add_argument("--blending_alpha", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224) # latent size, for 24G
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024) # image size, for 13G
    parser.add_argument("--latent_tiled_size", type=int, default=96) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=32) 
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')
    parser.add_argument("--start_steps", type=int, default=999) # defaults set to 999.
    parser.add_argument("--start_point", type=str, choices=['lr', 'noise'], default='lr') # LR Embedding Strategy, choose 'lr latent + 999 steps noise' as diffusion start point. 
    parser.add_argument("--save_prompts", action='store_true')
    parser.add_argument("--local_des_pkl_path", type=str, default=None)
    parser.add_argument("--des_path", type=str, default=None)
    parser.add_argument("--start_index", type=int, default=-1)
    parser.add_argument("--end_index", type=int, default=-1)
    args = parser.parse_args()
    main(args)



