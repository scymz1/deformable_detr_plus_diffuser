import torch
from diffusers import AutoPipelineForImage2Image
from diffuser_reconstruct.sable_diffution_pipeline import add_function_get_tensor_outputs, freezeModels
# from sable_diffution_pipeline import add_function_get_tensor_outputs, freezeModels
from diffusers.utils import load_image, make_image_grid
from diffusers import DDPMScheduler
from torchvision.transforms import functional as TF
from torchvision.utils import save_image
import numpy as np
import inspect
from PIL import Image

def add_noise_to_image(image, noise_level=0.1):
    image_tensor = TF.to_tensor(image)
    noise = torch.randn_like(image_tensor) * noise_level
    noisy_image_tensor = image_tensor + noise
    noisy_image_tensor = torch.clamp(noisy_image_tensor, 0, 1)
    noisy_image = TF.to_pil_image(noisy_image_tensor)
    return noisy_image


def add_noise_to_image_by_step(image, num_steps=50, current_step=1, noise_schedule='linear'):
    if isinstance(image, Image.Image):
        image = np.array(image)
    if noise_schedule == 'linear':
        std_dev = (current_step / (num_steps - 1)) * 50
    elif noise_schedule == 'cosine':
        std_dev = (1 - np.cos((current_step / (num_steps - 1)) * np.pi / 2)) * 50
    else:
        raise ValueError('Unsupported noise schedule')
    h, w, c = image.shape
    noise = np.random.normal(0, std_dev, (h, w, c))
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image


def image_reconstruct(pipeline, image, object_queries, image_id, save_img=False, device="cuda", 
                      save_recon_tensor=False, epoch=None, num_inference_steps=1):
    
    # pipeline.enable_model_cpu_offload()

    # embedding = text tokenized + positional embedding
    # stable diffusion text -> embedding ([1, 77, 768]) -> image

    if type(image) == str:
        image = load_image(image)

    
    # noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    # noise = torch.randn(image.size)
    # timesteps = torch.LongTensor([50])
    # noisy_image = noise_scheduler.add_noise(image, noise, timesteps)


    # Add noise to the image
    # noisy_image = add_noise_to_image(image, noise_level=0.1)
    noisy_image = add_noise_to_image_by_step(image, current_step=1)
    # Convert the noisy image to tensor for the model
    noisy_image = TF.to_tensor(noisy_image).unsqueeze(0) * 255  # Model expects images in [0, 255]


    prompt = "an image of zebra" # "cls an image of zebra els" + 71 " " [77] string
    # -> [77, 768] -> [1, 768]
    prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(prompt, device=device, do_classifier_free_guidance=True, num_images_per_prompt=1)
    prompt_embeds[:, 4] = object_queries
    output = pipeline.get_tensor_outputs(prompt_embeds=prompt_embeds,negative_prompt_embeds=negative_prompt_embeds, image=noisy_image, num_inference_steps=num_inference_steps)
    if save_recon_tensor:
        save_image(output, f"./recon_img/{epoch}_{image_id}_recon.png")
        image.save(f"./recon_img/{epoch}_{image_id}_origin.png")
        save_image(noisy_image / 255.0, f"./recon_img/{epoch}_{image_id}_noise.png")
    return output


if __name__ == "__main__":
    # object_queries = torch.load("object_queries_test.pt")
    # object_queries = torch.squeeze(object_queries, 1)
    pipeline = AutoPipelineForImage2Image.from_pretrained(
        "./diffuser_reconstruct/sd-coco-model", torch_dtype=torch.float32, use_safetensors=True
    )
    add_function_get_tensor_outputs(pipeline)
    freezeModels(pipeline)
    object_queries = torch.rand((1, 768))
    # print(object_queries)
    device = "cuda:2"
    image_reconstruct(pipeline, "./test_mask.png", object_queries, 11, save_img=True, device=device)
    # pipelineClass = type(pipeline)
    # print(inspect.getfile(pipelineClass))

    # line 375326