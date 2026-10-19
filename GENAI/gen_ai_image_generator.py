from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)

image = pipe(
    "sports during rain"
).images[0]

image.save("sports.png")