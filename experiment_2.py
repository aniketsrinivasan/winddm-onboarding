import utils
import torch
from diffusers import DDIMScheduler, EulerDiscreteScheduler
from core import UNet2D, MasterModel, TrainingConfig


torch.set_default_dtype(torch.float32)

# Setting device:
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Initializing datasets:
images = utils.ImageDataset(directory_path="pistachio_dataset/data/Kirmizi_Pistachio",
                            img_size=(64, 64))
val_images = utils.ImageDataset(directory_path="pistachio_dataset/data/Siirt_Pistachio",
                                img_size=(64, 64))

# Initializing the model of choice:
model = UNet2D(down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
               up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
               block_out_channels=(64, 96, 128, 192),
               )
# Initializing the training configuration:
training_config = TrainingConfig(num_train_epochs=0,
                                 train_batch_size=4,
                                 learning_rate=1e-4,
                                 gradient_accumulation_steps=4,
                                 learning_warmup_ratio=0.1,
                                 output_dir="stubs/diffusion-pistachio-64",)
# Initializing MasterModel object for the model:
# Using different sampler:
ddim_sampler = DDIMScheduler(num_train_timesteps=1000)
ddim_sampler.set_timesteps(1000)
euler = EulerDiscreteScheduler(num_train_timesteps=1000)
euler.set_timesteps(1000)

master_model = MasterModel(model=model.model,
                           training_config=training_config,
                           train_dataset=images,
                           val_dataset=val_images,
                           stub_path="stubs/diffusion-pistachio-64/experiment-1-1",
                           sampler=None)

# (Manual) denoising process:
viz_intermediate = False
viz_intermediate_steps = 100
start_denoise_step = 999
num_denoising_steps = 1000
step_size = - start_denoise_step // num_denoising_steps

#   Image to run diffusion process:
this_image = images[0].unsqueeze(0).to(dtype=torch.float32)
#   Sampling random noise to add:
noise = torch.randn(this_image.shape, device=this_image.device)
#   Noised image:
noisy = master_model.sampler.add_noise(this_image, noise, torch.Tensor([start_denoise_step]).long())
#   Visualizing initial image(s):
if viz_intermediate:
    _, _ = utils.ImageUtils.plot_torch(this_image)
    _, _ = utils.ImageUtils.plot_torch(noisy)


denoised = master_model.sampler.add_noise(this_image, noise, torch.Tensor([start_denoise_step]).long())
count = 0
# Denoising loop
with torch.no_grad():
    for i in master_model.sampler.timesteps:
        denoised = master_model.forward(noisy, i, predict_denoised=True, device=device)
        if (count % viz_intermediate_steps == 0) and viz_intermediate:
            pred_noise = master_model.forward(noisy, i, predict_denoised=False, device=device)
            _, _ = utils.ImageUtils.plot_torch(pred_noise)
            _, _ = utils.ImageUtils.plot_torch(denoised)
        print(f"Step: {i}, shape: {denoised.shape}")
        # Setting next "noisy" image to be denoised, incrementing count:
        noisy = denoised
        del denoised        # free up memory where possible
        torch.mps.empty_cache()
        count += 1

# Plotting final denoised image:
_, _ = utils.ImageUtils.plot_torch(noisy)
print(noisy.shape)
