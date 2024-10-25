import utils
import torch
from core import UNet2D, MasterModel, TrainingConfig


# Setting device:
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Initializing datasets:
images = utils.ImageDataset(directory_path="pistachio_dataset/data/Kirmizi_Pistachio",
                            img_size=(32, 32))
val_images = utils.ImageDataset(directory_path="pistachio_dataset/data/Siirt_Pistachio",
                                img_size=(32, 32))

# Initializing the model of choice:
#   here, we have a fully convolutional UNet2D
model = UNet2D(down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
               up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
               block_out_channels=(64, 128, 256),
               )
# Initializing the training configuration:
training_config = TrainingConfig(num_train_epochs=0)
# Initializing MasterModel object for the model:
master_model = MasterModel(model=model.model,
                           training_config=training_config,
                           train_dataset=images,
                           val_dataset=val_images,
                           stub_path=None)

# (Manual) denoising process:
viz_intermediate = False
start_denoise_step = 200
num_denoising_steps = 200
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
    for i in range(start_denoise_step, 0, step_size):
        denoised = master_model.forward(noisy, torch.Tensor([i]), predict_denoised=True, device=device)
        if (count % 10 == 0) and viz_intermediate:
            pred_noise = master_model.forward(noisy, torch.Tensor([i]), predict_denoised=False, device=device)
            _, _ = utils.ImageUtils.plot_torch(pred_noise)
            _, _ = utils.ImageUtils.plot_torch(denoised)
        print(f"Step: {i}, shape: {denoised.shape}")
        # Setting next "noisy" image to be denoised, incrementing count:
        noisy = denoised
        count += 1

# Plotting final denoised image:
_, _ = utils.ImageUtils.plot_torch(denoised)
print(denoised.shape)
