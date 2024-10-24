import utils
import diffusers
import matplotlib.pyplot as plt
import torch
from core import UNet2D, MasterModel, TrainingConfig


device = 'mps' if torch.backends.mps.is_available() else 'cpu'

images = utils.ImageDataset(directory_path="pistachio_dataset/data/Kirmizi_Pistachio",
                            img_size=(32, 32))


model = UNet2D()
training_config = TrainingConfig(num_train_epochs=2,
                                 train_batch_size=8,
                                 output_dir="stubs/diffusion-pistachio-32")
master_model = MasterModel(model=model.model,
                           training_config=training_config,
                           train_dataset=images,
                           stub_path="stubs/diffusion-pistachio-32/model_trained_5")
master_model.train(save_name="model_trained_6", print_enabled=True)

start_denoise_step = 150
num_denoising_steps = 150
denoising_step = - start_denoise_step // num_denoising_steps

timestep_list = list(range(start_denoise_step, 0, denoising_step))
master_model.sampler.set_timesteps(timesteps=timestep_list)

this_image = images[0].unsqueeze(0).to(dtype=torch.float32)
print("Sample image: ", this_image)
img, _ = utils.ImageUtils.plot_torch(this_image)

noise = torch.randn(this_image.shape, device=this_image.device)
print(noise)
img, _ = utils.ImageUtils.plot_torch(noise)

noisy = master_model.sampler.add_noise(this_image, noise, torch.Tensor([start_denoise_step]).long())
print(noisy)
img, _ = utils.ImageUtils.plot_torch(noisy)

denoised = master_model.sampler.add_noise(this_image, noise, torch.Tensor([start_denoise_step]).long())
# Denoising loop
count = 0
for i in range(start_denoise_step, 0, denoising_step):
    if (count % 10 == 0):
        denoised_img, _ = utils.ImageUtils.plot_torch(denoised)
    denoised = master_model.forward(noisy, torch.Tensor([i]), predict_denoised=True, device=device)
    print(i, denoised.shape)
    noisy = denoised
    count += 1

denoised_img, _ = utils.ImageUtils.plot_torch(denoised)
print(denoised.shape)