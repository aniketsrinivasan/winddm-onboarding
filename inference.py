import utils
import matplotlib.pyplot as plt
import torch
from core import UNet2D, MasterModel, TrainingConfig

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

#
images = utils.ImageDataset(directory_path="pistachio_dataset/data/Kirmizi_Pistachio",
                            img_size=(32, 32))

model = UNet2D()
training_config = TrainingConfig(num_train_epochs=0)
master_model = MasterModel(model=model.model,
                           training_config=training_config,
                           train_dataset=images,
                           stub_path="stubs/diffusion-pistachio-32/model_trained_5")
# master_model.train(save_name="model_trained_3")


this_image = images[0].unsqueeze(0).to(dtype=torch.float32)
noise = torch.randn(this_image.shape, device=this_image.device)

num_denoising_steps = 500
master_model.sampler.set_timesteps(num_denoising_steps, device=device)
noisy = master_model.sampler.add_noise(this_image, noise, torch.Tensor([999]).long())
denoised = master_model.sampler.add_noise(this_image, noise, torch.Tensor([999]).long())

count = 0
# Denoising loop
with torch.no_grad():
    for t in master_model.sampler.timesteps:
        if (count % 50 == 0):
            pred_noise = master_model.forward(noisy, torch.Tensor([t]), predict_denoised=False, device=device)
            pred_noised_img, _ = utils.ImageUtils.plot_torch(pred_noise)
            denoised_img, _ = utils.ImageUtils.plot_torch(denoised)
        denoised = master_model.forward(noisy, torch.Tensor([t]), predict_denoised=True, device=device)
        print(t, denoised.shape)
        noisy = denoised
        count += 1

denoised_img, _ = utils.ImageUtils.plot_torch(denoised)
print(denoised.shape)
