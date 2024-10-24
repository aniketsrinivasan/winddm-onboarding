import utils
import matplotlib.pyplot as plt
import torch
from core import UNet2D, MasterModel, TrainingConfig

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

#
images = utils.ImageDataset(directory_path="pistachio_dataset/data/Kirmizi_Pistachio",
                            img_size=(32, 32))

val_images = utils.ImageDataset(directory_path="pistachio_dataset/data/Siirt_Pistachio",
                                img_size=(32, 32))


model = UNet2D()
training_config = TrainingConfig(num_train_epochs=0)
master_model = MasterModel(model=model.model,
                           training_config=training_config,
                           train_dataset=images,
                           val_dataset=val_images,
                           stub_path="stubs/diffusion-pistachio-32/model_trained_5")
# master_model.train(save_name="model_trained_3")


start_denoise_step = 200
num_denoising_steps = 150
denoising_step = - start_denoise_step // num_denoising_steps

this_image = images[0].unsqueeze(0).to(dtype=torch.float32)
img, _ = utils.ImageUtils.plot_torch(this_image)

noise = torch.randn(this_image.shape, device=this_image.device)
img, _ = utils.ImageUtils.plot_torch(noise)

noisy = master_model.sampler.add_noise(this_image, noise, torch.Tensor([start_denoise_step]).long())
img, _ = utils.ImageUtils.plot_torch(noisy)

denoised = master_model.sampler.add_noise(this_image, noise, torch.Tensor([start_denoise_step]).long())
count = 0
# Denoising loop
for i in range(start_denoise_step, 0, denoising_step):
    if (count % 10 == 0):
        pred_noise = master_model.forward(noisy, torch.Tensor([i]), predict_denoised=False, device=device)
        pred_noised_img, _ = utils.ImageUtils.plot_torch(pred_noise)
        denoised_img, _ = utils.ImageUtils.plot_torch(denoised)
    denoised = master_model.forward(noisy, torch.Tensor([i]), predict_denoised=True, device=device)
    print(i, denoised.shape)
    noisy = denoised
    count += 1

denoised_img, _ = utils.ImageUtils.plot_torch(denoised)
print(denoised.shape)

# master_model.validate(device=device)
