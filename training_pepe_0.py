import utils
import torch
from core import UNet2D, MasterModel, TrainingConfig


# Setting device:
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Initializing datasets:
images = utils.ImageDataset(directory_path="pepe_dataset/data",
                            img_size=(64, 64))

# Initializing the model of choice:
model = UNet2D(block_out_channels=(16, 32, 64, 128),
               attention_head_dimension=8)
# Initializing the training configuration:
training_config = TrainingConfig(num_train_epochs=5,
                                 train_batch_size=4,
                                 learning_rate=1e-4,
                                 learning_warmup_ratio=0.2,
                                 output_dir="stubs/diffusion-pepe-64",)
# Initializing MasterModel object for the model:
master_model = MasterModel(model=model.model,
                           training_config=training_config,
                           train_dataset=images,
                           stub_path="stubs/diffusion-pepe-64/trained-pepe-0")
master_model.train(save_name="trained-pepe-0")

# Running post-training inference:
RUN_INFERENCE = True

if RUN_INFERENCE:
    # (Manual) denoising process:
    viz_intermediate = True
    viz_intermediate_steps = 10
    start_denoise_step = 200
    num_denoising_steps = 50
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
    for i in range(start_denoise_step, 0, step_size):
        denoised = master_model.forward(noisy, torch.Tensor([i]), predict_denoised=True, device=device)
        if (count % viz_intermediate_steps == 0) and viz_intermediate:
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
