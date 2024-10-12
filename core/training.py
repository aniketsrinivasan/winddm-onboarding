import torch
import os
from tqdm.auto import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader
import torch.nn.functional as F
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from utils import ImageDataset, log_info, TrainingConfig


class MasterModel:
    @log_info()
    def __init__(self, model, training_config: TrainingConfig, train_dataset: ImageDataset,
                 lr_scheduler=None, sampler=None, stub_path=None):
        # Setting instance:
        self.model = model
        if stub_path is not None:
            print(f"Loading from pretrained at {stub_path}.")
            self.model.from_pretrained(stub_path)
        self.training_config = training_config
        self.train_dataset: ImageDataset = train_dataset
        self.stub_path = stub_path

        # Initializing DataLoader using ImageDataset provided:
        self.train_dataloader: DataLoader = DataLoader(self.train_dataset,
                                                       batch_size=self.training_config.train_batch_size)
        # Initializing optimizer (AdamW):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.training_config.learning_rate)
        # Initializing the learning rate scheduler:
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler
        else:
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=int(len(self.train_dataloader)*self.training_config.num_train_epochs),
            )
        # Initializing the sampler/scheduler:
        if sampler is not None:
            self.sampler = sampler
        else:
            self.sampler = DDPMScheduler(num_train_timesteps=self.training_config.num_train_timesteps)
        return

    @log_info()
    def train(self, save_name="model_trained", print_enabled=False):
        # Initialize Accelerator:
        accelerator = Accelerator(**self.training_config.get_accelerator_arguments())

        # Prepare the Accelerator:
        args = (self.model, self.optimizer, self.train_dataloader, self.lr_scheduler)
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(*args)
        # Set global steps:
        global_step = 0

        # Over epochs:
        for epoch in tqdm(range(self.training_config.num_train_epochs)):
            # Progress bar using TQDM:
            progress_bar = tqdm(total=len(train_dataloader),
                                disable=(not accelerator.is_local_main_process))
            progress_bar.set_description(f"Epoch {epoch}")

            # Over batches:
            for step, batch in enumerate(train_dataloader):
                # Getting clean images:
                clean_batch = batch

                # Sampling noise to add to the clean images:
                noise = torch.randn(clean_batch.shape, device=clean_batch.device)
                batch_size = clean_batch.shape[0]

                # Sample a random timestep for each image in the batch (using scheduler):
                timesteps = torch.randint(low=0,
                                          high=self.sampler.config["num_train_timesteps"],
                                          size=(batch_size,),
                                          device=clean_batch.device,
                                          dtype=torch.int64).long()
                # Forward diffusion (adding noise based on timesteps, using sampler):
                noisy_batch = self.sampler.add_noise(clean_batch, noise, timesteps)
                residual = noisy_batch - clean_batch

                # Back-propagation:
                with accelerator.accumulate(model):
                    noise_prediction = model(noisy_batch, timesteps, return_dict=False)[0]
                    # Loss can either be with noise, or with residual.
                    loss = F.mse_loss(noise_prediction, noise)
                    accelerator.backward(loss)

                    # Printing information every so often:
                    if (global_step % 10 == 0) and print_enabled:
                        print(f"Loss at step {global_step}: {loss.item()}")

                    # Stepping and resetting gradients:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Increment the number of global steps we've taken:
                global_step += 1

            # Saving process:
            if (((epoch + 1) % self.training_config.save_image_epochs == 0)
                or (epoch == self.training_config.num_train_epochs - 1)):
                pass
            if (((epoch + 1) % self.training_config.save_model_epochs == 0)
                or (epoch == self.training_config.num_train_epochs - 1)):
                print(f"Saving model at {os.path.join(self.training_config.output_dir, save_name)}")
                model.save_pretrained(os.path.join(self.training_config.output_dir, save_name))

        accelerator.end_training()
        print(f"Done training.")
        return

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, predict_denoised=False) -> torch.Tensor:
        """
        One denoising pass using MasterModel.model and MasterModel.sampler.

        :param x:                   (noise) sample at timestep of the image.
        :param timestep:            current timestep.
        :param predict_denoised:    whether to predict the denoised image (defaults to residual).
        :return:                    prediction of noise at current timestep.
        """
        # Convert timestep to an IntTensor:
        timestep = timestep.to(dtype=torch.int64)
        # Predicting noise:
        noise_prediction = self.model(x, timestep).sample

        # If we want to return the denoised prediction:
        if predict_denoised:
            denoised_prediction = self.sampler.step(model_output=noise_prediction,
                                                    timestep=int(timestep.item()),
                                                    sample=x,
                                                    return_dict=False)[0]
            return denoised_prediction
        return noise_prediction
