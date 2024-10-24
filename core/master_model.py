import torch
import os
from tqdm.auto import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader
import torch.nn.functional as F
from diffusers import DDPMScheduler, DDIMScheduler, EulerDiscreteScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from utils import ImageDataset, log_info, TrainingConfig


class MasterModel:
    @log_info()
    def __init__(self, model, training_config: TrainingConfig, train_dataset: ImageDataset,
                 val_dataset: ImageDataset = None, lr_scheduler=None, sampler=None, stub_path=None):
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Main Initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instance variables:
        self.model = model
        self.training_config: TrainingConfig = training_config
        self.sampler = (sampler if sampler is not None else
                        DDPMScheduler(num_train_timesteps=self.training_config.num_train_timesteps))
        self.train_dataset: ImageDataset = train_dataset
        self.val_dataset: ImageDataset = val_dataset
        self.stub_path = stub_path
        # Initializing DataLoader using ImageDataset provided:
        self.train_dataloader: DataLoader = DataLoader(self.train_dataset,
                                                       batch_size=self.training_config.train_batch_size,
                                                       shuffle=True)
        self.val_dataloader: DataLoader = DataLoader(self.val_dataset,
                                                     batch_size=self.training_config.train_batch_size)
        # Initializing optimizer (AdamW):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.training_config.learning_rate)
        self.lr_scheduler = lr_scheduler if lr_scheduler is not None else \
            get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=int(len(self.train_dataloader)*self.training_config.num_train_epochs),
            )

        # ~~~~~~~~~~~~~~~~~~~~ Accelerator Initialization and Model Loading ~~~~~~~~~~~~~~~~~~~~
        # Initialize the Accelerator (this is how we'll save and load models):
        self.accelerator = Accelerator(**self.training_config.get_accelerator_arguments())
        args = (self.model, self.optimizer, self.train_dataloader, self.lr_scheduler)
        # Prepare the Accelerator:
        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(*args)
        # Loading the model (if applicable):
        if stub_path is not None:
            self.load_from_stub()

    @log_info(print_enabled=True, display_args=True)
    def save_checkpoint(self, save_name=None, save_path=None):
        """
        Saves the model using Accelerator.save_state() to the provided path.
        This path defaults to os.path.join(output_dir, save_name) where output_dir is provided in TrainingConfig.
        Parameter save_path overrides save_name.
        """
        # Setting the path for the checkpoint to be saved:
        if save_path is not None:
            path = save_path
        else:
            path = os.path.join(self.training_config.output_dir, save_name)
        self.accelerator.save_state(path)
        return

    @log_info(print_enabled=True, display_args=True)
    def load_from_stub(self, custom_stub_path=None):
        """
        Loading models using Accelerator.load_state(), using the stub provided.
        Using custom_stub_path takes priority over using stub_path provided during initialization.
        """
        if self.stub_path is None and custom_stub_path is None:
            return
        # Get the stub_path we want:
        if custom_stub_path is not None:
            stub_path = custom_stub_path
        else:
            stub_path = self.stub_path
        # Loading from stub_path:
        if os.path.exists(stub_path):
            self.accelerator.load_state(stub_path)
            return
        else:
            print(f"SoftWarn: Unable to load model from {stub_path}.")
            return

    @log_info(print_enabled=True, display_args=True)
    def train(self, save_name: str, objective="noise", print_enabled=True):
        # Check whether objective is valid:
        if objective not in ("noise", "residual", "sample"):
            raise NotImplementedError(f"Objective {objective} not found.")

        # Set global steps:
        global_step = 0

        # Iterate over epochs:
        for epoch in tqdm(range(self.training_config.num_train_epochs)):
            # Iterate over batches:
            for step, batch in enumerate(self.train_dataloader):
                # Get clean images:
                clean_batch = batch
                batch_size = clean_batch.shape[0]
                # Sample random set of timesteps:
                timesteps = torch.randint(low=0,
                                          high=self.sampler.config["num_train_timesteps"],
                                          size=(batch_size,),
                                          device=clean_batch.device,
                                          dtype=torch.int64).long()

                # Sample noise and add it based on the timestep:
                noise = torch.randn(clean_batch.shape, device=clean_batch.device)
                noisy_batch = self.sampler.add_noise(clean_batch, noise, timesteps)

                # Define loss based on the objective chosen:
                if objective == "noise":
                    label = noise
                elif objective == "residual":
                    label = noisy_batch - clean_batch
                elif objective == "sample":
                    label = clean_batch
                else:
                    raise NotImplementedError(f"Objective {objective} not found.")

                # Propagation loop:
                with self.accelerator.accumulate(self.model):
                    prediction = self.model(noisy_batch, timesteps, return_dict=True).sample
                    loss = F.mse_loss(prediction, label)
                    self.accelerator.backward(loss)

                    # Printing information:
                    if (global_step % 10 == 0) and print_enabled:
                        if self.training_config.train_batch_size == 1:
                            print(f"Noised timesteps: {timesteps.item()}")
                        print(f"Loss at step {global_step}: {loss.item()}")

                    # Stepping the optimizer and resetting gradients:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Increment steps:
                global_step += 1

            # Checkpointing:
            if (((epoch + 1) % self.training_config.save_model_epochs == 0) or
                (epoch == self.training_config.num_train_epochs - 1)):
                self.save_checkpoint(save_name=save_name)

        # End the training process:
        self.accelerator.end_training()
        if print_enabled:
            print(f"Finished the training process for {save_name}.")
        return

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, predict_denoised=False, device="cpu"):
        # Convert timestep to IntTensor:
        if not isinstance(self.sampler, EulerDiscreteScheduler):
            timestep = torch.Tensor([timestep])
            timestep = timestep.to(dtype=torch.int64, device=device)
        timestep = timestep.to(device=device)
        x = x.to(device=device)
        # Predict (noise):
        noise_prediction = self.model(x, timestep, return_dict=True).sample

        # If we want to return the denoised prediction:
        if predict_denoised:
            if not isinstance(self.sampler, EulerDiscreteScheduler):
                timestep_int = int(timestep.item())
            else:
                timestep_int = timestep
                self.sampler.scale_model_input(x, timestep_int)
            denoised_prediction = self.sampler.step(model_output=noise_prediction,
                                                    timestep=timestep_int,
                                                    sample=x,
                                                    return_dict=True).prev_sample
            return denoised_prediction
        return noise_prediction

    def validate(self, val_dataloader=None, device="cpu"):
        if val_dataloader is None:
            val_dataloader = self.val_dataloader
        # Iterate over batches of the validation set:
        for _, batch in enumerate(val_dataloader):
            # Get clean images:
            clean_batch = batch.to(device=device)
            batch_size = clean_batch.shape[0]
            # Sample random set of timesteps:
            timesteps = torch.randint(low=0,
                                      high=self.sampler.config["num_train_timesteps"],
                                      size=(batch_size,),
                                      device=clean_batch.device,
                                      dtype=torch.int64).long()

            # Sample noise and add it based on the timestep:
            noise = torch.randn(clean_batch.shape, device=clean_batch.device)
            noisy_batch = self.sampler.add_noise(clean_batch, noise, timesteps).to(device=clean_batch.device)

            # Run prediction:
            with torch.no_grad():
                label = noise
                prediction = self.model(noisy_batch, timesteps, return_dict=True).sample
                loss = F.mse_loss(prediction, label)
                print(f"Loss: {loss:.4f}")
        return

