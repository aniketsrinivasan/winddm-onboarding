import torch
import cv2 as cv
import os
import functools
from transformers import TrainingArguments
from torch.utils.data import Dataset, DataLoader
import numpy as np
from .core_utils import log_info


class TrainingConfig:
    def __init__(self,
                 train_batch_size:int=8,
                 eval_batch_size:int=8,
                 num_train_epochs:int=1,
                 num_train_timesteps:int=1000,
                 gradient_accumulation_steps:int=1,
                 learning_rate:float=1e-4,
                 learning_warmup_ratio:float=0.2,
                 output_dir:str="stubs/diffusion-pepe-256",
                 save_model_epochs:int=1,
                 save_image_epochs:int=1):
        # Setting instance:
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_train_epochs = num_train_epochs
        self.num_train_timesteps = num_train_timesteps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.learning_warmup_ratio = learning_warmup_ratio
        self.output_dir = output_dir
        self.save_model_epochs = save_model_epochs
        self.save_image_epochs = save_image_epochs

        # Other training configurations (shouldn't need to be changed):
        self.overwrite_output_dir = False
        self.eval_strategy = "epoch"

        # Instantiating a HuggingFace TrainingArguments object:
        self.training_arguments = TrainingArguments(output_dir=self.output_dir,
                                                    overwrite_output_dir=self.overwrite_output_dir,
                                                    eval_strategy=self.eval_strategy,
                                                    per_device_train_batch_size=self.train_batch_size,
                                                    per_device_eval_batch_size=self.eval_batch_size,
                                                    num_train_epochs=self.num_train_epochs,
                                                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                                                    learning_rate=self.learning_rate,
                                                    warmup_ratio=self.learning_warmup_ratio,
                                                    )

    def get_training_arguments(self):
        return self.training_arguments

    def get_accelerator_arguments(self):
        accelerator_arguments = dict(
            mixed_precision=None,
            gradient_accumulation_steps=self.training_arguments.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=os.path.join(self.training_arguments.output_dir, "logs")
        )
        return accelerator_arguments


@functools.cache
def directory_to_tensor(directory_path: str, img_size:tuple=(256, 256),
                        transform=None) -> torch.Tensor:
    images = []
    for image_path in os.listdir(directory_path):
        # Read the image in, perform transformations:
        this_image = cv.imread(os.path.join(directory_path, image_path))
        this_image = cv.resize(this_image, dsize=img_size)
        this_image = torch.from_numpy(this_image)             # conversion to Tensor
        # We reshape our Tensor (currently (width, height, channels)):
        #   (width, height, channels) => (height, width, channels)
        this_image = torch.reshape(this_image, shape=(this_image.shape[2], *img_size))
        # Applying transform if applicable:
        if transform is not None:
            try:
                this_image = transform(this_image)            # custom transform
            except Exception as e:
                print(f"Could not transform {image_path} in {__qualname__}: {e}")
        images.append(this_image)                             # append to collection

    # Convert image collection to a torch.Tensor:
    images = torch.stack(images, dim=0)
    return images


# Class for storing images into a singular torch.Tensor.
class ImageStack:
    def __init__(self, directory_path: str, img_size:tuple=(256, 256), image_transform=None):
        # Setting instance variables:
        self.directory_path = directory_path
        self.img_size = img_size
        self.image_transform = image_transform

        # Loading in the data to self.data_tensor:
        self.data_tensor = directory_to_tensor(directory_path=self.directory_path,
                                               img_size=self.img_size,
                                               transform=self.image_transform)

    def __len__(self):
        return self.data_tensor.shape[0]

    def __getitem__(self, idx):
        return self.data_tensor[idx]


# Wrapper class inheriting from Dataset (Torch).
class ImageDataset(Dataset):
    @log_info(log_path="logs/testing", log_enabled=False, print_enabled=True, display_args=True)
    # log_info(__init__, log_path="logs
    def __init__(self, directory_path: str, img_size:tuple=(256, 256), image_transform=None):
        # Store information in an ImageStack object:
        self.image_stack = ImageStack(directory_path=directory_path,
                                      img_size=img_size,
                                      image_transform=image_transform)

    def __len__(self):
        return len(self.image_stack)

    def __getitem__(self, idx):
        return self.image_stack[idx]
