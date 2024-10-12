import utils
import diffusers
import matplotlib.pyplot as plt
import torch
from core import UNet2D, MasterModel, TrainingConfig

#
images = utils.ImageDataset(directory_path="pistachio_dataset/data/Kirmizi_Pistachio",
                            img_size=(32, 32))
test_image = images[0].reshape(32, 32, 3).detach().numpy()
plt.imshow(test_image)
plt.show()


model = UNet2D()
training_config = TrainingConfig(num_train_epochs=20,
                                 train_batch_size=8,
                                 output_dir="stubs/diffusion-pistachio-32")
master_model = MasterModel(model=model.model,
                           training_config=training_config,
                           train_dataset=images)
master_model.train(save_name="model_trained_2", print_enabled=True)