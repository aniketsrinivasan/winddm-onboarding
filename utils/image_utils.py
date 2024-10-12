import matplotlib.pyplot as plt
import torch


class ImageUtils:
    @staticmethod
    def plot_torch(image: torch.Tensor) -> tuple:
        # If the image has shape greater than 3:
        if len(image.shape) > 3:
            try:
                image = torch.squeeze(image, 0)
            except Exception as e:
                print(f"Unable to plot image of shape {image.shape}. Error: {e}")

        # Rescaling so we have things in a [0..255] range:
        image -= image.min()
        image /= image.max()
        image = (image * 255).long()
        # We permute the indices:
        #   (3, height, width) => (height, width, 3)
        image = torch.reshape(image, (image.shape[2], image.shape[1], image.shape[0]))
        # Detach Torch:
        image = image.detach().numpy()
        # Plot the image:
        plt.imshow(image)
        plt.show()
        return image, True


