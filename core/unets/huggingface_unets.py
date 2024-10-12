from diffusers import UNet2DModel


class UNet2D(UNet2DModel):
    def __init__(self,
                 image_size:tuple=(32, 32),
                 in_channels:int=3,
                 out_channels:int=3,
                 block_out_channels:tuple=(128, 256, 512, 512),
                 activation_function:str="silu",
                 attention_head_dimension:int=8,
                 norm_num_groups:int=8,
                 ):
        super().__init__()
        # Setting instance variables:
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_out_channels = block_out_channels
        self.activation_function = activation_function
        self.attention_head_dimension = attention_head_dimension
        self.norm_num_groups = norm_num_groups

        # Initializing the UNet2DModel:
        self.model = UNet2DModel(sample_size=image_size,
                                 in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 block_out_channels=self.block_out_channels,
                                 act_fn=self.activation_function,
                                 attention_head_dim=self.attention_head_dimension,
                                 norm_num_groups=self.norm_num_groups)
