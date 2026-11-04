import torch
import torch.nn as nn
class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU -> MaxPool -> Dropout2d
 
    MaxPool halves height and width so deeper blocks see lower resolution
    while the caller increases the channel count.
    """
 
    def __init__(self, in_ch, out_ch, batch_norm, dropout):
        super().__init__()
        layers = [
            # bias is redundant when BatchNorm follows, so drop it in that case
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not batch_norm)
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers += [nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)
 
    def forward(self, x):
        return self.block(x)
class SimpleCNN(nn.Module):

    """ 
    Configurable CNN for plant disease classification.
    Depth, channel widths, dense head, pooling, batch norm and dropout are all constructor parameters, 
    so experiments only need a YAML change.
    """
 
    def __init__(
        self,
        num_classes=39,
        in_channels=3,
        input_size=128,
        channels=(32, 64, 128),
        dense_units=(128,),
        pooling="flatten",
        batch_norm=True,
        conv_dropout=0.0,
        dense_dropout=0.3,
    ):
        super().__init__()
        if pooling not in ("flatten", "global_avg"):
            raise ValueError(f"pooling must be 'flatten' or 'global_avg', got {pooling!r}")
        if len(channels) == 0:
            raise ValueError("channels must contain at least one conv block")
        if input_size // (2 ** len(channels)) < 1:
            raise ValueError(
                f"input_size={input_size} is too small for {len(channels)} pooling stages"
            )
 
        # Feature extractor, each block halves resolution, channels grow.
        blocks, prev = [], in_channels
        for out_ch in channels:
            blocks.append(ConvBlock(prev, out_ch, batch_norm, conv_dropout))
            prev = out_ch
        self.features = nn.Sequential(*blocks)
 
        # Pool to a vector, flatten keeps spatial layout, global_avg shrinks the head.
        if pooling == "global_avg":
            self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
            flat_dim = prev
        else:
            self.pool = nn.Flatten()
            flat_dim = prev * (input_size // 2 ** len(channels)) ** 2
 
        # Classification head, optional hidden dense layers, then 39-way linear.
        head = []
        in_dim = flat_dim
        for units in dense_units:
            head.append(nn.Linear(in_dim, units, bias=not batch_norm))
            if batch_norm:
                head.append(nn.BatchNorm1d(units))
            head.append(nn.ReLU(inplace=True))
            if dense_dropout > 0:
                head.append(nn.Dropout(dense_dropout))
            in_dim = units
        head.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*head)
 
    def forward(self, x):
        # Return raw logits of shape (batch, num_classes)
        return self.classifier(self.pool(self.features(x)))


def build_transfer_model(num_classes):
    # TODO: pretrained model fine-tuning
    raise NotImplementedError


def build_model(cfg):
    #Build the CNN from the loaded YAML config
    m = cfg["model"]
    return SimpleCNN(
        num_classes=m["num_classes"],
        in_channels=m["in_channels"],
        input_size=m["input_size"],
        channels=m["channels"],
        dense_units=m["dense_units"],
        pooling=m["pooling"],
        batch_norm=m["batch_norm"],
        conv_dropout=m["conv_dropout"],
        dense_dropout=m["dense_dropout"],
    )

def count_parameters(model):
    # Number of trainable parameters (reported per variant in the report)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
