"""
EEG Signal Reconstruction Model with Attention Mechanisms
"""

import torch
import torch.nn as nn
from .blocks import ChannelAttention, TemporalAttention, MultiHeadSelfAttention, MultiscaleAttentionFusion


class EncoderBlock(nn.Module):
    """Encoder block with convolutional layers and attention mechanisms"""
    def __init__(self, in_channels, out_channels, num_heads=4):
        super(EncoderBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2)  # Downsample by factor of 2
        )

        # Attention mechanisms
        self.channel_attention = ChannelAttention(out_channels)
        self.temporal_attention = TemporalAttention(out_channels)
        self.mhsa = MultiHeadSelfAttention(out_channels, num_heads=num_heads)

    def forward(self, x):
        # Convolutional block
        conv_out = self.conv_block(x)

        # Apply attention mechanisms with residual connections
        channel_attn = self.channel_attention(conv_out)
        temp_attn = self.temporal_attention(conv_out)
        conv_attn = conv_out + channel_attn + temp_attn  # Residual connection

        # Apply multi-head attention with residual connection
        mhsa_out = self.mhsa(conv_attn)
        refined = conv_attn + mhsa_out  # Residual connection

        return refined, conv_out  # Return both refined output and intermediate feature for skip connection


class DecoderBlock(nn.Module):
    """Decoder block with transposed convolution and attention mechanisms"""
    def __init__(self, in_channels, out_channels, skip_channels=0, num_heads=4):
        super(DecoderBlock, self).__init__()

        # Upsampling conv transpose
        self.up_conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

        # Skip connection fusion (if applicable)
        self.has_skip = skip_channels > 0
        if self.has_skip:
            self.skip_fusion = MultiscaleAttentionFusion(out_channels)

        # Convolutional block after upsampling (and optional skip fusion)
        self.conv_block = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Attention mechanisms
        self.channel_attention = ChannelAttention(out_channels)
        self.temporal_attention = TemporalAttention(out_channels)
        self.mhsa = MultiHeadSelfAttention(out_channels, num_heads=num_heads)

    def forward(self, x, skip=None):
        # Upsampling
        up = self.up_conv(x)

        # Skip connection fusion
        if self.has_skip and skip is not None:
            # Ensure skip connection matches upsampled feature map size
            if skip.size(2) != up.size(2):
                skip = nn.functional.adaptive_max_pool1d(skip, up.size(2))

            up = self.skip_fusion(up, skip)

        # Convolutional block
        conv_out = self.conv_block(up)

        # Apply attention mechanisms with residual connections
        channel_attn = self.channel_attention(conv_out)
        temp_attn = self.temporal_attention(conv_out)
        conv_attn = conv_out + channel_attn + temp_attn  # Residual connection

        # Apply multi-head attention with residual connection
        mhsa_out = self.mhsa(conv_attn)
        refined = conv_attn + mhsa_out  # Residual connection

        return refined


class BottleneckBlock(nn.Module):
    """Bottleneck block for latent space processing"""
    def __init__(self, channels, num_heads=4):
        super(BottleneckBlock, self).__init__()

        # Bottleneck convolutions
        self.conv_block = nn.Sequential(
            nn.Conv1d(channels, channels*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels*2),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels*2, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )

        # Attention mechanisms for bottleneck
        self.channel_attention = ChannelAttention(channels)
        self.temporal_attention = TemporalAttention(channels)
        self.mhsa = MultiHeadSelfAttention(channels, num_heads=num_heads)

    def forward(self, x):
        # Convolutional bottleneck
        bottleneck = self.conv_block(x)

        # Apply attention mechanisms with residual connections
        channel_attn = self.channel_attention(bottleneck)
        temp_attn = self.temporal_attention(bottleneck)
        bottleneck_attn = bottleneck + channel_attn + temp_attn  # Residual connection

        # Apply multi-head attention with residual connection
        mhsa_out = self.mhsa(bottleneck_attn)
        refined = bottleneck_attn + mhsa_out  # Residual connection

        return refined


class EEGReconstructionModel(nn.Module):
    """
    Complete EEG Reconstruction Model with Encoder-Decoder Architecture

    Args:
        input_channels (int): Number of input EEG channels
        input_length (int): Length of input EEG signal
        base_channels (int): Base number of channels for first layer
        latent_dim (int): Dimension of latent space
        num_heads (int): Number of attention heads
    """
    def __init__(self, input_channels=1, input_length=19, base_channels=32, latent_dim=256, num_heads=4):
        super(EEGReconstructionModel, self).__init__()

        self.input_length = input_length

        # Encoder pathway
        self.encoder1 = EncoderBlock(input_channels, base_channels, num_heads=num_heads)  # Out: base_channels, length/2
        self.encoder2 = EncoderBlock(base_channels, base_channels*2, num_heads=num_heads)  # Out: base_channels*2, length/4
        self.encoder3 = EncoderBlock(base_channels*2, base_channels*4, num_heads=num_heads)  # Out: base_channels*4, length/8

        # Bottleneck
        self.bottleneck = BottleneckBlock(base_channels*4, num_heads=num_heads)  # Out: base_channels*4, length/8

        # Feature fusion for bottleneck features
        self.fusion = MultiscaleAttentionFusion(base_channels*4)

        # Linear projection to latent space and back
        # Calculate flattened size based on input length
        self.bottleneck_size = base_channels*4 * (input_length // 8)

        # Projection to latent space and back
        self.to_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.bottleneck_size, latent_dim),
            nn.ReLU(inplace=True)
        )

        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, self.bottleneck_size),
            nn.ReLU(inplace=True)
        )

        # Decoder pathway
        self.decoder3 = DecoderBlock(base_channels*4, base_channels*2, skip_channels=base_channels*4, num_heads=num_heads)  # Out: base_channels*2, length/4
        self.decoder2 = DecoderBlock(base_channels*2, base_channels, skip_channels=base_channels*2, num_heads=num_heads)  # Out: base_channels, length/2
        self.decoder1 = DecoderBlock(base_channels, base_channels//2, skip_channels=base_channels, num_heads=num_heads)  # Out: base_channels//2, length

        # Final output layer
        self.output_layer = nn.Sequential(
            nn.Conv1d(base_channels//2, input_channels, kernel_size=1),
            nn.Tanh()  # Tanh to constrain output to [-1, 1] range, common for EEG signals
        )

    def encode(self, x):
        """Encoding path"""
        # Encoding path
        enc1, skip1 = self.encoder1(x)
        enc2, skip2 = self.encoder2(enc1)
        enc3, skip3 = self.encoder3(enc2)

        # Bottleneck
        bottleneck = self.bottleneck(enc3)

        # Project to latent space
        latent = self.to_latent(bottleneck)

        return latent, [skip1, skip2, skip3]

    def decode(self, latent, skip_connections):
        """Decoding path"""
        # Reconstruct from latent space
        bottleneck_flat = self.from_latent(latent)
        bottleneck = bottleneck_flat.view(-1, self.bottleneck_size // (self.input_length // 8), self.input_length // 8)

        # Apply fusion at bottleneck for additional refinement
        bottleneck = self.fusion(bottleneck, bottleneck)

        # Decoding path with skip connections
        dec3 = self.decoder3(bottleneck, skip_connections[2])
        dec2 = self.decoder2(dec3, skip_connections[1])
        dec1 = self.decoder1(dec2, skip_connections[0])

        # Output layer
        output = self.output_layer(dec1)

        return output

    def forward(self, x):
        """Forward pass"""
        # Encoding
        latent, skip_connections = self.encode(x)

        # Decoding
        output = self.decode(latent, skip_connections)

        return output, latent


class EEGReconstructionLoss(nn.Module):
    """
    Combined loss function for EEG reconstruction

    Args:
        alpha (float): Weight for MSE loss
        beta (float): Weight for L1 loss
        gamma (float): Weight for frequency domain loss
    """
    def __init__(self, alpha=0.8, beta=0.1, gamma=0.1):
        super(EEGReconstructionLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.alpha = alpha  # Weight for MSE loss
        self.beta = beta    # Weight for L1 loss
        self.gamma = gamma  # Weight for frequency domain loss

    def frequency_loss(self, x, y):
        """Compute mean squared error in frequency domain"""
        # Convert signals to frequency domain using FFT
        x_fft = torch.fft.rfft(x, dim=2)
        y_fft = torch.fft.rfft(y, dim=2)

        # Compute mean squared error in frequency domain
        return torch.mean(torch.abs(x_fft - y_fft)**2)

    def forward(self, pred, target):
        """Combined loss computation"""
        # Time domain losses
        mse = self.mse_loss(pred, target)
        l1 = self.l1_loss(pred, target)

        # Frequency domain loss
        freq_loss = self.frequency_loss(pred, target)

        # Combined loss
        total_loss = self.alpha * mse + self.beta * l1 + self.gamma * freq_loss

        return total_loss


def create_eeg_reconstruction_model(input_channels=1, input_length=19, base_channels=32, latent_dim=256, num_heads=4):
    """
    Factory function to create EEG reconstruction model

    Args:
        input_channels (int): Number of input channels
        input_length (int): Input signal length
        base_channels (int): Base channels for first layer
        latent_dim (int): Latent space dimension
        num_heads (int): Number of attention heads

    Returns:
        tuple: (model, loss_function)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGReconstructionModel(
        input_channels=input_channels,
        input_length=input_length,
        base_channels=base_channels,
        latent_dim=latent_dim,
        num_heads=num_heads
    ).to(device)

    loss_fn = EEGReconstructionLoss().to(device)

    return model, loss_fn
