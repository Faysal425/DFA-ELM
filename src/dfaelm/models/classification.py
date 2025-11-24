"""
DFA-ELM Classification Model for Cognitive Workload Assessment
"""

import torch
import torch.nn as nn
from .blocks import ChannelAttention, TemporalAttention, MultiHeadSelfAttention, MultiscaleAttentionFusion


class ELMClassifier(nn.Module):
    """
    Extreme Learning Machine Classifier

    Args:
        input_size (int): Input feature dimension
        hidden_size (int): Hidden layer size
        num_classes (int): Number of output classes
    """
    def __init__(self, input_size, hidden_size, num_classes):
        super(ELMClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # Hidden layer with random weights (not trainable)
        self.hidden_layer = nn.Linear(input_size, hidden_size, bias=True)

        # Initialize weights randomly and freeze them
        self._init_weights()
        self._freeze_hidden_layer()

        # Output layer
        self.output_layer = nn.Linear(hidden_size, num_classes, bias=True)

        # Activation function
        self.activation = nn.ReLU()

    def _init_weights(self):
        """Initialize hidden layer weights randomly"""
        nn.init.xavier_uniform_(self.hidden_layer.weight)
        nn.init.zeros_(self.hidden_layer.bias)

    def _freeze_hidden_layer(self):
        """Freeze the hidden layer parameters"""
        for param in self.hidden_layer.parameters():
            param.requires_grad = False

    def fit(self, X, y, alpha=0.001):
        """
        Analytically compute the output layer weights using pseudoinverse

        Args:
            X: Input features (tensor)
            y: Target labels (tensor)
            alpha: Regularization parameter
        """
        # Get hidden layer output
        with torch.no_grad():
            hidden_output = self.activation(self.hidden_layer(X))

        # Convert target to one-hot encoded format if needed
        if y.dim() == 1:
            y_onehot = torch.zeros(y.size(0), self.num_classes, device=y.device)
            y_onehot.scatter_(1, y.unsqueeze(1), 1)
        else:
            y_onehot = y

        # Calculate output weights using ridge regression (pseudoinverse)
        H = hidden_output
        H_T = H.t()

        # Ridge regression solution with regularization
        # W = (H^T * H + alpha * I)^(-1) * H^T * T
        identity = torch.eye(H.size(1), device=H.device) * alpha
        temp = torch.inverse(H_T @ H + identity) @ H_T
        output_weights = temp @ y_onehot

        # Set output layer weights
        with torch.no_grad():
            self.output_layer.weight.copy_(output_weights.t())
            # Bias will be learned through backpropagation

    def forward(self, x):
        """Forward pass"""
        # Hidden layer feature mapping with frozen weights
        hidden_output = self.activation(self.hidden_layer(x))

        # Output layer
        output = self.output_layer(hidden_output)
        return output


class DFAELM_C(nn.Module):
    """
    DFA-ELM Classification Model with CNN + Attention + ELM

    Args:
        input_size (int): Input signal length
        num_classes (int): Number of classes
        num_heads (int): Number of attention heads
        elm_hidden_size (int): ELM hidden layer size
    """
    def __init__(self, input_size=28, num_classes=2, num_heads=4, elm_hidden_size=512):
        super(DFAELM_C, self).__init__()

        # First convolutional block
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=1)
        )

        # Channel attention and temporal attention after first block
        self.channel_attention1 = ChannelAttention(64)
        self.temporal_attention1 = TemporalAttention(64)

        # Multi-head self-attention for feature refinement
        self.mhsa1 = MultiHeadSelfAttention(64, num_heads=num_heads)

        # Second convolutional block
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=1)
        )

        # Channel attention and temporal attention after second block
        self.channel_attention2 = ChannelAttention(128)
        self.temporal_attention2 = TemporalAttention(128)

        # Multi-head self-attention for feature refinement
        self.mhsa2 = MultiHeadSelfAttention(128, num_heads=num_heads)

        # Define expansion convolutions as module components
        self.conv1_expander = nn.Conv1d(64, 256, kernel_size=1)
        self.conv2_expander = nn.Conv1d(128, 256, kernel_size=1)

        # Feature fusion module - Multiscale Attention Fusion
        self.feature_fusion = MultiscaleAttentionFusion(256, reduction_ratio=16)

        # Calculate output size after convolutions for the fully connected layer
        self.flatten_size = self._get_flatten_size(input_size)

        # Feature extractor (replaced classifier)
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # ELM classifier
        self.elm_classifier = ELMClassifier(256, elm_hidden_size, num_classes)

        # Flag to track if ELM has been fitted
        self.elm_fitted = False

    def _get_flatten_size(self, input_size):
        """Calculate the output size after all convolution operations"""
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, input_size)

            # First convolution block
            out1 = self.conv_block1(dummy_input)

            # Apply channel and temporal attention with residual connections
            channel_attn1 = self.channel_attention1(out1)
            temp_attn1 = self.temporal_attention1(out1)
            out1_attn = out1 + channel_attn1 + temp_attn1  # Residual connection

            # Apply multi-head attention with residual connection
            mhsa_out1 = self.mhsa1(out1_attn)
            out1_refined = out1_attn + mhsa_out1  # Residual connection

            # Second convolution block
            out2 = self.conv_block2(out1_refined)

            # Apply channel and temporal attention with residual connections
            channel_attn2 = self.channel_attention2(out2)
            temp_attn2 = self.temporal_attention2(out2)
            out2_attn = out2 + channel_attn2 + temp_attn2  # Residual connection

            # Apply multi-head attention with residual connection
            mhsa_out2 = self.mhsa2(out2_attn)
            out2_refined = out2_attn + mhsa_out2  # Residual connection

            # Apply expansion
            out1_expanded = self.conv1_expander(out1_refined)
            out2_expanded = self.conv2_expander(out2_refined)

            # Check if there's a size mismatch and adjust if needed
            if out1_expanded.shape[2] != out2_expanded.shape[2]:
                # Use the smaller dimension as target
                target_size = min(out1_expanded.shape[2], out2_expanded.shape[2])
                out1_expanded = nn.functional.adaptive_max_pool1d(out1_expanded, target_size)
                out2_expanded = nn.functional.adaptive_max_pool1d(out2_expanded, target_size)

            # Feature fusion with MultiscaleAttentionFusion
            fused = self.feature_fusion(out1_expanded, out2_expanded)

            return fused.shape[1] * fused.shape[2]

    def forward(self, x):
        """Forward pass"""
        # First convolutional block
        conv1_out = self.conv_block1(x)

        # Apply channel and temporal attention with residual connections
        channel_attn1 = self.channel_attention1(conv1_out)
        temp_attn1 = self.temporal_attention1(conv1_out)
        conv1_attn = conv1_out + channel_attn1 + temp_attn1  # Residual connection

        # Apply multi-head attention with residual connection
        mhsa_out1 = self.mhsa1(conv1_attn)
        conv1_refined = conv1_attn + mhsa_out1  # Residual connection

        # Second convolutional block
        conv2_out = self.conv_block2(conv1_refined)

        # Apply channel and temporal attention with residual connections
        channel_attn2 = self.channel_attention2(conv2_out)
        temp_attn2 = self.temporal_attention2(conv2_out)
        conv2_attn = conv2_out + channel_attn2 + temp_attn2  # Residual connection

        # Apply multi-head attention with residual connection
        mhsa_out2 = self.mhsa2(conv2_attn)
        conv2_refined = conv2_attn + mhsa_out2  # Residual connection

        # Create 256-channel tensors using the defined expanders
        conv1_expanded = self.conv1_expander(conv1_refined)
        conv2_expanded = self.conv2_expander(conv2_refined)

        # Check if there's a size mismatch and adjust if needed
        if conv1_expanded.shape[2] != conv2_expanded.shape[2]:
            # Use the smaller dimension as target
            target_size = min(conv1_expanded.shape[2], conv2_expanded.shape[2])
            conv1_expanded = nn.functional.adaptive_max_pool1d(conv1_expanded, target_size)
            conv2_expanded = nn.functional.adaptive_max_pool1d(conv2_expanded, target_size)

        # Feature fusion using MultiscaleAttentionFusion
        fused = self.feature_fusion(conv1_expanded, conv2_expanded)

        # Flatten
        flat = fused.view(fused.size(0), -1)

        # Extract features
        features = self.feature_extractor(flat)

        # ELM classification
        output = self.elm_classifier(features)

        return output

    def fit_elm(self, dataloader, device):
        """
        Fit the ELM classifier using the extracted features from the trained CNN

        Args:
            dataloader: DataLoader containing training data
            device: Device to run on
        """
        self.eval()  # Set model to evaluation mode
        all_features = []
        all_labels = []

        # Extract features and labels from dataloader
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Get features up to the feature extraction layer
                conv1_out = self.conv_block1(inputs)
                channel_attn1 = self.channel_attention1(conv1_out)
                temp_attn1 = self.temporal_attention1(conv1_out)
                conv1_attn = conv1_out + channel_attn1 + temp_attn1
                mhsa_out1 = self.mhsa1(conv1_attn)
                conv1_refined = conv1_attn + mhsa_out1

                conv2_out = self.conv_block2(conv1_refined)
                channel_attn2 = self.channel_attention2(conv2_out)
                temp_attn2 = self.temporal_attention2(conv2_out)
                conv2_attn = conv2_out + channel_attn2 + temp_attn2
                mhsa_out2 = self.mhsa2(conv2_attn)
                conv2_refined = conv2_attn + mhsa_out2

                # Create 256-channel tensors using the defined expanders
                conv1_expanded = self.conv1_expander(conv1_refined)
                conv2_expanded = self.conv2_expander(conv2_refined)

                # Check if there's a size mismatch and adjust if needed
                if conv1_expanded.shape[2] != conv2_expanded.shape[2]:
                    # Use the smaller dimension as target
                    target_size = min(conv1_expanded.shape[2], conv2_expanded.shape[2])
                    conv1_expanded = nn.functional.adaptive_max_pool1d(conv1_expanded, target_size)
                    conv2_expanded = nn.functional.adaptive_max_pool1d(conv2_expanded, target_size)

                # Feature fusion using MultiscaleAttentionFusion
                fused = self.feature_fusion(conv1_expanded, conv2_expanded)
                flat = fused.view(fused.size(0), -1)
                features = self.feature_extractor(flat)

                all_features.append(features)
                all_labels.append(labels)

        # Concatenate all features and labels
        all_features = torch.cat(all_features, 0)
        all_labels = torch.cat(all_labels, 0)

        # Fit ELM classifier
        self.elm_classifier.fit(all_features, all_labels, alpha=0.001)
        self.elm_fitted = True

        print("ELM classifier fitted successfully!")


def create_dfaelm_classifier(input_size=28, num_classes=2, num_heads=4, elm_hidden_size=512):
    """
    Factory function to create DFA-ELM classification model

    Args:
        input_size (int): Input signal length
        num_classes (int): Number of classes
        num_heads (int): Number of attention heads
        elm_hidden_size (int): ELM hidden layer size

    Returns:
        DFAELM_C: Classification model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DFAELM_C(
        input_size=input_size,
        num_classes=num_classes,
        num_heads=num_heads,
        elm_hidden_size=elm_hidden_size
    ).to(device)

    return model
