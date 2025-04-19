
import torch.nn as nn

class FlexibleCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.features = self._make_feature_extractor()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, config['image_size'], config['image_size'])
            feat_dim = self.features(dummy).view(1, -1).size(1)
        self.classifier = self._make_classifier(feat_dim)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _make_feature_extractor(self):
        layers = []
        in_channels = 3
        num_filters = self.config['num_of_filter']
        multiplier = self.config['filter_multiplier']
        actv = self._get_activation_name(self.config['actv_func'])
        for i in range(self.config['conv_layers']):
            out_channels = int(num_filters * (multiplier ** i))
            kernel_size = self.config['filter_size'][i]
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size))
            if self.config.get('batch_normalization', False):
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(getattr(nn, actv)())
            layers.append(nn.MaxPool2d(2))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _make_classifier(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, self.config['dense_layer_size']),
            nn.ReLU(),
            nn.Dropout(self.config['dropout']),
            nn.Linear(self.config['dense_layer_size'], self.config['num_classes'])
        )

    def _get_activation_name(self, actv_func):
        mapping = {
            'elu': 'ELU',
            'gelu': 'GELU',
            'silu': 'SiLU',
            'selu': 'SELU',
            'leaky_relu': 'LeakyReLU',
            'relu': 'ReLU'
        }
        return mapping.get(actv_func, 'ReLU')
