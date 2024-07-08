import timm
import torch
from timm.models.layers import SelectAdaptivePool2d
from torch import nn

__all__ = (
    "MultiModalNet",
    "MultiModalNetFullModalityFeatureFusion",
    "MultiModalNetFullModalityGeometricFusion",
)


class CompositionalLayer(nn.Module):
    def __init__(
        self,
        in_features,
    ):
        super().__init__()
        self.fc = nn.Linear(in_features * 2, in_features, bias=True)

    def forward(self, f1, f2):
        """
        :param f1: shared-modality fts
        :param f2: specific-modality fts
        :return:
        """
        residual = torch.cat((f1, f2), 1)
        residual = self.fc(residual)
        features = f1 + residual

        return features


def _get_extractor_in_features(
    encoder_name,
    in_chans=3,
):
    model = timm.create_model(
        encoder_name,
        in_chans=in_chans,
    )
    return model.get_classifier().in_features


class MultiModalNet(nn.Module):
    def __init__(
        self,
        encoder_name,
        num_classes,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_features = _get_extractor_in_features(encoder_name)

        self.shared_enc = timm.create_model(
            encoder_name,
            pretrained=True,
            num_classes=0,
        )

        self.s2_enc = timm.create_model(
            encoder_name,
            pretrained=True,
            num_classes=0,
        )

        self.ortho_enc = timm.create_model(
            encoder_name,
            pretrained=True,
            num_classes=0,
        )

        self.street_enc = timm.create_model(
            encoder_name,
            pretrained=True,
            num_classes=0,
        )

        self.compos_layer = CompositionalLayer(self.in_features)
        self.domain_classfier = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=3, bias=True),
        )

        self.fc = nn.Linear(self.in_features * 3 + self.num_classes, self.num_classes)

    def forward(
        self,
        images,
        s2_data,
        country_id,
    ):
        num_channel = images.shape[1]
        assert num_channel in [3, 6]
        num_modal = num_channel // 3 + 1

        spec_feats = self.s2_enc(s2_data)[None, ...]
        spec_feats = torch.cat((spec_feats, self.ortho_enc(images[:, :3, ...])[None, ...]), axis=0)
        if num_channel == 6:
            spec_feats = torch.cat((spec_feats, self.street_enc(images[:, 3:, ...])[None, ...]), axis=0)

        shared_feats = self.shared_enc(s2_data)[None, ...]
        shared_feats = torch.cat((shared_feats, self.shared_enc(images[:, :3, ...])[None, ...]), axis=0)
        if num_channel == 6:
            shared_feats = torch.cat((shared_feats, self.shared_enc(images[:, 3:, ...])[None, ...]), axis=0)

        fused_feats = self.compos_layer(shared_feats[0], spec_feats[0])[None, ...]
        for i in range(1, num_modal):
            fused_feats = torch.cat((fused_feats, self.compos_layer(shared_feats[i], spec_feats[i])[None, ...]), axis=0)

        if num_modal == 2:
            fused_feats = torch.cat((fused_feats, torch.mean(shared_feats, dim=0)[None, ...]), axis=0)

        fused_feats = fused_feats.transpose(0, 1).reshape(fused_feats.shape[1], -1)
        fused_feats = torch.cat(
            (fused_feats, nn.functional.one_hot(country_id, num_classes=self.num_classes).float()), axis=1
        )

        logits = self.fc(fused_feats)

        spec_logits = self.domain_classfier(spec_feats[0])
        for i in range(1, num_modal):
            spec_logits = torch.cat((spec_logits, self.domain_classfier(spec_feats[i])), axis=0)

        return logits, spec_logits, shared_feats


class MultiModalNetFullModalityFeatureFusion(nn.Module):
    def __init__(
        self,
        encoder_name,
        num_classes,
    ):
        super().__init__()

        self.in_features_rgb = _get_extractor_in_features(encoder_name)
        self.in_features_s2 = _get_extractor_in_features(encoder_name, in_chans=12)

        self.num_classes = num_classes

        self.s2_enc = timm.create_model(
            encoder_name,
            pretrained=True,
            num_classes=0,
            in_chans=12,
        )

        self.ortho_enc = timm.create_model(
            encoder_name,
            pretrained=True,
            num_classes=0,
        )

        self.street_enc = timm.create_model(
            encoder_name,
            pretrained=True,
            num_classes=0,
        )

        self.fc = nn.Sequential(
            nn.Linear(
                self.in_features_rgb * 2 + self.in_features_s2 + self.num_classes,
                self.in_features_rgb,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                self.in_features_rgb,
                self.num_classes,
            ),
        )

    def forward(
        self,
        images,
        s2_data,
        country_id,
    ):
        feat = torch.cat(
            (
                self.s2_enc(s2_data),
                self.ortho_enc(images[:, :3, ...]),
                self.street_enc(images[:, 3:, ...]),
                nn.functional.one_hot(country_id, num_classes=self.num_classes).float(),
            ),
            axis=1,
        )
        feat = self.fc(feat)

        return feat


class MultiModalNetFullModalityGeometricFusion(nn.Module):
    def __init__(
        self,
        encoder_name,
        num_classes,
    ):
        super().__init__()

        self.in_features = _get_extractor_in_features(encoder_name)
        self.num_classes = num_classes

        self.s2_enc = timm.create_model(
            encoder_name,
            pretrained=True,
            num_classes=0,
            global_pool="",
            output_stride=4,
            in_chans=12,
        )

        self.ortho_enc = timm.create_model(
            encoder_name,
            pretrained=True,
            global_pool="",
            num_classes=0,
        )

        self.street_enc = timm.create_model(
            encoder_name,
            pretrained=True,
            global_pool="",
            num_classes=0,
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_features * 3, self.in_features, kernel_size=1),
            nn.BatchNorm2d(self.in_features),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.in_features * 3, self.in_features, kernel_size=1),
            nn.BatchNorm2d(self.in_features),
            nn.ReLU(inplace=True),
        )
        self.global_pool = SelectAdaptivePool2d(pool_type="avg", flatten=nn.Flatten(start_dim=1, end_dim=-1))
        self.fc = nn.Linear(self.in_features + self.num_classes, self.num_classes)

    def forward(
        self,
        images,
        s2_data,
        country_id,
    ):
        s2_feat = self.s2_enc(s2_data)
        ortho_feat = self.ortho_enc(images[:, :3, ...])
        street_feat = self.street_enc(images[:, 3:, ...])

        feat = torch.cat((s2_feat, ortho_feat, street_feat), axis=1)
        feat = self.conv1(feat)

        s2_feat = torch.mul(s2_feat, feat)
        ortho_feat = torch.mul(ortho_feat, feat)
        street_feat = torch.mul(street_feat, feat)

        feat = torch.cat((s2_feat, ortho_feat, street_feat), axis=1)

        feat = self.conv2(feat)
        feat = self.global_pool(feat)

        feat = torch.cat((feat, nn.functional.one_hot(country_id, num_classes=self.num_classes).float()), axis=1)
        feat = self.fc(feat)

        return feat
