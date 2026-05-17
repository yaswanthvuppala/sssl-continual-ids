from classifiers.base_head import build_classifier_head

def build_exfiltration_head(embed_dim: int = 256):
    """Head for T3: Data Exfiltration (Binary: Benign vs Exfiltration)"""
    return build_classifier_head(embed_dim, num_classes=2, name="exfiltration_head")
