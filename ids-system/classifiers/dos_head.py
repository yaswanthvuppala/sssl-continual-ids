from classifiers.base_head import build_classifier_head

def build_dos_head(embed_dim: int = 256):
    """Head for T1: DoS / DDoS (Binary: Benign vs DoS)"""
    return build_classifier_head(embed_dim, num_classes=2, name="dos_head")
