from classifiers.base_head import build_classifier_head

def build_scan_head(embed_dim: int = 256):
    """Head for T2: Port Scan (Binary: Benign vs Scan)"""
    return build_classifier_head(embed_dim, num_classes=2, name="scan_head")
