import { useState } from "react";

// ── Color System ──────────────────────────────────────────────────────────────
const C = {
  bg:       "#080c14",
  bg1:      "#0d1220",
  bg2:      "#111827",
  bg3:      "#1a2235",
  border:   "#1e2d44",
  border2:  "#2a3f5f",
  cyan:     "#00d4ff",
  cyanDim:  "#00d4ff22",
  cyanBg:   "#00d4ff11",
  teal:     "#00ffc8",
  tealDim:  "#00ffc822",
  amber:    "#f59e0b",
  amberDim: "#f59e0b22",
  amberBg:  "#f59e0b11",
  green:    "#22c55e",
  greenDim: "#22c55e22",
  greenBg:  "#22c55e11",
  purple:   "#a855f7",
  purpleDim:"#a855f722",
  purpleBg: "#a855f711",
  orange:   "#f97316",
  orangeDim:"#f9731622",
  red:      "#ef4444",
  redDim:   "#ef444422",
  text:     "#e2e8f0",
  textMd:   "#94a3b8",
  textDim:  "#64748b",
  white:    "#ffffff",
};

// ── Tabs ──────────────────────────────────────────────────────────────────────
const TABS = ["Overview", "Architecture", "Modules", "Workflow", "Code", "Folder", "Limitations"];

// ── Module Data ───────────────────────────────────────────────────────────────
const MODULE_DATA = [
  {
    id: "ssl-encoder",
    name: "SSL Encoder (SimCLR / BERT-flow)",
    icon: "🧠",
    color: C.cyan,
    colorDim: C.cyanDim,
    colorBg: C.cyanBg,
    description: "Self-supervised learning backbone that learns rich traffic representations from millions of unlabeled flows without any annotation. Frozen after pretraining for stable, reusable embeddings.",
    tech: ["SimCLR", "BERT-flow", "Contrastive Loss", "Masked Autoencoding", "TensorFlow"],
    inputs: ["Unlabeled network flows (millions)", "PCAP captures", "Flow metadata (duration, bytes, ports)"],
    outputs: ["Rich embedding vectors (d=256–512)", "Frozen encoder weights", "Representation manifold"],
    details: "Stage 1 of the two-stage pipeline. Uses contrastive or masked prediction objectives on raw traffic features. Once trained, the encoder is frozen and serves as a feature extractor for all downstream classifiers. This design allows encoder reuse across new attack categories without retraining.",
  },
  {
    id: "gpm",
    name: "Gradient Projection Memory (GPM)",
    icon: "🔒",
    color: C.amber,
    colorDim: C.amberDim,
    colorBg: C.amberBg,
    description: "Novel continual learning mechanism that prevents catastrophic forgetting when new attack categories are introduced. Projects gradient updates into the null-space of past task gradients.",
    tech: ["Null-space Projection", "Gradient Capture", "Memory Bank (SVD)", "Continual Learning", "NumPy/TensorFlow"],
    inputs: ["Current task gradients", "Past task gradient basis vectors", "Memory bank"],
    outputs: ["Projected safe gradients", "Updated memory bank", "Null-space basis"],
    details: "After each task's training, GPM captures principal gradient directions into a memory bank using SVD decomposition. When training on a new task, gradient updates are orthogonally projected to avoid disturbing previously learned representations. This enables sequential task learning without forgetting.",
  },
  {
    id: "classifier-heads",
    name: "Task-Specific Classifier Heads",
    icon: "🎯",
    color: C.green,
    colorDim: C.greenDim,
    colorBg: C.greenBg,
    description: "Modular, per-task classification heads sitting atop the frozen encoder. Each head specializes in a distinct attack category, enabling fine-grained multi-task detection.",
    tech: ["FixMatch", "MixMatch", "Semi-supervised Learning", "Pseudo-labeling", "Softmax Heads"],
    inputs: ["Frozen encoder embeddings", "Few labeled flows (100–500 per task)", "Unlabeled pool"],
    outputs: ["Attack type labels", "Per-class confidence scores", "Anomaly score"],
    details: "Each head is trained with FixMatch/MixMatch semi-supervised objectives using the frozen encoder. Only 100–500 labeled examples per task are required. Heads for T1 (DoS/DDoS), T2 (Port scan), T3 (Exfiltration), Anomaly Detection, and extensible T4+ (Zero-day) operate independently.",
  },
  {
    id: "fixmatch",
    name: "FixMatch / MixMatch Trainer",
    icon: "📐",
    color: C.purple,
    colorDim: C.purpleDim,
    colorBg: C.purpleBg,
    description: "Semi-supervised training engine that generates pseudo-labels for unlabeled data using consistency regularization. Bridges the gap between few labeled examples and large unlabeled pools.",
    tech: ["FixMatch", "MixMatch", "Consistency Regularization", "Pseudo-labeling", "Augmentation"],
    inputs: ["Labeled batch (few shots)", "Unlabeled pool (bulk)", "Frozen encoder features"],
    outputs: ["Pseudo-labeled data", "Trained classifier head", "Confidence-filtered predictions"],
    details: "FixMatch applies strong augmentation to unlabeled flows and retains only high-confidence pseudo-labels (threshold ≥ 0.95) for consistency loss. MixMatch interpolates labeled and unlabeled representations. Together they maximize label efficiency and enable learning from sparse annotations.",
  },
  {
    id: "anomaly-head",
    name: "Anomaly Detection Head",
    icon: "🚨",
    color: C.orange,
    colorDim: C.orangeDim,
    colorBg: C.amberDim,
    description: "Zero-day threat detector operating in the encoder embedding space. Combines reconstruction-based and density-based anomaly scoring for detecting previously unseen attack patterns.",
    tech: ["Autoencoder", "One-Class SVM", "Isolation Forest", "Mahalanobis Distance", "Threshold Tuning"],
    inputs: ["Encoder embeddings", "Normal traffic baseline", "Configurable threshold"],
    outputs: ["Anomaly score (0–1)", "Reconstruction error", "Outlier flag"],
    details: "Unlike labeled classifier heads, the anomaly head learns the distribution of normal traffic in embedding space. It flags deviations as potential zero-day threats using reconstruction error and statistical distance measures. Calibrated thresholds balance false-positive rate with detection sensitivity.",
  },
  {
    id: "ids-output",
    name: "IDS Output Engine",
    icon: "📡",
    color: C.teal,
    colorDim: C.tealDim,
    colorBg: C.tealDim,
    description: "Unified output layer that aggregates predictions from all classifier heads and the anomaly detector into actionable alerts with attack type classification and anomaly scoring.",
    tech: ["Ensemble Fusion", "Confidence Calibration", "Alert Routing", "SIEM Integration", "REST API"],
    inputs: ["All head predictions", "Anomaly score", "Flow metadata"],
    outputs: ["Attack type + confidence", "Anomaly score", "Alert with severity", "SIEM event"],
    details: "Combines outputs from T1–T4+ heads and the anomaly detector using a confidence-weighted ensemble. Generates structured alerts with attack type, confidence score, anomaly magnitude, flow identifiers, and severity level. Exposes a REST API for SIEM and SOC integration.",
  },
  {
    id: "data-pipeline",
    name: "Traffic Ingestion Pipeline",
    icon: "📥",
    color: C.textMd,
    colorDim: C.border2,
    colorBg: C.bg3,
    description: "Real-time and batch traffic collection, preprocessing, and feature extraction layer that feeds both the pretraining encoder and the semi-supervised classifier training.",
    tech: ["Zeek / Suricata", "Apache Kafka", "Flow Feature Extraction", "Normalization", "Redis Cache"],
    inputs: ["Raw PCAP", "NetFlow/IPFIX records", "Syslog feeds"],
    outputs: ["Feature vectors", "Labeled batches", "Unlabeled pool"],
    details: "Zeek or Suricata extracts 80+ flow-level features including duration, bytes, packets, port numbers, protocol flags, and inter-arrival times. Kafka provides streaming ingestion at scale. Features are normalized and cached in Redis for efficient mini-batch sampling during training.",
  },
];

// ── Workflow Steps ────────────────────────────────────────────────────────────
const WORKFLOW = [
  {
    phase: "Stage 1 — Pretraining",
    color: C.cyan,
    colorDim: C.cyanDim,
    icon: "🧠",
    steps: [
      { label: "Collect unlabeled traffic", detail: "Millions of flows from network taps, mirrors, or logs. No annotation required." },
      { label: "Extract flow features", detail: "80+ statistical features per flow: bytes, duration, ports, flags, inter-arrival time." },
      { label: "Define pretext task", detail: "Choose between contrastive (SimCLR) or masked prediction (BERT-flow) objective." },
      { label: "Train SSL encoder", detail: "Encoder learns rich representations. Train until convergence on pretext loss." },
      { label: "Freeze encoder weights", detail: "Encoder is frozen. Reused as a stable feature extractor for all downstream tasks." },
    ],
  },
  {
    phase: "Stage 2 — Task Learning",
    color: C.green,
    colorDim: C.greenDim,
    icon: "🎯",
    steps: [
      { label: "Label few traffic samples", detail: "100–500 labeled flows per attack category. Can be done by SOC analysts." },
      { label: "Capture task gradients (GPM)", detail: "Before training, record baseline gradient directions into the memory bank." },
      { label: "Train classifier head (FixMatch)", detail: "Semi-supervised training on the frozen encoder's embeddings." },
      { label: "Project gradients (GPM null-space)", detail: "New task gradients are projected to not interfere with past task directions." },
      { label: "Evaluate on held-out set", detail: "Validate F1, precision, recall on each attack category independently." },
    ],
  },
  {
    phase: "Continual Task Expansion",
    color: C.amber,
    colorDim: C.amberDim,
    icon: "🔄",
    steps: [
      { label: "New attack category appears", detail: "Zero-day flagged by anomaly head; SOC identifies new threat signature." },
      { label: "Collect new labeled examples", detail: "Minimal labeling effort: 100–500 examples of the new attack type." },
      { label: "Add new classifier head", detail: "Initialize fresh head for T(n+1). Encoder remains frozen." },
      { label: "GPM protects past tasks", detail: "Gradient projection ensures T1–Tn accuracy is preserved during T(n+1) training." },
      { label: "Update anomaly baseline", detail: "Retune anomaly detector thresholds with knowledge of new known-attack class." },
    ],
  },
  {
    phase: "Inference",
    color: C.purple,
    colorDim: C.purpleDim,
    icon: "⚡",
    steps: [
      { label: "Ingest live traffic", detail: "Streaming flow from Kafka at line-rate. Redis caches recent windows." },
      { label: "Encode with frozen SSL encoder", detail: "Sub-millisecond forward pass through the pretrained encoder." },
      { label: "Run all classifier heads in parallel", detail: "T1–T4+ heads score the embedding simultaneously." },
      { label: "Compute anomaly score", detail: "Anomaly head returns reconstruction error and distance metric." },
      { label: "Emit IDS alert", detail: "Fused output: attack type + confidence + anomaly score → SIEM." },
    ],
  },
];

// ── Code Snippets ─────────────────────────────────────────────────────────────
const CODE = {
  "SSL Pretraining": `# ssl_pretrain.py
import tensorflow as tf
import numpy as np
from models import build_flow_encoder, build_projection_head
from losses import nt_xent_loss

class SSLPretrainer:
    def __init__(self, config):
        self.encoder = build_flow_encoder(
            input_dim=config.feature_dim,      # 80+ flow features
            hidden_dim=config.hidden_dim,       # 512
            embedding_dim=config.embed_dim,     # 256
        )
        self.projector = build_projection_head(
            in_dim=config.embed_dim,
            out_dim=config.proj_dim,            # 128
        )
        self.temperature = 0.07
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=3e-4, weight_decay=1e-4
        )
        # Combined model for gradient tape
        self.trainable_vars = (
            self.encoder.trainable_variables +
            self.projector.trainable_variables
        )

    @tf.function
    def train_step(self, x1, x2):
        with tf.GradientTape() as tape:
            z1 = self.projector(self.encoder(x1, training=True), training=True)
            z2 = self.projector(self.encoder(x2, training=True), training=True)
            loss = nt_xent_loss(z1, z2, temperature=self.temperature)
        grads = tape.gradient(loss, self.trainable_vars)
        self.optimizer.apply_gradients(zip(grads, self.trainable_vars))
        return loss

    def train_epoch(self, dataset: tf.data.Dataset):
        total_loss = 0.0
        steps = 0
        for x1, x2 in dataset:          # two augmented views of same flow
            loss = self.train_step(x1, x2)
            total_loss += float(loss)
            steps += 1
        return total_loss / steps

    def freeze_encoder(self):
        self.encoder.trainable = False
        self.encoder.save("encoder_frozen.keras")
        print("[SSL] Encoder frozen and saved.")`,

  "GPM Module": `# gpm.py — Gradient Projection Memory (TensorFlow)
import numpy as np
import tensorflow as tf

class GradientProjectionMemory:
    """
    Prevents catastrophic forgetting by projecting new-task gradients
    into the null-space of past task gradient subspaces.
    """
    def __init__(self, threshold: float = 0.97):
        self.threshold = threshold   # SVD energy threshold
        self.memory_bank: list[np.ndarray] = []  # basis per past task

    def capture_gradient_basis(self, model, dataset, loss_fn):
        """After task T, compute SVD basis of gradient vectors."""
        grad_matrix = []
        for x, y in dataset:
            with tf.GradientTape() as tape:
                preds = model(x, training=False)
                loss  = loss_fn(y, preds)
            grads = tape.gradient(loss, model.trainable_variables)
            flat  = np.concatenate([
                g.numpy().ravel() for g in grads if g is not None
            ])
            grad_matrix.append(flat)

        G = np.stack(grad_matrix)                          # [N, D]
        U, S, _ = np.linalg.svd(G.T, full_matrices=False)
        # Retain components up to threshold fraction of energy
        energy = np.cumsum(S ** 2) / np.sum(S ** 2)
        k = int(np.searchsorted(energy, self.threshold)) + 1
        self.memory_bank.append(U[:, :k])
        print(f"[GPM] Captured basis with {k} components.")

    def project_gradients(self, grads, variables):
        """Project current gradients onto null-space of stored bases."""
        if not self.memory_bank:
            return grads
        projected = []
        for g, v in zip(grads, variables):
            if g is None:
                projected.append(g)
                continue
            g_np = g.numpy().ravel()
            for basis in self.memory_bank:
                # Project out components in past-task subspace
                g_np = g_np - basis @ (basis.T @ g_np)
            projected.append(
                tf.reshape(tf.constant(g_np, dtype=v.dtype), v.shape)
            )
        return projected`,

  "FixMatch Trainer": `# fixmatch_trainer.py (TensorFlow)
import tensorflow as tf

CONFIDENCE_THRESHOLD = 0.95

class FixMatchTrainer:
    def __init__(self, encoder, classifier_head, gpm, config):
        self.encoder = encoder          # frozen SSL encoder
        self.head    = classifier_head  # task-specific dense head
        self.gpm     = gpm              # gradient projection memory
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate=config.lr, momentum=0.9, nesterov=True
        )
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )

    def _encode(self, x):
        return self.encoder(x, training=False)   # frozen pass

    @tf.function
    def train_step(self, x_l, y_l, x_u_weak, x_u_strong):
        # Generate pseudo-labels on weakly-augmented unlabeled data (no tape)
        emb_u_weak  = self._encode(x_u_weak)
        probs_weak  = tf.nn.softmax(self.head(emb_u_weak, training=False))
        conf        = tf.reduce_max(probs_weak, axis=-1)
        pseudo      = tf.argmax(probs_weak, axis=-1)
        mask        = tf.cast(conf >= CONFIDENCE_THRESHOLD, tf.float32)

        with tf.GradientTape() as tape:
            # Supervised loss on labeled data
            emb_l    = self._encode(x_l)
            logits_l = self.head(emb_l, training=True)
            loss_s   = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    y_l, logits_l, from_logits=True
                )
            )
            # Consistency loss on strongly-augmented unlabeled data
            emb_u_strong  = self._encode(x_u_strong)
            logits_strong = self.head(emb_u_strong, training=True)
            loss_u = tf.reduce_mean(
                self.loss_fn(pseudo, logits_strong) * mask
            )
            total_loss = loss_s + 1.0 * loss_u

        grads = tape.gradient(total_loss, self.head.trainable_variables)
        # GPM null-space projection before applying
        grads = self.gpm.project_gradients(grads, self.head.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.head.trainable_variables)
        )
        return {
            "loss_s": float(loss_s),
            "loss_u": float(loss_u),
            "mask_rate": float(tf.reduce_mean(mask)),
        }`,

  "Inference Pipeline": `# inference_engine.py (TensorFlow)
import tensorflow as tf
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class IDSAlert:
    flow_id: str
    attack_type: Optional[str]
    confidence: float
    anomaly_score: float
    severity: str    # LOW / MEDIUM / HIGH / CRITICAL

class IDSInferenceEngine:
    def __init__(self, encoder, heads: dict, anomaly_detector, config):
        self.encoder  = encoder          # frozen .keras model
        self.heads    = heads            # {attack_name: tf.keras.Model}
        self.anomaly  = anomaly_detector
        self.threshold_anomaly = config.anomaly_threshold    # e.g. 0.65
        self.threshold_attack  = config.attack_threshold     # e.g. 0.80

    @tf.function(input_signature=[tf.TensorSpec([1, None], tf.float32)])
    def _encode(self, x):
        return self.encoder(x, training=False)

    def score(self, flow_features: np.ndarray, flow_id: str) -> IDSAlert:
        x = tf.constant(flow_features[np.newaxis, :], dtype=tf.float32)

        # Stage 1 — encode (frozen, sub-ms, compiled graph)
        embedding = self._encode(x)

        # Stage 2 — run all heads in parallel
        best_conf, best_type = 0.0, None
        for attack_type, head in self.heads.items():
            probs = tf.nn.softmax(head(embedding, training=False))
            conf  = float(tf.reduce_max(probs))
            if conf > best_conf:
                best_conf, best_type = conf, attack_type

        # Anomaly score from reconstruction / distance
        anomaly_score = self.anomaly.score(embedding.numpy())

        # Resolve final label
        if best_conf >= self.threshold_attack:
            label = best_type
        elif anomaly_score >= self.threshold_anomaly:
            label = "zero-day / unknown"
        else:
            label = None   # benign

        severity = (
            "CRITICAL" if anomaly_score > 0.9 else
            "HIGH"     if anomaly_score > 0.7 else
            "MEDIUM"   if anomaly_score > 0.5 else
            "LOW"
        )

        return IDSAlert(flow_id, label, best_conf, anomaly_score, severity)`,

  "Data Loader": `# traffic_dataset.py (TensorFlow tf.data)
import tensorflow as tf
import numpy as np
from augmentations import augment_weak, augment_strong

FLOW_FEATURES = [
    "duration", "bytes_fwd", "bytes_bwd", "pkts_fwd", "pkts_bwd",
    "pkt_len_mean", "pkt_len_std", "iat_mean", "iat_std",
    "flags_syn", "flags_fin", "flags_rst", "flags_psh", "flags_ack",
    "port_src", "port_dst", "protocol", "flow_rate",
    # ... 80+ features total
]

def make_unlabeled_dataset(
    flow_records: np.ndarray,
    batch_size: int = 512,
) -> tf.data.Dataset:
    """Yields two augmented views of the same flow for SSL pretraining."""
    ds = tf.data.Dataset.from_tensor_slices(
        tf.constant(flow_records, dtype=tf.float32)
    )
    def two_views(x):
        return augment_strong(x), augment_strong(x)

    return (
        ds
        .shuffle(buffer_size=50_000, reshuffle_each_iteration=True)
        .map(two_views, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

def make_labeled_dataset(
    flows: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 64,
) -> tf.data.Dataset:
    """Small labeled set for semi-supervised fine-tuning."""
    ds = tf.data.Dataset.from_tensor_slices((
        tf.constant(flows,  dtype=tf.float32),
        tf.constant(labels, dtype=tf.int32),
    ))
    def weak_aug(x, y):
        return augment_weak(x), y

    return (
        ds
        .shuffle(buffer_size=5_000, reshuffle_each_iteration=True)
        .map(weak_aug, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

def build_datasets(config):
    unlabeled_ds = make_unlabeled_dataset(
        config.unlabeled_flows, batch_size=512
    )
    labeled_ds = make_labeled_dataset(
        config.labeled_flows, config.labels, batch_size=64
    )
    return unlabeled_ds, labeled_ds`,
};

// ── Folder Structure ──────────────────────────────────────────────────────────
const FOLDER = `ids-system/
├── encoder/
│   ├── models/
│   │   ├── flow_encoder.py        # SimCLR / BERT-flow backbone
│   │   ├── projection_head.py     # SSL projection head
│   │   └── bert_flow.py           # Masked flow prediction model
│   ├── losses/
│   │   ├── ntxent_loss.py         # NT-Xent contrastive loss
│   │   └── masked_pred_loss.py    # BERT-style masked loss
│   ├── augmentations.py           # Flow augmentation strategies
│   └── ssl_pretrain.py            # Stage-1 training script
│
├── gpm/
│   ├── gpm.py                     # Gradient Projection Memory core
│   ├── svd_utils.py               # SVD decomposition helpers
│   └── memory_bank.py             # Persistent basis storage
│
├── classifiers/
│   ├── heads/
│   │   ├── dos_ddos_head.py       # T1: DoS/DDoS classifier
│   │   ├── port_scan_head.py      # T2: Port scan classifier
│   │   ├── exfiltration_head.py   # T3: Data exfiltration
│   │   ├── anomaly_head.py        # Anomaly / Zero-day detector
│   │   └── base_head.py           # Abstract base class
│   ├── fixmatch_trainer.py        # Semi-supervised FixMatch trainer
│   ├── mixmatch_trainer.py        # MixMatch alternative trainer
│   └── pseudo_labeler.py          # Confidence-filtered pseudo labels
│
├── inference/
│   ├── inference_engine.py        # Real-time IDS scoring engine
│   ├── ensemble_fusion.py         # Multi-head prediction aggregation
│   ├── alert_router.py            # Alert generation + SIEM routing
│   └── threshold_calibration.py  # Operating point tuning
│
├── data/
│   ├── traffic_dataset.py         # tf.data pipelines for flows
│   ├── kafka_consumer.py          # Real-time Kafka ingestion
│   ├── feature_extractor.py       # Zeek/Suricata feature extraction
│   ├── normalizer.py              # Feature normalization + scaling
│   └── redis_cache.py             # Hot-path embedding cache
│
├── api/
│   ├── server.py                  # FastAPI REST server
│   ├── routers/
│   │   ├── alerts.py              # GET /alerts, POST /alert/ack
│   │   ├── models.py              # Model management endpoints
│   │   └── health.py              # Health + readiness probes
│   └── schemas.py                 # Pydantic request/response models
│
├── infrastructure/
│   ├── docker-compose.yml         # Full stack: Kafka, Redis, API, GPU
│   ├── k8s/
│   │   ├── deployment.yaml        # Kubernetes GPU deployment
│   │   ├── hpa.yaml               # Horizontal pod autoscaler
│   │   └── service.yaml           # Load balancer service
│   └── terraform/
│       ├── main.tf                # Cloud infra (GCP/AWS)
│       └── variables.tf
│
├── notebooks/
│   ├── 01_ssl_pretraining.ipynb   # Stage-1 experiment notebook
│   ├── 02_gpm_analysis.ipynb      # GPM forgetting metrics
│   ├── 03_fixmatch_ablation.ipynb # Semi-supervised label efficiency
│   └── 04_inference_benchmark.ipynb
│
├── tests/
│   ├── unit/                      # Component-level tests
│   └── integration/               # End-to-end pipeline tests
│
├── configs/
│   ├── pretrain.yaml              # SSL training hyperparams
│   ├── finetune.yaml              # Per-task fine-tune config
│   └── inference.yaml             # Deployment config
│
└── requirements.txt               # tensorflow, numpy, kafka-python, fastapi…`;

// ── Limitations ───────────────────────────────────────────────────────────────
const LIMITATIONS = [
  {
    icon: "⚡",
    title: "Label Scarcity Assumption",
    severity: "medium",
    severityColor: C.amber,
    desc: "Assumes 100–500 labeled samples per attack class are always obtainable. In practice, novel zero-day attacks may have zero available labels at detection time, breaking the supervised head assumption entirely.",
    improvements: ["Active learning loop for SOC-in-the-loop labeling", "Zero-shot attack classification via embedding space clustering", "LLM-assisted label generation from threat intelligence feeds"],
  },
  {
    icon: "💾",
    title: "GPM Memory Bank Scalability",
    severity: "high",
    severityColor: C.red,
    desc: "The SVD-based memory bank grows linearly with the number of past tasks. For systems with hundreds of attack categories over years, storing and computing null-space projections becomes computationally expensive.",
    improvements: ["Compressed memory bank via random sketching or low-rank approximation", "Periodic memory consolidation to merge similar past-task bases", "Approximate null-space projection with FAISS-accelerated nearest-neighbor lookup"],
  },
  {
    icon: "🌊",
    title: "Distribution Shift in Traffic",
    severity: "high",
    severityColor: C.red,
    desc: "Network traffic distributions shift over time due to infrastructure changes, new protocols, and seasonal patterns. The frozen SSL encoder may produce stale embeddings that degrade downstream classifier performance.",
    improvements: ["Periodic encoder fine-tuning with new unlabeled traffic windows", "Online continual learning with Elastic Weight Consolidation (EWC)", "Domain adaptation techniques for temporal distribution shift"],
  },
  {
    icon: "🔬",
    title: "Anomaly Detector Calibration",
    severity: "medium",
    severityColor: C.amber,
    desc: "The anomaly threshold must be manually calibrated per deployment environment. High false-positive rates in busy enterprise networks can overwhelm SOC analysts and erode trust in the system.",
    improvements: ["Automated threshold tuning via ROC curve optimization on validation traffic", "Environment-adaptive thresholding using quantile regression", "Alert fatigue reduction via alert clustering and deduplication"],
  },
  {
    icon: "⏱️",
    title: "Pretraining Cost",
    severity: "low",
    severityColor: C.green,
    desc: "Stage-1 SSL pretraining on millions of flows requires significant GPU compute (potentially days on a single A100). This is a one-time cost but creates a barrier for resource-constrained deployments.",
    improvements: ["Publish pretrained encoder checkpoints for common traffic domains", "Knowledge distillation to a lighter mobile encoder for edge deployment", "Federated pretraining across multiple network operators without data sharing"],
  },
  {
    icon: "🎭",
    title: "Adversarial Robustness",
    severity: "high",
    severityColor: C.red,
    desc: "The SSL encoder and classifier heads are not explicitly trained for adversarial robustness. Sophisticated attackers can craft flows that evade detection by adversarially perturbing flow statistics.",
    improvements: ["Adversarial training with PGD-augmented flow samples", "Certified robustness via randomized smoothing on flow features", "Anomaly score fusion with rule-based expert systems for defense-in-depth"],
  },
  {
    icon: "🔗",
    title: "Feature Engineering Dependency",
    severity: "medium",
    severityColor: C.amber,
    desc: "The system relies on 80+ hand-crafted flow features from Zeek/Suricata. These features may miss important temporal or packet-payload signals, and the feature pipeline adds operational complexity.",
    improvements: ["End-to-end raw packet learning with 1D CNNs or Transformers on byte sequences", "Automatic feature selection via mutual information with attack labels", "Graph neural network over flow interaction graphs for lateral movement detection"],
  },
];

// ── Small helper components ───────────────────────────────────────────────────
const Badge = ({ label, color = C.cyan }) => (
  <span style={{
    fontSize: 11, fontWeight: 600, padding: "2px 8px",
    borderRadius: 4, background: color + "22", color,
    border: `1px solid ${color}44`, letterSpacing: "0.04em",
    whiteSpace: "nowrap",
  }}>{label}</span>
);

const Tag = ({ label }) => (
  <span style={{
    fontSize: 11, padding: "2px 8px", borderRadius: 4,
    background: C.bg3, color: C.textDim, border: `1px solid ${C.border}`,
    whiteSpace: "nowrap",
  }}>{label}</span>
);

const SectionTitle = ({ children, color = C.cyan }) => (
  <h2 style={{
    fontSize: 13, fontWeight: 700, letterSpacing: "0.12em", textTransform: "uppercase",
    color, marginBottom: 16, display: "flex", alignItems: "center", gap: 8,
  }}>
    <span style={{ width: 20, height: 2, background: color, display: "inline-block" }} />
    {children}
  </h2>
);

// ── Architecture Flow Component ───────────────────────────────────────────────
function ArchitectureViz() {
  const box = (label, sub, color, w = 160) => ({
    label, sub, color, w,
  });

  const FlowBox = ({ node, style = {} }) => (
    <div style={{
      border: `1px solid ${node.color}55`,
      borderLeft: `3px solid ${node.color}`,
      background: node.color + "0d",
      borderRadius: 6, padding: "8px 12px",
      width: node.w, minWidth: node.w,
      ...style,
    }}>
      <div style={{ fontSize: 12, fontWeight: 600, color: node.color }}>{node.label}</div>
      {node.sub && <div style={{ fontSize: 10, color: C.textDim, marginTop: 2 }}>{node.sub}</div>}
    </div>
  );

  const Arrow = ({ color = C.border2, vertical = false }) => (
    <div style={{
      display: "flex", alignItems: "center", justifyContent: "center",
      ...(vertical ? { flexDirection: "column", height: 28, width: 24 } : { height: 24, width: 28 }),
    }}>
      <div style={{
        ...(vertical
          ? { width: 1, height: 16, background: color }
          : { height: 1, width: 16, background: color }),
      }} />
      <div style={{ color, fontSize: 10, lineHeight: 1 }}>{vertical ? "▼" : "▶"}</div>
    </div>
  );

  return (
    <div style={{ fontFamily: "'JetBrains Mono', 'Courier New', monospace" }}>

      {/* Stage 1 */}
      <div style={{ border: `1px dashed ${C.cyan}44`, borderRadius: 8, padding: 16, marginBottom: 20 }}>
        <div style={{ fontSize: 11, color: C.cyan, fontWeight: 700, marginBottom: 12, letterSpacing: "0.1em" }}>
          STAGE 1 — SELF-SUPERVISED PRETRAINING &nbsp;
          <span style={{ color: C.textDim, fontWeight: 400 }}>Frozen after training</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: 4 }}>
          <FlowBox node={box("Unlabeled traffic", "millions of flows", C.textDim, 150)} />
          <Arrow color={C.cyan} />
          <FlowBox node={box("Pretext task", "SimCLR / BERT-flow", C.cyan, 150)} />
          <Arrow color={C.cyan} />
          <FlowBox node={box("SSL Encoder", "rich embeddings", C.cyan, 140)} />
          <Arrow color={C.cyan} />
          <div style={{
            border: `2px solid ${C.cyan}88`, borderRadius: 6, padding: "8px 12px",
            background: C.cyanBg, width: 160,
          }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: C.cyan }}>🔒 Frozen Encoder</div>
            <div style={{ fontSize: 10, color: C.textDim, marginTop: 2 }}>stable, reusable, does not forget</div>
          </div>
        </div>
      </div>

      {/* GPM */}
      <div style={{ border: `1px dashed ${C.amber}66`, borderRadius: 8, padding: 16, marginBottom: 20 }}>
        <div style={{ fontSize: 11, color: C.amber, fontWeight: 700, marginBottom: 12, letterSpacing: "0.1em" }}>
          GRADIENT PROJECTION MEMORY (GPM) — NEW
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 4, flexWrap: "wrap" }}>
          <FlowBox node={box("Gradient capture", "per-task grad vectors", C.amber, 160)} />
          <Arrow color={C.amber} />
          <FlowBox node={box("Memory bank", "basis of past tasks", C.amber, 160)} />
          <Arrow color={C.amber} />
          <FlowBox node={box("Projection op", "null-space constraint", C.amber, 160)} />
        </div>
      </div>

      {/* Stage 2 */}
      <div style={{ border: `1px dashed ${C.green}44`, borderRadius: 8, padding: 16, marginBottom: 20 }}>
        <div style={{ fontSize: 11, color: C.green, fontWeight: 700, marginBottom: 12, letterSpacing: "0.1em" }}>
          STAGE 2 — TASK-SPECIFIC CLASSIFIER HEADS &nbsp;
          <span style={{ color: C.textDim, fontWeight: 400 }}>(frozen encoder + GPM-constrained updates)</span>
        </div>
        <div style={{ display: "flex", alignItems: "flex-start", gap: 8, flexWrap: "wrap" }}>
          {[
            { label: "Head T1", sub: "DoS/DDoS", color: C.green },
            { label: "Head T2", sub: "Port scan", color: C.green },
            { label: "Head T3", sub: "Exfiltration", color: C.green },
            { label: "Anomaly", sub: "Detection", color: C.orange },
            { label: "Head T4+", sub: "Zero-day", color: C.textDim },
          ].map((n, i) => (
            <div key={i} style={{
              border: `1px solid ${n.color}55`, borderLeft: `3px solid ${n.color}`,
              background: n.color + "0d", borderRadius: 6, padding: "8px 12px", minWidth: 100,
            }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: n.color }}>{n.label}</div>
              <div style={{ fontSize: 10, color: C.textDim }}>{n.sub}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Output */}
      <div style={{ display: "flex", justifyContent: "center", marginTop: 8 }}>
        <div style={{
          border: `2px solid ${C.teal}88`, borderRadius: 8, padding: "12px 32px",
          background: C.tealDim, textAlign: "center", width: "80%",
        }}>
          <div style={{ fontSize: 14, fontWeight: 700, color: C.teal }}>
            📡 IDS Output — Attack type + Anomaly score
          </div>
          <div style={{ fontSize: 11, color: C.textDim, marginTop: 4 }}>
            best of both: strong representations + label efficiency + zero-day anomalies · encoder pretrained once → reusable
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Modules Tab ───────────────────────────────────────────────────────────────
function ModulesTab() {
  const [expanded, setExpanded] = useState(null);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))", gap: 16 }}>
      {MODULE_DATA.map(m => {
        const open = expanded === m.id;
        return (
          <div
            key={m.id}
            onClick={() => setExpanded(open ? null : m.id)}
            style={{
              border: `1px solid ${open ? m.color + "88" : C.border}`,
              borderLeft: `3px solid ${m.color}`,
              borderRadius: 8, padding: 16, background: open ? m.colorBg : C.bg1,
              cursor: "pointer", transition: "all 0.2s",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
              <div>
                <div style={{ fontSize: 18, marginBottom: 4 }}>{m.icon}</div>
                <div style={{ fontSize: 13, fontWeight: 700, color: m.color, marginBottom: 4 }}>{m.name}</div>
                <div style={{ fontSize: 12, color: C.textMd, lineHeight: 1.5 }}>{m.description}</div>
              </div>
              <div style={{ color: C.textDim, marginLeft: 8, fontSize: 12 }}>{open ? "▲" : "▼"}</div>
            </div>

            <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 10 }}>
              {m.tech.map(t => <Tag key={t} label={t} />)}
            </div>

            {open && (
              <div style={{ marginTop: 14, borderTop: `1px solid ${C.border}`, paddingTop: 14 }}>
                <div style={{ fontSize: 12, color: C.textDim, lineHeight: 1.7, marginBottom: 12 }}>{m.details}</div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                  <div>
                    <div style={{ fontSize: 11, fontWeight: 700, color: C.purple, marginBottom: 6, letterSpacing: "0.08em" }}>INPUTS</div>
                    {m.inputs.map((inp, i) => (
                      <div key={i} style={{ fontSize: 11, color: C.textMd, display: "flex", gap: 6, marginBottom: 4 }}>
                        <span style={{ color: C.purple }}>→</span>{inp}
                      </div>
                    ))}
                  </div>
                  <div>
                    <div style={{ fontSize: 11, fontWeight: 700, color: C.teal, marginBottom: 6, letterSpacing: "0.08em" }}>OUTPUTS</div>
                    {m.outputs.map((out, i) => (
                      <div key={i} style={{ fontSize: 11, color: C.textMd, display: "flex", gap: 6, marginBottom: 4 }}>
                        <span style={{ color: C.teal }}>←</span>{out}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ── Workflow Tab ──────────────────────────────────────────────────────────────
function WorkflowTab() {
  const [activePhase, setActivePhase] = useState(0);
  const phase = WORKFLOW[activePhase];

  return (
    <div>
      {/* Phase tabs */}
      <div style={{ display: "flex", gap: 8, marginBottom: 24, flexWrap: "wrap" }}>
        {WORKFLOW.map((w, i) => (
          <button
            key={i}
            onClick={() => setActivePhase(i)}
            style={{
              padding: "8px 16px", borderRadius: 6, fontSize: 12, fontWeight: 600,
              cursor: "pointer", border: `1px solid ${activePhase === i ? w.color : C.border}`,
              background: activePhase === i ? w.color + "22" : C.bg2,
              color: activePhase === i ? w.color : C.textDim,
              transition: "all 0.15s",
            }}
          >
            {w.icon} {w.phase}
          </button>
        ))}
      </div>

      {/* Phase steps */}
      <div style={{ position: "relative" }}>
        <div style={{
          position: "absolute", left: 15, top: 0, bottom: 0,
          width: 2, background: phase.colorDim,
        }} />
        {phase.steps.map((step, i) => (
          <div key={i} style={{ display: "flex", gap: 16, marginBottom: 20, position: "relative" }}>
            <div style={{
              width: 32, height: 32, borderRadius: "50%", flexShrink: 0,
              border: `2px solid ${phase.color}`, background: phase.colorDim,
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 12, fontWeight: 700, color: phase.color, zIndex: 1,
            }}>
              {i + 1}
            </div>
            <div style={{
              border: `1px solid ${C.border}`, borderRadius: 8,
              padding: "12px 16px", background: C.bg1, flex: 1,
            }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: C.text, marginBottom: 4 }}>{step.label}</div>
              <div style={{ fontSize: 12, color: C.textDim, lineHeight: 1.6 }}>{step.detail}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Code Tab ──────────────────────────────────────────────────────────────────
function CodeTab() {
  const [activeSnippet, setActiveSnippet] = useState(Object.keys(CODE)[0]);
  const [copied, setCopied] = useState(false);

  const copyCode = () => {
    navigator.clipboard?.writeText(CODE[activeSnippet]);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div>
      <div style={{ display: "flex", gap: 6, marginBottom: 16, flexWrap: "wrap" }}>
        {Object.keys(CODE).map(k => (
          <button
            key={k}
            onClick={() => setActiveSnippet(k)}
            style={{
              padding: "6px 12px", borderRadius: 5, fontSize: 11, fontWeight: 600,
              cursor: "pointer", border: `1px solid ${activeSnippet === k ? C.cyan : C.border}`,
              background: activeSnippet === k ? C.cyanDim : C.bg2,
              color: activeSnippet === k ? C.cyan : C.textDim,
            }}
          >
            {k}
          </button>
        ))}
      </div>
      <div style={{ position: "relative" }}>
        <button
          onClick={copyCode}
          style={{
            position: "absolute", right: 12, top: 12, padding: "4px 10px",
            borderRadius: 4, fontSize: 11, cursor: "pointer",
            border: `1px solid ${C.border2}`, background: C.bg3, color: C.textMd,
          }}
        >
          {copied ? "✓ copied" : "copy"}
        </button>
        <pre style={{
          background: C.bg, border: `1px solid ${C.border}`, borderRadius: 8,
          padding: "20px 16px", overflowX: "auto", fontSize: 11.5,
          color: C.textMd, lineHeight: 1.65, margin: 0,
          fontFamily: "'JetBrains Mono', 'Fira Code', 'Courier New', monospace",
        }}>
          <code style={{ color: C.text }}>{CODE[activeSnippet]}</code>
        </pre>
      </div>
    </div>
  );
}

// ── Folder Tab ────────────────────────────────────────────────────────────────
function FolderTab() {
  const colorize = (line) => {
    if (line.includes("# ")) {
      const [code, comment] = line.split("# ");
      return (
        <>
          <span>{code}</span>
          <span style={{ color: C.textDim }}>{"# " + comment}</span>
        </>
      );
    }
    if (line.trim().endsWith("/")) return <span style={{ color: C.cyan }}>{line}</span>;
    return <span>{line}</span>;
  };

  return (
    <div style={{
      background: C.bg, border: `1px solid ${C.border}`, borderRadius: 8,
      padding: 20, overflowX: "auto",
    }}>
      <pre style={{ margin: 0, fontSize: 12, lineHeight: 1.8, color: C.text,
        fontFamily: "'JetBrains Mono', 'Courier New', monospace" }}>
        {FOLDER.split("\n").map((line, i) => (
          <div key={i}>{colorize(line)}</div>
        ))}
      </pre>
    </div>
  );
}

// ── Limitations Tab ───────────────────────────────────────────────────────────
function LimitationsTab() {
  const [expanded, setExpanded] = useState(null);

  const severityBadge = (severity, color) => (
    <span style={{
      fontSize: 10, fontWeight: 700, padding: "2px 7px", borderRadius: 3,
      background: color + "22", color, border: `1px solid ${color}44`,
      letterSpacing: "0.06em", textTransform: "uppercase",
    }}>{severity}</span>
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {LIMITATIONS.map((lim, i) => {
        const open = expanded === i;
        return (
          <div
            key={i}
            onClick={() => setExpanded(open ? null : i)}
            style={{
              border: `1px solid ${open ? lim.severityColor + "66" : C.border}`,
              borderLeft: `3px solid ${lim.severityColor}`,
              borderRadius: 8, padding: 16, background: open ? lim.severityColor + "08" : C.bg1,
              cursor: "pointer", transition: "all 0.2s",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span style={{ fontSize: 18 }}>{lim.icon}</span>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 700, color: C.text }}>{lim.title}</div>
                  {!open && <div style={{ fontSize: 11, color: C.textDim, marginTop: 2 }}>{lim.desc.slice(0, 80)}…</div>}
                </div>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                {severityBadge(lim.severity, lim.severityColor)}
                <span style={{ color: C.textDim, fontSize: 12 }}>{open ? "▲" : "▼"}</span>
              </div>
            </div>
            {open && (
              <div style={{ marginTop: 14, borderTop: `1px solid ${C.border}`, paddingTop: 14 }}>
                <div style={{ fontSize: 12, color: C.textMd, lineHeight: 1.7, marginBottom: 14 }}>{lim.desc}</div>
                <div style={{ fontSize: 11, fontWeight: 700, color: C.teal, marginBottom: 8, letterSpacing: "0.08em" }}>
                  PROPOSED IMPROVEMENTS
                </div>
                {lim.improvements.map((imp, j) => (
                  <div key={j} style={{ fontSize: 12, color: C.textMd, display: "flex", gap: 8, marginBottom: 6, lineHeight: 1.5 }}>
                    <span style={{ color: C.teal, flexShrink: 0 }}>✦</span>{imp}
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ── Overview Tab ──────────────────────────────────────────────────────────────
function OverviewTab() {
  const stats = [
    { label: "Labeled samples needed", value: "100–500", sub: "per attack class", color: C.cyan },
    { label: "SSL pretraining data", value: "Millions", sub: "unlabeled flows", color: C.purple },
    { label: "Training stages", value: "2", sub: "pretrain → finetune", color: C.teal },
    { label: "Classifier heads", value: "T1–T4+", sub: "modular + extensible", color: C.green },
    { label: "Forgetting prevention", value: "GPM", sub: "null-space projection", color: C.amber },
    { label: "Zero-day detection", value: "✦ Yes", sub: "anomaly scoring head", color: C.orange },
  ];

  const approaches = [
    { id: 1, name: "Self-Supervised Only", color: C.textDim, rec: false,
      pros: ["No labels needed", "Scalable pretraining"], cons: ["Hard to detect specific attacks", "No class labels → anomaly only"] },
    { id: 2, name: "Semi-Supervised Only", color: C.amber, rec: false,
      pros: ["Label-efficient", "Attack vs normal classifier"], cons: ["No pretraining → weak representations", "Starts from scratch"] },
    { id: 3, name: "Self-sup → Semi-sup", color: C.cyan, rec: true,
      pros: ["Strong representations", "Label efficiency", "Zero-day detection", "Encoder reusable for new tasks"], cons: ["Two-stage training pipeline", "GPM memory overhead"] },
  ];

  return (
    <div>
      {/* Stats grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))", gap: 12, marginBottom: 28 }}>
        {stats.map((s, i) => (
          <div key={i} style={{
            border: `1px solid ${s.color}33`, borderTop: `3px solid ${s.color}`,
            borderRadius: 8, padding: "14px 16px", background: C.bg1,
          }}>
            <div style={{ fontSize: 22, fontWeight: 800, color: s.color, marginBottom: 2 }}>{s.value}</div>
            <div style={{ fontSize: 12, fontWeight: 600, color: C.text, marginBottom: 2 }}>{s.label}</div>
            <div style={{ fontSize: 11, color: C.textDim }}>{s.sub}</div>
          </div>
        ))}
      </div>

      {/* Approach comparison */}
      <SectionTitle color={C.cyan}>Learning approach comparison</SectionTitle>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: 12, marginBottom: 28 }}>
        {approaches.map(a => (
          <div key={a.id} style={{
            border: `1px solid ${a.rec ? C.cyan + "88" : C.border}`,
            borderRadius: 8, padding: 16, background: a.rec ? C.cyanBg : C.bg1,
            position: "relative",
          }}>
            {a.rec && (
              <div style={{
                position: "absolute", top: -10, right: 12,
                background: C.cyan, color: C.bg, fontSize: 10, fontWeight: 800,
                padding: "2px 10px", borderRadius: 4, letterSpacing: "0.08em",
              }}>
                ★ RECOMMENDED
              </div>
            )}
            <div style={{ fontSize: 11, color: C.textDim, marginBottom: 4 }}>Approach {a.id}</div>
            <div style={{ fontSize: 14, fontWeight: 700, color: a.color, marginBottom: 12 }}>{a.name}</div>
            <div>
              {a.pros.map((p, i) => (
                <div key={i} style={{ fontSize: 11, color: C.textMd, display: "flex", gap: 6, marginBottom: 4 }}>
                  <span style={{ color: C.green }}>✓</span>{p}
                </div>
              ))}
              {a.cons.map((c, i) => (
                <div key={i} style={{ fontSize: 11, color: C.textMd, display: "flex", gap: 6, marginBottom: 4 }}>
                  <span style={{ color: C.red }}>⚠</span>{c}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Core insight */}
      <div style={{
        border: `1px solid ${C.teal}44`, borderLeft: `4px solid ${C.teal}`,
        borderRadius: 8, padding: 16, background: C.tealDim,
      }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: C.teal, marginBottom: 8 }}>Core Design Insight</div>
        <div style={{ fontSize: 13, color: C.textMd, lineHeight: 1.8 }}>
          By separating the <strong style={{ color: C.cyan }}>representation learning</strong> (Stage 1, self-supervised) from
          the <strong style={{ color: C.green }}>task-specific classification</strong> (Stage 2, semi-supervised), the system achieves:
          (1) <em>label efficiency</em> — only 100–500 samples per attack type,
          (2) <em>continual learning</em> — GPM prevents catastrophic forgetting when new attack categories arrive,
          (3) <em>zero-day detection</em> — the anomaly head operates in embedding space with no labels required, and
          (4) <em>reusability</em> — the frozen encoder can be reused for any new classifier head without retraining.
        </div>
      </div>
    </div>
  );
}

// ── Root Component ─────────────────────────────────────────────────────────────
export default function IDSArchDashboard() {
  const [tab, setTab] = useState("Overview");

  const renderTab = () => {
    switch (tab) {
      case "Overview":      return <OverviewTab />;
      case "Architecture":  return <ArchitectureViz />;
      case "Modules":       return <ModulesTab />;
      case "Workflow":      return <WorkflowTab />;
      case "Code":          return <CodeTab />;
      case "Folder":        return <FolderTab />;
      case "Limitations":   return <LimitationsTab />;
      default:              return null;
    }
  };

  return (
    <div style={{
      background: C.bg, minHeight: "100vh", fontFamily: "'Inter', 'Segoe UI', sans-serif",
      color: C.text, padding: "0 0 40px 0",
    }}>

      {/* ── Header ──────────────────────────────────────────────────────────── */}
      <div style={{
        background: C.bg1, borderBottom: `1px solid ${C.border}`,
        padding: "24px 32px 0",
      }}>
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", flexWrap: "wrap", gap: 12, marginBottom: 16 }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
              <span style={{
                fontSize: 11, fontWeight: 800, letterSpacing: "0.12em",
                color: C.bg, background: C.cyan, padding: "3px 10px", borderRadius: 4,
              }}>IDS SYSTEM</span>
              <span style={{ fontSize: 11, color: C.textDim }}>v2.0 · Proposed Architecture</span>
            </div>
            <h1 style={{ fontSize: 26, fontWeight: 800, color: C.white, margin: "0 0 6px", lineHeight: 1.2 }}>
              Self-Supervised → Semi-Supervised
            </h1>
            <h2 style={{ fontSize: 16, fontWeight: 400, color: C.textMd, margin: "0 0 12px" }}>
              Intrusion Detection System with Continual Learning via Gradient Projection Memory
            </h2>
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              {["SimCLR", "BERT-flow", "FixMatch", "MixMatch", "GPM", "TensorFlow", "FastAPI", "Kafka"].map(t => (
                <Badge key={t} label={t} color={C.cyan} />
              ))}
            </div>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 6, alignItems: "flex-end" }}>
            <Badge label="★ RECOMMENDED APPROACH" color={C.teal} />
            <Badge label="Zero-Day Detection" color={C.orange} />
            <Badge label="Continual Learning" color={C.purple} />
          </div>
        </div>

        {/* ── Summary bar ─────────────────────────────────────────────────── */}
        <div style={{
          display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
          gap: 0, borderTop: `1px solid ${C.border}`,
        }}>
          {[
            { label: "Architecture", value: "2-Stage Hybrid SSL + Semi-sup", color: C.cyan },
            { label: "Forgetting Prevention", value: "GPM Null-space Projection", color: C.amber },
            { label: "Semi-sup Method", value: "FixMatch / MixMatch", color: C.green },
            { label: "Anomaly Detection", value: "Embedding-space Zero-day", color: C.orange },
          ].map((s, i) => (
            <div key={i} style={{
              padding: "12px 16px",
              borderRight: i < 3 ? `1px solid ${C.border}` : "none",
            }}>
              <div style={{ fontSize: 10, color: C.textDim, letterSpacing: "0.1em", marginBottom: 3, textTransform: "uppercase" }}>{s.label}</div>
              <div style={{ fontSize: 12, fontWeight: 600, color: s.color }}>{s.value}</div>
            </div>
          ))}
        </div>

        {/* ── Tab bar ─────────────────────────────────────────────────────── */}
        <div style={{ display: "flex", gap: 0, marginTop: 16, borderTop: `1px solid ${C.border}` }}>
          {TABS.map(t => (
            <button
              key={t}
              onClick={() => setTab(t)}
              style={{
                padding: "12px 18px", background: "transparent", border: "none",
                borderBottom: `2px solid ${tab === t ? C.cyan : "transparent"}`,
                color: tab === t ? C.cyan : C.textDim, fontSize: 13, fontWeight: 600,
                cursor: "pointer", transition: "all 0.15s", letterSpacing: "0.02em",
              }}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      {/* ── Body ────────────────────────────────────────────────────────────── */}
      <div style={{ padding: "28px 32px" }}>
        {renderTab()}
      </div>

      {/* ── Footer ──────────────────────────────────────────────────────────── */}
      <div style={{
        borderTop: `1px solid ${C.border}`, padding: "16px 32px",
        display: "flex", justifyContent: "space-between", alignItems: "center",
        flexWrap: "wrap", gap: 8,
      }}>
        <div style={{ fontSize: 11, color: C.textDim }}>
          IDS Architecture Explorer · Self-Supervised + Semi-Supervised + GPM Continual Learning
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <Badge label="Stage 1: SSL Pretraining" color={C.cyan} />
          <Badge label="Stage 2: Semi-sup Fine-tuning" color={C.green} />
          <Badge label="GPM Anti-forgetting" color={C.amber} />
        </div>
      </div>
    </div>
  );
}
