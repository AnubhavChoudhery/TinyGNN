#!/usr/bin/env python3
# ============================================================================
#  TinyGNN — Phase 7: Train GCN / GraphSAGE / GAT on Cora & export weights
#
#  Usage (from WSL):
#    source /home/anubhav/tinygnn_venv/bin/activate
#    cd /mnt/c/Users/anubh/OneDrive/Desktop/TinyGNN
#    python3 scripts/train_cora.py
#
#  Outputs (in weights/):
#    cora_graph.bin   — Cora graph data (features, labels, masks, edges)
#    gcn_cora.bin     — GCN model weights + test accuracy
#    sage_cora.bin    — GraphSAGE model weights + test accuracy
#    gat_cora.bin     — GAT model weights + test accuracy
#
#  Binary weight format:
#    Magic  "TGNN"  (4 bytes)
#    Version        uint32_le = 1
#    TestAccuracy   float32_le
#    NumTensors     uint32_le
#    ── per tensor ──
#    NameLen  uint32_le
#    Name     char[NameLen]
#    Rows     uint32_le
#    Cols     uint32_le
#    Data     float32_le[ Rows × Cols ]   (row-major)
#
#  Binary graph format:
#    num_nodes    uint32_le
#    num_features uint32_le
#    num_classes  uint32_le
#    num_edges    uint32_le   (directed pairs)
#    features     float32_le[ N × F ]
#    labels       int32_le[ N ]
#    train_mask   uint8[ N ]
#    val_mask     uint8[ N ]
#    test_mask    uint8[ N ]
#    edge_src     int32_le[ E ]
#    edge_dst     int32_le[ E ]
# ============================================================================

import os
import struct
import warnings

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

# ═════════════════════════════════════════════════════════════════════════════
#  Configuration
# ═════════════════════════════════════════════════════════════════════════════
SEED = 42
HIDDEN_GCN  = 64
HIDDEN_SAGE = 64
GAT_HEADS   = 8       # number of attention heads in layer 1
GAT_HEAD_DIM = 8      # output features per head in layer 1
DROPOUT_GCN  = 0.5
DROPOUT_SAGE = 0.5
DROPOUT_GAT  = 0.6
LR_GCN  = 0.01
LR_SAGE = 0.01
LR_GAT  = 0.005
WD      = 5e-4
EPOCHS  = 200

# ═════════════════════════════════════════════════════════════════════════════
#  Paths
# ═════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
WEIGHTS_DIR = os.path.join(PROJECT_DIR, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
#  Load Cora
# ═════════════════════════════════════════════════════════════════════════════
torch.manual_seed(SEED)
np.random.seed(SEED)

dataset = Planetoid(root="/tmp/pyg_data", name="Cora")
data = dataset[0]

N     = data.num_nodes          # 2708
F_IN  = dataset.num_features    # 1433
C     = dataset.num_classes     # 7

print(f"Cora: {N} nodes, {data.num_edges} edges, {F_IN} features, {C} classes")
print(f"  Train: {data.train_mask.sum().item()}, "
      f"Val: {data.val_mask.sum().item()}, "
      f"Test: {data.test_mask.sum().item()}")

# ═════════════════════════════════════════════════════════════════════════════
#  Export Cora graph data (binary)
# ═════════════════════════════════════════════════════════════════════════════
def export_graph_data():
    path = os.path.join(WEIGHTS_DIR, "cora_graph.bin")
    ei = data.edge_index.numpy()
    num_edges = ei.shape[1]

    with open(path, "wb") as f:
        f.write(struct.pack("<IIII", N, F_IN, C, num_edges))
        f.write(data.x.numpy().astype(np.float32).tobytes())          # features
        f.write(data.y.numpy().astype(np.int32).tobytes())            # labels
        f.write(data.train_mask.numpy().astype(np.uint8).tobytes())   # masks
        f.write(data.val_mask.numpy().astype(np.uint8).tobytes())
        f.write(data.test_mask.numpy().astype(np.uint8).tobytes())
        f.write(ei[0].astype(np.int32).tobytes())                     # edge src
        f.write(ei[1].astype(np.int32).tobytes())                     # edge dst

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"\nExported graph → {os.path.basename(path)} ({size_mb:.1f} MB)")
    print(f"  {num_edges} directed edges, {N}×{F_IN} features, {C} classes")

export_graph_data()

# ═════════════════════════════════════════════════════════════════════════════
#  Binary weight writer
# ═════════════════════════════════════════════════════════════════════════════
def write_weights(path, test_acc, tensors):
    """Write named tensors + test accuracy to TGNN binary file."""
    with open(path, "wb") as f:
        f.write(b"TGNN")
        f.write(struct.pack("<I", 1))               # version
        f.write(struct.pack("<f", test_acc))         # test accuracy
        f.write(struct.pack("<I", len(tensors)))     # num tensors

        for name, t in tensors:
            arr = t.detach().cpu().numpy().astype(np.float32)
            if arr.ndim == 1:
                rows, cols = 1, arr.shape[0]
                arr = arr.reshape(1, -1)
            else:
                rows, cols = arr.shape

            name_b = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_b)))
            f.write(name_b)
            f.write(struct.pack("<II", rows, cols))
            f.write(arr.tobytes())

    print(f"  Exported {len(tensors)} tensors → {os.path.basename(path)}")

# ═════════════════════════════════════════════════════════════════════════════
#  Training helper
# ═════════════════════════════════════════════════════════════════════════════
def train_model(model, lr, epochs, label):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WD)
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                pred = out.argmax(dim=1)
                val_ok = (pred[data.val_mask] == data.y[data.val_mask]).sum().item()
                val_acc = val_ok / data.val_mask.sum().item()
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if epoch % 50 == 0 or epoch == epochs:
                print(f"  Epoch {epoch:3d}: loss={loss.item():.4f}  val_acc={val_acc:.4f}")

    # Restore best and evaluate on test
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        test_ok = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
        test_acc = test_ok / data.test_mask.sum().item()

    print(f"  {label} Test Accuracy: {test_acc:.4f} ({test_ok}/{data.test_mask.sum().item()})")
    return test_acc


# ═════════════════════════════════════════════════════════════════════════════
#  GAT parameter extraction (handles PyG version differences)
# ═════════════════════════════════════════════════════════════════════════════
def gat_linear_weight(conv):
    """Get the linear projection weight from a GATConv, shape (H*F, F_in)."""
    sd = dict(conv.named_parameters())
    for attr in ["lin_src.weight", "lin_l.weight", "lin.weight"]:
        parts = attr.split(".")
        obj = conv
        try:
            for p in parts:
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise RuntimeError(f"Cannot find GAT linear weight in: {list(sd.keys())}")


def gat_attention(conv):
    """Return (att_src, att_dst) each of shape (1, heads, out_channels)."""
    if hasattr(conv, "att_src") and hasattr(conv, "att_dst"):
        return conv.att_src, conv.att_dst
    if hasattr(conv, "att"):
        att = conv.att  # (1, heads, 2*C)
        C = att.shape[-1] // 2
        return att[:, :, :C], att[:, :, C:]
    raise RuntimeError("Cannot find GAT attention parameters")


# ═════════════════════════════════════════════════════════════════════════════
#  1) GCN
# ═════════════════════════════════════════════════════════════════════════════
class GCNNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(F_IN, HIDDEN_GCN)
        self.conv2 = GCNConv(HIDDEN_GCN, C)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=DROPOUT_GCN, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


print("\n" + "═" * 50 + " GCN " + "═" * 50)
torch.manual_seed(SEED)
gcn = GCNNet()
gcn_acc = train_model(gcn, LR_GCN, EPOCHS, "GCN")

print("  Parameters:")
for k, v in gcn.state_dict().items():
    print(f"    {k}: {tuple(v.shape)}")

# TinyGNN weight = PyG lin.weight^T,  bias reshaped to (1, out)
gcn_tensors = [
    ("layer0.weight", gcn.conv1.lin.weight.T),    # (1433, 64)
    ("layer0.bias",   gcn.conv1.bias),             # (64,) → 1×64
    ("layer1.weight", gcn.conv2.lin.weight.T),     # (64, 7)
    ("layer1.bias",   gcn.conv2.bias),             # (7,)  → 1×7
]
write_weights(os.path.join(WEIGHTS_DIR, "gcn_cora.bin"), gcn_acc, gcn_tensors)


# ═════════════════════════════════════════════════════════════════════════════
#  2) GraphSAGE
# ═════════════════════════════════════════════════════════════════════════════
class SAGENet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(F_IN, HIDDEN_SAGE)
        self.conv2 = SAGEConv(HIDDEN_SAGE, C)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=DROPOUT_SAGE, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


print("\n" + "═" * 50 + " GraphSAGE " + "═" * 50)
torch.manual_seed(SEED)
sage = SAGENet()
sage_acc = train_model(sage, LR_SAGE, EPOCHS, "SAGE")

print("  Parameters:")
for k, v in sage.state_dict().items():
    print(f"    {k}: {tuple(v.shape)}")

# SAGEConv: lin_l = neighbor transform, lin_r = self transform
sage_tensors = [
    ("layer0.weight_neigh", sage.conv1.lin_l.weight.T),  # (1433, 64)
    ("layer0.weight_self",  sage.conv1.lin_r.weight.T),  # (1433, 64)
    ("layer0.bias",         sage.conv1.lin_l.bias),       # (64,) → 1×64
    ("layer1.weight_neigh", sage.conv2.lin_l.weight.T),   # (64, 7)
    ("layer1.weight_self",  sage.conv2.lin_r.weight.T),   # (64, 7)
    ("layer1.bias",         sage.conv2.lin_l.bias),        # (7,)  → 1×7
]
write_weights(os.path.join(WEIGHTS_DIR, "sage_cora.bin"), sage_acc, sage_tensors)


# ═════════════════════════════════════════════════════════════════════════════
#  3) GAT
# ═════════════════════════════════════════════════════════════════════════════
class GATNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Layer 1: 8 heads × 8 features = 64 output (concatenated)
        self.conv1 = GATConv(F_IN, GAT_HEAD_DIM, heads=GAT_HEADS,
                             concat=True, dropout=DROPOUT_GAT)
        # Layer 2: 1 head × 7 features = 7 output
        self.conv2 = GATConv(GAT_HEADS * GAT_HEAD_DIM, C, heads=1,
                             concat=False, dropout=DROPOUT_GAT)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=DROPOUT_GAT, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


print("\n" + "═" * 50 + " GAT " + "═" * 50)
torch.manual_seed(SEED)
gat = GATNet()
gat_acc = train_model(gat, LR_GAT, EPOCHS, "GAT")

print("  Parameters:")
for k, v in gat.state_dict().items():
    print(f"    {k}: {tuple(v.shape)}")

# ─── Export GAT weights (multi-head layer 1, single-head layer 2) ─────────
gat_tensors = []

# Layer 0: 8 heads
conv1 = gat.conv1
w1_full = gat_linear_weight(conv1)   # (H*F_out, F_in) = (64, 1433)
a1_src, a1_dst = gat_attention(conv1) # each (1, 8, 8)
b1 = conv1.bias                       # (64,) — may be None

for h in range(GAT_HEADS):
    # Per-head weight: transpose rows [h*F:(h+1)*F] → (F_in, F_out)
    w_h = w1_full[h * GAT_HEAD_DIM : (h + 1) * GAT_HEAD_DIM, :].T
    al_h = a1_src[0, h, :]                    # (F_out,)
    ar_h = a1_dst[0, h, :]                    # (F_out,)
    gat_tensors.append((f"layer0.head{h}.weight",     w_h))
    gat_tensors.append((f"layer0.head{h}.attn_left",  al_h))
    gat_tensors.append((f"layer0.head{h}.attn_right", ar_h))
    if b1 is not None:
        b_h = b1[h * GAT_HEAD_DIM : (h + 1) * GAT_HEAD_DIM]
        gat_tensors.append((f"layer0.head{h}.bias", b_h))

# Layer 1: single head
conv2 = gat.conv2
w2 = gat_linear_weight(conv2).T       # (H*C, F_in).T → (F_in, C) = (64, 7)
a2_src, a2_dst = gat_attention(conv2)  # each (1, 1, 7)
b2 = conv2.bias                        # (7,)

gat_tensors.append(("layer1.head0.weight",     w2))
gat_tensors.append(("layer1.head0.attn_left",  a2_src[0, 0, :]))
gat_tensors.append(("layer1.head0.attn_right", a2_dst[0, 0, :]))
if b2 is not None:
    gat_tensors.append(("layer1.head0.bias", b2))

write_weights(os.path.join(WEIGHTS_DIR, "gat_cora.bin"), gat_acc, gat_tensors)


# ═════════════════════════════════════════════════════════════════════════════
#  Summary
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print(f"  GCN  Test Accuracy: {gcn_acc:.4f}")
print(f"  SAGE Test Accuracy: {sage_acc:.4f}")
print(f"  GAT  Test Accuracy: {gat_acc:.4f}")
print(f"  All weights exported to {WEIGHTS_DIR}/")
print("═" * 60)
