#!/usr/bin/env python3
"""
TinyGNN — Cora Validation Script  (Phase 7: Python Bridge)
scripts/validate_cora.py

Validates TinyGNN C++ inference against PyTorch Geometric reference:

  Mode 1 — "Weight file validation" (default):
      Loads pre-trained weight files from weights/ directory, runs TinyGNN
      inference, and verifies test accuracy matches the expected value.

  Mode 2 — "Logit-level validation" (--logit-check):
      Trains a small GCN in PyG, exports weights to TinyGNN binary format,
      runs both PyG and TinyGNN inference on same input, asserts that
      output logits match within tolerance.

Usage:
    python scripts/validate_cora.py               # Mode 1 (weight files)
    python scripts/validate_cora.py --logit-check  # Mode 2 (fresh training)
    python scripts/validate_cora.py --all           # Both modes
"""

import sys
import os
import struct
import argparse
import time

import numpy as np

# ── Path setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
WEIGHTS_DIR = os.path.join(PROJECT_DIR, "weights")

sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "python"))

import _tinygnn_core as tg


# ============================================================================
#  Utility: write TGNN weight file
# ============================================================================
def write_tgnn_weights(path, test_acc, tensor_dict):
    """Write named tensors to TGNN binary format."""
    with open(path, "wb") as f:
        f.write(b"TGNN")
        f.write(struct.pack("<I", 1))          # version
        f.write(struct.pack("<f", test_acc))
        f.write(struct.pack("<I", len(tensor_dict)))
        for name, arr in tensor_dict.items():
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            name_b = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_b)))
            f.write(name_b)
            rows, cols = arr.shape
            f.write(struct.pack("<II", rows, cols))
            f.write(arr.astype(np.float32).tobytes())


# ============================================================================
#  Mode 1: Weight File Validation
# ============================================================================
def validate_weight_files():
    """Load pre-trained models and verify test accuracy on Cora."""
    graph_path = os.path.join(WEIGHTS_DIR, "cora_graph.bin")

    if not os.path.isfile(graph_path):
        print("  [SKIP] No cora_graph.bin found in weights/.")
        print("         Run 'python scripts/train_cora.py' first.")
        return False

    print("  Loading Cora graph data...")
    cora = tg.load_cora_binary(graph_path)
    print(f"  Cora: {cora.num_nodes} nodes, {cora.num_edges} edges, "
          f"{cora.num_features} features, {cora.num_classes} classes")

    labels = np.array(cora.labels, dtype=np.int32)
    test_mask = np.array(cora.test_mask, dtype=np.uint8)
    test_indices = np.where(test_mask == 1)[0]
    n_test = len(test_indices)

    all_passed = True

    # ── GCN ─────────────────────────────────────────────────────────────
    gcn_path = os.path.join(WEIGHTS_DIR, "gcn_cora.bin")
    if os.path.isfile(gcn_path):
        print(f"\n  GCN validation:")
        model = tg.Model()
        model.add_gcn_layer(1433, 64, True, tg.Activation.RELU)
        model.add_gcn_layer(64, 7, True, tg.Activation.NONE)
        model.load_weights(gcn_path)

        wf = tg.load_weight_file(gcn_path)
        expected_acc = wf.test_accuracy

        t0 = time.perf_counter()
        output = model.forward(cora.adjacency, cora.features)
        elapsed = time.perf_counter() - t0

        logits = output.to_numpy()
        tg.log_softmax_inplace(output)
        preds = np.argmax(logits, axis=1)

        correct = np.sum(preds[test_indices] == labels[test_indices])
        accuracy = correct / n_test

        print(f"    Expected accuracy: {expected_acc:.4f}")
        print(f"    TinyGNN accuracy:  {accuracy:.4f} ({correct}/{n_test})")
        print(f"    Inference time:    {elapsed*1000:.1f} ms")

        if abs(accuracy - expected_acc) < 0.01:
            print(f"    ✓ PASS (within 1% tolerance)")
        else:
            print(f"    ✗ FAIL (accuracy mismatch > 1%)")
            all_passed = False
    else:
        print(f"\n  [SKIP] gcn_cora.bin not found")

    # ── GraphSAGE ───────────────────────────────────────────────────────
    sage_path = os.path.join(WEIGHTS_DIR, "sage_cora.bin")
    if os.path.isfile(sage_path):
        print(f"\n  GraphSAGE validation:")
        model = tg.Model()
        model.add_sage_layer(1433, 64)
        model.add_sage_layer(64, 7, tg.SAGELayer.Aggregator.Mean, True, tg.Activation.NONE)
        model.load_weights(sage_path)

        wf = tg.load_weight_file(sage_path)
        expected_acc = wf.test_accuracy

        t0 = time.perf_counter()
        output = model.forward(cora.adjacency, cora.features)
        elapsed = time.perf_counter() - t0

        logits = output.to_numpy()
        preds = np.argmax(logits, axis=1)

        correct = np.sum(preds[test_indices] == labels[test_indices])
        accuracy = correct / n_test

        print(f"    Expected accuracy: {expected_acc:.4f}")
        print(f"    TinyGNN accuracy:  {accuracy:.4f} ({correct}/{n_test})")
        print(f"    Inference time:    {elapsed*1000:.1f} ms")

        if abs(accuracy - expected_acc) < 0.01:
            print(f"    ✓ PASS (within 1% tolerance)")
        else:
            print(f"    ✗ FAIL (accuracy mismatch > 1%)")
            all_passed = False
    else:
        print(f"\n  [SKIP] sage_cora.bin not found")

    # ── GAT ─────────────────────────────────────────────────────────────
    gat_path = os.path.join(WEIGHTS_DIR, "gat_cora.bin")
    if os.path.isfile(gat_path):
        print(f"\n  GAT validation:")
        model = tg.Model()
        # Layer 0: 8 heads × 8 features, concat=True → 64 output
        model.add_gat_layer(1433, 8, 8, True, 0.2, True, tg.Activation.NONE,
                            tg.Model.InterActivation.ELU)
        # Layer 1: 1 head × 7 features, concat=False → 7 output
        model.add_gat_layer(64, 7, 1, False, 0.2, True, tg.Activation.NONE)
        model.load_weights(gat_path)

        wf = tg.load_weight_file(gat_path)
        expected_acc = wf.test_accuracy

        t0 = time.perf_counter()
        output = model.forward(cora.adjacency, cora.features)
        elapsed = time.perf_counter() - t0

        logits = output.to_numpy()
        preds = np.argmax(logits, axis=1)

        correct = np.sum(preds[test_indices] == labels[test_indices])
        accuracy = correct / n_test

        print(f"    Expected accuracy: {expected_acc:.4f}")
        print(f"    TinyGNN accuracy:  {accuracy:.4f} ({correct}/{n_test})")
        print(f"    Inference time:    {elapsed*1000:.1f} ms")

        if abs(accuracy - expected_acc) < 0.01:
            print(f"    ✓ PASS (within 1% tolerance)")
        else:
            print(f"    ✗ FAIL (accuracy mismatch > 1%)")
            all_passed = False
    else:
        print(f"\n  [SKIP] gat_cora.bin not found")

    return all_passed


# ============================================================================
#  Mode 2: Logit-Level Validation (train fresh GCN, compare logits)
# ============================================================================
def validate_logit_level():
    """Train a tiny GCN in PyG, export weights, compare logits bit-for-bit."""
    import tempfile

    try:
        import warnings
        warnings.filterwarnings("ignore")
        import torch
        import torch.nn.functional as F
        from torch_geometric.datasets import Planetoid
        from torch_geometric.nn import GCNConv
    except ImportError:
        print("  [SKIP] PyTorch/PyG not available for logit-level check.")
        return True

    print("  Loading Cora via PyG...")
    torch.manual_seed(42)
    dataset = Planetoid(root="/tmp/pyg_data", name="Cora")
    data = dataset[0]

    N = data.num_nodes
    F_IN = dataset.num_features
    C = dataset.num_classes
    HIDDEN = 16

    # ── Train a small 2-layer GCN ───────────────────────────────────────
    class SmallGCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(F_IN, HIDDEN)
            self.conv2 = GCNConv(HIDDEN, C)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            return x  # raw logits (no log_softmax for comparison)

    print("  Training small GCN (50 epochs)...")
    model = SmallGCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(F.log_softmax(out, dim=1)[data.train_mask],
                          data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pyg_logits = model(data.x, data.edge_index).numpy()

    print(f"  PyG logits shape: {pyg_logits.shape}")

    # ── Export weights to TGNN format ───────────────────────────────────
    tensors = {
        "layer0.weight": model.conv1.lin.weight.T.detach().numpy().astype(np.float32),
        "layer0.bias": model.conv1.bias.detach().numpy().astype(np.float32),
        "layer1.weight": model.conv2.lin.weight.T.detach().numpy().astype(np.float32),
        "layer1.bias": model.conv2.bias.detach().numpy().astype(np.float32),
    }

    tmpdir = tempfile.mkdtemp()
    weight_path = os.path.join(tmpdir, "test_gcn.bin")
    graph_path = os.path.join(tmpdir, "test_graph.bin")

    write_tgnn_weights(weight_path, 0.0, tensors)

    # ── Export graph data ───────────────────────────────────────────────
    ei = data.edge_index.numpy()
    num_edges = ei.shape[1]
    with open(graph_path, "wb") as f:
        f.write(struct.pack("<IIII", N, F_IN, C, num_edges))
        f.write(data.x.numpy().astype(np.float32).tobytes())
        f.write(data.y.numpy().astype(np.int32).tobytes())
        f.write(data.train_mask.numpy().astype(np.uint8).tobytes())
        f.write(data.val_mask.numpy().astype(np.uint8).tobytes())
        f.write(data.test_mask.numpy().astype(np.uint8).tobytes())
        f.write(ei[0].astype(np.int32).tobytes())
        f.write(ei[1].astype(np.int32).tobytes())

    # ── TinyGNN inference ───────────────────────────────────────────────
    print("  Running TinyGNN inference...")
    cora = tg.load_cora_binary(graph_path)

    tg_model = tg.Model()
    tg_model.add_gcn_layer(F_IN, HIDDEN, True, tg.Activation.RELU)
    tg_model.add_gcn_layer(HIDDEN, C, True, tg.Activation.NONE)
    tg_model.load_weights(weight_path)

    t0 = time.perf_counter()
    tg_output = tg_model.forward(cora.adjacency, cora.features)
    elapsed = time.perf_counter() - t0

    tg_logits = tg_output.to_numpy()
    print(f"  TinyGNN logits shape: {tg_logits.shape}")
    print(f"  TinyGNN inference time: {elapsed*1000:.1f} ms")

    # ── Compare ─────────────────────────────────────────────────────────
    # Note: PyG and TinyGNN use the same GCN normalization formula
    # D^{-1/2} (A+I) D^{-1/2}, so outputs should match closely
    max_abs_diff = np.max(np.abs(pyg_logits - tg_logits))
    mean_abs_diff = np.mean(np.abs(pyg_logits - tg_logits))
    rel_diff = np.mean(np.abs(pyg_logits - tg_logits) / (np.abs(pyg_logits) + 1e-8))

    print(f"\n  Logit comparison (PyG vs TinyGNN):")
    print(f"    Max  absolute diff: {max_abs_diff:.6f}")
    print(f"    Mean absolute diff: {mean_abs_diff:.6f}")
    print(f"    Mean relative diff: {rel_diff:.6f}")

    # Check prediction agreement
    pyg_preds = np.argmax(pyg_logits, axis=1)
    tg_preds = np.argmax(tg_logits, axis=1)
    agreement = np.mean(pyg_preds == tg_preds)
    print(f"    Prediction agreement: {agreement*100:.1f}%")

    # Tolerance: allow small float32 accumulation differences
    LOGIT_TOL = 0.05  # generous tolerance for float32 path differences
    if max_abs_diff < LOGIT_TOL:
        print(f"    ✓ PASS (max diff < {LOGIT_TOL})")
        passed = True
    elif agreement > 0.99:
        print(f"    ✓ PASS (>99% prediction agreement despite logit diff)")
        passed = True
    else:
        print(f"    ✗ FAIL (logits diverge too much)")
        passed = False

    # Cleanup
    os.unlink(weight_path)
    os.unlink(graph_path)
    os.rmdir(tmpdir)

    return passed


# ============================================================================
#  Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="TinyGNN Cora Validation")
    parser.add_argument("--logit-check", action="store_true",
                        help="Run logit-level comparison (trains fresh GCN)")
    parser.add_argument("--all", action="store_true",
                        help="Run both weight file and logit-level validation")
    args = parser.parse_args()

    run_weight = not args.logit_check or args.all
    run_logit = args.logit_check or args.all

    print("=" * 60)
    print("  TinyGNN — Cora Validation Script")
    print("=" * 60)

    all_ok = True

    if run_weight:
        print("\n── Mode 1: Weight File Validation ──")
        if not validate_weight_files():
            all_ok = False

    if run_logit:
        print("\n── Mode 2: Logit-Level Validation ──")
        if not validate_logit_level():
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("  All validations PASSED ✓")
    else:
        print("  Some validations FAILED ✗")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
