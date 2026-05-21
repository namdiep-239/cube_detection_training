"""
Generate a sample-detections grid image from the test split.

Usage:
  python scripts/gen_sample_detections.py                  # 12 random images, 3 cols
  python scripts/gen_sample_detections.py --n 18 --cols 3
  python scripts/gen_sample_detections.py --n 6  --cols 3 --seed 42
  python scripts/gen_sample_detections.py --images ur_2cube3cyl-24.jpg ur_1cyl1cube-36.jpg
  python scripts/gen_sample_detections.py --all   # every test image, one big grid
"""

import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# -----------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Generate sample detection grid")
parser.add_argument("--model",     type=Path,
                    default=Path("./models/cylinder_detector_int8_edgetpu.tflite"),
                    help="Path to TFLite model (EdgeTPU or CPU)")
parser.add_argument("--edgetpu",   action="store_true",
                    help="Run on Google Coral EdgeTPU (requires pycoral)")
parser.add_argument("--test-dir",  type=Path,
                    default=Path("./outputs/voc_split/test"))
parser.add_argument("--output",    type=Path,
                    default=Path("./outputs/sample_detections_report.png"))
parser.add_argument("--n",         type=int, default=12,
                    help="Number of images to show (ignored if --images or --all)")
parser.add_argument("--cols",      type=int, default=3,
                    help="Number of columns in the grid")
parser.add_argument("--threshold", type=float, default=0.3)
parser.add_argument("--seed",      type=int, default=None,
                    help="Random seed for reproducible sample selection")
parser.add_argument("--images",    nargs="+", default=None,
                    help="Specific image filenames to show (e.g. ur_2cube3cyl-24.jpg)")
parser.add_argument("--all",       action="store_true",
                    help="Show every image in the test split")
parser.add_argument("--fig-width",   type=float, default=18.0,
                    help="Figure width in inches (height scales automatically)")
parser.add_argument("--show-cubes",  action="store_true",
                    help="Also draw cube predictions (orange) with label and score")
args = parser.parse_args()

args.output.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------
# Load model — Coral EdgeTPU or CPU TFLite
# -----------------------------------------------------------------------

import tensorflow as tf

if not args.model.exists():
    raise SystemExit(f"Model not found: {args.model}")

USE_CORAL = False
if args.edgetpu:
    try:
        from pycoral.utils.edgetpu import make_interpreter
        interpreter = make_interpreter(str(args.model))
        interpreter.allocate_tensors()
        USE_CORAL = True
        print(f"[Coral] Loaded EdgeTPU model: {args.model.name}")
    except ImportError:
        print("[Coral] pycoral not found — falling back to CPU TFLite")
    except Exception as e:
        print(f"[Coral] Failed to load EdgeTPU model ({e}) — falling back to CPU TFLite")

if not USE_CORAL:
    interpreter = tf.lite.Interpreter(model_path=str(args.model))
    interpreter.allocate_tensors()
    print(f"[CPU]   Loaded TFLite model: {args.model.name}")
input_detail  = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()
INPUT_SIZE = input_detail['shape'][1]


def _suffix(d):
    try:
        return int(d.get("name", "").rsplit(":", 1)[-1])
    except ValueError:
        return 999


def run_inference(img_bgr):
    h, w = img_bgr.shape[:2]
    rgb  = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (INPUT_SIZE, INPUT_SIZE))
    inp  = rgb.astype(np.uint8)[np.newaxis] if input_detail['dtype'] == np.uint8 \
           else (rgb / 255.0).astype(np.float32)[np.newaxis]
    interpreter.set_tensor(input_detail['index'], inp)
    interpreter.invoke()

    s3 = [d for d in output_details if len(d['shape']) == 3 and d['shape'][2] == 4]
    s2 = sorted([d for d in output_details if len(d['shape']) == 2], key=_suffix)
    if not s3 or not s2:
        return [], []

    bd, sd = s3[0], s2[0]
    cd = s2[1] if len(s2) > 1 else None

    rb = interpreter.get_tensor(bd['index'])[0]
    rs = interpreter.get_tensor(sd['index'])[0]
    if sd['dtype'] == np.uint8:
        sc, zp = sd['quantization']; rs = sc * (rs.astype(np.float32) - zp)
    else:
        rs = rs.astype(np.float32)
    if bd['dtype'] == np.uint8:
        sc, zp = bd['quantization']; rb = sc * (rb.astype(np.float32) - zp)
    else:
        rb = rb.astype(np.float32)
    if cd is not None:
        rc = interpreter.get_tensor(cd['index'])[0]
        if cd['dtype'] == np.uint8:
            sc, zp = cd['quantization']
            class_ids = (sc * (rc.astype(np.float32) - zp)).round().astype(int)
        else:
            class_ids = rc.astype(int)
    else:
        class_ids = np.zeros(len(rs), dtype=int)

    cyl_boxes, cyl_scores, cube_boxes, cube_scores = [], [], [], []
    for i, (ymin, xmin, ymax, xmax) in enumerate(rb):
        box = [int(xmin*w), int(ymin*h), int(xmax*w), int(ymax*h)]
        if class_ids[i] == 0:
            cyl_boxes.append(box);  cyl_scores.append(float(rs[i]))
        elif class_ids[i] == 1:
            cube_boxes.append(box); cube_scores.append(float(rs[i]))
    return cyl_boxes, cyl_scores, cube_boxes, cube_scores

# -----------------------------------------------------------------------
# Collect test items
# -----------------------------------------------------------------------

img_dir = args.test_dir / "images"
ann_dir = args.test_dir / "annotations"

if not img_dir.exists():
    raise SystemExit(f"Test split not found at {args.test_dir}\nRun train.py first.")

all_items = []
for xml_path in sorted(ann_dir.glob("*.xml")):
    for ext in (".jpg", ".png"):
        ip = img_dir / (xml_path.stem + ext)
        if ip.exists():
            all_items.append((ip, xml_path))
            break

if not all_items:
    raise SystemExit("No annotated test images found.")

# Select subset
if args.images:
    lookup = {it[0].stem: it for it in all_items}
    items = []
    for name in args.images:
        stem = Path(name).stem
        if stem in lookup:
            items.append(lookup[stem])
        else:
            print(f"  [WARN] {name} not found in test split — skipping")
    if not items:
        raise SystemExit("None of the requested images found in test split.")
elif args.all:
    items = all_items
else:
    if args.seed is not None:
        random.seed(args.seed)
    n = min(args.n, len(all_items))
    items = random.sample(all_items, n)
    items.sort(key=lambda x: x[0].name)

print(f"Generating grid for {len(items)} image(s)  →  {args.output}")

# -----------------------------------------------------------------------
# Draw grid
# -----------------------------------------------------------------------

cols  = args.cols
rows  = (len(items) + cols - 1) // cols
h_per = args.fig_width / cols * 0.75   # approximate aspect ratio
fig, axes = plt.subplots(rows, cols,
                         figsize=(args.fig_width, h_per * rows),
                         squeeze=False)
legend = "(--- green GT: cylinder  |  blue GT: cube  |  red pred: cylinder"
if args.show_cubes:
    legend += "  |  orange pred: cube"
legend += f"  |  threshold={args.threshold})"
fig.suptitle(f"Sample Detections on Test Set  {legend}", fontsize=11)

for idx, ax in enumerate(axes.flatten()):
    if idx >= len(items):
        ax.axis("off")
        continue

    img_path, xml_path = items[idx]
    img_bgr = cv2.imread(str(img_path))
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ax.set_title(img_path.name, fontsize=7)
    ax.axis("off")

    # Ground truth (green dashed) — all objects
    tree = ET.parse(str(xml_path))
    for obj in tree.getroot().findall("object"):
        bb   = obj.find("bndbox")
        x1   = int(bb.find("xmin").text); y1 = int(bb.find("ymin").text)
        x2   = int(bb.find("xmax").text); y2 = int(bb.find("ymax").text)
        name = obj.find("name").text
        color = "#4CAF50" if name == "cylinder" else "#2196F3"
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=1.5, edgecolor=color, facecolor="none", linestyle="--"))

    # Predictions — cylinder (red), cube (orange, optional)
    cyl_boxes, cyl_scores, cube_boxes, cube_scores = run_inference(img_bgr)
    for (x1, y1, x2, y2), score in zip(cyl_boxes, cyl_scores):
        if score < args.threshold:
            continue
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor="#FF3333", facecolor="none"))
        ax.text(x1, max(0, y1 - 4), f"cylinder {score:.2f}",
                fontsize=7, fontweight="bold", color="white",
                bbox=dict(facecolor="#FF3333", alpha=0.85, pad=1, edgecolor="none"))
    if args.show_cubes:
        for (x1, y1, x2, y2), score in zip(cube_boxes, cube_scores):
            if score < args.threshold:
                continue
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor="#FF9800", facecolor="none"))
            ax.text(x1, max(0, y1 - 4), f"cube {score:.2f}",
                    fontsize=7, fontweight="bold", color="white",
                    bbox=dict(facecolor="#FF9800", alpha=0.85, pad=1, edgecolor="none"))

plt.tight_layout()
fig.savefig(str(args.output), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {args.output}  ({rows}×{cols} grid)")
