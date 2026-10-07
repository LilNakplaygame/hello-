import os
import sys
import glob
import random
import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

CLASS_NAMES = ["Obstacle", "Right", "No_Right", "Straight", "Left"]
CLASS_COLORS = [
    (255, 0, 0),     # Đỏ: Vật cản (Obstacle)
    (0, 140, 255),   # Xanh dương: Biển Rẽ Phải (Right)
    (255, 200, 0),   # Vàng cam: Biển Cấm Rẽ (No_Right)
    (0, 255, 255),   # Xanh lơ (Cyan): Biển Đi Thẳng (Straight)
    (255, 0, 255)    # Tím hồng: Biển Rẽ Trái (Left)
]

def parse_args():
    parser = argparse.ArgumentParser(description="TriLiteNet Random Test Inference")
    parser.add_argument("--weights", type=str, default="", help="Path to .pth checkpoint file")
    parser.add_argument("--config", type=str, default="small", choices=["tiny", "small", "base"], help="Model variant")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of random test images to sample")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold for NMS")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--output", type=str, default="output_random_test.png", help="Path to save output figure")
    parser.add_argument("--no-show", action="store_true", help="Do not call plt.show() (headless mode)")
    return parser.parse_args()

def find_checkpoint():
    """Auto-detect latest trained checkpoint."""
    candidates = [
        "/kaggle/working/TriLiteNet/runs/final_state.pth",
        "/kaggle/working/TriLiteNet/runs/checkpoint.pth",
        "./runs/final_state.pth",
        "./runs/checkpoint.pth",
        "../runs/final_state.pth",
        "runs/final_state.pth",
        "runs/checkpoint.pth"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # Search recursively in runs/
    matches = glob.glob("runs/**/*.pth", recursive=True) + glob.glob("/kaggle/working/**/runs/**/*.pth", recursive=True)
    if matches:
        matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return matches[0]
    return None

def find_val_images():
    """Auto-detect validation images."""
    search_dirs = [
        "/kaggle/working/dataset_ucr/images/val/*.jpg",
        "/kaggle/input/**/images/val/*.jpg",
        "../dataset_ucr/images/val/*.jpg",
        "./dataset_ucr/images/val/*.jpg",
        "dataset_ucr/images/val/*.jpg"
    ]
    for pattern in search_dirs:
        imgs = glob.glob(pattern, recursive=True)
        if imgs:
            return imgs
    # Fallback to train images if val is missing
    fallback_dirs = [
        "/kaggle/working/dataset_ucr/images/train/*.jpg",
        "/kaggle/input/**/images/train/*.jpg",
        "../dataset_ucr/images/train/*.jpg",
        "./dataset_ucr/images/train/*.jpg"
    ]
    for pattern in fallback_dirs:
        imgs = glob.glob(pattern, recursive=True)
        if imgs:
            return imgs
    return []

def run_random_test(weights=None, config="small", num_samples=5, conf_thres=0.25, iou_thres=0.45, output="output_random_test.png", show=True):
    import torch
    import torchvision.transforms as transforms
    from lib.models import get_net
    from lib.config import cfg
    from lib.core.general import non_max_suppression, scale_coords

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Running on device: {device}")

    # Set config
    cfg.config = config
    cfg.num_det_class = 5
    model = get_net(cfg).to(device)

    # Resolve checkpoint
    ckpt_path = weights or find_checkpoint()
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"[-] Error: Checkpoint not found at '{ckpt_path}'. Please train the model first or specify --weights.")
        return

    print(f"[*] Loading weights from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    # Support both state_dict directly or checkpoint dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("[+] Model loaded successfully (5 Classes: Obstacle, Right, No_Right, Straight, Left)!")

    # Find and sample images
    val_images = find_val_images()
    if not val_images:
        print("[-] Error: No test images found in dataset. Please run convert_dataset.py first.")
        return

    val_candidates = [img for img in val_images if '1050' not in img]
    if not val_candidates:
        val_candidates = val_images

    k = min(num_samples, len(val_candidates))
    sampled_images = random.sample(val_candidates, k)
    print(f"[+] Randomly sampled {k} images for visualization.")

    inp_w, inp_h = cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    fig = plt.figure(figsize=(16, 4.5 * k))

    for idx, img_path in enumerate(sampled_images):
        file_name = os.path.basename(img_path)
        orig_img = cv2.imread(img_path)
        if orig_img is None:
            continue
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        h0, w0, _ = orig_img.shape

        img_resized = cv2.resize(orig_img, (inp_w, inp_h))
        img_tensor = transform(img_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            det_out, da_seg_out, ll_seg_out = model(img_tensor)

        # 1. Drivable road mask
        pred_mask = torch.argmax(da_seg_out, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        pred_mask = cv2.resize(pred_mask, (w0, h0), interpolation=cv2.INTER_NEAREST)

        overlay = orig_img.copy()
        overlay[pred_mask == 1] = [0, 255, 0]  # Green overlay
        blended = cv2.addWeighted(orig_img, 0.7, overlay, 0.3, 0)

        # 2. Object detection bounding boxes
        inf_out = det_out[0]
        det_pred = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres)
        det = det_pred[0]

        if len(det):
            det[:, :4] = scale_coords((inp_h, inp_w), det[:, :4], (h0, w0)).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                class_id = int(cls)
                class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"ID_{class_id}"
                box_color = CLASS_COLORS[class_id] if class_id < len(CLASS_COLORS) else (255, 255, 255)

                cv2.rectangle(blended, (x1, y1), (x2, y2), box_color, 2)
                label_text = f"{class_name}: {conf:.2f}"
                cv2.putText(blended, label_text, (x1, max(y1 - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

        # Plot 3 columns
        plt.subplot(k, 3, idx * 3 + 1)
        plt.title(f"1. Ảnh gốc\n({file_name[:22]}...)" if idx == 0 else f"{file_name[:22]}...")
        plt.imshow(orig_img)
        plt.axis("off")

        plt.subplot(k, 3, idx * 3 + 2)
        plt.title("2. Mask mặt đường" if idx == 0 else "Mask dự đoán")
        plt.imshow(pred_mask, cmap='gray')
        plt.axis("off")

        plt.subplot(k, 3, idx * 3 + 3)
        plt.title("3. Đa nhiệm (5 Classes)" if idx == 0 else "Đường + Bounding Box")
        plt.imshow(blended)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"[+] Output visualization saved to: {output}")

    if show:
        try:
            plt.show()
        except Exception:
            pass

def main():
    # Clean old cache modules
    for mod in list(sys.modules.keys()):
        if 'lib.models' in mod or 'lib.config' in mod:
            del sys.modules[mod]

    args = parse_args()
    run_random_test(
        weights=args.weights or None,
        config=args.config,
        num_samples=args.num_samples,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        output=args.output,
        show=not args.no_show
    )

if __name__ == "__main__":
    main()
