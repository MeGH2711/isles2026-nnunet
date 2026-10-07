"""
Animated Slice Viewer for ISLES26 .b2nd Brain MRI Data
=======================================================
MODE 1 – Single orientation, 3 panels (Original | Mask | Overlay):
    python animate_slices.py --case r035s001 --axis 0   → axial
    python animate_slices.py --case r035s001 --axis 1   → coronal
    python animate_slices.py --case r035s001 --axis 2   → sagittal
    python animate_slices.py --case r035s001 --out .gif → save as GIF

MODE 2 – All 3 orientations side-by-side (MRI + Overlay):
    python animate_slices.py --case r035s001 --all_views

Output is always saved to:
    visualizations/<case_id>/<case_id>_axial.mp4          (mode 1)
    visualizations/<case_id>/<case_id>_all_views.mp4      (mode 2)
"""

import argparse
import numpy as np
import blosc2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path


# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(
    "data/nnunet_preprocessed/Dataset001_ISLES26/nnUNetPlans_3d_fullres"
)
LESION_COLOUR = (1.0, 0.2, 0.2)
AXIS_NAMES    = {0: "Axial", 1: "Coronal", 2: "Sagittal"}


# ── Helpers ────────────────────────────────────────────────────────────────────
def find_case(data_dir: Path, case_name: str | None):
    if case_name:
        img = data_dir / f"{case_name}.b2nd"
        seg = data_dir / f"{case_name}_seg.b2nd"
        if not img.exists():
            raise FileNotFoundError(f"Image not found: {img}")
        if not seg.exists():
            raise FileNotFoundError(f"Segmentation not found: {seg}")
        return img, seg
    candidates = sorted(p for p in data_dir.glob("*.b2nd") if "_seg" not in p.name)
    if not candidates:
        raise FileNotFoundError(f"No .b2nd files found in {data_dir}")
    for img in candidates:
        seg = data_dir / img.name.replace(".b2nd", "_seg.b2nd")
        if seg.exists():
            return img, seg
    raise FileNotFoundError("No case with a matching _seg.b2nd file was found.")


def load_b2nd(path: Path) -> np.ndarray:
    return blosc2.open(str(path))[:]


def normalise(vol: np.ndarray) -> np.ndarray:
    p2, p98 = np.percentile(vol, 2), np.percentile(vol, 98)
    vol = np.clip(vol, p2, p98)
    rng = p98 - p2
    return (vol - p2) / rng if rng > 0 else np.zeros_like(vol)


def make_lesion_cmap():
    return ListedColormap([(0, 0, 0, 1), (*LESION_COLOUR, 1)])

def make_overlay_cmap():
    return ListedColormap([(0, 0, 0, 0), (*LESION_COLOUR, 0.55)])

def get_slice(arr, axis, idx):
    return np.take(arr, idx, axis=axis)

def out_path(case_id: str, suffix: str, tag: str) -> Path:
    """Build and create output path: visualizations/<case_id>/<case_id>_<tag><suffix>"""
    d = Path("visualizations") / case_id
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{case_id}_{tag}{suffix}"

def save_animation(ani, path: Path, interval_ms: int):
    print(f"Saving → {path.resolve()}")
    if path.suffix.lower() == ".gif":
        writer = animation.PillowWriter(fps=max(1, 1000 // interval_ms))
    else:
        writer = animation.FFMpegWriter(fps=max(1, 1000 // interval_ms), bitrate=2400)
    ani.save(str(path), writer=writer, dpi=130)
    plt.close()
    print("Done ✓")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 – Single orientation: Original | Mask | Overlay
# ══════════════════════════════════════════════════════════════════════════════
def animate_single(img_path, seg_path, axis=0, interval_ms=80, suffix=".mp4"):
    vol = normalise(load_b2nd(img_path).astype(np.float32))
    seg = load_b2nd(seg_path)

    if vol.ndim == 4 and vol.shape[0] in (1, 2, 4): vol = vol[0]
    if seg.ndim == 4 and seg.shape[0] == 1:          seg = seg[0]
    seg_bin = (seg > 0).astype(np.uint8)

    case_id    = img_path.stem
    axis_label = AXIS_NAMES[axis]
    n_slices   = vol.shape[axis]
    print(f"Volume shape : {vol.shape}  →  {n_slices} {axis_label} slices")

    lesion_cmap  = make_lesion_cmap()
    overlay_cmap = make_overlay_cmap()

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor="black",
                              gridspec_kw={"wspace": 0.04})
    fig.subplots_adjust(top=0.88, bottom=0.06, left=0.02, right=0.98)

    for ax, t in zip(axes, ["Original MRI", "Lesion Mask", "MRI + Lesion Overlay"]):
        ax.set_facecolor("black"); ax.axis("off")
        ax.set_title(t, color="white", fontsize=11, pad=5, fontweight="bold")

    sl0  = get_slice(vol,     axis, 0)
    seg0 = get_slice(seg_bin, axis, 0)

    im1  = axes[0].imshow(sl0,  cmap="gray",        vmin=0, vmax=1, interpolation="bilinear")
    im2  = axes[1].imshow(seg0, cmap=lesion_cmap,   vmin=0, vmax=1, interpolation="nearest")
    im3a = axes[2].imshow(sl0,  cmap="gray",        vmin=0, vmax=1, interpolation="bilinear")
    im3b = axes[2].imshow(seg0, cmap=overlay_cmap,  vmin=0, vmax=1, interpolation="nearest")

    patch = mpatches.Patch(color=LESION_COLOUR, label="Lesion")
    axes[2].legend(handles=[patch], loc="lower right", fontsize=8,
                   framealpha=0.4, labelcolor="white", facecolor="black")

    slice_text = fig.text(0.5, 0.01, "", ha="center", va="bottom",
                          color="white", fontsize=10)

    fig.suptitle(f"{case_id}  —  {axis_label}",
                 color="white", fontsize=13, fontweight="bold", y=0.97)

    def update(frame):
        sl     = get_slice(vol,     axis, frame)
        seg_sl = get_slice(seg_bin, axis, frame)
        im1.set_data(sl);  im2.set_data(seg_sl)
        im3a.set_data(sl); im3b.set_data(seg_sl)
        slice_text.set_text(f"{axis_label}  {frame + 1} / {n_slices}")
        return im1, im2, im3a, im3b, slice_text

    ani  = animation.FuncAnimation(fig, update, frames=n_slices,
                                    interval=interval_ms, blit=True)
    tag  = axis_label.lower()           # "axial" / "coronal" / "sagittal"
    path = out_path(case_id, suffix, tag)
    save_animation(ani, path, interval_ms)


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 – All 3 views side-by-side (MRI + Overlay only)
# ══════════════════════════════════════════════════════════════════════════════
def animate_all_views(img_path, seg_path, interval_ms=80, suffix=".mp4"):
    vol = normalise(load_b2nd(img_path).astype(np.float32))
    seg = load_b2nd(seg_path)

    if vol.ndim == 4 and vol.shape[0] in (1, 2, 4): vol = vol[0]
    if seg.ndim == 4 and seg.shape[0] == 1:          seg = seg[0]
    seg_bin = (seg > 0).astype(np.uint8)

    case_id  = img_path.stem
    overlay_cmap = make_overlay_cmap()

    # Each axis can have a different number of slices → pad shorter ones
    n = [vol.shape[a] for a in (0, 1, 2)]        # [n_axial, n_coronal, n_sagittal]
    n_frames = max(n)
    print(f"Volume shape : {vol.shape}  →  frames: {n_frames} (max across views)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor="black",
                              gridspec_kw={"wspace": 0.06})
    fig.subplots_adjust(top=0.88, bottom=0.06, left=0.02, right=0.98)

    view_labels = ["Axial", "Coronal", "Sagittal"]
    for ax, lbl in zip(axes, view_labels):
        ax.set_facecolor("black"); ax.axis("off")
        ax.set_title(lbl, color="white", fontsize=12, pad=5, fontweight="bold")

    def init_panel(ax, axis_idx):
        idx  = min(0, n[axis_idx] - 1)
        sl   = get_slice(vol,     axis_idx, idx)
        seg0 = get_slice(seg_bin, axis_idx, idx)
        bg   = ax.imshow(sl,   cmap="gray",        vmin=0, vmax=1, interpolation="bilinear")
        ov   = ax.imshow(seg0, cmap=overlay_cmap,  vmin=0, vmax=1, interpolation="nearest")
        return bg, ov

    bg0, ov0 = init_panel(axes[0], 0)
    bg1, ov1 = init_panel(axes[1], 1)
    bg2, ov2 = init_panel(axes[2], 2)

    patch = mpatches.Patch(color=LESION_COLOUR, label="Lesion")
    for ax in axes:
        ax.legend(handles=[patch], loc="lower right", fontsize=7,
                  framealpha=0.4, labelcolor="white", facecolor="black")

    slice_text = fig.text(0.5, 0.01, "", ha="center", va="bottom",
                          color="white", fontsize=10)

    fig.suptitle(f"{case_id}  —  Axial | Coronal | Sagittal",
                 color="white", fontsize=13, fontweight="bold", y=0.97)

    def update(frame):
        # Map global frame → per-axis index (clamp to that axis's slice count)
        for (bg, ov, ax_idx) in [(bg0, ov0, 0), (bg1, ov1, 1), (bg2, ov2, 2)]:
            idx    = min(frame, n[ax_idx] - 1)
            sl     = get_slice(vol,     ax_idx, idx)
            seg_sl = get_slice(seg_bin, ax_idx, idx)
            bg.set_data(sl)
            ov.set_data(seg_sl)

        slice_text.set_text(
            f"Axial {min(frame+1, n[0])}/{n[0]}   "
            f"Coronal {min(frame+1, n[1])}/{n[1]}   "
            f"Sagittal {min(frame+1, n[2])}/{n[2]}"
        )
        return bg0, ov0, bg1, ov1, bg2, ov2, slice_text

    ani  = animation.FuncAnimation(fig, update, frames=n_frames,
                                    interval=interval_ms, blit=True)
    path = out_path(case_id, suffix, "all_views")
    save_animation(ani, path, interval_ms)


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3-panel animated slice viewer for ISLES26 .b2nd data"
    )
    parser.add_argument("--data_dir",  default=str(DATA_DIR))
    parser.add_argument("--case",      default=None,
                        help="Case stem e.g. r035s001 (default: first with seg)")
    parser.add_argument("--axis",      type=int, default=0, choices=[0, 1, 2],
                        help="0=Axial 1=Coronal 2=Sagittal  [MODE 1 only]")
    parser.add_argument("--all_views", action="store_true",
                        help="MODE 2: all 3 orientations side-by-side with overlay")
    parser.add_argument("--interval",  type=int, default=80,
                        help="Frame interval ms (default 80 → ~12 fps)")
    parser.add_argument("--out",       default=".mp4",
                        help="Extension only: .mp4 (needs ffmpeg) or .gif")
    args = parser.parse_args()

    # Normalise --out to just the extension suffix, defaulting to .mp4
    raw = args.out.strip()
    if raw in (".mp4", ".gif"):
        suffix = raw
    elif raw.lower().endswith(".gif"):
        suffix = ".gif"
    else:
        suffix = ".mp4"

    img_path, seg_path = find_case(Path(args.data_dir), args.case)
    print(f"Case          : {img_path.stem}")

    if args.all_views:
        animate_all_views(img_path, seg_path,
                          interval_ms=args.interval, suffix=suffix)
    else:
        animate_single(img_path, seg_path,
                       axis=args.axis, interval_ms=args.interval, suffix=suffix)