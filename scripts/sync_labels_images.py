#!/usr/bin/env python3
import argparse
from pathlib import Path
import shutil, sys, zipfile, time

IMG_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

def index_images_recursive(images_dir: Path):
    idx = {}
    for p in images_dir.rglob('*'):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            idx[p.stem] = p
    return idx

def find_image(images_index, stem):
    if stem in images_index:
        return images_index[stem]
    for name, p in images_index.items():
        if name.startswith(stem) or stem.startswith(name):
            return p
    return None

def collect_labels(labels_dir):
    mapping = {}
    for d in sorted(labels_dir.iterdir()):
        if d.is_dir():
            mapping[d.name] = sorted(d.glob("*.txt"))
    return mapping

def ensure_dirs(out_dir, subsets):
    for kind in ('images','labels'):
        for s in subsets + ['unlabeled']:
            (out_dir/'data'/kind/s).mkdir(parents=True, exist_ok=True)

def zip_folder(src, dst):
    with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in src.rglob('*'):
            z.write(p, p.relative_to(src))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--yolo", default="yolo")
    p.add_argument("--out", default="out")
    p.add_argument("--move", action="store_true")
    p.add_argument("--create-empty-labels", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--backup", action="store_true")
    a = p.parse_args()

    y = Path(a.yolo).resolve()
    o = Path(a.out).resolve()
    imgs = y/"images"
    labels = y/"labels"

    if a.backup:
        t = time.strftime("%Y%m%d-%H%M%S")
        zip_folder(y, o/f"backup_{t}.zip")

    label_map = collect_labels(labels)
    subsets = sorted(label_map.keys())
    ensure_dirs(o, subsets)

    images_index = index_images_recursive(imgs)
    stats = {'paired':0,'miss':0,'unlabeled':0,'copied':0,'moved':0}

    for subset, txts in label_map.items():
        di = o/'data'/'images'/subset
        dl = o/'data'/'labels'/subset
        for lbl in txts:
            stem = lbl.stem
            img = find_image(images_index, stem)
            if img is None:
                stats['miss'] += 1
                if not a.dry_run:
                    shutil.copy2(lbl, dl/lbl.name)
                continue
            dst_img = di/img.name
            dst_lbl = dl/lbl.name
            if not a.dry_run:
                if a.move:
                    shutil.move(img, dst_img)
                    shutil.move(lbl, dst_lbl)
                    stats['moved'] += 1
                else:
                    shutil.copy2(img, dst_img)
                    shutil.copy2(lbl, dst_lbl)
                    stats['copied'] += 1
            stats['paired'] += 1
            images_index.pop(img.stem, None)

    ui = o/'data'/'images'/'unlabeled'
    ul = o/'data'/'labels'/'unlabeled'

    for stem, img in images_index.items():
        dst = ui/img.name
        if not a.dry_run:
            if a.move:
                shutil.move(img, dst)
                stats['moved'] += 1
            else:
                shutil.copy2(img, dst)
                stats['copied'] += 1
            stats['unlabeled'] += 1
            if a.create_empty_labels:
                (ul/(stem+".txt")).write_text("")

    print("paired", stats['paired'])
    print("labels_without_image", stats['miss'])
    print("images_without_label", stats['unlabeled'])
    print("copied", stats['copied'])
    print("moved", stats['moved'])
    print("done")

if __name__ == "__main__":
    main()
