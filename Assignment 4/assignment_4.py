"""
assignment_4.py
Extended version:
 - Harris corner detection
 - Feature-based alignment using SIFT, SURF, and ORB (each saved separately)
 - Combine all outputs into a single PDF

Outputs:
    harris.png
    aligned_sift.png, matches_sift.png
    aligned_surf.png, matches_surf.png
    aligned_orb.png, matches_orb.png
    assignment4_output.pdf
"""

import cv2
import numpy as np
from PIL import Image
import os
import sys

# ===============================
# 1. Harris Corner Detection
# ===============================
def harris_edge_save(reference_image_path, out_path='harris.png', blockSize=2, ksize=3, k=0.04, threshold_ratio=0.01):
    img = cv2.imread(reference_image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read {reference_image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, blockSize, ksize, k)
    dst = cv2.dilate(dst, None)
    out = img.copy()
    thresh = threshold_ratio * dst.max()
    out[dst > thresh] = [0, 0, 255]
    cv2.imwrite(out_path, out)
    print(f"Harris result saved to {out_path}")
    return out

# ===============================
# 2. Helper to draw matches
# ===============================
def _draw_matches(img1, kp1, img2, kp2, matches, max_display=50):
    display_matches = matches[:max_display]
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, display_matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img_matches

# ===============================
# 3. Image Alignment Function
# ===============================
def align_images(image_to_align_path, reference_image_path,
                 method='sift', max_features=1500, good_match_percent=0.15,
                 out_aligned='aligned.png', out_matches='matches.png'):

    img = cv2.imread(image_to_align_path)
    ref = cv2.imread(reference_image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read {image_to_align_path}")
    if ref is None:
        raise FileNotFoundError(f"Could not read {reference_image_path}")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    method_lower = method.lower()

    # Choose feature detector
    if method_lower == 'sift':
        if hasattr(cv2, 'SIFT_create'):
            detector = cv2.SIFT_create(nfeatures=max_features)
        else:
            detector = cv2.ORB_create(nfeatures=max_features)
            method_lower = 'orb'
    elif method_lower == 'surf':
        try:
            detector = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
        except Exception as e:
            print("SURF not available; falling back to ORB.")
            detector = cv2.ORB_create(nfeatures=max_features)
            method_lower = 'orb'
    else:
        detector = cv2.ORB_create(nfeatures=max_features)

    kp1, des1 = detector.detectAndCompute(ref_gray, None)
    kp2, des2 = detector.detectAndCompute(img_gray, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        raise RuntimeError(f"{method} failed: not enough keypoints/descriptors found.")

    # Match features
    if method_lower in ['sift', 'surf']:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance, reverse=False)

    num_good = max(4, int(len(matches) * good_match_percent))
    good_matches = matches[:num_good]
    print(f"[{method.upper()}] Total matches: {len(matches)}, keeping {num_good} good matches.")

    # Extract coordinates
    points_ref = np.zeros((len(good_matches), 2), dtype=np.float32)
    points_img = np.zeros((len(good_matches), 2), dtype=np.float32)
    for i, m in enumerate(good_matches):
        points_ref[i, :] = kp1[m.queryIdx].pt
        points_img[i, :] = kp2[m.trainIdx].pt

    H, mask = cv2.findHomography(points_img, points_ref, cv2.RANSAC)
    if H is None:
        raise RuntimeError(f"{method} failed: could not compute homography.")

    height, width = ref.shape[:2]
    aligned = cv2.warpPerspective(img, H, (width, height))
    matches_img = _draw_matches(ref, kp1, img, kp2, good_matches)

    cv2.imwrite(out_aligned, aligned)
    cv2.imwrite(out_matches, matches_img)
    print(f"[{method.upper()}] Results saved: {out_aligned}, {out_matches}")
    return aligned, matches_img

# ===============================
# 4. Combine Images into PDF
# ===============================
def make_pdf_from_images(image_paths, out_pdf='assignment4_output.pdf'):
    pil_images = []
    for p in image_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found for PDF creation.")
        im = Image.open(p)
        if im.mode != "RGB":
            im = im.convert("RGB")
        pil_images.append(im)
    first, rest = pil_images[0], pil_images[1:]
    first.save(out_pdf, "PDF", resolution=100.0, save_all=True, append_images=rest)
    print(f"✅ PDF saved to {out_pdf}")
    return out_pdf

# ===============================
# MAIN EXECUTION
# ===============================
if __name__ == ("__main__"):
    ref_img = r"C:\Users\Sanjana\Downloads\reference_img.png"
    align_img = r"C:\Users\Sanjana\Downloads\align_this.jpg"

    # 1️⃣ Harris Corner Detection
    harris_edge_save(ref_img, out_path='harris.png')

    # 2️⃣ Feature Alignment - SIFT, SURF, ORB (separately)
    methods = [
        ('sift', 10, 0.7),
        ('surf', 10, 0.7),
        ('orb', 1500, 0.15)
    ]

    generated_images = ['harris.png']
    for m, f, g in methods:
        try:
            aligned_path = f"aligned_{m}.png"
            matches_path = f"matches_{m}.png"
            align_images(align_img, ref_img, method=m, max_features=f, good_match_percent=g,
                         out_aligned=aligned_path, out_matches=matches_path)
            generated_images.extend([aligned_path, matches_path])
        except Exception as e:
            print(f"{m.upper()} failed: {e}")

    # 3️⃣ Create PDF with all results
    make_pdf_from_images(generated_images, out_pdf='assignment4_output.pdf')