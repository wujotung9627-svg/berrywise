# ============================================================
#  BerryWise — 戶外影像強化模組 v2
#  enhance_outdoor.py
#  針對室外草莓園五大干擾：強光過曝、陰影、色偏、模糊、複雜背景
# ============================================================

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import io

# ────────────────────────────────────────────────────────────
#  1. 自適應直方圖均衡化 (CLAHE)
#  解決：強光過曝 / 陰影不均 → 局部對比自動拉伸
# ────────────────────────────────────────────────────────────
def apply_clahe(pil_img: Image.Image, clip_limit=2.5, tile_size=8) -> Image.Image:
    img_np = np.array(pil_img)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return Image.fromarray(result)

# ────────────────────────────────────────────────────────────
#  2. 白平衡校正（灰世界假設）
#  解決：戶外色偏（黃昏偏橘、陰天偏藍）
# ────────────────────────────────────────────────────────────
def white_balance(pil_img: Image.Image) -> Image.Image:
    img_np = np.array(pil_img).astype(np.float32)
    r_mean = np.mean(img_np[:,:,0])
    g_mean = np.mean(img_np[:,:,1])
    b_mean = np.mean(img_np[:,:,2])
    gray = (r_mean + g_mean + b_mean) / 3
    img_np[:,:,0] = np.clip(img_np[:,:,0] * (gray / r_mean), 0, 255)
    img_np[:,:,1] = np.clip(img_np[:,:,1] * (gray / g_mean), 0, 255)
    img_np[:,:,2] = np.clip(img_np[:,:,2] * (gray / b_mean), 0, 255)
    return Image.fromarray(img_np.astype(np.uint8))

# ────────────────────────────────────────────────────────────
#  3. 去模糊銳化（Unsharp Mask）
#  解決：手持拍攝模糊、快門速度不足
# ────────────────────────────────────────────────────────────
def unsharp_mask(pil_img: Image.Image, radius=2, amount=1.4, threshold=3) -> Image.Image:
    img_np = np.array(pil_img).astype(np.float32)
    blurred = cv2.GaussianBlur(img_np, (0, 0), radius)
    sharpened = img_np + amount * (img_np - blurred)
    # threshold：只銳化差異明顯的區域（避免放大雜訊）
    low_contrast = np.abs(img_np - blurred) < threshold
    sharpened[low_contrast] = img_np[low_contrast]
    return Image.fromarray(np.clip(sharpened, 0, 255).astype(np.uint8))

# ────────────────────────────────────────────────────────────
#  4. 過曝區域恢復（Highlight Recovery）
#  解決：強光直射導致葉面白化、失去紋理
# ────────────────────────────────────────────────────────────
def recover_highlights(pil_img: Image.Image, threshold=235) -> Image.Image:
    img_np = np.array(pil_img).astype(np.float32)
    # 找出過曝區域（所有通道都很亮）
    overexposed = np.all(img_np > threshold, axis=2)
    # 在過曝區域降低亮度並微增色彩飽和度
    img_np[overexposed] = img_np[overexposed] * 0.82
    return Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))

# ────────────────────────────────────────────────────────────
#  5. 病害色彩增強（針對紅棕色病斑）
#  解決：戶外綠色背景壓制病斑視覺對比
# ────────────────────────────────────────────────────────────
def enhance_lesion_colors(pil_img: Image.Image) -> Image.Image:
    img_np = np.array(pil_img).astype(np.float32)
    hsv = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    # 增強紅棕色範圍（病斑色調：H=0~30 或 H=160~180）
    lesion_mask = ((h < 30) | (h > 160)) & (s > 40)
    s[lesion_mask] = np.clip(s[lesion_mask] * 1.35, 0, 255)
    v[lesion_mask] = np.clip(v[lesion_mask] * 0.92, 0, 255)

    # 稍微壓制過飽和的綠色背景（H=40~90）
    green_mask = (h > 40) & (h < 90) & (s > 80)
    s[green_mask] = np.clip(s[green_mask] * 0.85, 0, 255)

    hsv_merged = cv2.merge([h, s, v]).astype(np.uint8)
    result = cv2.cvtColor(hsv_merged, cv2.COLOR_HSV2RGB)
    return Image.fromarray(result)

# ────────────────────────────────────────────────────────────
#  6. 智慧裁切（中央權重，去掉邊緣雜亂背景）
#  解決：手機拍攝常有邊緣雜草、土壤干擾
# ────────────────────────────────────────────────────────────
def smart_crop(pil_img: Image.Image, margin=0.08) -> Image.Image:
    w, h = pil_img.size
    left   = int(w * margin)
    top    = int(h * margin)
    right  = int(w * (1 - margin))
    bottom = int(h * (1 - margin))
    cropped = pil_img.crop((left, top, right, bottom))
    return cropped.resize((w, h), Image.LANCZOS)

# ────────────────────────────────────────────────────────────
#  整合管線：根據影像特性自動選擇處理強度
# ────────────────────────────────────────────────────────────
def analyze_image_condition(pil_img: Image.Image) -> dict:
    """分析影像狀況，回傳建議的處理策略"""
    img_np = np.array(pil_img).astype(np.float32)
    brightness = np.mean(img_np)
    contrast   = np.std(img_np)
    # 過曝比例
    overexp_ratio = np.mean(np.all(img_np > 235, axis=2))
    # 模糊程度（Laplacian 變異數）
    gray = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    return {
        "brightness":     brightness,
        "contrast":       contrast,
        "overexp_ratio":  overexp_ratio,
        "blur_score":     blur_score,
        "is_overexposed": brightness > 180 or overexp_ratio > 0.08,
        "is_dark":        brightness < 80,
        "is_blurry":      blur_score < 80,
        "is_low_contrast":contrast < 35,
    }

def enhance_outdoor_v2(
    pil_img: Image.Image,
    use_clahe: bool = True,
    use_wb: bool = True,
    use_sharpen: bool = True,
    use_highlight: bool = True,
    use_lesion: bool = True,
    use_crop: bool = False,  # 預設關閉，避免裁到目標
    verbose: bool = False,
) -> tuple:
    """
    戶外影像強化完整管線。
    回傳 (enhanced_image, report_dict)
    """
    cond = analyze_image_condition(pil_img)
    report = {"條件分析": cond, "套用步驟": []}
    img = pil_img.copy()

    # 步驟順序很重要：先白平衡 → 再均衡化 → 再銳化 → 最後色彩
    if use_wb:
        img = white_balance(img)
        report["套用步驟"].append("✅ 白平衡校正")

    if use_highlight and cond["is_overexposed"]:
        img = recover_highlights(img)
        report["套用步驟"].append("✅ 過曝恢復（偵測到強光）")

    if use_clahe:
        # 依對比度動態調整 clip_limit
        clip = 3.5 if cond["is_dark"] else (2.0 if cond["is_overexposed"] else 2.5)
        img = apply_clahe(img, clip_limit=clip)
        report["套用步驟"].append(f"✅ CLAHE 自適應均衡（clip={clip}）")

    if use_lesion:
        img = enhance_lesion_colors(img)
        report["套用步驟"].append("✅ 病斑色彩增強")

    if use_sharpen and cond["is_blurry"]:
        img = unsharp_mask(img, amount=1.6)
        report["套用步驟"].append("✅ 去模糊銳化（偵測到模糊）")
    elif use_sharpen:
        img = unsharp_mask(img, amount=1.1)
        report["套用步驟"].append("✅ 輕度銳化")

    if use_crop:
        img = smart_crop(img)
        report["套用步驟"].append("✅ 智慧裁切（去除邊緣雜訊）")

    if verbose:
        print(f"\n📊 影像條件分析：")
        print(f"   亮度：{cond['brightness']:.1f}  {'⚠️ 過亮' if cond['is_overexposed'] else ('⚠️ 過暗' if cond['is_dark'] else '✅ 正常')}")
        print(f"   對比：{cond['contrast']:.1f}  {'⚠️ 低對比' if cond['is_low_contrast'] else '✅ 正常'}")
        print(f"   清晰：{cond['blur_score']:.1f}  {'⚠️ 模糊' if cond['is_blurry'] else '✅ 清晰'}")
        print(f"   過曝：{cond['overexp_ratio']:.1%}")
        print(f"\n🔧 套用步驟：")
        for s in report["套用步驟"]:
            print(f"   {s}")

    return img, report


# ────────────────────────────────────────────────────────────
#  快速測試：比較原圖 vs 強化後的差異
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法：python3 enhance_outdoor.py 你的草莓照片.jpg")
        print("將產生 enhanced_output.jpg 供對比")
        sys.exit(0)

    src = sys.argv[1]
    print(f"📁 載入：{src}")
    original = Image.open(src).convert("RGB")

    enhanced, report = enhance_outdoor_v2(original, verbose=True)

    # 並排輸出對比圖
    w, h = original.size
    compare = Image.new("RGB", (w * 2, h))
    compare.paste(original, (0, 0))
    compare.paste(enhanced, (w, 0))
    out_path = src.rsplit(".", 1)[0] + "_enhanced_compare.jpg"
    compare.save(out_path, quality=95)
    print(f"\n💾 對比圖已存至：{out_path}")
    print("左半 = 原圖，右半 = 強化後")

