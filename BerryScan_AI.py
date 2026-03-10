# ============================================================
#  BerryWise — 草莓園小助手 v2
#  手機優先 · 純 requests API · Streamlit Cloud 部署就緒
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import datetime
import requests
import numpy as np
try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False
from pathlib import Path

# ────────────────────────────────────────────────────────────
#  設定讀取（Streamlit Secrets 優先，次則預設值）
# ────────────────────────────────────────────────────────────
def get_api_key():
    try:
        return st.secrets["ROBOFLOW_API_KEY"]
    except Exception:
        return "VOhMaw0JTEKqryA0pM3p"

DEFAULT_MODEL_ID  = "-strawberry-disease-hrfcc/2"
ROBOFLOW_DATASET  = "strawberry-disease-hrfcc"   # 上傳到同一個 dataset 的待標記區
ROBOFLOW_WORKSPACE = "jojos-workspace-mudmq"
APP_TITLE         = "草莓園小助手"

# ────────────────────────────────────────────────────────────
#  Roboflow 上傳（用於回饋訓練）
# ────────────────────────────────────────────────────────────
def upload_to_roboflow(pil_image: Image.Image, api_key: str,
                       suggested_label: str = "", note: str = "") -> bool:
    """將圖片上傳到 Roboflow dataset 待標記區。回傳是否成功。"""
    try:
        import base64, io as _io
        buf = _io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=88)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"feedback_{timestamp}.jpg"
        tag       = suggested_label.replace(" ", "-").lower() if suggested_label else "unlabeled"

        url = f"https://api.roboflow.com/dataset/{ROBOFLOW_DATASET}/upload"
        params = {
            "api_key": api_key,
            "name": filename,
            "tag": tag,
            "split": "train",
        }
        resp = requests.post(
            url,
            params=params,
            data=b64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=20,
        )
        return resp.status_code == 200
    except Exception:
        return False

# ────────────────────────────────────────────────────────────
#  iOS PWA Meta Tags
# ────────────────────────────────────────────────────────────
IOS_PWA_META = """
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="草莓園小助手">
<meta name="mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#050505">
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
<link rel="apple-touch-icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E🍓%3C/text%3E%3C/svg%3E">
"""

# ────────────────────────────────────────────────────────────
#  農事建議資料庫
# ────────────────────────────────────────────────────────────
ADVICE_DB = {
    "angular leafspot": {
        "zh_name": "角斑病",
        "en_name": "Angular Leafspot",
        "severity": "⚠️ 中度警示",
        "color": "#E74C3C",
        "visual_check": (
            "【目測確認特徵】\n"
            "  ✓ 病斑受葉脈限制，呈不規則多角形（非圓形）\n"
            "  ✓ 初期水浸狀、半透明，後轉黃褐色至深褐色\n"
            "  ✓ 潮濕時病斑背面可見白色菌膿\n"
            "  ✗ 若病斑為圓形、有同心圓，可能是葉斑病"
        ),
        "confused_with": "葉斑病（Leaf Spot）",
        "advice": (
            "【立即處置】\n"
            "  • 移除並銷毀出現水浸狀、褐色多角形病斑的葉片。\n"
            "  • 禁止從上方澆水，改採滴灌，減少葉面積水。\n\n"
            "【環境管理】\n"
            "  • 加強植株間通風，行距維持 30cm 以上。\n"
            "  • 雨後 24 小時內巡視積水情形。\n\n"
            "【藥劑參考】\n"
            "  • 可使用銅基殺菌劑（如氧化亞銅）進行保護性噴施。"
        ),
    },
    "anthracnose fruit rot": {
        "zh_name": "炭疽病（果實）",
        "en_name": "Anthracnose Fruit Rot",
        "severity": "🚨 高度警示",
        "color": "#C0392B",
        "visual_check": (
            "【目測確認特徵】\n"
            "  ✓ 果面出現圓形、黑褐色、明顯凹陷的病斑\n"
            "  ✓ 病斑邊緣清晰，呈「燒灼感」外觀\n"
            "  ✓ 濕潤時病斑中央可見橘紅色孢子堆\n"
            "  ✗ 若果面為白色粉末狀，應為果實白粉病\n"
            "  ✗ 若黴層灰色蓬鬆，應為灰黴病"
        ),
        "confused_with": "灰黴病（Gray Mold）、果實白粉病（Powdery Mildew Fruit）",
        "advice": (
            "【立即處置】\n"
            "  • 移除出現黑色凹陷圓形病斑的果實，套袋帶出田區銷毀。\n"
            "  • 避免病果與健康果實接觸，防止接觸傳染。\n\n"
            "【環境管理】\n"
            "  • 高溫多濕環境需每日巡檢，雨後立即清查。\n"
            "  • 減少果實受傷，同步進行蟲害防治。\n\n"
            "【藥劑參考】\n"
            "  • 苯醚甲環唑或咪鮮胺類藥劑交替噴施。"
        ),
    },
    "blossom blight": {
        "zh_name": "花凋病 / 花腐病",
        "en_name": "Blossom Blight",
        "severity": "🚨 高度警示",
        "color": "#8E44AD",
        "visual_check": (
            "【目測確認特徵】\n"
            "  ✓ 花瓣出現褐色水浸狀腐爛，快速萎凋\n"
            "  ✓ 花萼及花梗變褐，嚴重時整花枯死\n"
            "  ✓ 潮濕條件下可見灰色黴層（與灰黴病相似）\n"
            "  ✗ 若症狀在花謝後果實上擴展，留意是否轉為灰黴病"
        ),
        "confused_with": "灰黴病（Gray Mold）",
        "advice": (
            "【立即處置】\n"
            "  • 立刻摘除出現褐化、枯萎的花朵與花梗，集中銷毀。\n"
            "  • 避免在開花期噴水，減少花部濕潤時間。\n\n"
            "【環境管理】\n"
            "  • 開花期間特別注意通風，降低棚內濕度至 70% 以下。\n"
            "  • 避免過度密植，確保花叢間有足夠氣流。\n\n"
            "【藥劑參考】\n"
            "  • 可於開花前預防性噴施腐黴利或撲克拉等藥劑。"
        ),
    },
    "gray mold": {
        "zh_name": "灰黴病",
        "en_name": "Gray Mold",
        "severity": "🚨 高度警示",
        "color": "#9B59B6",
        "visual_check": (
            "【目測確認特徵】\n"
            "  ✓ 黴層呈灰褐色、蓬鬆絨毛狀（非白色粉末）\n"
            "  ✓ 輕拍患部可見灰色孢子雲飄散\n"
            "  ✓ 感染部位組織軟腐，有腐爛氣味\n"
            "  ✓ 常從花萼或傷口處開始發病\n"
            "  ✗ 若表面為白色粉末且無軟腐，應為果實白粉病\n"
            "  ✗ 白粉病組織不軟爛，可用手指輕擦確認"
        ),
        "confused_with": "果實白粉病（Powdery Mildew Fruit）",
        "advice": (
            "【立即處置】\n"
            "  • 立刻摘除並套袋帶出所有出現灰色黴層的果實與葉片。\n"
            "  • 重點檢查花萼附近是否有初期霉斑。\n\n"
            "【環境管理】\n"
            "  • 目標相對濕度維持 70% 以下；多雨期可搭設防雨網。\n"
            "  • 及時疏果，避免果實相互擠壓。\n\n"
            "【藥劑參考】\n"
            "  • 輪替使用 SDHI 類或 QoI 類殺菌劑，防止抗藥性。"
        ),
    },
    "leaf spot": {
        "zh_name": "葉斑病（蛇眼病）",
        "en_name": "Leaf Spot",
        "severity": "⚠️ 中度警示",
        "color": "#E67E22",
        "visual_check": (
            "【目測確認特徵】\n"
            "  ✓ 病斑為圓形至橢圓形，有明顯紫紅色邊緣\n"
            "  ✓ 中心灰白色，外圍紫褐色（蛇眼狀）\n"
            "  ✓ 嚴重時多個病斑融合，葉片黃化脫落\n"
            "  ✗ 若病斑多角形且受葉脈限制，應為角斑病\n"
            "  ✗ 若葉背有白粉，應為葉片白粉病"
        ),
        "confused_with": "角斑病（Angular Leafspot）、葉片白粉病（Powdery Mildew Leaf）",
        "advice": (
            "【立即處置】\n"
            "  • 摘除出現蛇眼狀褐色圓形病斑的葉片，集中銷毀。\n"
            "  • 優先清除老葉與貼地葉片，減少病原殘留。\n\n"
            "【環境管理】\n"
            "  • 避免葉面積水，改採滴灌方式。\n"
            "  • 加強田間通風，降低濕度至 75% 以下。\n\n"
            "【藥劑參考】\n"
            "  • 代森錳鋅或百菌清等保護性殺菌劑，每 7～10 天噴一次。"
        ),
    },
    "powdery mildew fruit": {
        "zh_name": "白粉病（果實）",
        "en_name": "Powdery Mildew Fruit",
        "severity": "⚠️ 中度警示",
        "color": "#F39C12",
        "visual_check": (
            "【目測確認特徵】\n"
            "  ✓ 果面出現白色至灰白色粉末狀覆蓋物\n"
            "  ✓ 用手指輕擦可抹去白粉，下方果皮完整\n"
            "  ✓ 果實發育停滯、變硬、失去光澤\n"
            "  ✗ 若黴層灰褐蓬鬆、組織軟腐，應為灰黴病\n"
            "  ✗ 灰黴病有腐爛氣味，白粉病無"
        ),
        "confused_with": "灰黴病（Gray Mold）",
        "advice": (
            "【立即處置】\n"
            "  • 摘除表面出現白色粉末的果實，套袋帶出避免孢子飛散。\n"
            "  • 同步檢查葉片背面是否也有白粉症狀。\n\n"
            "【環境管理】\n"
            "  • 保持通風乾燥，套袋栽培可有效隔離病菌。\n"
            "  • 避免過度施用氮肥，植株過嫩易感染。\n\n"
            "【藥劑參考】\n"
            "  • 微粒硫磺或亞磷酸製劑，避免高溫（32°C 以上）時段施藥。"
        ),
    },
    "powdery mildew leaf": {
        "zh_name": "白粉病（葉片）",
        "en_name": "Powdery Mildew Leaf",
        "severity": "⚠️ 中度警示",
        "color": "#F1C40F",
        "visual_check": (
            "【目測確認特徵】\n"
            "  ✓ 葉背出現白色粉末狀菌絲層\n"
            "  ✓ 葉面對應位置出現紫紅色至褐色斑塊\n"
            "  ✓ 嫩葉受害後向上捲曲\n"
            "  ✗ 若病斑為蛇眼狀圓形、無白粉，應為葉斑病\n"
            "  ✗ 若病斑多角形，應為角斑病"
        ),
        "confused_with": "葉斑病（Leaf Spot）",
        "advice": (
            "【立即處置】\n"
            "  • 摘除葉背出現白色粉末狀菌絲的葉片，避免孢子飛散。\n"
            "  • 特別注意嫩葉與頂葉，白粉病偏好幼嫩組織。\n\n"
            "【環境管理】\n"
            "  • 保持通風乾燥，避免過度施用氮肥。\n"
            "  • 日夜溫差大的季節加強巡視。\n\n"
            "【藥劑參考】\n"
            "  • 硫磺製劑或三唑類殺菌劑交替使用，每 5～7 天一次。"
        ),
    },
}

def get_advice(label: str) -> dict:
    key = label.lower().strip()
    if key in ADVICE_DB:
        return ADVICE_DB[key]
    return {
        "zh_name": label,
        "en_name": label,
        "severity": "❓ 請進一步確認",
        "color": "#3498DB",
        "visual_check": "",
        "confused_with": "",
        "advice": (
            "【通用建議】\n"
            "  • 本系統尚未收錄此標籤，請對照目測特徵確認病害種類。\n"
            "  • 建議拍攝多角度照片，記錄發病部位、面積與時間。\n"
            "  • 若症狀快速擴散，請儘速隔離受染植株。"
        ),
    }

# ────────────────────────────────────────────────────────────
#  模型未涵蓋病害參考（用於偵測為 0 時提示）
# ────────────────────────────────────────────────────────────
UNCOVERED_LEAF = [
    {
        "zh_name": "枯葉病（葉緣焦枯）",
        "en_name": "Leaf Scorch",
        "visual": "葉緣或葉尖出現紅褐色至暗褐色乾枯，病健交界不清晰，嚴重時整葉枯焦。與葉斑病差異：無圓形病斑、無紫紅色邊緣。",
        "action": "移除嚴重枯葉，調整水分供應（乾旱或鹽分過高易誘發），可噴施銅基殺菌劑預防。",
    },
    {
        "zh_name": "蛇眼病",
        "en_name": "Common Leaf Spot (Mycosphaerella)",
        "visual": "圓形病斑，中心灰白或淡褐，外圍有明顯紫紅色環，外觀像蛇眼。與葉斑病外觀相似但邊緣更清晰。",
        "action": "摘除病葉，噴施代森錳鋅或百菌清，加強通風降濕。",
    },
    {
        "zh_name": "炭疽葉枯病",
        "en_name": "Anthracnose Crown Rot / Leaf Blight",
        "visual": "葉片出現不規則深褐色至黑褐色壞死斑，葉柄基部常見黑化，嚴重時全株萎凋。濕潤時可見橘紅色孢子堆。",
        "action": "立即移除病株，避免連作，噴施苯醚甲環唑或咪鮮胺，重點保護植株基部。",
    },
    {
        "zh_name": "輪斑病",
        "en_name": "Leaf Blight (Phomopsis)",
        "visual": "葉片出現大型 V 字形或不規則褐色壞死，常從葉尖或葉緣向內擴展。與枯葉病差異：病斑內常有小黑點（分生孢子器）。",
        "action": "摘除病葉，改善通風，噴施百菌清或腐黴利預防。",
    },
]

UNCOVERED_FRUIT = [
    {
        "zh_name": "軟腐病",
        "en_name": "Soft Rot (Rhizopus)",
        "visual": "果實快速水爛、表面出現白色棉絮狀黴層，最終出現黑色孢子囊，腐爛快速且有酸臭味。",
        "action": "立即移除並銷毀，保持採後低溫（4°C），避免果實受傷，加強通風。",
    },
    {
        "zh_name": "疫病果腐",
        "en_name": "Phytophthora Fruit Rot",
        "visual": "果實出現深褐色、皮革狀硬化病斑，不像炭疽病的凹陷。通常從近地面果實先發病，濕潤時有白色黴層。",
        "action": "架高果實避免接觸土壤，使用甲霜靈或烯醯嗎啉，加強排水。",
    },
    {
        "zh_name": "日燒果（非病害）",
        "en_name": "Sunscald",
        "visual": "果實向陽面出現白化、褪色或淡黃色硬化區域，無黴層，無孢子。非侵染性，屬生理障礙。",
        "action": "非病害，不需噴藥。加裝遮陰網，避免強烈直射光照射果實。",
    },
]

# ────────────────────────────────────────────────────────────
#  影像前處理 v2（五層戶外強化管線）
# ────────────────────────────────────────────────────────────
def _white_balance(img_np):
    f = img_np.astype(np.float32)
    gray = np.mean(f)
    for c in range(3):
        m = np.mean(f[:, :, c])
        if m > 0:
            f[:, :, c] = np.clip(f[:, :, c] * (gray / m), 0, 255)
    return f.astype(np.uint8)

def _clahe(img_np, clip=2.5):
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    lab = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def _recover_highlights(img_np, thr=232):
    f = img_np.astype(np.float32)
    mask = np.all(f > thr, axis=2)
    f[mask] *= 0.80
    return np.clip(f, 0, 255).astype(np.uint8)

def _enhance_lesions(img_np):
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    lesion = ((h < 30) | (h > 160)) & (s > 40)
    s[lesion] = np.clip(s[lesion] * 1.35, 0, 255)
    v[lesion] = np.clip(v[lesion] * 0.92, 0, 255)
    green = (h > 40) & (h < 90) & (s > 80)
    s[green] = np.clip(s[green] * 0.85, 0, 255)
    return cv2.cvtColor(cv2.merge([h, s, v]).astype(np.uint8), cv2.COLOR_HSV2RGB)

def _unsharp(img_np, amount=1.2):
    f = img_np.astype(np.float32)
    blur = cv2.GaussianBlur(f, (0, 0), 2.0)
    sharp = f + amount * (f - blur)
    low = np.abs(f - blur) < 4
    sharp[low] = f[low]
    return np.clip(sharp, 0, 255).astype(np.uint8)

def enhance_outdoor_image(pil_image: Image.Image) -> Image.Image:
    if not CV2_AVAILABLE:
        return pil_image  # cv2 不可用時直接回傳原圖
    img = np.array(pil_image.convert("RGB"))
    brightness = np.mean(img)
    blur_score = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()

    # 1. 白平衡（輕度）
    img = _white_balance(img)

    # 2. 過曝恢復（只在非常過曝時才做）
    if np.mean(np.all(img.astype(np.float32) > 240, axis=2)) > 0.10:
        img = _recover_highlights(img, thr=240)

    # 3. CLAHE（輕度，避免過度改變訓練資料分布）
    clip = 1.5 if brightness < 80 else (1.2 if brightness > 175 else 1.5)
    img = _clahe(img, clip=clip)

    # 4. 跳過 _enhance_lesions（會嚴重破壞模型色彩分布）

    # 5. 輕度銳化（只在真的模糊時做）
    if blur_score < 50:
        img = _unsharp(img, amount=0.6)

    return Image.fromarray(img)

# ────────────────────────────────────────────────────────────
#  Roboflow API（inference_sdk）
# ────────────────────────────────────────────────────────────
def run_inference(pil_image: Image.Image, api_key: str, model_id: str, confidence: float):
    try:
        from inference_sdk import InferenceHTTPClient
        import tempfile, os

        # 存成暫存檔再傳入
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            pil_image.save(tmp_path, format="JPEG", quality=90)

        client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=api_key,
        )
        result = client.infer(tmp_path, model_id=model_id)
        os.unlink(tmp_path)
        # 全部回傳，由 UI 層決定顯示策略
        return result

    except Exception as e:
        st.error(f"❌ 診斷失敗：{e}")
        return None

# ────────────────────────────────────────────────────────────
#  繪製偵測框
# ────────────────────────────────────────────────────────────
def draw_detections(pil_image: Image.Image, predictions: list, confidence_threshold: float) -> Image.Image:
    draw = ImageDraw.Draw(pil_image)
    w, h = pil_image.size
    COLOR_MAP = {
        "angular leafspot":      "#E74C3C",
        "anthracnose fruit rot": "#C0392B",
        "blossom blight":        "#8E44AD",
        "gray mold":             "#9B59B6",
        "leaf spot":             "#E67E22",
        "powdery mildew fruit":  "#F39C12",
        "powdery mildew leaf":   "#F1C40F",
    }
    line_w   = max(3, int(min(w, h) * 0.004))
    font_sz  = max(18, int(min(w, h) * 0.028))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_sz)
    except Exception:
        font = ImageFont.load_default()

    for pred in predictions:
        conf  = pred.get("confidence", 0)
        label = pred.get("class", "unknown")
        cx, cy, bw, bh = pred.get("x",0), pred.get("y",0), pred.get("width",0), pred.get("height",0)
        x1, y1 = int(cx - bw/2), int(cy - bh/2)
        x2, y2 = int(cx + bw/2), int(cy + bh/2)
        color = COLOR_MAP.get(label.lower(), "#3498DB")

        if conf >= confidence_threshold:
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)
            text = f"{label} {conf:.0%}"
            try:
                bbox = draw.textbbox((x1, y1 - font_sz - 4), text, font=font)
                draw.rectangle(bbox, fill=color)
                draw.text((x1, y1 - font_sz - 4), text, fill="white", font=font)
            except Exception:
                draw.text((x1, max(0, y1 - 20)), text, fill=color, font=font)
        else:
            dash = max(8, line_w * 3)
            for x in range(x1, x2, dash * 2):
                draw.line([(x, y1), (min(x+dash, x2), y1)], fill="#888", width=line_w)
                draw.line([(x, y2), (min(x+dash, x2), y2)], fill="#888", width=line_w)
            for y in range(y1, y2, dash * 2):
                draw.line([(x1, y), (x1, min(y+dash, y2))], fill="#888", width=line_w)
                draw.line([(x2, y), (x2, min(y+dash, y2))], fill="#888", width=line_w)
    return pil_image

# ────────────────────────────────────────────────────────────
#  診斷報告
# ────────────────────────────────────────────────────────────
def generate_report(predictions: list, confidence_threshold: float, mode: str, manual_override: str = None) -> str:
    now   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    valid = [p for p in predictions if p.get("confidence", 0) >= confidence_threshold]
    below = [p for p in predictions if p.get("confidence", 0) < confidence_threshold]
    HIGH_CONF = 0.75

    lines = [
        "══════════════════════════════",
        f"  🍓 草莓園小助手 診斷報告",
        f"  模式：{mode}　{now}",
        "══════════════════════════════",
        "",
    ]

    # 手動確認優先
    if manual_override and manual_override in ADVICE_DB:
        minfo = ADVICE_DB[manual_override]
        ai_label = get_advice(valid[0].get("class","unknown"))["zh_name"] if valid else "無"
        ai_conf  = f"{valid[0].get('confidence',0):.1%}" if valid else "—"
        lines += [
            "【診斷來源：農民手動確認】",
            f"  AI 建議：{ai_label}（{ai_conf}）→ 農民判斷修正為以下結果",
            "",
            f"─── 確認病害 ────────────────",
            f"  病害名稱：{minfo['zh_name']} ({minfo['en_name']})",
            f"  警示等級：{minfo['severity']}",
            "",
        ]
        confused = minfo.get("confused_with","")
        if confused:
            lines += [f"  ⚡ 請再確認是否排除：{confused}", ""]
        visual = minfo.get("visual_check","")
        if visual:
            lines += [visual, ""]
        lines += [minfo["advice"], ""]

    elif not predictions:
        lines += [
            "未偵測到病害目標。",
            "",
            "【建議】",
            "  · 拉近距離至 20～30cm，讓病斑佔畫面 2/3",
            "  · 選陰天或遮陰環境拍攝，避免強烈反光",
            "  · 確認畫面對焦清晰後重新拍攝",
        ]
    elif not valid:
        best_low = max((p.get("confidence", 0) for p in below), default=0)
        lines += [
            f"偵測到 {len(below)} 個候選，確信度均未達門檻（最高 {best_low:.1%}）。",
            "",
            "【拍攝改善建議】",
            "  · 拉近距離，讓單一病斑填滿畫面",
            "  · 側光或散射光拍攝，避免正面強光",
            "  · 可嘗試葉片背面拍攝",
        ]
    else:
        valid_sorted = sorted(valid, key=lambda p: p.get("confidence", 0), reverse=True)
        lines += [f"偵測到 {len(valid_sorted)} 個有效目標（依確信度排列）", ""]

        for i, pred in enumerate(valid_sorted, 1):
            label = pred.get("class", "unknown")
            conf  = pred.get("confidence", 0)
            info  = get_advice(label)
            conf_tag = "● 高可信" if conf >= HIGH_CONF else "○ 待確認"
            lines += [
                f"─── #{i} {info['zh_name']} ({info['en_name']})",
                f"  確信度：{conf:.1%}  {conf_tag}",
                f"  警示　：{info['severity']}",
                "",
            ]
            confused = info.get("confused_with", "")
            if confused:
                lines += [f"  ⚡ 易混淆病害：{confused}", ""]
            visual = info.get("visual_check", "")
            if visual:
                lines += [visual, ""]
            lines += [info["advice"], ""]
            if conf < HIGH_CONF:
                lines += [
                    "  【拍攝改善建議】",
                    "  · 拉近距離至 20cm，讓病斑佔畫面 2/3 以上",
                    "  · 選散射光或陰天拍攝，避免反光與陰影",
                    "",
                ]

        if below:
            below_sorted = sorted(below, key=lambda p: p.get("confidence", 0), reverse=True)
            lines += ["─── 其他候選（未達門檻，依確信度）────"]
            for p in below_sorted:
                info = get_advice(p.get("class", "unknown"))
                lines.append(f"  · {info['zh_name']}（{p.get('confidence', 0):.1%}）")
            lines.append("")

    lines += [
        "══════════════════════════════",
        "【現場備註】",
        "  田區位置：",
        "  發病面積：",
        "  備註說明：",
        "══════════════════════════════",
    ]
    return "\n".join(lines)

# ────────────────────────────────────────────────────────────
#  主程式
# ────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="草莓園小助手",
        page_icon="🍓",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    # 載入 CSS
    _style_path = Path(__file__).resolve().parent / "style.css"
    try:
        with open(_style_path, "r", encoding="utf-8") as f:
            st.markdown(f.read(), unsafe_allow_html=True)
    except FileNotFoundError:
        pass

    st.markdown(f"<head>{IOS_PWA_META}</head>", unsafe_allow_html=True)

    # ── 側邊欄：實用設定 ──
    with st.sidebar:
        st.markdown('<p class="sidebar-title">⚙️ 設定</p>', unsafe_allow_html=True)

        confidence_threshold = st.slider(
            "確信度門檻",
            min_value=0.10,
            max_value=1.0,
            value=0.55,
            step=0.05,
            format="%.2f",
            help="數值越高越嚴格，建議 0.50～0.65",
        )

        enable_enhance = st.toggle("✨ 戶外強光優化", value=False)

        with st.expander("📷 拍攝技巧", expanded=False):
            st.markdown("""
**提高診斷準確率**

· 距離 20～30cm，讓病斑填滿畫面 2/3 以上
· 選擇陰天或遮陰拍攝，避免強烈直射光
· 確保畫面清晰不模糊，必要時手動對焦
· 拍攝葉片背面，角斑病、白粉病症狀更明顯
· 果實拍攝時可用白紙墊底，提高對比

**確信度說明**

· 75% 以上：可信度高，建議立即處置
· 55～74%：可能性高，建議複拍確認
· 55% 以下：建議重新拍攝
""")

        with st.expander("📖 使用說明", expanded=False):
            st.markdown("""
1. 選擇診斷模式（葉片 / 果實）
2. 拍照或上傳草莓影像
3. 點擊「開始 AI 診斷」
4. 檢視結果並填寫現場備註後下載報告

**iPhone 加入主畫面**
Safari → 分享 → 加入主畫面
""")

        st.markdown("""
        <div class="sidebar-footer">
            <span class="footer-logo">BerryWise</span>
            <span class="footer-version">v2.0</span>
        </div>
        """, unsafe_allow_html=True)

    # 固定使用預設 API Key / Model ID
    api_key  = get_api_key()
    model_id = DEFAULT_MODEL_ID

    # ── 主畫面 ──
    st.markdown("""
    <div class="title-fullwidth">
        <div class="logo-emoji">🍓</div>
        <div class="main-title">草莓園小助手</div>
        <div class="sub-title">BerryWise · AI 病害診斷</div>
    </div>
    """, unsafe_allow_html=True)

    # ── 模式選擇（用 session_state + 自訂按鈕，完全可控）──
    if "mode" not in st.session_state:
        st.session_state.mode = 0
    if "inp" not in st.session_state:
        st.session_state.inp = 0

    # 模式按鈕
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🌿 葉片診斷", width="stretch",
                     type="primary" if st.session_state.mode==0 else "secondary"):
            st.session_state.mode = 0
    with c2:
        if st.button("🍓 果實分析", width="stretch",
                     type="primary" if st.session_state.mode==1 else "secondary"):
            st.session_state.mode = 1

    mode = "🌿 葉片診斷" if st.session_state.mode == 0 else "🍓 果實分析"

    # 模式提示
    if st.session_state.mode == 0:
        st.markdown('<div class="mode-tip tip-leaf">對準草莓葉片拍攝，確保病斑在畫面中央</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="mode-tip tip-fruit">在自然光下對準果實拍攝，避免強烈背光</div>', unsafe_allow_html=True)

    # ── 輸入方式按鈕 ──
    i1, i2 = st.columns(2)
    with i1:
        if st.button("📷 即時拍照", width="stretch",
                     type="primary" if st.session_state.inp==0 else "secondary"):
            st.session_state.inp = 0
    with i2:
        if st.button("📁 上傳圖片", width="stretch",
                     type="primary" if st.session_state.inp==1 else "secondary"):
            st.session_state.inp = 1

    captured_image = None
    if st.session_state.inp == 0:
        camera_image = st.camera_input("拍攝", label_visibility="collapsed")
        if camera_image:
            captured_image = Image.open(camera_image).convert("RGB")
    else:
        uploaded_file = st.file_uploader(
            "上傳",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        if uploaded_file:
            captured_image = Image.open(uploaded_file).convert("RGB")

    # ── 影像預覽與診斷 ──
    if captured_image is not None:

        if enable_enhance:
            col_orig, col_proc = st.columns(2)
            with col_orig:
                st.caption("原始影像")
                st.image(captured_image, width="stretch")
            processed_image = enhance_outdoor_image(captured_image)
            with col_proc:
                st.caption("✨ 優化後")
                st.image(processed_image, width="stretch")
        else:
            processed_image = captured_image
            st.image(captured_image, width="stretch", caption="原始影像")

        # 診斷按鈕
        st.markdown('<div class="btn-spacer"></div>', unsafe_allow_html=True)
        if st.button("🔬 開始 AI 診斷", type="primary", width="stretch"):
            st.session_state.feedback_state = None
            st.session_state.manual_disease = None
            st.session_state.confirmed_disease = None
            loading_placeholder = st.empty()
            loading_placeholder.markdown('<div class="ai-loading-bar"><div class="ai-loading-shimmer"></div></div>', unsafe_allow_html=True)

            with st.spinner("AI 分析中，約需 3～5 秒..."):
                result = run_inference(processed_image, api_key, model_id, confidence_threshold)

            loading_placeholder.empty()  # 診斷完成後移除動畫

            if result is not None:
                predictions = result.get("predictions", [])
                predictions_sorted = sorted(predictions, key=lambda p: p.get("confidence", 0), reverse=True)
                HIGH_CONF = 0.75
                valid_preds = [p for p in predictions_sorted if p.get("confidence", 0) >= confidence_threshold]
                high_preds  = [p for p in valid_preds if p.get("confidence", 0) >= HIGH_CONF]

                # ── 統計數字 ──
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("偵測數", len(predictions))
                with c2:
                    st.metric("有效", len(valid_preds))
                with c3:
                    best = max((p.get("confidence", 0) for p in valid_preds), default=None)
                    st.metric("最高確信度", f"{best:.0%}" if best else "—")

                # ── 字體大小調整 ──
                if "font_size" not in st.session_state:
                    st.session_state.font_size = 16
                font_size = st.slider(
                    "🔡 字體大小",
                    min_value=14, max_value=22,
                    value=st.session_state.font_size,
                    step=1, format="%dpx",
                    key="font_slider",
                )
                st.session_state.font_size = font_size
                st.markdown(f"""
                <style>
                p, .stMarkdown p, .stMarkdown, div[data-testid="stMarkdownContainer"] p
                    {{ font-size: {font_size}px !important; line-height: 1.7 !important; }}
                .disease-name {{ font-size: {font_size + 2}px !important; }}
                .disease-meta {{ font-size: {font_size - 1}px !important; }}
                .stButton > button {{ font-size: {font_size}px !important; min-height: {max(48, font_size * 3)}px !important; }}
                [data-testid="stExpander"] summary {{ font-size: {font_size - 1}px !important; min-height: {max(44, font_size * 2 + 14)}px !important; }}
                pre {{ font-size: {font_size - 2}px !important; }}
                label {{ font-size: {font_size - 1}px !important; }}
                .empty-text {{ font-size: {font_size + 1}px !important; }}
                .empty-hint {{ font-size: {font_size - 2}px !important; }}
                [data-testid="stMetricValue"] {{ font-size: {font_size + 8}px !important; }}
                [data-testid="stMetric"] label {{ font-size: {font_size - 3}px !important; }}
                </style>
                """, unsafe_allow_html=True)

                # ── 低確信度：顯示手動選擇區 ──
                if valid_preds and not high_preds:
                    best_conf = max(p.get("confidence", 0) for p in valid_preds)
                    if best_conf < 0.65:
                        tips = "距離拉近至 20cm、確保病斑佔畫面 2/3、避免強烈背光或反光"
                    else:
                        tips = "建議複拍一張確認，可嘗試從不同角度或葉片背面拍攝"
                    st.markdown(f"""
                    <div style="background:rgba(255,165,0,0.08);border:1px solid rgba(255,165,0,0.3);
                    border-radius:12px;padding:12px 16px;margin:8px 0;">
                    <div style="color:#f5a623;font-weight:600;margin-bottom:4px;">⚠️ 確信度偏低（{best_conf:.0%}）— 請手動確認病害</div>
                    <div style="font-size:12px;color:rgba(255,200,100,0.8);">📷 拍攝改善：{tips}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # 手動選擇按鈕（依模式顯示對應病害）
                    leaf_diseases = ["angular leafspot","leaf spot","powdery mildew leaf"]
                    fruit_diseases = ["anthracnose fruit rot","blossom blight","gray mold","powdery mildew fruit"]
                    all_diseases = list(ADVICE_DB.keys())
                    relevant = (leaf_diseases if st.session_state.mode == 0 else fruit_diseases)
                    others   = [k for k in all_diseases if k not in relevant]

                    st.markdown("<div style='font-size:13px;color:rgba(255,255,255,0.5);margin:10px 0 6px;'>👆 您目測判斷是哪種病害？</div>", unsafe_allow_html=True)

                    # 主要候選（依模式）
                    cols = st.columns(len(relevant))
                    for i, key in enumerate(relevant):
                        info = ADVICE_DB[key]
                        selected = st.session_state.get("manual_disease") == key
                        with cols[i]:
                            if st.button(
                                f"{'✓ ' if selected else ''}{info['zh_name']}",
                                key=f"btn_main_{key}",
                                type="primary" if selected else "secondary",
                                width="stretch",
                            ):
                                st.session_state.manual_disease = key
                                st.rerun()

                    # 其他病害（收合）
                    with st.expander("其他病害選項"):
                        cols2 = st.columns(2)
                        for i, key in enumerate(others):
                            info = ADVICE_DB[key]
                            selected = st.session_state.get("manual_disease") == key
                            with cols2[i % 2]:
                                if st.button(
                                    f"{'✓ ' if selected else ''}{info['zh_name']}",
                                    key=f"btn_other_{key}",
                                    type="primary" if selected else "secondary",
                                    width="stretch",
                                ):
                                    st.session_state.manual_disease = key
                                    st.rerun()

                    # 清除選擇
                    if st.session_state.get("manual_disease"):
                        if st.button("✕ 清除手動選擇", key="clear_manual"):
                            st.session_state.manual_disease = None
                            st.rerun()

                    # 顯示手動選擇的病害資訊
                    manual_key = st.session_state.get("manual_disease")
                    if manual_key and manual_key in ADVICE_DB:
                        minfo = ADVICE_DB[manual_key]
                        st.markdown(f"""
                        <div class="disease-card" style="border-left:4px solid {minfo['color']};margin-top:10px;">
                            <div class="disease-name">{minfo['zh_name']} <span class="disease-en">({minfo['en_name']})</span>
                            <span style="font-size:11px;background:rgba(52,152,219,0.2);color:#5dade2;padding:2px 8px;border-radius:6px;margin-left:6px;">農民確認</span></div>
                            <div class="disease-meta">{minfo['severity']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        confused = minfo.get("confused_with","")
                        if confused:
                            st.markdown(f"<div style='font-size:12px;color:rgba(255,200,100,0.7);padding:4px 0;'>⚡ 易混淆：{confused}</div>", unsafe_allow_html=True)
                        with st.expander("🔍 目測確認特徵"):
                            st.markdown(f"<pre style='font-size:12px;color:rgba(255,255,255,0.7);white-space:pre-wrap;background:none;border:none;padding:0;'>{minfo.get('visual_check','')}</pre>", unsafe_allow_html=True)
                        with st.expander("📋 農事建議"):
                            st.markdown(f"<pre style='font-size:12px;color:rgba(255,255,255,0.75);white-space:pre-wrap;background:none;border:none;padding:0;'>{minfo['advice']}</pre>", unsafe_allow_html=True)

                # ── 標注圖 ──
                st.markdown("**偵測結果**")
                annotated = draw_detections(processed_image.copy(), predictions, confidence_threshold)
                st.image(annotated, width="stretch")

                # ══════════════════════════════════════════════
                # ── 診斷結果 + 農民確認（統一流程）──
                # ══════════════════════════════════════════════
                st.markdown("---")
                st.markdown("""
                <div style="font-size:15px;font-weight:700;color:rgba(255,255,255,0.6);
                margin:4px 0 10px;letter-spacing:0.3px;">
                ✅ 這張圖是哪種病害？請點選確認
                </div>
                <div style="font-size:13px;color:rgba(255,255,255,0.35);margin-bottom:12px;">
                點選後系統將自動記錄並用於模型訓練
                </div>
                """, unsafe_allow_html=True)

                confirmed = st.session_state.get("confirmed_disease")
                feedback_done = st.session_state.get("feedback_state")

                # ── AI 偵測到的候選（依確信度排列）──
                if predictions_sorted:
                    st.markdown(f"<div style='font-size:13px;color:rgba(255,255,255,0.4);margin-bottom:6px;'>🤖 AI 偵測到的病害</div>", unsafe_allow_html=True)
                    ai_keys = [p.get("class","").lower() for p in predictions_sorted]
                    n_ai = len(predictions_sorted)
                    cols = st.columns(min(n_ai, 3))
                    for i, pred in enumerate(predictions_sorted[:3]):
                        info = get_advice(pred.get("class","unknown"))
                        conf = pred.get("confidence", 0)
                        key  = pred.get("class","").lower()
                        is_selected = confirmed == key
                        badge_color = "#2ecc71" if conf >= HIGH_CONF else "#f5a623"
                        with cols[i]:
                            if st.button(
                                f"{'✓ ' if is_selected else ''}{info['zh_name']}\n{conf:.0%}",
                                key=f"confirm_ai_{key}_{i}",
                                type="primary" if is_selected else "secondary",
                                width="stretch",
                            ):
                                st.session_state.confirmed_disease = key
                                st.session_state.feedback_state = None
                                st.rerun()

                # ── 全部 7 種病害選項（收合）──
                st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
                with st.expander("🔍 不在上方？點此選擇其他病害"):
                    all_keys = list(ADVICE_DB.keys())
                    cols2 = st.columns(2)
                    for i, key in enumerate(all_keys):
                        info = ADVICE_DB[key]
                        is_selected = confirmed == key
                        with cols2[i % 2]:
                            if st.button(
                                f"{'✓ ' if is_selected else ''}{info['zh_name']}",
                                key=f"confirm_all_{key}",
                                type="primary" if is_selected else "secondary",
                                width="stretch",
                            ):
                                st.session_state.confirmed_disease = key
                                st.session_state.feedback_state = None
                                st.rerun()

                    # 健康植株選項
                    is_healthy = confirmed == "healthy"
                    if st.button(
                        f"{'✓ ' if is_healthy else ''}🌱 健康植株（無病害）",
                        key="confirm_healthy",
                        type="primary" if is_healthy else "secondary",
                        width="stretch",
                    ):
                        st.session_state.confirmed_disease = "healthy"
                        st.session_state.feedback_state = None
                        st.rerun()

                # ── 確認後顯示詳情 + 上傳 Roboflow ──
                if confirmed:
                    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)

                    if confirmed == "healthy":
                        st.markdown("""
                        <div class="disease-card" style="border-left:4px solid #2ecc71;">
                            <div class="disease-name">🌱 健康植株</div>
                            <div class="disease-meta">無明顯病害特徵</div>
                        </div>
                        """, unsafe_allow_html=True)
                        conf_label = "healthy"
                    elif confirmed in ADVICE_DB:
                        info = ADVICE_DB[confirmed]
                        conf_label = info.get("en_name", confirmed)
                        st.markdown(f"""
                        <div class="disease-card" style="border-left:4px solid {info['color']};">
                            <div class="disease-name">{info['zh_name']} <span class="disease-en">({info['en_name']})</span>
                            <span style="font-size:11px;background:rgba(52,152,219,0.2);color:#5dade2;
                            padding:2px 8px;border-radius:6px;margin-left:6px;">農民確認</span></div>
                            <div class="disease-meta">{info['severity']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        confused = info.get("confused_with","")
                        if confused:
                            st.markdown(f"<div style='font-size:12px;color:rgba(255,200,100,0.7);padding:3px 0 4px;'>⚡ 易混淆：{confused}</div>", unsafe_allow_html=True)
                        with st.expander("🔍 目測確認特徵"):
                            st.markdown(f"<pre style='font-size:12px;color:rgba(255,255,255,0.7);white-space:pre-wrap;background:none;border:none;padding:0;'>{info.get('visual_check','')}</pre>", unsafe_allow_html=True)
                        with st.expander("📋 農事建議"):
                            st.markdown(f"<pre style='font-size:12px;color:rgba(255,255,255,0.75);white-space:pre-wrap;background:none;border:none;padding:0;'>{info['advice']}</pre>", unsafe_allow_html=True)
                    else:
                        conf_label = confirmed

                    # ── 上傳 Roboflow ──
                    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
                    if feedback_done == "uploaded":
                        st.markdown("""
                        <div style="background:rgba(52,152,219,0.08);border:1px solid rgba(52,152,219,0.2);
                        border-radius:10px;padding:12px 14px;font-size:13px;color:#5dade2;">
                        🚀 感謝您的協助！這張圖片將被用於 BerryWise 的下一次自我進化訓練。
                        </div>
                        """, unsafe_allow_html=True)
                    elif feedback_done == "upload_fail":
                        if st.button("⚠️ 上傳失敗，點此重試", key="retry_upload", type="secondary", width="stretch"):
                            with st.spinner("上傳中..."):
                                ok = upload_to_roboflow(processed_image, api_key, suggested_label=conf_label)
                            st.session_state.feedback_state = "uploaded" if ok else "upload_fail"
                            st.rerun()
                    else:
                        if st.button("📤 送出確認並協助訓練", key="upload_confirmed", type="primary", width="stretch"):
                            with st.spinner("上傳中..."):
                                ok = upload_to_roboflow(processed_image, api_key, suggested_label=conf_label)
                            st.session_state.feedback_state = "uploaded" if ok else "upload_fail"
                            st.rerun()

                elif not predictions:
                    uncovered = UNCOVERED_LEAF if st.session_state.mode == 0 else UNCOVERED_FRUIT
                    st.markdown("""
                    <div style="background:rgba(52,152,219,0.08);border:1px solid rgba(52,152,219,0.25);
                    border-radius:12px;padding:12px 16px;margin:8px 0;">
                    <div style="color:#5dade2;font-weight:600;margin-bottom:4px;">🔍 模型未偵測到已知病害</div>
                    <div style="font-size:12px;color:rgba(150,210,240,0.8);">
                    可能原因：病害類型不在訓練範圍內、拍攝角度或光線問題<br>
                    以下為模型目前未涵蓋的常見病害，請對照目測特徵自行確認：
                    </div></div>
                    """, unsafe_allow_html=True)
                    for d in uncovered:
                        with st.expander(f"📌 {d['zh_name']}　{d['en_name']}"):
                            st.markdown(f"""
                            <div style="font-size:13px;color:rgba(255,255,255,0.75);line-height:1.8;">
                            <b style="color:rgba(255,255,255,0.5);font-size:11px;">目測特徵</b><br>
                            {d['visual']}<br><br>
                            <b style="color:rgba(255,255,255,0.5);font-size:11px;">處置方向</b><br>
                            {d['action']}
                            </div>
                            """, unsafe_allow_html=True)

                # ── Top-3 其他可能（依確信度排列）──
                other_candidates = [p for p in predictions_sorted if p not in valid_preds][:3]
                if other_candidates:
                    with st.expander(f"🔎 其他候選病害（{len(other_candidates)} 項）"):
                        st.caption("確信度未達門檻，可調低滑桿後重新診斷，或複拍後比對")
                        for p in other_candidates:
                            info = get_advice(p.get("class", "unknown"))
                            conf = p.get("confidence", 0)
                            st.markdown(f"""
                            <div class="disease-card" style="border-left:4px solid #333;opacity:0.7">
                                <div class="disease-name" style="font-size:14px">{info['zh_name']} <span class="disease-en">({info['en_name']})</span></div>
                                <div class="disease-meta">確信度 {conf:.1%}</div>
                            </div>
                            """, unsafe_allow_html=True)

                # ── 可編輯診斷報告（納入手動選擇）──
                st.markdown("**📋 診斷報告**")
                st.caption("可直接編輯備註後下載")
                manual_key = st.session_state.get("manual_disease")
                report_text = generate_report(
                    predictions, confidence_threshold, mode,
                    manual_override=manual_key
                )
                edited_report = st.text_area(
                    "報告",
                    value=report_text,
                    height=360,
                    label_visibility="collapsed",
                )
                filename = f"草莓園小助手_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                st.download_button(
                    "💾 下載診斷報告",
                    data=edited_report.encode("utf-8"),
                    file_name=filename,
                    mime="text/plain",
                    width="stretch",
                )

    else:
        # 空狀態引導
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📷</div>
            <div class="empty-text">請拍照或上傳草莓影像</div>
            <div class="empty-hint">支援 JPG、PNG、WEBP，單檔限 200MB</div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
