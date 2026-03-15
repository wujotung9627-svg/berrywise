# ============================================================
#  BerryWise — 草莓園小助手 v4
#  ✅ 真實 Roboflow 推論 + BBox 標注（PIL，無 cv2 依賴）
#  ✅ Groq Vision 交叉驗證（免費，llama-3.2-11b-vision-preview）
#  ✅ 農民回饋一鍵上傳 Roboflow（2 步驟確認）
#  ✅ PWA manifest 動態注入（iOS Safari + Android Chrome）
#  ✅ 手機優先 UI · 設定搬到頁面內 · 純 requests
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io, base64, datetime, json, requests, numpy as np
from pathlib import Path

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

# ──────────────────────────────────────────────────────────────
#  API Keys & 常數
# ──────────────────────────────────────────────────────────────
def get_api_key() -> str:
    """從 Streamlit secrets 讀取 Roboflow API Key。
    不提供 fallback，缺少設定時明確告知使用者，避免憑證暴露於原始碼。
    """
    try:
        key = st.secrets["ROBOFLOW_API_KEY"]
        if not key or not isinstance(key, str):
            raise KeyError
        return key
    except (KeyError, FileNotFoundError):
        st.error(
            "⚠️ **缺少 Roboflow API Key**\n\n"
            "請在 Streamlit Cloud → Settings → Secrets 加入：\n"
            "```\nROBOFLOW_API_KEY = \"your-api-key\"\n```"
        )
        st.stop()

def get_groq_key() -> str:
    """從 Streamlit secrets 讀取 Groq API Key。
    Groq 為選配功能，未設定時回傳空字串並靜默停用交叉驗證。
    """
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
        return key if isinstance(key, str) else ""
    except Exception:
        return ""

DEFAULT_MODEL_ID   = "-strawberry-disease-hrfcc/2"
ROBOFLOW_DATASET   = "-strawberry-disease-hrfcc"
ROBOFLOW_WORKSPACE = "jojos-workspace-mudmq"

# ──────────────────────────────────────────────────────────────
#  PWA HTML（iOS + Android，動態 manifest blob）
# ──────────────────────────────────────────────────────────────
PWA_HTML = """
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="草莓園小助手">
<meta name="mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#0a0a0a">
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
<link rel="apple-touch-icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Crect width='100' height='100' rx='22' fill='%23C0392B'/%3E%3Ctext y='.85em' font-size='72' text-anchor='middle' x='50'%3E🍓%3C/text%3E%3C/svg%3E">
<link rel="manifest" id="berry-manifest">
<script>
(function(){
  var manifest = {
    "name":"草莓園小助手","short_name":"BerryWise",
    "description":"AI 草莓病害診斷","start_url":"/",
    "display":"standalone","background_color":"#0a0a0a","theme_color":"#C0392B",
    "orientation":"portrait",
    "icons":[
      {"src":"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 192 192'%3E%3Crect width='192' height='192' rx='40' fill='%23C0392B'/%3E%3Ctext y='.85em' font-size='138' text-anchor='middle' x='96'%3E🍓%3C/text%3E%3C/svg%3E","sizes":"192x192","type":"image/svg+xml","purpose":"any maskable"},
      {"src":"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 512 512'%3E%3Crect width='512' height='512' rx='100' fill='%23C0392B'/%3E%3Ctext y='.85em' font-size='370' text-anchor='middle' x='256'%3E🍓%3C/text%3E%3C/svg%3E","sizes":"512x512","type":"image/svg+xml","purpose":"any maskable"}
    ]
  };
  var blob = new Blob([JSON.stringify(manifest)],{type:"application/json"});
  var url  = URL.createObjectURL(blob);
  var el   = document.getElementById("berry-manifest");
  if(el){ el.href=url; } else {
    var l=document.createElement("link"); l.rel="manifest"; l.href=url;
    document.head.appendChild(l);
  }
  window.addEventListener("beforeinstallprompt",function(e){
    e.preventDefault(); window._deferredPWA=e;
    var b=document.getElementById("pwa-install-btn");
    if(b) b.style.display="inline-flex";
  });
})();
</script>
"""

PWA_BANNER_HTML = """
<div id="pwa-banner" style="display:none;position:fixed;bottom:0;left:0;right:0;
  background:linear-gradient(135deg,#160808,#161616);border-top:1px solid rgba(192,57,43,0.35);
  padding:11px 16px;z-index:9999;align-items:center;gap:10px;font-family:system-ui,-apple-system;">
  <span style="font-size:20px;">🍓</span>
  <div style="flex:1;">
    <div style="font-size:13px;color:rgba(255,255,255,0.9);font-weight:600;">加入主畫面</div>
    <div style="font-size:10px;color:rgba(255,255,255,0.4);">一鍵安裝，無需上架 App Store</div>
  </div>
  <button onclick="if(window._deferredPWA)window._deferredPWA.prompt();"
    style="padding:8px 18px;border-radius:10px;border:none;background:#C0392B;
    color:#fff;font-size:13px;font-weight:600;cursor:pointer;flex-shrink:0;">安裝</button>
  <button onclick="document.getElementById('pwa-banner').style.display='none';"
    style="padding:8px 10px;border-radius:8px;border:1px solid rgba(255,255,255,0.14);
    background:transparent;color:rgba(255,255,255,0.4);font-size:12px;cursor:pointer;flex-shrink:0;">✕</button>
</div>
"""

# ──────────────────────────────────────────────────────────────
#  全域 CSS
# ──────────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
/* 基底 */
.stApp { background:#0a0a0a !important; }
.block-container {
  padding-top:0.75rem !important;
  padding-bottom:2rem !important;
  max-width:480px !important;
}

/* 隱藏 Streamlit 多餘元素 */
#MainMenu, footer, header { visibility:hidden; }
[data-testid="stToolbar"] { display:none !important; }

/* 按鈕 */
.stButton>button {
  border-radius:12px !important;
  min-height:46px !important;
  font-size:14px !important;
  font-weight:600 !important;
  transition:all 0.18s ease !important;
  letter-spacing:0.2px !important;
}
.stButton>button[kind="primary"],
.stButton>button[data-testid="baseButton-primary"] {
  background:#C0392B !important;
  border:none !important;
  color:#fff !important;
  box-shadow:0 2px 14px rgba(192,57,43,0.38) !important;
}
.stButton>button[kind="primary"]:hover {
  background:#a93226 !important;
  box-shadow:0 4px 20px rgba(192,57,43,0.52) !important;
}
.stButton>button[kind="secondary"],
.stButton>button[data-testid="baseButton-secondary"] {
  background:#161616 !important;
  border:1px solid rgba(255,255,255,0.1) !important;
  color:rgba(255,255,255,0.5) !important;
}
.stButton>button[kind="secondary"]:hover {
  border-color:rgba(255,255,255,0.22) !important;
  color:rgba(255,255,255,0.75) !important;
}

/* Metric */
div[data-testid="stMetric"] {
  background:#161616;
  border:1px solid rgba(255,255,255,0.07);
  border-radius:12px;
  padding:10px 14px !important;
}
div[data-testid="stMetricValue"] {
  font-size:22px !important;
  font-weight:800 !important;
  color:rgba(255,255,255,0.92) !important;
}
div[data-testid="stMetricLabel"] {
  font-size:11px !important;
  color:rgba(255,255,255,0.32) !important;
}

/* Image */
img { border-radius:12px !important; }

/* Expander */
.stExpander {
  border:1px solid rgba(255,255,255,0.07) !important;
  border-radius:12px !important;
  background:#161616 !important;
}
.stExpanderHeader {
  background:#161616 !important;
  border-radius:12px !important;
  font-weight:600 !important;
}

/* Text area (報告) */
.stTextArea textarea {
  background:#111 !important;
  border:1px solid rgba(255,255,255,0.08) !important;
  border-radius:10px !important;
  color:rgba(255,255,255,0.82) !important;
  font-size:13px !important;
  font-family:'Courier New', monospace !important;
}

/* Download button */
.stDownloadButton>button {
  border-radius:12px !important;
  font-weight:600 !important;
  background:#161616 !important;
  border:1px solid rgba(255,255,255,0.12) !important;
  color:rgba(255,255,255,0.6) !important;
}

/* 分隔線 */
hr { border-color:rgba(255,255,255,0.07) !important; }

/* Toggle */
.stToggle { padding:4px 0 !important; }

/* Slider */
.stSlider [data-testid="stSlider"] { padding:0 !important; }

/* Camera input */
[data-testid="stCameraInput"] video,
[data-testid="stCameraInput"] img {
  border-radius:12px !important;
}

/* Sidebar（桌機備用）*/
[data-testid="stSidebar"] {
  background:#0d0d0d !important;
  border-right:1px solid rgba(255,255,255,0.06) !important;
}
</style>
"""

# ──────────────────────────────────────────────────────────────
#  病害資料庫
# ──────────────────────────────────────────────────────────────
ADVICE_DB = {
    "angular leafspot": {
        "zh_name":"角斑病","en_name":"Angular Leafspot",
        "severity":"⚠️ 中度","color":"#E74C3C","icon":"🍃",
        "visual_check":(
            "✓ 病斑受葉脈限制，呈不規則多角形（非圓形）\n"
            "✓ 初期水浸狀、半透明，後轉黃褐色至深褐色\n"
            "✓ 潮濕時病斑背面可見白色菌膿\n"
            "✗ 若病斑為圓形、有同心圓 → 可能是葉斑病"
        ),
        "confused_with":"葉斑病（Leaf Spot）",
        "advice":(
            "【立即處置】移除出現水浸狀、褐色多角形病斑的葉片。\n"
            "禁止從上方澆水，改採滴灌，減少葉面積水。\n\n"
            "【環境管理】加強植株間通風，行距維持 30cm 以上。\n"
            "雨後 24 小時內巡視積水情形。\n\n"
            "【藥劑參考】可使用銅基殺菌劑（如氧化亞銅）進行保護性噴施。"
        ),
    },
    "anthracnose fruit rot": {
        "zh_name":"炭疽病（果實）","en_name":"Anthracnose Fruit Rot",
        "severity":"🚨 高度","color":"#C0392B","icon":"🍓",
        "visual_check":(
            "✓ 果面出現圓形、黑褐色、明顯凹陷的病斑\n"
            "✓ 病斑邊緣清晰，呈「燒灼感」外觀\n"
            "✓ 濕潤時病斑中央可見橘紅色孢子堆\n"
            "✗ 若果面為白色粉末狀 → 應為果實白粉病"
        ),
        "confused_with":"灰黴病（Gray Mold）、果實白粉病",
        "advice":(
            "【立即處置】移除出現黑色凹陷圓形病斑的果實，套袋帶出田區銷毀。\n"
            "避免病果與健康果實接觸，防止接觸傳染。\n\n"
            "【環境管理】高溫多濕環境需每日巡檢，雨後立即清查。\n\n"
            "【藥劑參考】苯醚甲環唑或咪鮮胺類藥劑交替噴施。"
        ),
    },
    "blossom blight": {
        "zh_name":"花凋病","en_name":"Blossom Blight",
        "severity":"🚨 高度","color":"#8E44AD","icon":"🌸",
        "visual_check":(
            "✓ 花瓣出現褐色水浸狀腐爛，快速萎凋\n"
            "✓ 花萼及花梗變褐，嚴重時整花枯死\n"
            "✗ 若症狀在花謝後擴展至果實 → 留意是否轉為灰黴病"
        ),
        "confused_with":"灰黴病（Gray Mold）",
        "advice":(
            "【立即處置】立刻摘除出現褐化、枯萎的花朵與花梗，集中銷毀。\n"
            "避免在開花期噴水，減少花部濕潤時間。\n\n"
            "【藥劑參考】可於開花前預防性噴施腐黴利或撲克拉等藥劑。"
        ),
    },
    "gray mold": {
        "zh_name":"灰黴病","en_name":"Gray Mold",
        "severity":"🚨 高度","color":"#9B59B6","icon":"🌫️",
        "visual_check":(
            "✓ 黴層呈灰褐色、蓬鬆絨毛狀（非白色粉末）\n"
            "✓ 輕拍患部可見灰色孢子雲飄散\n"
            "✓ 感染部位組織軟腐，有腐爛氣味\n"
            "✗ 若表面為白色粉末且無軟腐 → 應為果實白粉病"
        ),
        "confused_with":"果實白粉病（Powdery Mildew Fruit）",
        "advice":(
            "【立即處置】立刻摘除並套袋帶出所有出現灰色黴層的果實與葉片。\n\n"
            "【環境管理】目標相對濕度維持 70% 以下；多雨期可搭設防雨網。\n\n"
            "【藥劑參考】輪替使用 SDHI 類或 QoI 類殺菌劑，防止抗藥性。"
        ),
    },
    "leaf spot": {
        "zh_name":"葉斑病（蛇眼病）","en_name":"Leaf Spot",
        "severity":"⚠️ 中度","color":"#E67E22","icon":"👁️",
        "visual_check":(
            "✓ 病斑為圓形至橢圓形，有明顯紫紅色邊緣\n"
            "✓ 中心灰白色，外圍紫褐色（蛇眼狀）\n"
            "✗ 若病斑多角形且受葉脈限制 → 應為角斑病"
        ),
        "confused_with":"角斑病（Angular Leafspot）、葉片白粉病",
        "advice":(
            "【立即處置】摘除出現蛇眼狀褐色圓形病斑的葉片，集中銷毀。\n\n"
            "【環境管理】避免葉面積水，改採滴灌方式。\n\n"
            "【藥劑參考】代森錳鋅或百菌清，每 7～10 天噴一次。"
        ),
    },
    "powdery mildew fruit": {
        "zh_name":"白粉病（果實）","en_name":"Powdery Mildew Fruit",
        "severity":"⚠️ 中度","color":"#F39C12","icon":"🍓",
        "visual_check":(
            "✓ 果面出現白色至灰白色粉末狀覆蓋物\n"
            "✓ 用手指輕擦可抹去白粉，下方果皮完整\n"
            "✗ 若黴層灰褐蓬鬆、組織軟腐 → 應為灰黴病"
        ),
        "confused_with":"灰黴病（Gray Mold）",
        "advice":(
            "【立即處置】摘除表面出現白色粉末的果實，套袋帶出避免孢子飛散。\n\n"
            "【藥劑參考】微粒硫磺或亞磷酸製劑，避免高溫時段施藥。"
        ),
    },
    "powdery mildew leaf": {
        "zh_name":"白粉病（葉片）","en_name":"Powdery Mildew Leaf",
        "severity":"⚠️ 中度","color":"#F1C40F","icon":"🌿",
        "visual_check":(
            "✓ 葉背出現白色粉末狀菌絲層\n"
            "✓ 葉面對應位置出現紫紅色至褐色斑塊\n"
            "✗ 若病斑為蛇眼狀圓形、無白粉 → 應為葉斑病"
        ),
        "confused_with":"葉斑病（Leaf Spot）",
        "advice":(
            "【立即處置】摘除葉背出現白色粉末狀菌絲的葉片，避免孢子飛散。\n\n"
            "【藥劑參考】硫磺製劑或三唑類殺菌劑交替使用，每 5～7 天一次。"
        ),
    },
}

def _html_escape(text: str) -> str:
    """HTML-escape 外部來源字串，防止 XSS 注入。"""
    return (str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;"))

def get_advice(label: str) -> dict:
    """取得病害建議資料。
    若 label 不在 ADVICE_DB（即來自 Roboflow 的未知標籤），
    將 zh_name / en_name 進行 HTML escape，防止未知標籤注入 HTML。
    """
    key = label.lower().strip()
    if key in ADVICE_DB:
        return ADVICE_DB[key]
    # 外部來源的 label 必須 escape 後才能放入 HTML 模板
    safe_label = _html_escape(label)
    return {
        "zh_name": safe_label, "en_name": safe_label,
        "severity": "❓ 待確認", "color": "#3498DB", "icon": "❓",
        "visual_check": "", "confused_with": "",
        "advice": "本系統尚未收錄此標籤，請對照目測特徵確認病害種類。",
    }

UNCOVERED = {
    0: [  # 葉片
        {"zh_name":"枯葉病（葉緣焦枯）","en_name":"Leaf Scorch",
         "visual":"葉緣或葉尖出現紅褐色至暗褐色乾枯，病健交界不清晰，嚴重時整葉枯焦。",
         "action":"移除嚴重枯葉，調整水分供應，可噴施銅基殺菌劑預防。"},
        {"zh_name":"炭疽葉枯病","en_name":"Anthracnose Crown Rot",
         "visual":"葉片出現不規則深褐色至黑褐色壞死斑，濕潤時可見橘紅色孢子堆。",
         "action":"立即移除病株，避免連作，噴施苯醚甲環唑。"},
    ],
    1: [  # 果實
        {"zh_name":"軟腐病","en_name":"Soft Rot",
         "visual":"果實快速水爛、表面白色棉絮狀黴層，最終出現黑色孢子囊，有酸臭味。",
         "action":"立即移除並銷毀，保持低溫（4°C），加強通風。"},
        {"zh_name":"疫病果腐","en_name":"Phytophthora Fruit Rot",
         "visual":"果實出現深褐色、皮革狀硬化病斑，通常從近地面果實先發病。",
         "action":"架高果實避免接觸土壤，使用甲霜靈，加強排水。"},
    ],
}

# ──────────────────────────────────────────────────────────────
#  影像前處理（戶外強光優化）
# ──────────────────────────────────────────────────────────────
def enhance_outdoor_image(pil_image: Image.Image) -> Image.Image:
    if not CV2_AVAILABLE:
        return pil_image
    img = np.array(pil_image.convert("RGB"))
    try:
        f = img.astype(np.float32)
        gray = np.mean(f)
        for c in range(3):
            m = np.mean(f[:, :, c])
            if m > 0:
                f[:, :, c] = np.clip(f[:, :, c] * (gray / m), 0, 255)
        img = f.astype(np.uint8)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        lab = cv2.merge([clahe.apply(l), a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        blur_score = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        if blur_score < 50:
            f2   = img.astype(np.float32)
            blur = cv2.GaussianBlur(f2, (0, 0), 2.0)
            img  = np.clip(f2 + 0.6 * (f2 - blur), 0, 255).astype(np.uint8)
        return Image.fromarray(img)
    except Exception:
        return pil_image

# ──────────────────────────────────────────────────────────────
#  Roboflow 推論（純 requests，POST base64）
# ──────────────────────────────────────────────────────────────
def run_inference(pil_image: Image.Image, api_key: str,
                  model_id: str, confidence: float):
    try:
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=90)
        b64     = base64.b64encode(buf.getvalue()).decode("utf-8")
        url     = f"https://detect.roboflow.com/{model_id}"
        params  = {"api_key": api_key, "confidence": int(confidence * 100)}
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        resp = requests.post(url, params=params, data=b64,
                             headers=headers, timeout=30)
        if resp.status_code != 200:
            st.error(f"❌ 推論失敗（HTTP {resp.status_code}）：{resp.text[:200]}")
            return None
        result = resp.json()
        result.setdefault("predictions", [])
        return result
    except requests.exceptions.Timeout:
        st.error("❌ 連線逾時，請檢查網路後重試")
        return None
    except Exception as e:
        st.error(f"❌ 推論失敗：{e}")
        return None

# ──────────────────────────────────────────────────────────────
#  Groq Vision 交叉驗證（免費）
# ──────────────────────────────────────────────────────────────
def ask_groq_vision(pil_image: Image.Image, mode: str,
                    yolo_top: dict | None, groq_key: str) -> dict:
    empty = {"groq_zh": "", "agree": None, "groq_raw": ""}
    if not groq_key:
        return empty
    try:
        from groq import Groq
        client = Groq(api_key=groq_key)
        buf = io.BytesIO()
        img_s = pil_image.copy()
        img_s.thumbnail((800, 800), Image.LANCZOS)
        img_s.save(buf, format="JPEG", quality=82)
        b64  = base64.b64encode(buf.getvalue()).decode()
        part = "葉片" if "葉片" in mode else "果實"
        if yolo_top:
            yolo_zh   = get_advice(yolo_top.get("class", ""))["zh_name"]
            yolo_conf = yolo_top.get("confidence", 0)
            cross_note = (
                f"\n\n【參考】物件偵測模型判斷最可能為「{yolo_zh}」"
                f"（確信度 {yolo_conf:.0%}）。請你獨立看圖判斷並說明是否認同及原因。"
            )
        else:
            cross_note = "\n\n【參考】物件偵測模型未偵測到已知病害，請你獨立判斷。"

        prompt = (
            f"這是一張台灣草莓{part}的照片。請仔細觀察病斑、顏色、紋理等特徵，"
            f"判斷最可能是哪種草莓病害（或健康植株）。\n"
            f"請用繁體中文回答，格式如下：\n"
            f"【判斷結果】（病害名稱；若健康則填「健康植株」）\n"
            f"【觀察到的特徵】（條列視覺線索，2～4 點）\n"
            f"【農民建議】（1～2 句簡短處置建議）"
            f"{cross_note}"
        )
        resp = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]}],
            temperature=0.3, max_tokens=600,
        )
        raw = resp.choices[0].message.content.strip() if resp.choices else ""
        if not raw:
            return empty
        agree = None
        if yolo_top:
            yolo_info = get_advice(yolo_top.get("class", ""))
            raw_lower = raw.lower()
            agree = (
                yolo_info["zh_name"] in raw or
                yolo_info["en_name"].lower() in raw_lower or
                yolo_top.get("class", "").lower() in raw_lower
            )
        return {"groq_zh": raw, "agree": agree, "groq_raw": raw}
    except Exception:
        return empty

# ──────────────────────────────────────────────────────────────
#  BBox 標注繪製（PIL RGBA，不依賴 cv2）
# ──────────────────────────────────────────────────────────────
COLOR_MAP = {
    "angular leafspot":      "#E74C3C",
    "anthracnose fruit rot": "#C0392B",
    "blossom blight":        "#8E44AD",
    "gray mold":             "#9B59B6",
    "leaf spot":             "#E67E22",
    "powdery mildew fruit":  "#F39C12",
    "powdery mildew leaf":   "#F1C40F",
}

def _hex_rgba(h: str, a: int = 255) -> tuple:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), a)

def draw_detections(pil_image: Image.Image, predictions: list,
                    confidence_threshold: float) -> Image.Image:
    img     = pil_image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    W, H    = img.size
    lw      = max(3, int(min(W, H) * 0.005))
    fsz     = max(16, int(min(W, H) * 0.026))
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fsz)
    except Exception:
        try:   font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", fsz)
        except: font = ImageFont.load_default()

    for pred in predictions:
        conf  = pred.get("confidence", 0)
        label = pred.get("class", "unknown")
        cx, cy = pred.get("x", 0), pred.get("y", 0)
        bw, bh = pred.get("width", 0), pred.get("height", 0)
        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
        chex  = COLOR_MAP.get(label.lower(), "#3498DB")
        info  = get_advice(label)

        if conf >= confidence_threshold:
            draw.rectangle([x1, y1, x2, y2], fill=_hex_rgba(chex, 36))
            draw.rectangle([x1, y1, x2, y2], outline=_hex_rgba(chex, 215), width=lw)
            # 四角加粗
            cs = max(18, int(min(bw, bh) * 0.2))
            cc = _hex_rgba(chex, 255)
            for (ox, oy), (hx, hy), (vx, vy) in [
                ((x1,y1),(x1+cs,y1),(x1,y1+cs)),
                ((x2,y1),(x2-cs,y1),(x2,y1+cs)),
                ((x1,y2),(x1+cs,y2),(x1,y2-cs)),
                ((x2,y2),(x2-cs,y2),(x2,y2-cs)),
            ]:
                draw.line([(ox,oy),(hx,hy)], fill=cc, width=lw+2)
                draw.line([(ox,oy),(vx,vy)], fill=cc, width=lw+2)
            # 標籤
            text = f" {info['zh_name']}  {conf:.0%} "
            try:    tb = draw.textbbox((0,0), text, font=font)
            except: tb = (0, 0, len(text)*8, fsz+4)
            tw, th = tb[2]-tb[0]+4, tb[3]-tb[1]+4
            lx = max(0, x1)
            ly = y1 - th - 4 if y1 - th - 4 >= 0 else y1 + 4
            draw.rectangle([lx, ly, lx+tw, ly+th], fill=_hex_rgba(chex, 210))
            draw.text((lx+2, ly+2), text, fill=(255,255,255,255), font=font)
        else:
            # 灰色虛線（未達門檻）
            dash = max(8, lw*3)
            gray = (160, 160, 160, 120)
            for x in range(x1, x2, dash*2):
                draw.line([(x,y1),(min(x+dash,x2),y1)], fill=gray, width=lw)
                draw.line([(x,y2),(min(x+dash,x2),y2)], fill=gray, width=lw)
            for y in range(y1, y2, dash*2):
                draw.line([(x1,y),(x1,min(y+dash,y2))], fill=gray, width=lw)
                draw.line([(x2,y),(x2,min(y+dash,y2))], fill=gray, width=lw)

    return Image.alpha_composite(img, overlay).convert("RGB")

# ──────────────────────────────────────────────────────────────
#  回饋上傳 Roboflow
# ──────────────────────────────────────────────────────────────
def upload_to_roboflow(pil_image: Image.Image, api_key: str,
                       suggested_label: str = "") -> tuple[bool, str]:
    try:
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fn  = f"feedback_{ts}.jpg"
        tag = suggested_label.replace(" ", "-").replace("_", "-").lower() if suggested_label else "unlabeled"
        url = f"https://api.roboflow.com/dataset/{ROBOFLOW_DATASET}/upload"
        params = {"api_key": api_key, "name": fn, "tag": tag,
                  "split": "train", "batch": "BerryWise-Feedback"}
        files  = {"file": (fn, buf, "image/jpeg")}
        resp   = requests.post(url, params=params, files=files, timeout=30)
        if resp.status_code in (200, 201):
            return True, ""
        try:    err = resp.json().get("error", {}).get("message", resp.text) or resp.text
        except: err = resp.text or f"HTTP {resp.status_code}"
        return False, err
    except Exception as e:
        return False, str(e) or repr(e)

# ──────────────────────────────────────────────────────────────
#  診斷報告文字生成
# ──────────────────────────────────────────────────────────────
def generate_report(predictions, confidence_threshold, mode, manual_override=None):
    now   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    valid = [p for p in predictions if p.get("confidence", 0) >= confidence_threshold]
    below = [p for p in predictions if p.get("confidence", 0) < confidence_threshold]
    HIGH  = 0.75
    lines = [
        "══════════════════════════════",
        "  🍓 草莓園小助手 診斷報告",
        f"  模式：{mode}　{now}",
        "══════════════════════════════", "",
    ]
    if manual_override and manual_override in ADVICE_DB:
        minfo = ADVICE_DB[manual_override]
        ai_lbl  = get_advice(valid[0].get("class","unknown"))["zh_name"] if valid else "無"
        ai_conf = f"{valid[0].get('confidence',0):.1%}" if valid else "—"
        lines += [
            "【診斷來源：農民手動確認】",
            f"  AI 建議：{ai_lbl}（{ai_conf}）→ 農民判斷修正",
            "", f"─── 確認病害 ────────────────",
            f"  病害名稱：{minfo['zh_name']} ({minfo['en_name']})",
            f"  警示等級：{minfo['severity']}", "",
        ]
        if minfo.get("confused_with"):
            lines += [f"  ⚡ 請確認是否排除：{minfo['confused_with']}", ""]
        if minfo.get("visual_check"):
            lines += [minfo["visual_check"], ""]
        lines += [minfo["advice"], ""]
    elif not predictions:
        lines += ["未偵測到病害目標。", "",
                  "【建議】",
                  "  · 拉近距離，讓病斑佔畫面 2/3",
                  "  · 選陰天或遮陰環境拍攝",
                  "  · 確認畫面對焦清晰後重新拍攝"]
    elif not valid:
        best_low = max((p.get("confidence", 0) for p in below), default=0)
        lines += [f"偵測到 {len(below)} 個候選，確信度均未達門檻（最高 {best_low:.1%}）。",
                  "", "【建議】", "  · 拉近距離讓病斑填滿畫面",
                  "  · 側光或散射光拍攝"]
    else:
        valid_s = sorted(valid, key=lambda p: p.get("confidence",0), reverse=True)
        lines += [f"偵測到 {len(valid_s)} 個有效目標（依確信度排列）", ""]
        for i, pred in enumerate(valid_s, 1):
            info = get_advice(pred.get("class", "unknown"))
            conf = pred.get("confidence", 0)
            lines += [
                f"─── #{i} {info['zh_name']} ({info['en_name']})",
                f"  確信度：{conf:.1%}  {'● 高可信' if conf >= HIGH else '○ 待確認'}",
                f"  警示　：{info['severity']}", "",
            ]
            if info.get("confused_with"):
                lines += [f"  ⚡ 易混淆：{info['confused_with']}", ""]
            if info.get("visual_check"):
                lines += [info["visual_check"], ""]
            lines += [info["advice"], ""]
        if below:
            lines += ["─── 其他候選（未達門檻）────"]
            for p in sorted(below, key=lambda p: p.get("confidence",0), reverse=True):
                info = get_advice(p.get("class","unknown"))
                lines.append(f"  · {info['zh_name']}（{p.get('confidence',0):.1%}）")
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

# ──────────────────────────────────────────────────────────────
#  HTML 元件輔助函式
# ──────────────────────────────────────────────────────────────
def _card(content: str, border_color: str = "rgba(255,255,255,0.07)",
          bg: str = "#161616", padding: str = "13px 16px",
          extra_style: str = "") -> None:
    st.markdown(
        f'<div style="background:{bg};border:1px solid {border_color};'
        f'border-radius:14px;padding:{padding};{extra_style}">'
        f'{content}</div>',
        unsafe_allow_html=True,
    )

def _divider():
    st.markdown(
        '<hr style="border:none;border-top:1px solid rgba(255,255,255,0.06);'
        'margin:16px 0;">',
        unsafe_allow_html=True,
    )

def _section(title: str):
    st.markdown(
        f'<div style="font-size:12px;font-weight:700;color:rgba(255,255,255,0.3);'
        f'letter-spacing:1.2px;text-transform:uppercase;margin:16px 0 8px;">'
        f'{title}</div>',
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────
#  主程式
# ──────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="草莓園小助手",
        page_icon="🍓",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    # 注入 PWA + CSS
    st.markdown(f"<head>{PWA_HTML}</head>", unsafe_allow_html=True)
    st.markdown(PWA_BANNER_HTML, unsafe_allow_html=True)
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # Session 初始化
    _defaults = {
        "mode": 0, "inp": 0,
        "diagnosis_data": None, "feedback_state": None,
        "manual_disease": None, "confirmed_disease": None,
        "groq_suggestion": None, "upload_error": None,
        "_upload_cycle": 0,
        "conf_thresh": 0.55,
        "enable_groq": True,
        "enable_enhance": False,
    }
    for k, v in _defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    api_key  = get_api_key()
    groq_key = get_groq_key()

    # ════════════════════════════════════════════════════════
    #  標題
    # ════════════════════════════════════════════════════════
    st.markdown("""
    <div style="text-align:center;padding:18px 0 12px;">
      <div style="font-size:40px;margin-bottom:6px;">🍓</div>
      <div style="font-size:20px;font-weight:800;color:rgba(255,255,255,0.92);
        letter-spacing:-0.5px;">草莓園小助手</div>
      <div style="font-size:11px;color:rgba(255,255,255,0.28);letter-spacing:1.2px;
        margin-top:3px;">BerryWise · AI 病害診斷</div>
    </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    #  模式選擇（葉片 / 果實）
    # ════════════════════════════════════════════════════════
    def _set_mode(m):
        st.session_state.mode = m
        st.session_state.diagnosis_data = None

    c1, c2 = st.columns(2)
    with c1:
        st.button("🌿 葉片診斷", use_container_width=True,
                  type="primary" if st.session_state.mode == 0 else "secondary",
                  on_click=_set_mode, args=(0,))
    with c2:
        st.button("🍓 果實分析", use_container_width=True,
                  type="primary" if st.session_state.mode == 1 else "secondary",
                  on_click=_set_mode, args=(1,))

    mode = "🌿 葉片診斷" if st.session_state.mode == 0 else "🍓 果實分析"

    # 模式提示條
    tip_map = {
        0: ("對準草莓葉片拍攝，確保病斑在畫面中央",
            "rgba(39,174,96,0.08)", "rgba(39,174,96,0.2)", "#6dbe91"),
        1: ("在自然光下對準果實拍攝，避免強烈背光",
            "rgba(192,57,43,0.08)", "rgba(192,57,43,0.2)", "#d9847a"),
    }
    tip_txt, tip_bg, tip_bd, tip_cl = tip_map[st.session_state.mode]
    st.markdown(
        f'<div style="background:{tip_bg};border:1px solid {tip_bd};border-radius:9px;'
        f'padding:6px 14px;font-size:12px;color:{tip_cl};text-align:center;margin:5px 0 10px;">'
        f'{tip_txt}</div>',
        unsafe_allow_html=True,
    )

    # ════════════════════════════════════════════════════════
    #  設定（折疊，手機友善）
    # ════════════════════════════════════════════════════════
    with st.expander("⚙️ 診斷設定", expanded=False):
        st.session_state.conf_thresh = st.slider(
            "確信度門檻", 0.10, 1.0,
            st.session_state.conf_thresh, 0.05,
            format="%.2f",
            help="建議 0.50～0.65；過低會產生誤報",
        )
        col_g, col_e = st.columns(2)
        with col_g:
            st.session_state.enable_groq = st.toggle(
                "🧠 Groq 交叉驗證", value=st.session_state.enable_groq,
                help="llama-3.2-11b-vision-preview · 完全免費",
            )
        with col_e:
            st.session_state.enable_enhance = st.toggle(
                "✨ 強光優化", value=st.session_state.enable_enhance,
                help="戶外強光拍攝時開啟（需 cv2）",
            )
        st.markdown(
            '<div style="font-size:11px;color:rgba(255,255,255,0.2);padding:4px 0 0;">'
            '📱 手機安裝：iOS → Safari 分享 → 加入主畫面　·　'
            'Android → Chrome → 安裝應用程式</div>',
            unsafe_allow_html=True,
        )

    confidence_threshold = st.session_state.conf_thresh
    enable_groq          = st.session_state.enable_groq
    enable_enhance       = st.session_state.enable_enhance

    # ════════════════════════════════════════════════════════
    #  輸入方式切換
    # ════════════════════════════════════════════════════════
    def _set_inp(i):
        st.session_state.inp = i
        st.session_state.diagnosis_data = None
        st.session_state.feedback_state = None

    i1, i2 = st.columns(2)
    with i1:
        st.button("📷 即時拍照", use_container_width=True,
                  type="primary" if st.session_state.inp == 0 else "secondary",
                  on_click=_set_inp, args=(0,))
    with i2:
        st.button("📁 上傳圖片", use_container_width=True,
                  type="primary" if st.session_state.inp == 1 else "secondary",
                  on_click=_set_inp, args=(1,))

    # ════════════════════════════════════════════════════════
    #  圖片輸入
    # ════════════════════════════════════════════════════════
    captured_image = None
    _cycle = st.session_state._upload_cycle
    if st.session_state.inp == 0:
        cam = st.camera_input("拍攝", label_visibility="collapsed",
                               key=f"cam_{_cycle}")
        if cam:
            captured_image = Image.open(cam).convert("RGB")
    else:
        upl = st.file_uploader(
            "上傳", type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed", key=f"upl_{_cycle}",
        )
        if upl:
            captured_image = Image.open(upl).convert("RGB")

    # ════════════════════════════════════════════════════════
    #  有圖片 → 診斷流程
    # ════════════════════════════════════════════════════════
    if captured_image is not None:

        # 影像前處理
        if enable_enhance:
            col_a, col_b = st.columns(2)
            with col_a:
                st.caption("📷 原始")
                st.image(captured_image, use_container_width=True)
            processed_image = enhance_outdoor_image(captured_image)
            with col_b:
                st.caption("✨ 優化後")
                st.image(processed_image, use_container_width=True)
        else:
            processed_image = captured_image
            st.image(captured_image, use_container_width=True)

        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

        # ── 診斷按鈕 ──────────────────────────────────────
        if st.button("🔬 開始 AI 診斷", type="primary",
                     use_container_width=True, key="diagnose_btn"):
            for k in ["feedback_state", "manual_disease", "confirmed_disease",
                      "groq_suggestion", "upload_error", "diagnosis_data"]:
                st.session_state[k] = None

            # Loading UI
            loading_ph = st.empty()
            loading_ph.markdown("""
            <div style="background:#161616;border:1px solid rgba(192,57,43,0.28);
              border-radius:16px;padding:22px 18px;text-align:center;margin:10px 0;">
              <div style="font-size:26px;margin-bottom:10px;">🔬</div>
              <div style="font-size:14px;color:rgba(255,255,255,0.82);font-weight:700;
                margin-bottom:4px;">Roboflow YOLO 分析中...</div>
              <div style="font-size:11px;color:rgba(255,255,255,0.3);margin-bottom:16px;">
                物件偵測 + Groq Vision 交叉驗證</div>
              <div style="height:4px;background:rgba(255,255,255,0.05);border-radius:3px;
                overflow:hidden;position:relative;">
                <div style="position:absolute;height:100%;width:45%;
                  background:linear-gradient(90deg,transparent,#C0392B,transparent);
                  border-radius:3px;animation:_scan 1.4s ease-in-out infinite;left:-45%;"></div>
              </div>
            </div>
            <style>
              @keyframes _scan{0%{left:-45%}100%{left:100%}}
            </style>
            """, unsafe_allow_html=True)

            # ① Roboflow 推論
            result = run_inference(processed_image, api_key,
                                   DEFAULT_MODEL_ID, confidence_threshold)

            # ② Groq Vision（若有結果且啟用）
            groq_result = {"groq_zh": "", "agree": None}
            if result is not None and groq_key and enable_groq:
                preds_s  = sorted(result.get("predictions", []),
                                  key=lambda p: p.get("confidence", 0), reverse=True)
                yolo_top = preds_s[0] if preds_s else None
                groq_result = ask_groq_vision(
                    processed_image, mode, yolo_top, groq_key)

            st.session_state.groq_suggestion = groq_result
            if result is not None:
                preds = result.get("predictions", [])
                st.session_state.diagnosis_data = {
                    "result": result,
                    "predictions": preds,
                    "predictions_sorted": sorted(
                        preds, key=lambda p: p.get("confidence", 0), reverse=True),
                    "processed_image": processed_image,
                }
            loading_ph.empty()
            st.rerun()

        # ════════════════════════════════════════════════════
        #  診斷結果顯示
        # ════════════════════════════════════════════════════
        diag = st.session_state.get("diagnosis_data")
        if diag:
            predictions        = diag["predictions"]
            predictions_sorted = diag["predictions_sorted"]
            processed_image    = diag["processed_image"]
            HIGH_CONF          = 0.75
            valid_preds = [p for p in predictions_sorted
                           if p.get("confidence", 0) >= confidence_threshold]
            high_preds  = [p for p in valid_preds
                           if p.get("confidence", 0) >= HIGH_CONF]

            # ── ① 統計卡 ──────────────────────────────────
            _divider()
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.metric("偵測到", len(predictions))
            with mc2:
                st.metric("有效框", len(valid_preds))
            with mc3:
                best = max((p.get("confidence", 0) for p in valid_preds), default=None)
                st.metric("最高確信度", f"{best:.0%}" if best else "—")

            st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

            # ── ② BBox 標注圖（核心顯示）──────────────────
            annotated = draw_detections(
                processed_image.copy(), predictions, confidence_threshold)
            st.markdown(
                '<div style="font-size:11px;color:rgba(255,255,255,0.28);'
                'margin-bottom:4px;">🎯 彩色框 = 有效偵測 · 灰色虛線 = 未達門檻</div>',
                unsafe_allow_html=True,
            )
            st.image(annotated, use_container_width=True)

            # ── ③ 低確信度手動確認 ────────────────────────
            if valid_preds and not high_preds:
                best_conf = max(p.get("confidence", 0) for p in valid_preds)
                tip_photo = ("拉近至 20cm、病斑佔畫面 2/3、避免強烈背光"
                             if best_conf < 0.65 else "建議複拍一張確認")
                st.markdown(f"""
                <div style="background:rgba(255,165,0,0.07);border:1px solid rgba(255,165,0,0.28);
                  border-radius:12px;padding:11px 14px;margin:8px 0;">
                  <div style="color:#f5a623;font-weight:700;font-size:13px;margin-bottom:3px;">
                    ⚠️ 確信度偏低（{best_conf:.0%}）— 請手動確認病害</div>
                  <div style="font-size:11px;color:rgba(255,200,100,0.7);">📷 {tip_photo}</div>
                </div>""", unsafe_allow_html=True)

                leaf_d  = ["angular leafspot", "leaf spot", "powdery mildew leaf"]
                fruit_d = ["anthracnose fruit rot", "blossom blight",
                           "gray mold", "powdery mildew fruit"]
                relevant = leaf_d if st.session_state.mode == 0 else fruit_d
                others   = [k for k in ADVICE_DB if k not in relevant]

                st.markdown(
                    '<div style="font-size:12px;color:rgba(255,255,255,0.38);'
                    'margin:10px 0 6px;">👆 您目測判斷是哪種病害？</div>',
                    unsafe_allow_html=True,
                )
                manual_cols = st.columns(len(relevant))
                for i, key in enumerate(relevant):
                    info = ADVICE_DB[key]
                    sel  = st.session_state.get("manual_disease") == key
                    with manual_cols[i]:
                        def _set_manual(k=key):
                            st.session_state.manual_disease = k
                            st.rerun()
                        st.button(
                            f"{'✓ ' if sel else ''}{info['zh_name']}",
                            key=f"btn_m_{key}",
                            type="primary" if sel else "secondary",
                            use_container_width=True,
                            on_click=_set_manual,
                        )

                with st.expander("🔍 其他病害選項"):
                    oc = st.columns(2)
                    for i, key in enumerate(others):
                        info = ADVICE_DB[key]
                        sel  = st.session_state.get("manual_disease") == key
                        with oc[i % 2]:
                            def _set_other(k=key):
                                st.session_state.manual_disease = k
                                st.rerun()
                            st.button(
                                f"{'✓ ' if sel else ''}{info['zh_name']}",
                                key=f"btn_o_{key}",
                                type="primary" if sel else "secondary",
                                use_container_width=True,
                                on_click=_set_other,
                            )

                if st.session_state.get("manual_disease"):
                    if st.button("✕ 清除手動選擇", key="clear_manual"):
                        st.session_state.manual_disease = None
                        st.rerun()

            # ── ④ 病害結果卡 ──────────────────────────────
            if valid_preds:
                _divider()
                _section("診斷結果")
                shown = set()
                for pred in predictions_sorted:
                    conf  = pred.get("confidence", 0)
                    label = pred.get("class", "unknown")
                    if conf < confidence_threshold or label in shown:
                        continue
                    shown.add(label)
                    info   = get_advice(label)
                    bbox_n = sum(1 for p in valid_preds
                                 if p.get("class", "") == label)
                    chex   = info.get("color", "#3498DB")
                    conf_p = f"{conf:.0%}"
                    conf_lbl = ("● 高可信" if conf >= HIGH_CONF else
                                "○ 待確認" if conf >= confidence_threshold
                                else "△ 建議複拍")

                    st.markdown(f"""
                    <div style="background:#161616;border:1px solid {chex}44;
                      border-left:4px solid {chex};border-radius:14px;
                      padding:13px 15px;margin:8px 0;">
                      <div style="display:flex;align-items:center;
                        justify-content:space-between;margin-bottom:10px;">
                        <div>
                          <span style="font-size:18px;">{info.get('icon','🌿')}</span>
                          <span style="font-size:15px;font-weight:700;
                            color:rgba(255,255,255,0.92);margin-left:7px;">
                            {info['zh_name']}</span><br>
                          <span style="font-size:10px;color:rgba(255,255,255,0.28);
                            margin-left:26px;">{info['en_name']}</span>
                        </div>
                        <div style="background:{chex}1a;border:1px solid {chex}44;
                          border-radius:11px;padding:5px 13px;text-align:center;
                          min-width:58px;flex-shrink:0;">
                          <div style="font-size:18px;font-weight:800;color:{chex};
                            line-height:1.1;">{conf_p}</div>
                          <div style="font-size:9px;color:rgba(255,255,255,0.28);">
                            {conf_lbl}</div>
                        </div>
                      </div>
                      <div style="display:flex;gap:6px;flex-wrap:wrap;">
                        <span style="font-size:11px;padding:2px 10px;border-radius:6px;
                          background:{chex}18;color:{chex};
                          border:1px solid {chex}33;font-weight:600;">
                          {info['severity']}</span>
                        <span style="font-size:11px;padding:2px 10px;border-radius:6px;
                          background:rgba(39,174,96,0.08);color:#6dbe91;
                          border:1px solid rgba(39,174,96,0.2);">
                          🎯 {bbox_n} 個框標定</span>
                      </div>
                    </div>""", unsafe_allow_html=True)

                    with st.expander(f"🌿 {info['zh_name']} — 目測特徵與農事建議"):
                        if info.get("confused_with"):
                            st.markdown(f"""
                            <div style="background:rgba(255,165,0,0.07);
                              border:1px solid rgba(255,165,0,0.22);border-radius:8px;
                              padding:8px 12px;margin-bottom:8px;font-size:12px;
                              color:#f5a623;">
                              ⚡ 請先確認排除：{info['confused_with']}
                            </div>""", unsafe_allow_html=True)
                        if info.get("visual_check"):
                            st.code(info["visual_check"], language=None)
                        st.code(info["advice"], language=None)

            elif predictions:
                _card(
                    '<div style="color:#5dade2;font-weight:700;font-size:13px;'
                    'margin-bottom:4px;">🔍 偵測到候選，但確信度未達門檻</div>'
                    '<div style="font-size:12px;color:rgba(150,210,240,0.75);">'
                    '可調低設定中的確信度滑桿，或複拍後重新診斷</div>',
                    border_color="rgba(52,152,219,0.25)",
                    bg="rgba(52,152,219,0.06)",
                )
            else:
                uncovered = UNCOVERED[st.session_state.mode]
                _card(
                    '<div style="color:#5dade2;font-weight:700;font-size:13px;'
                    'margin-bottom:4px;">🔍 未偵測到已知病害</div>'
                    '<div style="font-size:12px;color:rgba(150,210,240,0.75);">'
                    '可能為模型未涵蓋的病害型別，請對照下方目測特徵</div>',
                    border_color="rgba(52,152,219,0.25)",
                    bg="rgba(52,152,219,0.06)",
                )
                for d in uncovered:
                    with st.expander(f"📌 {d['zh_name']}　{d['en_name']}"):
                        st.markdown(
                            f'<div style="font-size:13px;color:rgba(255,255,255,0.72);'
                            f'line-height:1.8;"><b style="font-size:11px;'
                            f'color:rgba(255,255,255,0.35);">目測特徵</b><br>'
                            f'{d["visual"]}<br><br>'
                            f'<b style="font-size:11px;color:rgba(255,255,255,0.35);">'
                            f'處置方向</b><br>{d["action"]}</div>',
                            unsafe_allow_html=True,
                        )

            # ── ⑤ Groq Vision 交叉驗證 ────────────────────
            groq_result = st.session_state.get("groq_suggestion") or {}
            groq_txt    = groq_result.get("groq_zh", "") if isinstance(groq_result, dict) else ""
            if groq_txt:
                agree = groq_result.get("agree", None) if isinstance(groq_result, dict) else None
                badge_map = {
                    True:  ('<span style="background:rgba(39,174,96,0.15);color:#6dbe91;'
                            'border:1px solid rgba(39,174,96,0.28);border-radius:6px;'
                            'padding:3px 10px;font-size:11px;font-weight:600;">'
                            '✅ 視覺模型一致</span>'),
                    False: ('<span style="background:rgba(255,165,0,0.12);color:#f5a623;'
                            'border:1px solid rgba(255,165,0,0.28);border-radius:6px;'
                            'padding:3px 10px;font-size:11px;font-weight:600;">'
                            '⚠️ 視覺模型有異議</span>'),
                    None:  ('<span style="background:rgba(52,152,219,0.12);color:#5dade2;'
                            'border:1px solid rgba(52,152,219,0.28);border-radius:6px;'
                            'padding:3px 10px;font-size:11px;font-weight:600;">'
                            '👁️ 視覺模型獨立判斷</span>'),
                }
                badge_html = badge_map.get(agree, badge_map[None])
                escaped    = (groq_txt.replace("<","&lt;").replace(">","&gt;")
                              .replace("\n","<br>"))
                _divider()
                st.markdown(f"""
                <div style="background:rgba(155,89,182,0.07);
                  border:1px solid rgba(155,89,182,0.25);
                  border-radius:14px;padding:13px 15px;margin:6px 0;">
                  <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
                    <span style="font-size:15px;">🧠</span>
                    <span style="font-size:12px;color:rgba(255,255,255,0.35);flex:1;">
                      Groq Vision（llama-3.2-11b-vision · 免費）</span>
                    {badge_html}
                  </div>
                  <div style="font-size:13px;line-height:1.8;
                    color:rgba(255,255,255,0.78);">{escaped}</div>
                </div>""", unsafe_allow_html=True)

            # ── ⑥ 農民回饋（2 步驟）────────────────────────
            _divider()
            _section("📊 診斷回饋 — 協助 AI 進化")
            st.markdown(
                '<div style="font-size:11px;color:rgba(255,255,255,0.25);'
                'margin-bottom:10px;">回饋圖片將上傳 Roboflow 訓練資料集（完全免費）</div>',
                unsafe_allow_html=True,
            )

            fb = st.session_state.get("feedback_state")

            if fb == "uploaded":
                st.markdown("""
                <div style="background:rgba(39,174,96,0.06);
                  border:1px solid rgba(39,174,96,0.2);border-radius:14px;
                  padding:18px;text-align:center;">
                  <div style="font-size:24px;margin-bottom:8px;">🎉</div>
                  <div style="color:#6dbe91;font-weight:700;font-size:14px;
                    margin-bottom:4px;">感謝您的回饋！</div>
                  <div style="font-size:11px;color:rgba(255,255,255,0.3);">
                    圖片已上傳至 Roboflow（BerryWise-Feedback）<br>
                    預計下次訓練後生效</div>
                </div>""", unsafe_allow_html=True)

            elif fb == "upload_fail":
                err_msg = st.session_state.get("upload_error", "未知錯誤")
                st.error(f"上傳失敗：{err_msg}")
                def _retry():
                    top_lbl = (predictions_sorted[0].get("class", "")
                               if predictions_sorted else "")
                    ok, err = upload_to_roboflow(
                        processed_image, api_key, suggested_label=top_lbl)
                    st.session_state.feedback_state = "uploaded" if ok else "upload_fail"
                    st.session_state.upload_error   = err if not ok else None
                st.button("⚠️ 重試上傳", key="retry_upload",
                          use_container_width=True, on_click=_retry)

            else:
                # 步驟一：正確 / 不正確
                fb1, fb2 = st.columns(2)
                with fb1:
                    def _fb_ok():
                        top_lbl = (predictions_sorted[0].get("class", "")
                                   if predictions_sorted else "")
                        ok, err = upload_to_roboflow(
                            processed_image, api_key, suggested_label=top_lbl)
                        st.session_state.feedback_state = "uploaded" if ok else "upload_fail"
                        st.session_state.upload_error   = err if not ok else None
                    st.button("✅ 正確，上傳協助訓練",
                              use_container_width=True, key="fb_ok",
                              type="primary", on_click=_fb_ok)
                with fb2:
                    def _fb_no():
                        st.session_state.feedback_state = "picking"
                    st.button("❌ 不正確，手動標注",
                              use_container_width=True, key="fb_no",
                              type="secondary", on_click=_fb_no)

                # 步驟二：選擇正確病害
                if fb == "picking":
                    st.markdown(
                        '<div style="font-size:12px;color:rgba(255,255,255,0.38);'
                        'margin:10px 0 6px;">👆 請選擇實際病害：</div>',
                        unsafe_allow_html=True,
                    )
                    confirmed = st.session_state.get("confirmed_disease")
                    pick_cols = st.columns(2)
                    for i, (key, info) in enumerate(ADVICE_DB.items()):
                        sel = confirmed == key
                        with pick_cols[i % 2]:
                            def _pick(k=key):
                                st.session_state.confirmed_disease = k
                                st.rerun()
                            st.button(
                                f"{'✓ ' if sel else ''}{info['icon']} {info['zh_name']}",
                                key=f"pick_{key}",
                                type="primary" if sel else "secondary",
                                use_container_width=True,
                                on_click=_pick,
                            )
                    if confirmed:
                        def _submit():
                            ok, err = upload_to_roboflow(
                                processed_image, api_key,
                                suggested_label=confirmed)
                            st.session_state.feedback_state = "uploaded" if ok else "upload_fail"
                            st.session_state.upload_error   = err if not ok else None
                        st.button("📤 確認並上傳回饋", key="fb_submit",
                                  type="primary", use_container_width=True,
                                  on_click=_submit)

            # ── ⑦ 診斷報告 ────────────────────────────────
            _divider()
            _section("📄 診斷報告")
            report_override = (st.session_state.get("confirmed_disease") or
                               st.session_state.get("manual_disease"))
            report_text   = generate_report(
                predictions, confidence_threshold, mode,
                manual_override=report_override)
            edited_report = st.text_area(
                "報告內容", value=report_text, height=290,
                label_visibility="collapsed",
            )
            filename = (f"草莓園小助手_"
                        f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

            def _reset_all():
                for k in ["diagnosis_data", "confirmed_disease", "feedback_state",
                          "manual_disease", "groq_suggestion", "upload_error"]:
                    st.session_state[k] = None
                st.session_state._upload_cycle = (
                    st.session_state.get("_upload_cycle", 0) + 1)
                st.rerun()

            rr1, rr2, rr3 = st.columns(3)
            with rr1:
                st.download_button(
                    "💾 下載報告",
                    data=edited_report.encode("utf-8"),
                    file_name=filename, mime="text/plain",
                    key="dl_report", use_container_width=True,
                )
            with rr2:
                if st.button("📋 複製", key="copy_btn", use_container_width=True):
                    st.session_state["_copy_txt"] = edited_report
                    st.rerun()
            with rr3:
                st.button("🔄 重新診斷", key="refresh_btn",
                          use_container_width=True, on_click=_reset_all)

            if st.session_state.get("_copy_txt"):
                # 確認 session 值為純字串再注入，防止 session 被污染後執行任意 JS
                copy_val = st.session_state["_copy_txt"]
                if isinstance(copy_val, str):
                    esc = json.dumps(copy_val)
                    st.components.v1.html(
                        f"<script>(function(){{var t={esc};"
                        f"if(navigator.clipboard)navigator.clipboard.writeText(t);"
                        f"}})();</script>",
                        height=0,
                    )
                    st.toast("✓ 已複製到剪貼簿", icon="📋")
                del st.session_state["_copy_txt"]

    # ════════════════════════════════════════════════════════
    #  空狀態（未上傳圖片）
    # ════════════════════════════════════════════════════════
    else:
        st.markdown("""
        <div style="text-align:center;padding:52px 20px 32px;">
          <div style="font-size:52px;margin-bottom:16px;opacity:0.18;">📷</div>
          <div style="font-size:15px;color:rgba(255,255,255,0.3);
            font-weight:600;margin-bottom:6px;">
            請拍照或上傳草莓影像</div>
          <div style="font-size:12px;color:rgba(255,255,255,0.16);
            line-height:1.7;">
            支援 JPG、PNG、WEBP<br>
            對準病斑，距離約 15–20 cm
          </div>
        </div>""", unsafe_allow_html=True)

        # 拍攝技巧卡
        _card(
            '<div style="font-size:12px;color:rgba(255,255,255,0.45);'
            'font-weight:600;margin-bottom:8px;">📷 拍攝小技巧</div>'
            '<div style="font-size:12px;color:rgba(255,255,255,0.3);line-height:1.9;">'
            '· 距離 15–20cm，病斑填滿畫面 2/3<br>'
            '· 選擇陰天或遮陰，避免強烈直射光<br>'
            '· 手持穩定，確認畫面對焦清晰<br>'
            '· 葉片建議拍攝背面效果更佳'
            '</div>',
            padding="12px 15px",
        )

    # Footer
    st.markdown(
        '<div style="text-align:center;padding:20px 0 8px;opacity:0.2;">'
        '<div style="font-size:11px;">BerryWise v4.0 · mAP 91.6%</div>'
        '<div style="font-size:10px;margin-top:2px;">7 種病害 · 2,462 訓練樣本</div>'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
