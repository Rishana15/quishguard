from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
import pickle
import os
import base64
import re
from urllib.parse import urlparse

app = Flask(__name__, static_folder='static')
CORS(app)

MODEL_PATH = 'model.pkl'
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"[✓] Model loaded from {MODEL_PATH}")
else:
    print(f"[!] model.pkl not found — using dummy model for testing")


SAFE_DOMAINS = {
    'google.com', 'youtube.com', 'facebook.com', 'instagram.com',
    'twitter.com', 'x.com', 'linkedin.com', 'github.com',
    'microsoft.com', 'apple.com', 'amazon.com', 'wikipedia.org',
    'stackoverflow.com', 'reddit.com', 'whatsapp.com', 'zoom.us',
    'teams.microsoft.com', 'docs.google.com', 'drive.google.com',
    'maps.google.com', 'paypal.com', 'netflix.com',
}

# Each pattern has: (regex, human readable label, severity)
PHISHING_PATTERNS = [
    (r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '🔴 Raw IP address used instead of domain name', 'high'),
    (r'secure.{0,10}login',   '🔴 Suspicious keyword: secure-login', 'high'),
    (r'verify.{0,10}account', '🔴 Suspicious keyword: verify-account', 'high'),
    (r'account.{0,10}suspended', '🔴 Suspicious keyword: account-suspended', 'high'),
    (r'update.{0,10}payment', '🔴 Suspicious keyword: update-payment', 'high'),
    (r'confirm.{0,10}identity', '🔴 Suspicious keyword: confirm-identity', 'high'),
    (r'password|passwd|pwd',  '🔴 Password keyword found in URL', 'high'),
    (r'\.tk$|\.ml$|\.ga$|\.cf$|\.gq$', '🟠 Free suspicious domain extension (.tk/.ml/.ga)', 'medium'),
    (r'paypal.{0,20}\.(?!com)', '🔴 Fake PayPal domain detected', 'high'),
    (r'apple.{0,20}\.(?!com)',  '🔴 Fake Apple domain detected', 'high'),
    (r'google.{0,20}\.(?!com)', '🔴 Fake Google domain detected', 'high'),
    (r'microsoft.{0,20}\.(?!com)', '🔴 Fake Microsoft domain detected', 'high'),
    (r'bit\.ly|tinyurl|goo\.gl', '🟠 URL shortener used (hides real destination)', 'medium'),
    (r'@', '🔴 @ symbol in URL (classic phishing trick)', 'high'),
    (r'login.{0,10}redirect', '🟠 Suspicious login redirect detected', 'medium'),
    (r'token=.{10,}',         '🟡 Long token parameter in URL', 'low'),
    (r'bank.{0,20}secure',    '🔴 Suspicious keyword: bank-secure', 'high'),
    (r'urgent.{0,10}action',  '🟠 Urgency tactic: urgent-action', 'medium'),
    (r'free.{0,10}gift',      '🟠 Social engineering: free-gift', 'medium'),
    (r'winner.{0,10}prize',   '🟠 Social engineering: winner-prize', 'medium'),
    (r'\.ru/|\.cn/|\.xyz/',   '🟡 Suspicious country/generic TLD', 'low'),
]

def analyze_url(url_text):
    if not url_text or len(url_text.strip()) < 4:
        return 'Unknown', 'No URL detected', 0.0, []

    url = url_text.strip().lower()
    flags = []

    # check safe domains
    try:
        parsed = urlparse(url if url.startswith('http') else 'https://' + url)
        domain = parsed.netloc.lower().replace('www.', '').split(':')[0]
        if domain in SAFE_DOMAINS or any(domain.endswith('.' + s) for s in SAFE_DOMAINS):
            return 'Safe', f'trusted domain: {domain}', -1.0, [f'✅ Verified trusted domain: {domain}']
    except:
        pass

    # check phishing patterns — collect ALL flags
    for pattern, label, severity in PHISHING_PATTERNS:
        if re.search(pattern, url, re.IGNORECASE):
            flags.append(label)

    if len(flags) >= 2:
        return 'Malicious', f'{len(flags)} phishing indicators detected', 1.0, flags
    elif len(flags) == 1:
        return 'Malicious', 'phishing indicator detected', 0.5, flags

    # structural checks
    suspicion_score = 0
    if len(url) > 100:
        suspicion_score += 1
        flags.append('🟡 Unusually long URL')
    try:
        parts = urlparse(url if url.startswith('http') else 'https://' + url).netloc.split('.')
        if len(parts) > 4:
            suspicion_score += 1
            flags.append('🟠 Too many subdomains')
    except:
        pass
    if '?' in url and len(url.split('?')[1]) > 50:
        suspicion_score += 1
        flags.append('🟡 Suspiciously long query parameters')

    if not url.startswith('http') and '.' not in url:
        return 'Unknown', 'not a URL', 0.0, []

    if suspicion_score >= 2:
        return 'Malicious', 'suspicious URL structure', 0.3, flags

    return 'Unknown', 'no clear signals', 0.0, flags


def combine_predictions(pixel_pred, pixel_prob, url_verdict, url_reason):
    if url_verdict == 'Safe':
        return 'Safe', f'URL verified safe ({url_reason})', 0.95
    elif url_verdict == 'Malicious' and pixel_prob > 0.6:
        return 'Malicious', f'Both pixel pattern and URL analysis flagged ({url_reason})', min(0.99, pixel_prob + 0.1)
    elif url_verdict == 'Malicious':
        return 'Malicious', f'URL analysis flagged ({url_reason})', 0.82
    elif pixel_prob > 0.85:
        return 'Malicious', 'Pixel pattern analysis flagged', pixel_prob
    else:
        return 'Safe', 'No threats detected', 1.0 - pixel_prob


def preprocess_qr(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    img_resized = cv2.resize(img, (69, 69))
    img_norm    = img_resized / 255.0
    features    = img_norm.flatten().reshape(1, -1)
    return features, img_resized


def generate_heatmap(img_69, features, prediction_label):
    H, W = 69, 69
    OUTPUT_SIZE = 320

    if model is not None and hasattr(model, 'feature_importances_'):
        base_importance = model.feature_importances_
    else:
        flat = features.flatten()
        base_importance = np.abs(flat - flat.mean())

    pixel_vals = features.flatten()
    combined   = base_importance * (0.4 + 0.6 * pixel_vals)
    imp_map    = combined.reshape(H, W).astype(np.float32)
    imp_map    = cv2.GaussianBlur(imp_map, (7, 7), 0)
    p80        = np.percentile(imp_map, 80)
    imp_map    = np.clip((imp_map - p80) * 3 + p80, imp_map.min(), imp_map.max())
    mn, mx     = imp_map.min(), imp_map.max()
    imp_norm   = ((imp_map - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8)

    colormap      = cv2.COLORMAP_BONE if prediction_label == 'Malicious' else cv2.COLORMAP_OCEAN
    heatmap_color = cv2.applyColorMap(imp_norm, colormap)
    qr_big        = cv2.resize(img_69, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_NEAREST)
    heat_big      = cv2.resize(heatmap_color, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_LANCZOS4)
    qr_bgr        = cv2.cvtColor(qr_big, cv2.COLOR_GRAY2BGR)
    qr_dark       = cv2.convertScaleAbs(qr_bgr, alpha=0.5, beta=0)
    overlay       = cv2.addWeighted(qr_dark, 0.35, heat_big, 0.65, 0)

    if prediction_label == 'Malicious':
        overlay = cv2.copyMakeBorder(overlay, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(80, 80, 220))
        overlay = cv2.copyMakeBorder(overlay, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(40, 40, 180))
    else:
        overlay = cv2.copyMakeBorder(overlay, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(180, 220, 80))
        overlay = cv2.copyMakeBorder(overlay, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(0, 160, 60))

    _, buffer = cv2.imencode('.png', overlay)
    return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"


def get_region_label(imp_map):
    zones = {
        'top-left finder pattern':     imp_map[0:20,  0:20],
        'top-right finder pattern':    imp_map[0:20,  49:69],
        'bottom-left finder pattern':  imp_map[49:69, 0:20],
        'timing pattern (horizontal)': imp_map[6:7,   20:49],
        'timing pattern (vertical)':   imp_map[20:49, 6:7],
        'data region (center)':        imp_map[20:49, 20:49],
        'format information':          imp_map[0:9,   20:29],
    }
    return max(zones, key=lambda z: zones[z].mean())


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image received'}), 400

    image_file  = request.files['image']
    qr_text     = request.form.get('qr_text', '')
    image_bytes = image_file.read()

    features, img_69 = preprocess_qr(image_bytes)
    if features is None:
        return jsonify({'error': 'Could not process image'}), 400

    if model is not None:
        proba         = model.predict_proba(features)[0]
        phishing_prob = float(proba[1])
        pixel_pred    = 'Malicious' if phishing_prob > 0.85 else 'Safe'
    else:
        phishing_prob = 0.3
        pixel_pred    = 'Safe'

    url_verdict, url_reason, _, url_flags = analyze_url(qr_text)
    prediction_label, detection_reason, confidence = combine_predictions(
        pixel_pred, phishing_prob, url_verdict, url_reason
    )
    heatmap_b64 = generate_heatmap(img_69, features, prediction_label)

    imp_map = model.feature_importances_.reshape(69, 69) if (
        model is not None and hasattr(model, 'feature_importances_')
    ) else np.abs(features.flatten() - features.flatten().mean()).reshape(69, 69)

    suspicious_zone = get_region_label(imp_map)

    print(f"[→] Pixel model   : {pixel_pred} ({phishing_prob:.2%})")
    print(f"[→] URL verdict   : {url_verdict} — {url_reason}")
    print(f"[→] URL flags     : {url_flags}")
    print(f"[→] FINAL         : {prediction_label} ({confidence:.2%})")

    return jsonify({
        'prediction':       prediction_label,
        'confidence':       confidence,
        'qr_text':          qr_text,
        'heatmap':          heatmap_b64,
        'suspicious_zone':  suspicious_zone,
        'detection_reason': detection_reason,
        'url_flags':        url_flags,
        'pixel_prob':       round(phishing_prob * 100, 1),
    })


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  QuishGuard Flask Server")
    print("="*50)
    print("  Local:   http://localhost:5000")
    print("="*50 + "\n")
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port)
