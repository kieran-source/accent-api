from flask import Flask, request, jsonify
import tempfile
import subprocess
import os
import json

app = Flask(__name__)

# Load model once at startup
from speechbrain.pretrained import EncoderClassifier

print("Loading accent classifier model...")
classifier = EncoderClassifier.from_hparams(
    source="Jzuluaga/accent-id-commonaccent_ecapa",
    savedir="/app/model_cache"
)
print("Model loaded!")

# Accent groups for matching — only broad groups where interchangeability is acceptable.
# British Isles is grouped: SpeechBrain's regional accuracy isn't reliable enough to
# enforce exact-match for scotland vs ireland vs wales vs england. The gate's job is
# to catch broad-category mismatches (UK vs US vs India), not regional drift.
# Manual spot-check + /regen-scene handles regional issues.
ACCENT_GROUPS = {
    'american': ['us', 'canada'],
    'british_isles': ['england', 'scotland', 'wales', 'ireland'],
}

def extract_accent_requirement(voice_description):
    """Extract the likely intended accent from voice description"""
    voice_lower = voice_description.lower()
    
    if any(x in voice_lower for x in ['scottish', 'scots', 'glasgow', 'edinburgh']):
        return 'scotland'
    elif any(x in voice_lower for x in ['irish', 'dublin', 'cork']):
        return 'ireland'
    elif any(x in voice_lower for x in ['welsh', 'cardiff']):
        return 'wales'
    elif any(x in voice_lower for x in ['british', 'english', 'london', 'uk accent', 'uk']):
        return 'england'
    elif any(x in voice_lower for x in ['american', 'us accent', 'usa']):
        return 'us'
    elif any(x in voice_lower for x in ['australian', 'aussie']):
        return 'australia'
    elif any(x in voice_lower for x in ['indian']):
        return 'india'
    elif any(x in voice_lower for x in ['canadian']):
        return 'canada'
    return None

LOW_CONF_FLOOR = 0.50  # if top1 prob < this, fall back to top-3 group match

def _is_group_match(a, b):
    for members in ACCENT_GROUPS.values():
        if a in members and b in members:
            return True
    return False

def check_accent_match(top3, requested):
    """Check if detected accent matches or is acceptable for requested.

    top3: list of (accent, prob) tuples sorted desc. SpeechBrain regional
    confidence is weak (~0.4-0.6 range), so top-1 alone is unstable. When the
    model is uncertain (top1 < LOW_CONF_FLOOR), we treat any top-3 member of
    the requested group as a pass — the right answer is usually in top-3.
    """
    detected, top1_prob = top3[0]

    if detected == requested:
        return True, "exact_match"

    if _is_group_match(detected, requested):
        for group, members in ACCENT_GROUPS.items():
            if detected in members and requested in members:
                return True, f"group_match_{group}"

    if top1_prob < LOW_CONF_FLOOR:
        for i in range(1, len(top3)):
            cand, _cand_prob = top3[i]
            if cand == requested:
                return True, f"top3_low_conf_exact_pos{i+1}"
            if _is_group_match(cand, requested):
                for group, members in ACCENT_GROUPS.items():
                    if cand in members and requested in members:
                        return True, f"top3_low_conf_{group}_pos{i+1}"

    british = ['england', 'scotland', 'wales', 'ireland']
    american = ['us', 'canada']

    if requested in british and detected in american:
        return False, "american_instead_of_british"
    if requested in american and detected in british:
        return False, "british_instead_of_american"
    if requested in british and detected == 'india':
        return False, "indian_instead_of_british"

    return False, "mismatch"

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model": "accent-id-commonaccent_ecapa"})

@app.route('/classify', methods=['POST'])
def classify_accent():
    """
    Classify accent from video/audio URL
    
    POST JSON:
    {
        "video_url": "https://...",
        "requested_accent": "scottish accent, warm and friendly"
    }
    """
    try:
        data = request.json
        video_url = data.get('video_url')
        requested_accent_raw = data.get('requested_accent', '')
        
        if not video_url:
            return jsonify({"error": "video_url required"}), 400
        
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "video.mp4")
            audio_path = os.path.join(tmpdir, "audio.wav")
            
            # Download video
            result = subprocess.run([
                'curl', '-sL', '-o', video_path, video_url
            ], check=True, timeout=120)
            
            # Extract audio
            subprocess.run([
                'ffmpeg', '-y', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                audio_path
            ], check=True, capture_output=True, timeout=60)
            
            # Classify
            out_prob, score, index, text_lab = classifier.classify_file(audio_path)
            
            # FIX: Get labels from the model's label encoder (correct order!)
            # Don't use hardcoded list - the order must match the model's internal order
            probs = out_prob[0].tolist()
            labels = classifier.hparams.label_encoder.decode_ndim(range(len(probs)))
            
            # Build sorted predictions
            sorted_predictions = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)
            top3 = sorted_predictions[:3]
            
            # Use highest probability prediction (not text_lab which can be wrong)
            detected = top3[0][0].lower()
            confidence = top3[0][1]
            
            # Parse requested
            requested_parsed = extract_accent_requirement(requested_accent_raw)
            
            # Check match — pass the full top-3 so the low-confidence escape rule
            # can fall back to top-3 group match when top1 is unstable.
            top3_pairs = [(a.lower(), p) for a, p in top3]
            if requested_parsed:
                is_match, match_type = check_accent_match(top3_pairs, requested_parsed)
            else:
                is_match = True
                match_type = "no_requirement_parsed"
            
            return jsonify({
                "detected_accent": detected,
                "confidence": round(confidence, 3),
                "top_3": [{"accent": a, "prob": round(p, 3)} for a, p in top3],
                "requested_accent_raw": requested_accent_raw,
                "requested_accent_parsed": requested_parsed,
                "match": is_match,
                "match_type": match_type,
                "verdict": "PASS" if is_match else "FAIL"
            })
            
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Processing timeout"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
