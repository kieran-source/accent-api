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

# Accent groups for matching
ACCENT_GROUPS = {
    'british': ['england', 'scotland', 'wales', 'ireland'],
    'american': ['us', 'canada'],
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

def check_accent_match(detected, requested):
    """Check if detected accent matches or is acceptable for requested"""
    if detected == requested:
        return True, "exact_match"
    
    for group, members in ACCENT_GROUPS.items():
        if detected in members and requested in members:
            return True, f"group_match_{group}"
    
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
            
            detected = text_lab[0].lower() if text_lab else "unknown"
            confidence = float(score[0]) if score is not None else 0.0
            
            # Get top 3
            probs = out_prob[0].tolist()
            labels = ['africa', 'australia', 'bermuda', 'canada', 'england', 'hongkong', 
                     'india', 'ireland', 'malaysia', 'newzealand', 'philippines', 
                     'scotland', 'singapore', 'southatlandtic', 'us', 'wales']
            top3 = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)[:3]
            
            # Parse requested
            requested_parsed = extract_accent_requirement(requested_accent_raw)
            
            # Check match
            if requested_parsed:
                is_match, match_type = check_accent_match(detected, requested_parsed)
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
