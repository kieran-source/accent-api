# Accent Detection API - Railway Deployment

## Quick Deploy (5 minutes)

### 1. Create Railway Account
Go to [railway.app](https://railway.app) and sign up (free tier available)

### 2. Create New Project
- Click "New Project"
- Select "Deploy from GitHub repo" OR "Empty Project"

### 3. If using GitHub:
- Fork/upload these files to a GitHub repo
- Connect Railway to your repo
- It will auto-deploy

### 4. If deploying manually:
- Click "New" → "Empty Service"  
- Go to Settings → Deploy → Select "Dockerfile"
- Upload these files or connect to GitHub

### 5. Get Your API URL
Once deployed, Railway gives you a URL like:
```
https://your-app-name.up.railway.app
```

### 6. Test It
```bash
curl -X POST https://your-app-name.up.railway.app/classify \
  -H "Content-Type: application/json" \
  -d '{"video_url": "YOUR_VIDEO_URL", "requested_accent": "British accent"}'
```

---

## Use in n8n

Add an **HTTP Request** node:

- **Method:** POST
- **URL:** `https://your-app-name.up.railway.app/classify`
- **Body Type:** JSON
- **Body:**
```json
{
  "video_url": "{{ $json.video_url }}",
  "requested_accent": "{{ $json.voice_description }}"
}
```

### Response Format
```json
{
  "detected_accent": "india",
  "confidence": 0.847,
  "top_3": [
    {"accent": "india", "prob": 0.847},
    {"accent": "england", "prob": 0.092},
    {"accent": "us", "prob": 0.031}
  ],
  "requested_accent_raw": "polished London accent",
  "requested_accent_parsed": "england",
  "match": false,
  "match_type": "indian_instead_of_british",
  "verdict": "FAIL"
}
```

---

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check if API is running |
| `/classify` | POST | Classify accent from video URL |

---

## Cost

Railway free tier: $5 credit/month
Typical usage: ~$2-3/month for occasional accent checks

---

## Files Included

- `app.py` - Flask API server
- `Dockerfile` - Container configuration  
- `requirements.txt` - Python dependencies
- `README.md` - This file
