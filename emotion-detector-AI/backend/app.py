from flask import Flask, request, jsonify, render_template, Response, redirect, url_for
import requests
from fer import FER
import cv2
import numpy as np
import base64
import json
import time

app = Flask(__name__)

def detect_emotion_from_base64(image_base64):
    try:
        img_data = base64.b64decode(image_base64.split(",")[1])
        
        np_arr = np.frombuffer(img_data, np.uint8)
        
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        img = cv2.resize(img, (400, 400))
        
        detector = FER()
        result = detector.top_emotion(img)
        return result[0] if result else "neutral"
        
    except Exception as e:
        print(f"Error deteksi emosi: {e}")
        return "neutral"
    
@app.route('/reset')
def reset():
    return redirect(url_for('index'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

# API Endpoints
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    data = request.json
    image_base64 = data.get("image")
    if not image_base64:
        return jsonify({"error": "No image provided"}), 400
    emotion = detect_emotion_from_base64(image_base64)
    print(f"Deteksi emosi selesai dalam {time.time() - start_time:.2f} detik")
    return jsonify({"emotion": emotion})

@app.route('/stream-auto-chat')
def stream_auto_chat():
    emotion = request.args.get('emotion', 'neutral')
    user_message = f"Hai, saya sedang merasa {emotion}. Bisa kamu beri saya saran yang sesuai?"
    
    system_prompt = (
        f"Kamu adalah asisten yang peka terhadap emosi. Pengguna sedang merasa {emotion}. "
        f"Tanggapi dengan empati dan gunakan Bahasa Indonesia. "
        f"Tawarkan beberapa aktivitas yang cocok, seperti wisata, film, musik, atau podcast. "
        f"Tulis dengan rapi dan mudah dibaca. Jangan gunakan Bahasa Inggris."
    )
    
    payload = {
        "model": "llama3:latest",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "stream": True,
        "options": {
            "temperature": 0.7,
            "num_predict": 512
        }
    }

    def generate():
        try:
            with requests.post(
                "http://localhost:11434/api/chat",
                json=payload,
                stream=True,
                timeout=70
            ) as r:
                for line in r.iter_lines():
                    if line:
                        try:
                            decoded_line = line.decode('utf-8')
                            if '"content":' in decoded_line:
                                data = json.loads(decoded_line)
                                content = data.get("message", {}).get("content", "")
                                if content:
                                    yield f"data: {json.dumps({'content': content})}\n\n"
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            print(f"Stream error: {e}")
                            break
        except Exception as e:
            print(f"Connection error: {e}")
            yield "data: {\"error\": \"Koneksi bermasalah\"}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route('/stream-chat')
def stream_chat():
    user_message = request.args.get('message', '')
    emotion = request.args.get('emotion', 'neutral')
    
    system_prompt = (
        f"Kamu adalah asisten yang peka terhadap emosi. Pengguna sedang merasa {emotion}. "
        f"Tanggapi dengan empati dan gunakan Bahasa Indonesia. "
        f"Tawarkan beberapa aktivitas yang cocok, seperti wisata, film, musik, atau podcast. "
        f"Jika user merasa sedih maka hibur dia dengan lelucon."
        f"Tulis dengan rapi dan mudah dibaca. Jangan gunakan Bahasa Inggris."
    )
    
    payload = {
        "model": "llama3:latest",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "stream": True,
        "options": {
            "temperature": 0.7,
            "num_predict": 256
        }
    }

    def generate():
        with requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            stream=True,
            timeout=70
        ) as r:
            for line in r.iter_lines():
                if line:
                    try:
                        decoded_line = line.decode('utf-8')
                        if '"content":' in decoded_line:
                            data = json.loads(decoded_line)
                            content = data.get("message", {}).get("content", "")
                            if content:
                                yield f"data: {json.dumps({'content': content})}\n\n"
                    except Exception:
                        continue
        yield "data: [DONE]\n\n"

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)