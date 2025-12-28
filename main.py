import os
import sys
import time
import unicodedata
import threading
from collections import Counter
from io import BytesIO

import cv2
import numpy as np
import sounddevice as sd
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from dotenv import load_dotenv
from fer import FER
from google import genai


# ------------------------------
# GLOBAL
# ------------------------------
last_phq9_score = None
last_phq9_result = None
last_question = None
CABLE_INPUT_INDEX = 13
LOG_FILE = "phq9_log.txt"
conversation_history = []
DEBUG = True

shared_state = {"emotion": "neutral", "face_detected": False, "last_seen": time.time()}

# ------------------------------
# LOAD CONFIG
# ------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("❌ ERROR: ไม่พบ GEMINI_API_KEY")
    sys.exit(1)
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.0-flash"

# ------------------------------
# UTILS
# ------------------------------
def log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if DEBUG:
        print(f"[{timestamp}] {msg}")

def safe_text(text: str) -> str:
    normalized = unicodedata.normalize('NFC', text)
    return normalized.encode('utf-8', errors='ignore').decode('utf-8')

def check_audio_output(device_index: int):
    try:
        devices = sd.query_devices()
        if device_index >= len(devices) or devices[device_index]['max_output_channels'] == 0:
            log(f"❌ ERROR: ไม่พบอุปกรณ์เสียง index {device_index}")
            sys.exit(1)
        else:
            log(f"✅ พบอุปกรณ์เสียง index {device_index}: {devices[device_index]['name']}")
    except Exception as e:
        log(f"❌ ERROR ตรวจสอบอุปกรณ์เสียง: {e}")
        sys.exit(1)

def text_to_speech(text: str, device_index: int = CABLE_INPUT_INDEX):
    log(f"🤖 อลิษาพูดว่า: {text}")
    try:
        tts = gTTS(text=safe_text(text), lang="th")
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        audio = AudioSegment.from_file(mp3_fp, format="mp3")
        samples = np.array(audio.get_array_of_samples())
        sd.play(samples, samplerate=audio.frame_rate, device=device_index)
        sd.wait()
    except Exception as e:
        log(f"TTS Error: {e}")

def speech_to_text(recognizer: sr.Recognizer, mic: sr.Microphone):
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
        except sr.WaitTimeoutError:
            log("🗣️ ผู้ใช้: (ไม่ได้พูด/ไม่จับเสียง)")
            return None
    try:
        text = recognizer.recognize_google(audio, language="th-TH")
        log(f"🗣️ ผู้ใช้พูดว่า: {text}")
        return text
    except Exception as e:
        log(f"🗣️ ผู้ใช้: (ฟังไม่ชัด) Error: {e}")
        return None

# ------------------------------
# GEMINI REPLY
# ------------------------------
MAX_HISTORY = 3

def gemini_reply(prompt: str, persona: str = "") -> str:
    try:
        recent_history = conversation_history[-MAX_HISTORY:]

        context = ""
        for c in recent_history:
            context += f"{c['role']}: {c['text']}\n"

        full_prompt = (
            f"{persona}\n"
            f"{context}"
            f"user: {prompt}\n"
            "ตอบสั้น กระชับ"
        )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=full_prompt,
            config={
                "temperature": 0.2,
                "max_output_tokens": 120,
            }
        )

        reply_text = response.text.strip()

        conversation_history.append({
            "role": "assistant",
            "text": reply_text[:300]
        })

        return reply_text

    except Exception as e:
        log(f"Gemini Error: {e}")
        return "ขออภัยค่ะ ระบบประมวลผลขัดข้องชั่วคราว"

# ------------------------------
# INTENT DETECTION
# ------------------------------
def detect_intent(user_text: str) -> str:
    """
    ให้ Gemini วิเคราะห์ว่า user ต้องการอะไร:
    - start_phq9   → ต้องการเริ่มแบบประเมิน
    - general      → พูดคุยทั่วไป
    """
    intent_prompt = f"""
คุณคือตัวตรวจจับเจตนา (Intent Detection)
ข้อความจากผู้ใช้:
"{user_text}"
ให้ตอบเพียง 1 คำจากลิสต์:
- start_phq9    (เมื่อผู้ใช้ต้องการเริ่มทำแบบประเมิน PHQ-9 ไม่ว่าจะใช้คำพูดแบบไหน)
- general       (เมื่อผู้ใช้ไม่ได้ต้องการเริ่ม PHQ-9)
ตอบเฉพาะคำเดียวเท่านั้น ห้ามตอบอย่างอื่นเด็ดขาด
    """
    reply = gemini_reply(intent_prompt)
    reply = reply.replace("\n", "").strip()
    if reply not in ["start_phq9", "general"]:
        return "general"
    return reply

# ------------------------------
# PHQ9 SYSTEM
# ------------------------------
PHQ9_QUESTIONS = [
    "ในช่วง 2 สัปดาห์ที่ผ่านมา คุณรู้สึกไม่สนใจหรือไม่อยากทำอะไรบ่อยแค่ไหนคะ?",
    "รู้สึกเศร้าหม่นหมองหรือท้อแท้บ่อยแค่ไหนคะ?",
    "มีปัญหาเรื่องการนอน เช่น นอนไม่หลับหรือนอนมากเกินไปไหมคะ?",
    "รู้สึกเหนื่อยง่ายหรือไม่ค่อยมีแรงบ่อยแค่ไหนคะ?",
    "รู้สึกเบื่ออาหารหรือกินมากเกินไปไหมคะ?",
    "รู้สึกไม่ดีกับตัวเอง หรือคิดว่าตัวเองล้มเหลวบ่อยแค่ไหนคะ?",
    "มีปัญหาสมาธิ เช่น สมาธิสั้นลงไหมคะ?",
    "เคลื่อนไหวช้าลง พูดช้าลง หรือทำอะไรเร็วขึ้นเพราะความกระสับกระส่ายไหมคะ?",
    "คิดทำร้ายตัวเองหรือคิดว่าตายไปซะจะดีกว่าไหมคะ?",
]

CHOICE_MAP = {"ไม่เลย":0, "หลายวัน":1, "บ่อย":2, "เกือบทุกวัน":3}

def classify_phq9(score: int) -> str:
    if score <= 4: return "ไม่มีภาวะซึมเศร้า"
    elif score <= 9: return "มีภาวะซึมเศร้าเล็กน้อย"
    elif score <= 14: return "ภาวะซึมเศร้าปานกลาง"
    elif score <= 19: return "ภาวะซึมเศร้ารุนแรง"
    else: return "ภาวะซึมเศร้ารุนแรงมาก"

def recommendation(result: str) -> str:
    if result in ["ภาวะซึมเศร้ารุนแรง","ภาวะซึมเศร้ารุนแรงมาก"]:
        return "ระดับค่อนข้างสูงค่ะ ควรพบแพทย์หรือผู้เชี่ยวชาญด้านสุขภาพจิต หากมีความคิดทำร้ายตัวเอง ติดต่อสายด่วน 1323 ทันที"
    else:
        return "อยู่ในระดับที่ยังไม่รุนแรงมาก พักผ่อนให้เพียงพอ พูดคุยกับคนใกล้ชิด หากอาการยังอยู่ควรปรึกษาผู้เชี่ยวชาญค่ะ"

def run_phq9_ai(recognizer: sr.Recognizer, mic: sr.Microphone):
    global last_phq9_score, last_phq9_result, last_question
    total_score = 0

    text_to_speech("เราจะทำแบบประเมิน PHQ9 ทั้งหมด 9 ข้อนะคะ ตอบตามจริงได้เลยค่ะ")

    time.sleep(1)

    for q in PHQ9_QUESTIONS:
        last_question = q
        while True:
            text_to_speech(q)
            ans = speech_to_text(recognizer, mic)
            if not ans:
                text_to_speech("ไม่ได้ยินค่ะ ลองตอบอีกครั้งนะคะ")
                continue

            # ----- NEW PROMPT HERE -----
            prompt_ai = f"""
ผู้ใช้ตอบว่า: '{ans}'

โปรดประเมินคำตอบนี้ว่าเข้ากับคำตอบใดมากที่สุดของแบบประเมิน PHQ-9:

ตัวเลือก:
- ไม่เลย (0)
- หลายวัน (1)
- บ่อย (2)
- เกือบทุกวัน (3)

ให้ตอบเพียงหนึ่งคำต่อไปนี้เท่านั้น:
ไม่เลย, หลายวัน, บ่อย, เกือบทุกวัน

ตัวอย่าง:
- "นอนไม่ค่อยหลับสองสามครั้ง" → หลายวัน
- "แทบทุกวันเลย เหนื่อยมาก" → เกือบทุกวัน
- "ยังโอเคนะ ไม่ได้รู้สึกอะไร" → ไม่เลย
- "มีบ้างเป็นบางครั้ง" → หลายวัน
- "บ่อยมาก ช่วงนี้หนักเลย" → บ่อย

คำตอบของคุณ:
"""

            ai_choice = gemini_reply(prompt_ai).strip()

            if ai_choice not in CHOICE_MAP:
                ai_choice = "ไม่เลย"  # fallback ปลอดภัย

            score = CHOICE_MAP[ai_choice]
            total_score += score
            log(f"ผู้ใช้ตอบ: {ans} → AI วิเคราะห์เป็น: {ai_choice} (คะแนน {score})")
            break

    last_phq9_score = total_score
    last_phq9_result = classify_phq9(total_score)

    summary = f"คุณได้ {last_phq9_score} คะแนน ผลคือ {last_phq9_result}"
    advice = recommendation(last_phq9_result)

    text_to_speech(summary)
    text_to_speech(advice)

def safety_override(user_text: str) -> int | None:
    """
    ถ้าเจอคำเสี่ยงสูง ให้ Override คะแนนเป็นระดับ 3 ทันที
    """
    danger_keywords = [
        "ฆ่าตัวตาย", "อยากตาย", "ไม่อยากอยู่แล้ว", "จบชีวิต",
        "ทำร้ายตัวเอง", "สิ้นหวังมาก", "อยากหายไป", "ไม่เห็นคุณค่า",
        "จะฆ่าตัวเอง", "ไม่อยากมีชีวิต"
    ]
    for word in danger_keywords:
        if word in user_text:
            return 3   # ระดับสูงสุด
    return None

def classify_phq9_answer(answer_text: str) -> int:
    """
    วิเคราะห์คำตอบ PHQ-9 แบบปลอดภัย (Hybrid: Rule-based + LLM)
    """
    # 1) RULE-BASED SAFETY
    danger_score = safety_override(answer_text)
    if danger_score is not None:
        return danger_score
    # 2) LLM CLASSIFICATION
    prompt = f"""
คุณคือนักจิตวิทยาที่ทำแบบประเมิน PHQ-9
ข้อความจากผู้ใช้:
"{answer_text}"
ให้เลือกคำตอบที่ตรงที่สุด:
0 = ไม่เลย
1 = หลายวัน
2 = บ่อยกว่า 50% ของวัน
3 = เกือบทุกวัน
ตอบเป็นตัวเลขเท่านั้น (0,1,2หรือ3) ห้ามมีคำอธิบาย
    """
    reply = gemini_reply(prompt).strip()
    # ป้องกัน LLM ตอบผิด
    if reply not in ["0", "1", "2", "3"]:
        return 0
    return int(reply)

# ------------------------------
# EMOTION CAMERA THREAD
# ------------------------------
def emotion_thread():
    detector = FER(mtcnn=True)
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        log("❌ ERROR: ไม่สามารถเปิดกล้องได้!")
        return

    log("✅ กล้องเปิดเรียบร้อย")

    while True:
        ret, frame = cap.read()
        if not ret:
            log("❌ ไม่สามารถอ่าน frame จากกล้องได้")
            continue

        results = detector.detect_emotions(frame)
        if results:
            emotions_list = [max(face["emotions"], key=face["emotions"].get) for face in results]
            emotion_count = Counter(emotions_list)
            main_emotion = emotion_count.most_common(1)[0][0]
            shared_state["emotion"] = main_emotion
            shared_state["face_detected"] = True
            shared_state["last_seen"] = time.time()

            for face in results:
                (x, y, w, h) = face["box"]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, main_emotion, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        else:
            shared_state["face_detected"] = False
            cv2.putText(frame, "ไม่พบใบหน้า", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.imshow("AI Emotion Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------------------------
# MAIN LOOP
# ------------------------------
if __name__ == "__main__":
    check_audio_output(CABLE_INPUT_INDEX)

    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    persona = """คุณคือ “อลิษา” ผู้ช่วย AI แบบเสียง สุภาพ อ่อนโยน และเป็นกันเอง  
ทำหน้าที่ช่วยประเมิน PHQ-9  
ตอบสั้น กระชับ ไม่เกิน 5 บรรทัด  
หากผู้ใช้มีแนวคิดทำร้ายตัวเอง ให้แนะนำติดต่อสายด่วน 1323"""

    threading.Thread(target=emotion_thread, daemon=True).start()

    log("=== อลิษาเริ่มทำงานแล้ว ===")
    text_to_speech("อลิษาอยู่ตรงนี้ค่ะ เริ่มคุยได้เลยนะคะ")

    while True:
        user_text = speech_to_text(recognizer, mic)
        if not user_text:
            continue

        conversation_history.append({"role": "user", "text": user_text})

        if "ออก" in user_text:
            text_to_speech("ไว้คุยกันใหม่นะคะ บ๊ายบายค่ะ")
            conversation_history.clear()
            last_phq9_score = None
            last_phq9_result = None
            last_question = None
            break

        intent = detect_intent(user_text)
        if intent == "start_phq9":
            text_to_speech("ได้เลยค่ะ พร้อมเริ่มทำแบบประเมิน PHQ-9 แล้วนะคะ")
            run_phq9_ai(recognizer, mic)
            continue

        if "คะแนน" in user_text and "เท่าไหร่" in user_text:
            if last_phq9_score is not None:
                reply = f"คุณได้ {last_phq9_score} คะแนน ผลคือ {last_phq9_result} ค่ะ"
            else:
                reply = "ยังไม่มีผลคะแนนค่ะ ต้องทำแบบประเมินก่อนนะคะ"
            text_to_speech(reply)
            continue

        reply = gemini_reply(user_text, persona)
        text_to_speech(reply)