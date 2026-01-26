import streamlit as st
import time
import re
import unicodedata
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
from io import BytesIO
from gtts import gTTS
from pydub import AudioSegment
from google import genai
from google.genai import types
from obswebsocket import obsws, requests

# --- 1. CONFIGURATION ---
# ⚠️ เปลี่ยนเป็น API Key ตัวใหม่ของคุณที่นี่
MY_KEY = "AIzaSyA-nRi2vf0xpyUUvwtRJ9vOaRMcral77dw" 
OBS_HOST = "localhost"
OBS_PORT = 4455
OBS_PASSWORD = "์Namo011213" 
SCENE_NAME = "โครงงาน"
CABLE_INPUT_INDEX = 14 # ตรวจสอบ Index ของ Virtual Cable ในเครื่องคุณ
MODEL_ID = "gemini-2.0-flash"
SAMPLE_RATE = 16000

client = genai.Client(api_key=MY_KEY, http_options={'api_version': 'v1'})

# --- 2. INITIALIZE SESSION STATE ---
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'phq_step' not in st.session_state: st.session_state.phq_step = 0 
if 'total_score' not in st.session_state: st.session_state.total_score = 0
if 'input_mode' not in st.session_state: st.session_state.input_mode = "🎤 Voice"

PHQ9_QUESTIONS = [
    "เบื่อ หรือไม่สนใจอยากทำอะไรเลยไหมคะ?", "รู้สึกไม่สบายใจ เศร้า หรือท้อแท้ไหมคะ?",
    "หลับยาก หรือหลับมากเกินไปไหมคะ?", "เหนื่อยง่าย หรือไม่ค่อยมีแรงไหมคะ?",
    "เบื่ออาหาร หรือกินมากเกินไปไหมคะ?", "รู้สึกไม่ดีกับตัวเอง หรือล้มเหลวไหมคะ?",
    "สมาธิไม่ดีเวลาอ่านหนังสือหรือดูทีวีไหมคะ?", "พูดหรือทำอะไรช้าลง หรือกระสับกระส่ายไหมคะ?",
    "คิดทำร้ายตัวเอง หรือคิดว่าตายไปจะดีกว่าไหมคะ?"
]

# --- 3. FUNCTIONS ---

@st.cache_resource
def get_obs_client():
    try:
        ws = obsws(OBS_HOST, OBS_PORT, OBS_PASSWORD)
        ws.connect()
        return ws
    except: return None

def get_obs_frame():
    ws = get_obs_client()
    if ws:
        try:
            response = ws.call(requests.GetSourceScreenshot(sourceName=SCENE_NAME, imageFormat="jpg", imageCompression=40, width=800))
            return response.getImageData()
        except: return None
    return None

def speak(text):
    """ส่งเสียงและบันทึกประวัติแชท"""
    clean_text = re.sub(r'\[.*?\]', '', text).strip()
    if not clean_text: return
    st.session_state.chat_history.append({"role": "assistant", "content": clean_text})
    
    def run_speech():
        try:
            tts = gTTS(text=unicodedata.normalize("NFC", clean_text), lang="th")
            fp = BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            audio = AudioSegment.from_file(fp, format="mp3")
            samples = np.array(audio.get_array_of_samples())
            sd.play(samples, samplerate=audio.frame_rate, device=CABLE_INPUT_INDEX)
        except: pass
    
    threading.Thread(target=run_speech, daemon=True).start()

def listen_and_record():
    """บันทึกเสียงแบบง่าย 4 วินาที (ปรับปรุง VAD ได้ในอนาคต)"""
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1) as stream:
            st.toast("👂 อลิษากำลังฟังอยู่ค่ะ...")
            data, _ = stream.read(int(SAMPLE_RATE * 4)) 
            buffer = BytesIO()
            sf.write(buffer, data, SAMPLE_RATE, format='WAV')
            return buffer.getvalue()
    except:
        st.error("❌ ไม่พบไมโครโฟนหรืออุปกรณ์บันทึกเสียง")
        return None

# --- 4. UI LAYOUT ---
st.set_page_config(page_title="ALISA VTuber Dashboard", layout="wide")

col1, col2 = st.columns([0.6, 0.4])

with col1:
    st.subheader("📺 Live Preview")
    image_holder = st.empty()
@st.fragment(run_every=0.1) # จากเดิม 0.05
def sync_view():
    frame = get_obs_frame()
    if frame: 
        image_holder.image(frame, use_container_width=True)

with col2:
    st.subheader("💬 Chat Interface")
    
    # --- ปุ่มควบคุมหลัก (แถวบน) ---
    ctl_col1, ctl_col2 = st.columns([0.6, 0.4])
    with ctl_col1:
        if st.button("📝 เริ่มทำ PHQ-9", use_container_width=True):
            st.session_state.phq_step = 1
            st.session_state.total_score = 0
            speak("ได้เลยค่ะ อลิษาจะถาม 9 ข้อนะคะ ข้อแรก... " + PHQ9_QUESTIONS[0])
            st.rerun()
    with ctl_col2:
        st.session_state.input_mode = st.radio("Mode", ["🎤 Voice", "⌨️ Text"], horizontal=True, label_visibility="collapsed")

    # --- แสดงประวัติแชท ---
    chat_container = st.container(height=450)
    with chat_container:
        for chat in st.session_state.chat_history:
            st.chat_message(chat["role"]).write(chat["content"])

    # --- ส่วนรับ Input ---
    user_input_content = None

    if st.session_state.input_mode == "🎤 Voice":
        btn_label = "🎤 กดเพื่อตอบคำถาม" if st.session_state.phq_step > 0 else "🎤 กดเพื่อคุยกับอลิษา"
        if st.button(btn_label, use_container_width=True, type="primary"):
            audio_bytes = listen_and_record()
            if audio_bytes:
                user_input_content = [types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")]
    else:
        placeholder = "พิมพ์คำตอบที่นี่..." if st.session_state.phq_step > 0 else "พิมพ์คุยกับอลิษา..."
        if prompt_text := st.chat_input(placeholder):
            st.session_state.chat_history.append({"role": "user", "content": prompt_text})
            user_input_content = [types.Part.from_text(text=prompt_text)]

    # --- LOGIC การประมวลผลโดย Gemini ---
    if user_input_content:
        try:
            if st.session_state.phq_step > 0:
                # โหมดทำแบบทดสอบ
                idx = st.session_state.phq_step - 1
                instruction = f"คำถามคือ '{PHQ9_QUESTIONS[idx]}' วิเคราะห์คำตอบและให้คะแนน [SCORE:0-3] พร้อมคำปลอบโยนสั้นๆ 1 ประโยค"
                contents = [types.Part.from_text(text=instruction)] + user_input_content
                
                res = client.models.generate_content(model=MODEL_ID, contents=contents)
                
                score_match = re.search(r'\[SCORE:(\d)\]', res.text)
                if score_match:
                    st.session_state.total_score += int(score_match.group(1))
                    st.session_state.phq_step += 1
                    
                    if st.session_state.phq_step <= 9:
                        next_q = PHQ9_QUESTIONS[st.session_state.phq_step - 1]
                        speak(f"{res.text} ข้อต่อไปนะคะ... {next_q}")
                    else:
                        score = st.session_state.total_score
                        result_msg = f"ทำครบแล้วค่ะ คะแนนรวมคือ {score} คะแนน "
                        if score >= 9: result_msg += "อลิษาเป็นห่วงนะคะ แนะนำให้ลองปรึกษาผู้เชี่ยวชาญดูนะคะ"
                        else: result_msg += "คุณเก่งมากเลยค่ะ รักษาใจให้แข็งแรงแบบนี้ต่อไปนะคะ"
                        speak(result_msg)
                        st.session_state.phq_step = 0
                st.rerun()
                
            else:
                # โหมดคุยปกติ
                instruction = "คุณคืออลิษา ตอบโต้สั้นๆ เป็นกันเอง ถ้าผู้ใช้ขอประเมินสุขภาพจิตให้ตอบรับแล้วใส่ [PHQ9]"
                contents = [types.Part.from_text(text=instruction)] + user_input_content
                res = client.models.generate_content(model=MODEL_ID, contents=contents)
                
                if "[PHQ9]" in res.text:
                    st.session_state.phq_step = 1
                    st.session_state.total_score = 0
                    speak("ได้ค่ะ เริ่มข้อแรกนะคะ... " + PHQ9_QUESTIONS[0])
                else:
                    speak(res.text)
                st.rerun()
        except Exception as e:
            st.error(f"⚠️ เกิดข้อผิดพลาด: {e}")

    if st.button("🧹 ล้างการสนทนา", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.phq_step = 0
        st.rerun()