import streamlit as st
import threading
import re
import numpy as np
import sounddevice as sd
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from io import BytesIO
from google import genai
from obswebsocket import obsws, requests
from streamlit_mic_recorder import mic_recorder
import base64
import time

# --- 1. CONFIGURATION ---
MY_KEY = "AIzaSyA-nRi2vf0xpyUUvwtRJ9vOaRMcral77dw"
CABLE_INPUT_INDEX = 14 
OBS_HOST = "localhost"
OBS_PORT = 4455
OBS_PASSWORD = "" 
SCENE_NAME = "Project" 
MODEL_ID = "gemini-2.0-flash"

client = genai.Client(api_key=MY_KEY)

PHQ9_QUESTIONS = [
    "เบื่อ ทำอะไรๆ ก็ไม่เพลิดเพลิน",
    "ไม่สบายใจ ซึมเศร้า หรือท้อแท้",
    "หลับยาก หรือหลับๆ ตื่นๆ หรือหลับมากเกินไป",
    "เหนื่อยง่าย หรือไม่ค่อยมีแรง",
    "เบื่ออาหาร หรือกินมากเกินไป",
    "รู้สึกไม่ดีกับตัวเอง คิดว่าตัวเองล้มเหลว หรือทำให้ตนเองหรือครอบครัวผิดหวัง",
    "สมาธิในการทำบางอย่างแย่ลง เช่น ดูโทรทัศน์ หรืออ่านหนังสือพิมพ์",
    "พูดหรือทำอะไรช้าจนคนอื่นสังเกตเห็นได้ หรือกระสับกระส่ายจนอยู่ไม่นิ่ง",
    "คิดทำร้ายตนเอง หรือคิดว่าถ้าตายๆ ไปเสียคงจะดี"
]

# --- 2. INITIALIZE SESSION STATE ---
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'app_mode' not in st.session_state: st.session_state.app_mode = None 
if 'phq9_history' not in st.session_state: st.session_state.phq9_history = []
if 'waiting_confirmation' not in st.session_state: st.session_state.waiting_confirmation = False
if 'is_processing' not in st.session_state: st.session_state.is_processing = False
if 'last_audio_id' not in st.session_state: st.session_state.last_audio_id = None 
if 'phq9_final' not in st.session_state: st.session_state.phq9_final = 0
if 'q9_final' not in st.session_state: st.session_state.q9_final = 0

# --- 3. FUNCTIONS ---

@st.cache_resource
def get_obs_client():
    try:
        ws = obsws(OBS_HOST, OBS_PORT, OBS_PASSWORD)
        ws.connect()
        return ws
    except: return None

def speak_to_discord(text):
    try:
        clean_text = re.sub(r'\*+', '', text)
        tts = gTTS(text=clean_text, lang="th")
        
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        # โหลดไฟล์เสียง
        audio = AudioSegment.from_file(fp, format="mp3")
        
        # --- เพิ่มความเร็วเสียงตรงนี้ ---
        speed_factor = 1.10  # ปรับค่าได้ (เช่น 1.15 หรือ 1.2)
        audio = audio.speedup(playback_speed=speed_factor)
        # ----------------------------

        samples = np.array(audio.get_array_of_samples())
        
        st.toast(f"อลิษากำลังตอบกลับ...", icon="🔊")
        
        sd.stop() 
        sd.play(samples, samplerate=audio.frame_rate, device=CABLE_INPUT_INDEX)
    except Exception as e:
        print(f"Audio Output Error: {e}")

def typewriter_effect(text):
    message_placeholder = st.empty()
    full_response = ""
    for char in text:
        full_response += char
        message_placeholder.markdown(full_response + "▌")
        time.sleep(0.03)
    message_placeholder.markdown(full_response)

def process_ai_response(user_text, mode="voice"):
    if not user_text or st.session_state.is_processing: return

    # --- 1. ระบบดักจับคำสั่ง "เริ่มทำแบบประเมิน" แบบตรงตัว ---
    # จะเปลี่ยนหน้าก็ต่อเมื่อผู้ใช้สั่งชัดเจนเท่านั้น
    start_keywords = ["เริ่มทำแบบประเมิน", "ขอทำแบบประเมิน", "เริ่ม phq9", "ทำแบบทดสอบซึมเศร้า"]
    if any(k in user_text.lower() for k in start_keywords):
        st.session_state.app_mode = "phq9"
        st.rerun()
        return

    # --- 2. การสนทนาปกติ (AI จะไม่เสนอหน้าแบบประเมินเอง) ---
    st.session_state.is_processing = True
    st.session_state.chat_history.append({"role": "user", "content": user_text})

    try:
        # กำหนด Instruction ให้ AI เป็นผู้ช่วยที่อบอุ่น 
        # และถ้าผู้ใช้กังวล ให้แนะนำให้ 'กดปุ่ม' หรือ 'บอกว่าต้องการทำ' แทนการเด้งไปเอง
        instruction = """คุณคือ 'อลิษา' AI ผู้ช่วยสุขภาพจิต 
        กฎเหล็ก: 
        1. ห้ามเปลี่ยนหน้าจอไปที่แบบประเมินเองเด็ดขาด
        2. หากผู้ใช้ดูมีความกังวล ให้แนะนำสั้นๆ ว่า 'หากคุณต้องการประเมินความเสี่ยงเบื้องต้น สามารถบอกอลิษาว่าขอทำแบบประเมิน หรือกดปุ่มที่แถบด้านข้างได้นะคะ'
        3. ตอบสั้นและอบอุ่น 1-2 ประโยค"""

        chat = client.chats.create(model=MODEL_ID, config={"system_instruction": instruction})
        response = chat.send_message(user_text)
        reply = response.text.strip()

        st.session_state.chat_history.append({"role": "assistant", "content": reply, "new": True})
        
        if mode == "voice":
            threading.Thread(target=speak_to_discord, args=(reply,), daemon=True).start()

    except Exception as e:
        st.error(f"AI Error: {e}")
    finally:
        st.session_state.is_processing = False
        st.rerun()

@st.fragment(run_every=0.5)
def sync_obs_display():
    ws = get_obs_client()
    if ws:
        try:
            res = ws.call(requests.GetSourceScreenshot(sourceName=SCENE_NAME, imageFormat="jpg", imageWidth=640))
            img_b64 = res.datain.get('imageData') or res.getImg()
            if "," in img_b64: img_b64 = img_b64.split(",")[1]
            st.image(base64.b64decode(img_b64), use_container_width=True)
        except: pass

# --- 4. UI LAYOUT ---
st.set_page_config(page_title="Alisa Hybrid AI", layout="wide")

with st.sidebar:
    st.title("⚙️ แผงควบคุม")
    if st.button("🔄 กลับหน้าแรก/ล้างประวัติ", use_container_width=True):
        st.session_state.app_mode = None; st.session_state.chat_history = []; st.rerun()
    st.divider()
    if st.button("📊 ทำ PHQ-9 ทันที", use_container_width=True, type="primary"):
        st.session_state.app_mode = "phq9"; st.rerun()

if st.session_state.app_mode is None:
    st.title("ยินดีต้อนรับสู่ พื้นที่ใจของอลิษา 💙")
    col_a, col_b = st.columns(2)
    if col_a.button("🎤 VTuber Mode", use_container_width=True): st.session_state.app_mode = "voice"; st.rerun()
    if col_b.button("💬 Silent Mode", use_container_width=True): st.session_state.app_mode = "text"; st.rerun()

elif st.session_state.app_mode == "phq9":
    st.subheader("📊 แบบประเมินภาวะซึมเศร้า (PHQ-9)")
    with st.form("phq9_form"):
        scores = []
        q9_val = 0
        all_answered = True
        for i, q in enumerate(PHQ9_QUESTIONS):
            choice = st.radio(f"{i+1}. {q}", ["ไม่มีเลย", "มีบางวัน", "มีบ่อย", "มีทุกวัน"], horizontal=True, index=None, key=f"q_{i}")
            if choice:
                val = {"ไม่มีเลย":0, "มีบางวัน":1, "มีบ่อย":2, "มีทุกวัน":3}[choice]
                scores.append(val)
                if i == 8: q9_val = val
            else: all_answered = False
            
        if st.form_submit_button("ส่งและประมวลผล", use_container_width=True):
            if all_answered:
                prog_bar = st.progress(0)
                for p in range(1, 101):
                    time.sleep(0.02)
                    prog_bar.progress(p)
                st.session_state.phq9_final = sum(scores)
                st.session_state.q9_final = q9_val
                st.session_state.app_mode = "phq9_result"; st.rerun()
            else: st.error("กรุณาตอบคำถามให้ครบทุกข้อก่อนนะคะ")

elif st.session_state.app_mode == "phq9_result":
    score = st.session_state.phq9_final
    q9 = st.session_state.q9_final
    st.balloons()
    if score <= 4: level, color, msg = "ปกติ", "#28a745", "สุขภาพจิตดี"
    elif score <= 9: level, color, msg = "ระดับน้อย", "#ffc107", "ควรพักผ่อน"
    elif score <= 14: level, color, msg = "ระดับปานกลาง", "#fd7e14", "ควรปรึกษาคนใกล้ชิด"
    else: level, color, msg = "ระดับรุนแรง", "#dc3545", "แนะนำให้พบแพทย์"

    st.markdown(f"""<div style="background-color:{color}; padding:20px; border-radius:15px; text-align:center; color:white;">
        <h2>คะแนน: {score}</h2><h3>ระดับ: {level}</h3><p>{msg}</p></div>""", unsafe_allow_html=True)
    
    if q9 > 0:
        st.error("⚠️ แนะนำให้ติดต่อสายด่วน 1323 หรือพบแพทย์โดยเร็วที่สุดนะคะ")

    if st.button("กลับไปคุยกับอลิษา"): st.session_state.app_mode = "voice"; st.rerun()

elif st.session_state.app_mode == "voice":
    st.subheader("📺 Visual Interface (Voice Mode)")
    col_v = st.columns([0.1, 0.8, 0.1])[1]
    with col_v:
        sync_obs_display()
        st.divider()
        if st.session_state.chat_history:
            last_msg = st.session_state.chat_history[-1]
            st.caption(f"**{'อลิษา' if last_msg['role'] == 'assistant' else 'คุณ'}:** {last_msg['content']}")

        audio = mic_recorder(start_prompt="🎤 คลิกเพื่อพูด", stop_prompt="⏹️ ส่ง", key='v_mic')
        
        if audio and hash(audio['bytes']) != st.session_state.last_audio_id:
            st.session_state.last_audio_id = hash(audio['bytes'])
            try:
                wav_io = BytesIO(audio['bytes'])
                audio_segment = AudioSegment.from_file(wav_io)
                wav_data = BytesIO()
                audio_segment.export(wav_data, format="wav")
                wav_data.seek(0)
                
                r = sr.Recognizer()
                with sr.AudioFile(wav_data) as source:
                    audio_recorded = r.record(source) # FIXED LINE
                    text = r.recognize_google(audio_recorded, language="th-TH")
                    
                    if text:
                        st.info(f"คุณพูดว่า: {text}")
                        process_ai_response(text, mode="voice")
                        st.rerun()
            except sr.UnknownValueError:
                st.toast("ฟังไม่ชัดเลยค่ะ", icon="❓")
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {e}")

else:
    st.subheader("💬 Silent Mode")
    chat_holder = st.container(height=500)
    with chat_holder:
        for m in st.session_state.chat_history:
            if m["role"] == "assistant" and m.get("new"):
                with st.chat_message("assistant"): typewriter_effect(m["content"]); m["new"] = False
            else:
                with st.chat_message(m["role"]): st.write(m["content"])
    if user_input := st.chat_input("พิมพ์ข้อความที่นี่..."): 
        process_ai_response(user_input, mode="text")
        st.rerun()