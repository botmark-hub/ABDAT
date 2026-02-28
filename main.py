import streamlit as st
import re
import numpy as np
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from io import BytesIO
from google import genai
from obswebsocket import obsws, requests
from streamlit_mic_recorder import mic_recorder
import base64
import time
import sounddevice as sd
import streamlit.components.v1 as components

# --- HTML & JS FOR CAMERA ---
CAMERA_HTML = """
<div style="margin-bottom: 10px;">
    <video id="video" width="100%" autoplay playsinline style="border-radius: 15px; border: 2px solid #4A90E2;"></video>
    <canvas id="canvas" width="640" height="480" style="display:none;"></canvas>
</div>
<script>
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');

    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => { video.srcObject = stream; })
        .catch(err => { console.error("Error: ", err); });

    window.addEventListener('message', function(event) {
        if (event.data.type === 'SNAPSHOT') {
            const context = canvas.getContext('2d');
            context.drawImage(video, 0, 0, 640, 480);
            const dataURL = canvas.toDataURL('image/jpeg', 0.7);
            window.parent.postMessage({
                type: 'streamlit:set_component_value',
                value: dataURL,
                key: 'captured_image'
            }, '*');
        }
    });
</script>
"""

# --- 1. CONFIGURATION ---
MY_KEY = "YOUR_GEMINI_API" 
OBS_HOST = "localhost"
OBS_PORT = 4455
OBS_PASSWORD = "" 
SCENE_NAME = "Project" 
MODEL_ID = "gemini-2.0-flash"

CABLE_INPUT_INDEX = 12
client = genai.Client(api_key=MY_KEY)

PHQ9_QUESTIONS = [
    "เบื่อ ทำอะไรๆ ก็ไม่เพลิดเพลิน", "ไม่สบายใจ ซึมเศร้า หรือท้อแท้",
    "หลับยาก หรือหลับๆ ตื่นๆ หรือหลับมากเกินไป", "เหนื่อยง่าย หรือไม่ค่อยมีแรง",
    "เบื่ออาหาร หรือกินมากเกินไป", "รู้สึกไม่ดีกับตัวเอง คิดว่าตัวเองล้มเหลว",
    "สมาธิแย่ลง", "พูดหรือทำอะไรช้าลง หรือกระสับกระส่าย",
    "คิดทำร้ายตนเอง"
]

# --- 2. INITIALIZE SESSION STATE ---
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'app_mode' not in st.session_state: st.session_state.app_mode = None 
if 'is_processing' not in st.session_state: st.session_state.is_processing = False
if 'last_audio_id' not in st.session_state: st.session_state.last_audio_id = None 
if 'phq9_final' not in st.session_state: st.session_state.phq9_final = 0
if 'q9_val' not in st.session_state: st.session_state.q9_val = 0
if 'audio_to_play' not in st.session_state: st.session_state.audio_to_play = None

# --- 3. FUNCTIONS ---

def play_audio_logic(text):
    try:
        clean_text = re.sub(r'\*+', '', text)
        tts = gTTS(text=clean_text, lang="th")
        fp = BytesIO()
        tts.write_to_fp(fp)
        audio_content = fp.getvalue()
        st.session_state.audio_to_play = audio_content
        try:
            audio_seg = AudioSegment.from_file(BytesIO(audio_content), format="mp3")
            audio_seg = audio_seg.speedup(playback_speed=1.15)
            samples = np.array(audio_seg.get_array_of_samples())
            if audio_seg.sample_width == 2:
                samples = samples.astype(np.float32) / 32768.0
            sd.play(samples, samplerate=audio_seg.frame_rate, device=CABLE_INPUT_INDEX)
        except Exception as e: print(f"CABLE Error: {e}")
    except Exception as e: st.error(f"TTS Error: {e}")

@st.cache_resource
def get_obs_client():
    try:
        ws = obsws(OBS_HOST, OBS_PORT, OBS_PASSWORD)
        ws.connect()
        return ws
    except: return None

def typewriter_effect(text):
    message_placeholder = st.empty()
    full_response = ""
    for char in text:
        full_response += char
        message_placeholder.markdown(f"### {full_response}▌")
        time.sleep(0.02)
    message_placeholder.markdown(f"### {full_response}")

def process_ai_response(user_text, image_b64=None, mode="voice"):
    if not user_text or st.session_state.is_processing: return
    st.session_state.is_processing = True
    
    try:
        contents = [user_text]
        if image_b64:
            if "," in image_b64: image_b64 = image_b64.split(",")[1]
            contents.append({"mime_type": "image/jpeg", "data": image_b64})

        st.session_state.chat_history.append({"role": "user", "content": user_text})
        
        instruction = """คุณคือ 'อลิษา' AI เพื่อนคู่คิด คุณเห็นหน้าผู้ใช้ผ่านกล้อง Browser 
        ให้สังเกตสีหน้าและแววตาประกอบกับการตอบ ถ้าเขาสีหน้าดูเศร้าแต่พิมพ์ว่าสบายดี 
        ให้ทักด้วยความห่วงใยอย่างอ่อนโยน เป็นธรรมชาติที่สุด ตอบไม่ต้องยาว"""
        
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=contents,
            config={"system_instruction": instruction}
        )
        
        reply = response.text.strip()
        st.session_state.chat_history.append({"role": "assistant", "content": reply, "new": True})
        
        if mode == "voice": play_audio_logic(reply)
            
    except Exception as e: st.error(f"AI Error: {e}")
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
st.set_page_config(page_title="อลิษา AI เพื่อนข้างกาย", layout="wide")
st.markdown("<style>audio { display: none !important; }</style>", unsafe_allow_html=True)

if st.session_state.audio_to_play:
    st.audio(st.session_state.audio_to_play, format="audio/mp3", autoplay=True)
    st.session_state.audio_to_play = None 

with st.sidebar:
    st.title("⚙️ เมนูของอลิษา")
    if st.button("🔄 เริ่มคุยใหม่", use_container_width=True):
        st.session_state.app_mode = None
        st.session_state.chat_history = []
        st.rerun()
    st.divider()
    if st.button("📊 ทำแบบประเมินสุขภาพจิต", use_container_width=True, type="primary"):
        st.session_state.app_mode = "confirm_phq9"
        st.rerun()
    st.divider()
    st.subheader("📸 อลิษากำลังมองคุณ")
    components.html(CAMERA_HTML, height=250)

# --- APP MODES ---

if st.session_state.app_mode is None:
    st.title("สวัสดีค่ะ อลิษาพร้อมรับฟังคุณนะคะ 💙")
    col_a, col_b = st.columns(2)
    if col_a.button("🎤 คุยผ่านเสียง", use_container_width=True): 
        st.session_state.app_mode = "voice"
        st.rerun()
    if col_b.button("💬 พิมพ์ข้อความ", use_container_width=True): 
        st.session_state.app_mode = "text"
        st.rerun()

elif st.session_state.app_mode == "confirm_phq9":
    st.markdown("## อลิษาขอถามเพื่อความแน่ใจนะคะ\n### คุณต้องการเริ่มทำแบบประเมินความรู้สึกตอนนี้เลยไหมคะ?")
    col_y, col_n = st.columns(2)
    if col_y.button("✅ เริ่มเลยค่ะ", use_container_width=True, type="primary"): 
        st.session_state.app_mode = "phq9"; st.rerun()
    if col_n.button("❌ ไว้คราวหลังนะ", use_container_width=True): 
        st.session_state.app_mode = "voice"; st.rerun()

elif st.session_state.app_mode == "phq9":
    st.markdown("## 📊 แบบประเมินความรู้สึก")
    with st.form("phq9_form"):
        temp_scores = []
        all_answered = True
        for i, q in enumerate(PHQ9_QUESTIONS):
            choice = st.radio(f"### {i+1}. {q}", ["ไม่มีเลย", "มีบางวัน", "มีบ่อย", "มีทุกวัน"], horizontal=True, index=None, key=f"q_{i}")
            if choice:
                val = {"ไม่มีเลย":0, "มีบางวัน":1, "มีบ่อย":2, "มีทุกวัน":3}[choice]
                temp_scores.append(val)
            else: all_answered = False
        
        if st.form_submit_button("ส่งคำตอบให้อลิษา", use_container_width=True):
            if all_answered:
                st.session_state.phq9_final = sum(temp_scores)
                st.session_state.q9_val = temp_scores[8]
                st.session_state.app_mode = "phq9_result"; st.rerun()
            else: st.error("รบกวนตอบให้ครบทุกข้อน้า")

elif st.session_state.app_mode == "phq9_result":
    score = st.session_state.phq9_final
    q9_val = st.session_state.q9_val
    
    with st.status("อลิษากำลังวิเคราะห์ความมั่นใจของผลลัพธ์...", expanded=False) as status:
        running_avg = []
        current_sum = 0
        for i in range(1, 1001):
            sim_score = max(0, min(27, score + np.random.normal(0, 0.5)))
            current_sum += sim_score
            running_avg.append(current_sum / i)
        final_avg = running_avg[-1]
        status.update(label="วิเคราะห์เสร็จสิ้น!", state="complete")

    if final_avg <= 4.49: res, color, emoji = "ดีเยี่ยม", "#28a745", "😊"
    elif final_avg <= 8.49: res, color, emoji = "กังวลเล็กน้อย", "#17a2b8", "😐"
    elif final_avg <= 14.49: res, color, emoji = "เริ่มเศร้า", "#ffc107", "😟"
    elif final_avg <= 19.49: res, color, emoji = "เศร้ามาก", "#fd7e14", "😰"
    else: res, color, emoji = "วิกฤต", "#dc3545", "🆘"

    st.markdown(f"""
        <div style="background-color:{color}; padding:25px; border-radius:20px; text-align:center; color:white;">
            <h1 style="margin:0; font-size: 50px;">{emoji} {res}</h1>
            <p style="font-size:24px;">คะแนนความเสี่ยงของคุณ: {score} / 27</p>
        </div>
    """, unsafe_allow_html=True)

    st.write("### 📈 กราฟแสดงความมั่นใจในการวิเคราะห์")
    st.line_chart(running_avg) 

    if q9_val > 0:
        st.error("### ⚠️ พบความเสี่ยงในการทำร้ายตนเอง")
        st.markdown(f"**กรุณาติดต่อสายด่วนสุขภาพจิต 1323 หรือคุยกับคนที่คุณไว้ใจทันทีนะคะ**")

    if st.button("⬅️ กลับไปคุยกับอลิษาต่อ", use_container_width=True, type="primary"):
        st.session_state.app_mode = "voice"; st.rerun()

elif st.session_state.app_mode == "voice":
    st.subheader("🎤 คุยกับอลิษาผ่านเสียง")
    sync_obs_display()
    
    if st.session_state.chat_history:
        st.info(f"อลิษา: {st.session_state.chat_history[-1]['content']}")
    
    # สั่ง Snapshot อัตโนมัติ (เพื่อให้ AI เห็นหน้าปัจจุบัน)
    components.html("<script>window.parent.postMessage({type: 'SNAPSHOT'}, '*');</script>", height=0)
    img_from_js = st.session_state.get('captured_image')

    audio = mic_recorder(start_prompt="🎤 แตะเพื่อเริ่มพูด", stop_prompt="⏹️ เสร็จแล้ว", key='v_mic_main')

    if audio and hash(audio['bytes']) != st.session_state.last_audio_id:
        st.session_state.last_audio_id = hash(audio['bytes'])
        with st.spinner("อลิษากำลังฟังนะ..."):
            try:
                raw_audio = BytesIO(audio['bytes'])
                audio_segment = AudioSegment.from_file(raw_audio)
                wav_buffer = BytesIO()
                audio_segment.export(wav_buffer, format="wav")
                wav_buffer.seek(0)
                
                r = sr.Recognizer()
                with sr.AudioFile(wav_buffer) as source:
                    r.adjust_for_ambient_noise(source, duration=0.5)
                    audio_data = r.record(source)
                    text = r.recognize_google(audio_data, language="th-TH")
                    if text:
                        process_ai_response(text, image_b64=img_from_js, mode="voice")
                    else:
                        st.warning("เหมือนอลิษาจะไม่ได้ยินเสียงพูดเลยค่ะ")
            except sr.UnknownValueError:
                st.warning("อลิษาไม่ค่อยเข้าใจที่พูดเลยค่ะ รบกวนพูดอีกรอบได้ไหมคะ?")
            except Exception as e:
                st.toast(f"เกิดข้อผิดพลาด: {e}")

else: # Text Mode
    st.subheader("💬 คุยกับอลิษาผ่านการพิมพ์")
    
    # --- ส่วนที่เพิ่มใหม่: ทักทายเมื่อเริ่มคุยครั้งแรก ---
    if not st.session_state.chat_history:
        greeting_text = "สวัสดีค่ะ! อลิษาเองนะคะ เป็น AI เพื่อนคู่คิดที่จะคอยรับฟังและดูแลใจคุณ คุณสามารถระบายความรู้สึก ปรึกษาปัญหา หรือให้ความช่วยเหลือในการทำแบบประเมินสุขภาพจิต (PHQ-9) ก็ได้นะ อยากเล่าอะไรให้อลิษาฟังไหมคะ?"
        st.session_state.chat_history.append({"role": "assistant", "content": greeting_text, "new": True})
    # -------------------------------------------

    chat_holder = st.container(height=500)
    with chat_holder:
        for m in st.session_state.chat_history:
            with st.chat_message(m["role"]):
                if m["role"] == "assistant" and m.get("new"):
                    typewriter_effect(m["content"])
                    m["new"] = False
                else:
                    st.write(m["content"])
                    
    if user_input := st.chat_input("เล่าอะไรให้อลิษาฟังได้เลย..."): 
        process_ai_response(user_input, mode="text")