import os
import sys
import time
import re
import unicodedata
import threading
from io import BytesIO

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
from gtts import gTTS
from pydub import AudioSegment
from fer import FER
from google import genai
from google.genai import types

# --- CONFIG ---
MY_KEY = "AIzaSyCFDro1aHHN1Q-RptrWlO7-eBx5vY7uDAI" 
CABLE_INPUT_INDEX = 14 
OBS_TEXT_FILE = "alisa_obs.txt"
MODEL_ID = "gemini-2.0-flash"
SAMPLE_RATE = 16000 

client = genai.Client(api_key=MY_KEY, http_options={'api_version': 'v1'})
shared_state = {"emotion": "neutral"}

def text_to_speech(text: str):
    clean_text = re.sub(r'\[.*?\]', '', text).strip()
    if not clean_text: return
    print(f"🤖 อลิษา: {clean_text}")
    try:
        with open(OBS_TEXT_FILE, "w", encoding="utf-8") as f:
            f.write(clean_text)
        tts = gTTS(text=unicodedata.normalize("NFC", clean_text), lang="th")
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        audio = AudioSegment.from_file(mp3_fp, format="mp3")
        samples = np.array(audio.get_array_of_samples())
        sd.play(samples, samplerate=audio.frame_rate, device=CABLE_INPUT_INDEX)
        sd.wait()
    except: pass

def listen_and_record():
    threshold = 0.012
    silence_limit = 1.8
    recording = []
    is_speaking = False
    silent_frames = 0
    
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1) as stream:
        print("👂 ฟังอยู่...")
        while True:
            data, overflowed = stream.read(1024)
            volume = np.linalg.norm(data) / np.sqrt(len(data))
            if volume > threshold:
                if not is_speaking: is_speaking = True
                silent_frames = 0
            elif is_speaking:
                silent_frames += 1024 / SAMPLE_RATE
            if is_speaking:
                recording.append(data.copy())
                if silent_frames > silence_limit: break
    
    if not recording: return None
    audio_data = np.concatenate(recording, axis=0)
    buffer = BytesIO()
    sf.write(buffer, audio_data, SAMPLE_RATE, format='WAV')
    return buffer.getvalue()

def run_phq9_assessment():
    questions = [
        "เบื่อ หรือไม่สนใจอยากทำอะไรเลยไหมคะ?", "รู้สึกไม่สบายใจ เศร้า หรือท้อแท้ไหมคะ?",
        "หลับยาก หรือหลับมากเกินไปไหมคะ?", "เหนื่อยง่าย หรือไม่ค่อยมีแรงไหมคะ?",
        "เบื่ออาหาร หรือกินมากเกินไปไหมคะ?", "รู้สึกไม่ดีกับตัวเอง หรือล้มเหลวไหมคะ?",
        "สมาธิไม่ดีเวลาอ่านหนังสือหรือดูทีวีไหมคะ?", "พูดหรือทำอะไรช้าลง หรือกระสับกระส่ายไหมคะ?",
        "คิดทำร้ายตัวเอง หรือคิดว่าตายไปจะดีกว่าไหมคะ?"
    ]
    total_score = 0
    q9_score = 0 # แยกคะแนนข้อ 9 ไว้ตรวจสอบความปลอดภัย
    
    text_to_speech("อลิษาขอเริ่มการประเมินเบื้องต้นนะคะ เล่าความรู้สึกให้ฟังได้เต็มที่เลยค่ะ")
    
    for i, q in enumerate(questions):
        text_to_speech(f"ข้อ {i+1}: {q}")
        answered = False
        while not answered:
            audio_bytes = listen_and_record()
            if not audio_bytes: continue
            
            prompt = (
                f"คำถาม: '{q}'\n"
                "วิเคราะห์เสียงตอบ:\n"
                "1. ถอดความใส่ [USER_SAID:...]\n"
                "2. ให้คะแนน 0-3 ใส่ [SCORE:ตัวเลข]\n"
                "3. ตอบกลับด้วยความเห็นอกเห็นใจ (ถ้าเป็นข้อ 9 และคะแนน > 0 ต้องปลอบโยนเป็นพิเศษและแสดงความห่วงใยทันที)"
            )
            
            try:
                res = client.models.generate_content(
                    model=MODEL_ID,
                    contents=[types.Part.from_text(text=prompt), types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")]
                )
                reply = res.text.strip()
                
                user_match = re.search(r'\[USER_SAID:(.*?)\]', reply)
                if user_match: print(f"🗣️ คุณพูดว่า: {user_match.group(1).strip()}")

                score_match = re.search(r'\[SCORE:(\d)\]', reply)
                if score_match:
                    current_score = int(score_match.group(1))
                    total_score += current_score
                    if i == 8: q9_score = current_score # เก็บคะแนนข้อ 9
                    
                    text_to_speech(reply)
                    answered = True
                else:
                    text_to_speech("อลิษารับรู้นะคะ แต่ขอชัดๆ อีกนิดว่าความรู้สึกนี้เกิดขึ้นบ่อยไหมคะ?")
            except:
                text_to_speech("ขออภัยค่ะ ระบบขัดข้องนิดหน่อย รบกวนเล่าใหม่อีกทีนะ")

    # --- ส่วนสรุปผล (ปรับปรุงตามคะแนนข้อ 9) ---
    text_to_speech(f"ขอบคุณที่ไว้วางใจเล่าให้ฟังนะคะ คะแนนรวมของคุณคือ {total_score} คะแนน")
    
    if q9_score > 0:
        # หากข้อ 9 มีคะแนน (มีความคิดอยากทำร้ายตัวเอง) ให้เข้าโหมดดูแลด่วน
        crisis_msg = (
            "อลิษาเป็นห่วงคุณมากนะคะ ความรู้สึกเหนื่อยจนอยากพักไปตลอดมันหนักหนามาก "
            "แต่อยากให้รู้ว่าคุณไม่ต้องเผชิญเรื่องนี้คนเดียวนะคะ อลิษาอยากให้คุณลองหาที่ปรึกษา "
            "หรือโทรสายด่วนสุขภาพจิต 1 3 2 3 จะมีเจ้าหน้าที่คอยรับฟังคุณตลอด 2 4 ชั่วโมงเลยค่ะ "
            "กอดแน่นๆ นะคะ คุณเก่งมากแล้วที่ผ่านวันนี้มาได้"
        )
        text_to_speech(crisis_msg)
    elif total_score >= 9:
        text_to_speech("ผลประเมินอยู่ในเกณฑ์เสี่ยง อลิษาแนะนำให้ลองหาเวลาพักผ่อนหรือปรึกษาผู้เชี่ยวชาญเพื่อดูแลใจนะคะ")
    else:
        text_to_speech("คุณยังดูเข้มแข็งดีค่ะ อย่าลืมหาเวลาทำสิ่งที่ชอบเพื่อผ่อนคลายด้วยนะคะ")
    
if __name__ == "__main__":
    # (Thread FER สำหรับวิเคราะห์อารมณ์ใบหน้า...)
    text_to_speech("สวัสดีค่ะ อลิษามาแล้ว วันนี้อยากคุยเล่นหรืออยากให้ช่วยประเมินสุขภาพจิตดีคะ?")

    while True:
        audio_bytes = listen_and_record()
        if not audio_bytes: continue
        
        instruction = (
            f"คุณคืออลิษา AI ที่ปรึกษา (อารมณ์ผู้ใช้: {shared_state['emotion']})\n"
            "กฎการทำงาน:\n"
            "1. ถอดความคำพูดผู้ใช้ใส่ใน [USER_SAID:...]\n"
            "2. หากผู้ใช้ต้องการ 'เริ่ม' ทำแบบประเมินสุขภาพจิตจริงๆ ให้ใส่แท็ก [PHQ9] มาในคำตอบ\n"
            "3. หากผู้ใช้ขอให้เล่าเรื่องตลก คุยเล่น หรือถามคำถามทั่วไป ห้ามใส่ [PHQ9] เด็ดขาด\n"
            "4. ตอบโต้ด้วยความเป็นกันเองและเห็นอกเห็นใจ"
        )
        
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[
                    types.Part.from_text(text=instruction),
                    types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
                ]
            )
            reply = response.text.strip()
            
            user_speech = re.search(r'\[USER_SAID:(.*?)\]', reply)
            if user_speech:
                print(f"🗣️ คุณพูดว่า: {user_speech.group(1).strip()}")

            if "[PHQ9]" in reply:
                text_to_speech("ได้เลยค่ะ เดี๋ยวอลิษาพาทำแบบประเมินนะคะ")
                run_phq9_assessment()
                continue

            text_to_speech(reply)
            
        except Exception as e:
            print(f"Error: {e}")