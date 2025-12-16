import pyaudio

def get_cable_output_device_index(target_name: str = "CABLE Input") -> int | None:
    """
    ค้นหา Audio Device Index ของอุปกรณ์ที่มีชื่อว่า 'CABLE Input'
    โดยใช้หลักการเดียวกับโค้ดในภาพที่คุณแนบมา
    
    Returns:
        int: Index ของอุปกรณ์ หรือ None หากไม่พบ
    """
    p = pyaudio.PyAudio()
    host_api_index = 0  # มักจะเป็นค่าเริ่มต้นของ Windows/MME
    
    # ดึงข้อมูล Host API หลัก
    try:
        info = p.get_host_api_info_by_index(host_api_index)
        num_devices = info.get('deviceCount')
    except Exception as e:
        print(f"❌ Error getting host API info: {e}")
        p.terminate()
        return None

    print(f"--- เริ่มค้นหา Output Devices ใน Host API Index {host_api_index} ---")
    
    for i in range(num_devices):
        # ใช้ get_device_info_by_host_api_device_index ตามโค้ดในภาพ
        device_info = p.get_device_info_by_host_api_device_index(host_api_index, i)
        
        device_name = device_info['name']
        max_output = device_info['maxOutputChannels']
        
        # แสดงข้อมูลอุปกรณ์ทั้งหมด
        print(f"Index: {i}, Name: '{device_name}', Max Output Channels: {max_output}")
        
        # ตรวจสอบชื่ออุปกรณ์และต้องเป็นอุปกรณ์ที่รองรับ Output
        if target_name in device_name and max_output > 0:
            print(f"\n✅ พบอุปกรณ์ '{target_name}' ที่ Index: {i}")
            p.terminate()
            return i
            
    p.terminate()
    print(f"\n❌ ไม่พบอุปกรณ์ '{target_name}' ที่ใช้งานได้ในระบบ")
    return None

if __name__ == "__main__":
    cable_index = get_cable_output_device_index()
    if cable_index is not None:
        print(f"🎉 CABLE INPUT DEVICE INDEX คือ: {cable_index}")