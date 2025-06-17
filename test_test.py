import pvporcupine
from pvrecorder import PvRecorder
from dotenv import load_dotenv
import os

load_dotenv()

access_key = os.getenv("ACCESS_KEY")
keywords = os.getenv("KEYWORDS")

# for keyword in pvporcupine.KEYWORDS:
#     print(keyword)

porcupine = pvporcupine.create(
    access_key=access_key, 
    keywords=["hey barista"]
    # keyword_paths=["./wake_words/Hey-Coffee-Bot_en_linux_v3_0_0.ppn"]
)
recoder = PvRecorder(device_index=-1, frame_length=porcupine.frame_length)

try:
    recoder.start()

    while True:
        keyword_index = porcupine.process(recoder.read())
        if keyword_index >= 0:
            print(f"Detected {keywords[keyword_index]}")

except KeyboardInterrupt:
    recoder.stop()
finally:
    porcupine.delete()
    recoder.delete()

