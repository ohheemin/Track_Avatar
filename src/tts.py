from gtts import gTTS
import playsound
import os

# 1. 출력할 텍스트 및 설정
text = "빅나티를 변기에 넣고서 내려"
filename = "announce.mp3"

# 2. gTTS 객체 생성 (언어를 'ko'로 설정하여 한국어 음성 사용)
print("음성을 생성 중입니다...")
tts = gTTS(text=text, lang='ko')

# 3. 음성을 mp3 파일로 저장
tts.save(filename)

# 4. 저장된 음성 파일 재생
print("음성을 재생합니다.")
playsound.playsound(filename)

# (선택 사항) 재생이 끝난 후 생성된 mp3 파일 삭제
if os.path.exists(filename):
    os.remove(filename)