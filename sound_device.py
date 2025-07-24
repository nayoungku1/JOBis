import os
from dotenv import load_dotenv
import sounddevice as sd
import numpy as np
import azure.cognitiveservices.speech as speechsdk
from langchain_openai import AzureChatOpenAI

# ── 1) 환경 변수 로드 ─────────────────────────────────
load_dotenv()
speech_key     = os.getenv("SPEECH_KEY")
service_region = os.getenv("REGION")
if not speech_key or not service_region:
    raise RuntimeError(".env에 SPEECH_KEY, REGION 설정 필요")

# ── 2) 녹음 기본 설정 ─────────────────────────────────
SAMPLE_RATE = 16000   # Azure 권장 16kHz
CHANNELS    = 1       # 모노
DTYPE       = 'int16' # 16비트 정수

# ── 3) 녹음 버퍼 준비 ─────────────────────────────────
frames = []  # 녹음된 조각들을 저장할 리스트

def callback(indata, frames_count, time_info, status):
    """InputStream 에서 불리는 함수: indata는 (frames_count, channels) ndarray"""
    if status:
        print(f"⚠ 녹음 에러: {status}")
    # 복사해서 저장 (sounddevice는 내부 버퍼를 재사용함)
    frames.append(indata.copy())

# ── 4) 스트림 열고 녹음 시작 ─────────────────────────────
stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype=DTYPE,
    callback=callback
)
stream.start()
print("● 녹음 중… 말이 끝나면 엔터를 눌러주세요")
input()  # 엔터 입력 대기
stream.stop()
stream.close()
print("✔ 녹음 종료")

# ── 5) 프레임 합치기 ────────────────────────────────────
audio_data = np.concatenate(frames, axis=0)  # (총샘플수, 채널) ndarray

# ── 6) Azure로 보낼 바이트 변환 ─────────────────────────
audio_bytes = audio_data.tobytes()
audio_format = speechsdk.audio.AudioStreamFormat(
    samples_per_second=SAMPLE_RATE,
    bits_per_sample=16,
    channels=CHANNELS
)
push_stream = speechsdk.audio.PushAudioInputStream(audio_format)
push_stream.write(audio_bytes)
push_stream.close()

# ── 7) Azure Speech 설정 및 인식 ────────────────────────
speech_config = speechsdk.SpeechConfig(
    subscription=speech_key,
    region=service_region
)
speech_config.speech_recognition_language = "ko-KR"

audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
recognizer   = speechsdk.SpeechRecognizer(
    speech_config=speech_config,
    audio_config=audio_config
)

print("🔊 Azure Speech 인식 중…")
result = recognizer.recognize_once()

# ── 8) 결과 출력 ───────────────────────────────────────
if result.reason == speechsdk.ResultReason.RecognizedSpeech:
    print("▶ 인식된 텍스트:", result.text)
elif result.reason == speechsdk.ResultReason.NoMatch:
    print("⚠ 음성을 인식할 수 없습니다.")
else:
    print("❌ 오류 발생:", result.reason)


def refine_text(background: str, question: str, transcript: str,
                model: str = "gpt-4o-mini",
                temperature: float = 0.2) -> str:
    """
    배경지식(background), 질문(question), STT 텍스트(transcript)를
    받아서 깔끔하게 다듬은 텍스트를 반환합니다.
    """
    system_prompt = f"""
당신은 한국어 텍스트 교정과 문장 다듬기에 특화된 LLM입니다.
다음 내용을 참고하여, 주어진 STT 결과를 자연스럽고 정확한 문장으로 수정해 주세요.

배경지식:
{background}

질문:
{question}

(STT 결과에는 오타, 띄어쓰기, 문장부호 오류 등이 있을 수 있습니다.)
"""
    messages = [
        {"role": "system",  "content": system_prompt.strip()},
        {"role": "user",    "content": transcript.strip()}
    ]

    resp = AzureChatOpenAI.chat.completions.create(
        model = "gpt-4o-mini",
        messages=messages,
        temperature=temperature,
        max_tokens= max( len(transcript)*2,  256 )
    )
    return resp.choices[0].message.content.strip()


if __name__ == "__main__":
    # 예시 사용
    # background = "이 통화는 고객상담 시나리오입니다."
    # question   = "인사말을 올바르게 교정해 주세요."
    # transcript = "안녕하하세요 고객님 무엇을 도와드릴까요"
    # transcript = result.text
    
    # refined = refine_text(background, question, transcript)
    # print("🔄 다듬어진 문장:", refined)
    llm = AzureChatOpenAI(model = "gpt-4o-mini")
    query1 = result.text+"이 말을 문법에 맞게 수정해줘. 예를들면, 떨림이 있어서 **아안녕하하세요** 라면 **안녕하세요**로 출력해줘. 출력은 반드시 수정한 말만 해. 추가 정보를 덧붙이지마."
    print("인식된 텍스트 : ", result.text)
    response = llm.invoke(query1)
    print("수정된 텍스트: ", response.content)
    query2 = f"""너는 면접관이야. {result.text} 이 말과 {response.content} 이 말을 비교에서 후자에 비해 전자에서 개선해야할 부분들을 출력해줘. 예를들면, **아안녕하하세요**와 **안녕하세요**라면 '떨림이 있으니 줄이세요'로 출력해줘. 출력은 말하는 사람 기준으로 간결하게 해"""
    res = llm.invoke(query2)
    print("피드백: ", res.content)