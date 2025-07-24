import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk

# .env 파일에서 환경 변수 불러오기
load_dotenv()

# 1. Azure Speech 서비스 키와 지역을 환경 변수에서 가져오기
speech_key = os.getenv("AZURE_SPEECH_KEY")
speech_region = os.getenv("AZURE_SPEECH_REGION")

# --- 나머지 코드는 이전과 동일 ---
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
reference_text = "안녕하세요"
pronunciation_config = speechsdk.PronunciationAssessmentConfig(
    reference_text=reference_text,
    grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
    granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme
)

audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

pronunciation_config.apply_to(speech_recognizer)

print(f'"{reference_text}" 문장을 마이크에 대고 말해주세요...')
result = speech_recognizer.recognize_once()

if result.reason == speechsdk.ResultReason.RecognizedSpeech:
    print(f"인식된 텍스트: {result.text}")
    pronunciation_result = speechsdk.PronunciationAssessmentResult(result)
    print(f"  - 종합 정확도 점수: {pronunciation_result.accuracy_score}")
    print(f"  - 유창성 점수: {pronunciation_result.fluency_score}")
else:
    print("음성을 인식하지 못했습니다.")