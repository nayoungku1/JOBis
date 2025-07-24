# J.O.B.I.S. - AI 면접 준비 도우미

## 프로젝트 개요
Jobis는 개인화된 면접 준비를 지원하는 AI 기반 도구입니다. 사용자의 이력서, 채용 공고, 웹 데이터를 분석하여 직무에 특화된 맞춤형 면접 질문을 생성하고, 기존 AI 면접 프로그램의 한계를 극복하여 개인화된 피드백과 동적 데이터 활용을 제공합니다.

### 주요 기능
- **개인화된 면접 질문 생성**: 이력서(PDF/Docx), 개인 홈페이지(URL), 채용 공고를 기반으로 맞춤형 질문 생성
- **웹 데이터 분석**: 인터넷 검색 및 웹 스크래핑을 통해 회사 및 직무 맥락 분석
- **기존 서비스와의 차별점**:
  - 단순 질문 리스트 제공에서 벗어나 직무 특화 질문 제공
  - 개인 특성 반영 및 동적 데이터베이스 기반 질문 생성
  - 답변 개선 지원 기능 포함

## 설치 및 환경 설정

### Prerequisites
- **Conda**: 프로젝트는 Conda 환경을 사용합니다. [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 또는 [Anaconda](https://www.anaconda.com/products/distribution)를 설치하세요.

### 환경 설정
1. **Conda 환경 생성**:
   `environment.yml` 파일을 사용하여 `jobis` 환경을 생성합니다.
   ```bash
   conda env create -f environment.yml
   ```
   `environment.yml` 파일은 아래와 같습니다:

   ```yaml
   name: jobis
   channels:
     - defaults
     - conda-forge
   dependencies:
     - python=3.9
     - streamlit=1.31.0
     - pandas=2.2.0
     - requests=2.31.0
     - beautifulsoup4=4.12.3
     - python-docx=1.1.0
     - pypdf=4.0.1
     - pip
     - pip:
         - wikipedia-api==0.6.0
   ```

2. **환경 활성화**:
   ```bash
   conda activate jobis
   ```

3. **의존성 설치 확인**:
   환경 생성 후 설치된 패키지를 확인하려면:
   ```bash
   conda list
   ```

### `environment.yml` 생성 방법
현재 환경을 `environment.yml`로 내보내고 싶을 경우, 활성화된 환경에서 다음 명령어를 실행하세요:
```bash
conda env export --no-builds > environment.yml
```
이 명령은 플랫폼 독립적인 환경 파일을 생성합니다.

## 실행 방법
1. **환경 활성화**:
   ```bash
   conda activate jobis
   ```
2. **애플리케이션 실행**:
   프로젝트 디렉토리에서 다음 명령어를 실행하여 Streamlit 앱을启动합니다:
   ```bash
   streamlit run app.py
   ```
3. **브라우저에서 확인**:
   Streamlit 서버가 시작되면, 브라우저에서 `http://localhost:8501`로 접속하여 Jobis 인터페이스를 확인하세요.

## 프로젝트 구조
```
jobis/
├── app.py                # Streamlit 애플리케이션 메인 파일
├── environment.yml       # Conda 환경 설정 파일
├── README.md            # 프로젝트 설명 문서
└── data/                # 이력서, 채용 공고 등 입력 데이터 저장 폴더
```

## 기대 효과 및 향후 발전 방향
- **기대 효과**:
  - 개인화된 면접 준비로 지원자의 경쟁력 강화
  - 데이터 기반의 직무 특화 질문으로 실질적인 면접 연습 제공
- **향후 발전 방향**:
  - 자체 데이터 수집 기반 마케팅 인프라 확보
  - 외부 파트너와의 협업을 통한 기능 확장
  - 트렌드 변화에 빠르게 대응하는 시스템 구축

## 문서 기반 정보
- **문제 정의**: "면접, 혼자서 잘 준비할 수 있을까?"라는 고민을 해결하기 위해 설계
- **차별점**:
  - 기존 서비스: 단순 질문 제공, 표면적 평가, 고정 DB 기반
  - Jobis: 개인화된 질문, 동적 데이터 활용, 답변 개선 지원
- **작동 원리**: 이력서, 채용 공고, 웹 데이터 스크래핑을 통해 맥락 분석 후 질문 생성
- **시연**: 이력서 및 채용 공고 파일 업로드 후 분석 및 질문 생성 데모

## 기여 방법
1. 저장소를 클론합니다:
   ```bash
   git clone <repository-url>
   ```
2. 새로운 브랜치를 생성하여 기능 개발 후 풀 리퀘스트를 제출하세요.

## 문의
궁금한 점이 있으시면 [이메일](mailto:example@jobis.com)로 연락 주세요.

---

**Thank you**  
함께 해주셔서 감사합니다!