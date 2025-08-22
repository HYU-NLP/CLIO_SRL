# CLIO_SRL

CLIO_SRL은 한국어 서사 복원 프로젝트의 SRL(Narrative Semantic Role Labeling)을 위한 시스템입니다.  
이 저장소는 SRL 태스크를 위한 **데이터 준비**, **모델 학습**, 그리고 **추론 과정**을 포함합니다.

---

## 🧐 Pip install 

    ```bash
    pip install -r requirements.txt
    ```

---

## 📂 Data Preparation

1. 다음 링크에서 [`CLIO SRL dataset_ver2.json`](https://github.com/clioisds/Narrative_mining/tree/main) 파일을 다운로드합니다.
2. 다운로드한 파일을 현재 레포지토리의 루트 디렉토리에 저장합니다.
3. 아래 명령어를 실행하여 데이터를 학습/검증/평가용으로 분할합니다:

    ```bash
    python3 data_split.py
    ```

---

## 🏋️ Train

모델 학습을 위해 아래 쉘 스크립트를 실행합니다:

```bash
sh train.sh
```

> 참고: 하이퍼파라미터나 경로 설정 등은 `train.sh` 내에서 수정 가능합니다.

---

## 🔍 Inference

1. 아래 Google Drive 링크에서 추론용 CSV 파일을 다운로드합니다:  
   👉 [📥 social_line.csv 다운로드](https://drive.google.com/file/d/1kq9_K7CwQJ_k7XOhq4p3Czm-AcS_ePy9/view?usp=sharing)

2. 파일 이름을 `social_line.csv`로 변경한 후, `data/` 디렉토리에 저장합니다.

3. 추론을 위해 아래 쉘 스크립트를 실행합니다:

    ```bash
    sh inference.sh
    ```

---

## 📁 디렉토리 구조 예시

```text
.
├── data/
│   ├── CLIO SRL dataset_ver2.json
│   ├── clio_train.json
│   ├── clio_val.json
│   ├── clio_test.json
│   └── social_line.csv
├── data_split.py
├── train.sh
├── inference.sh
└── ...
```
