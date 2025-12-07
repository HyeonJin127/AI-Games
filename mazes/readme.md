# 🚀 AI Q-Learning Maze Demo 프로젝트

이 프로젝트는 Q-Learning 알고리즘을 사용하여 미로를 탈출하는 AI(AI1)를 학습시키고, 규칙 기반의 방해 AI(AI2)와 경쟁하는 환경에서 그 성능을 시각적으로 시뮬레이션합니다. 학습된 Q-테이블은 Supabase 데이터베이스를 통해 관리됩니다.

-----

## 🛠️ 1. 개발 환경 설정

### 1.1. Python 환경 및 의존성

이 프로젝트는 Python 3.11 환경에서 테스트되었습니다.

1.  **가상 환경 생성 및 활성화:**

    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

2.  **필수 패키지 설치:**

    ```bash
    pip install -r requirements.txt
    ```

### 1.2. Supabase 및 환경 변수 설정 (필수)

데이터베이스 연결 정보는 보안을 위해 `.env` 파일을 통해 관리합니다.

1.  프로젝트 루트 폴더에 **`.env`** 파일을 생성합니다.

2.  Supabase 프로젝트 설정에서 다음 세 가지 키를 **큰따옴표 없이** 정확하게 입력합니다.

    ```env
    # .env 파일 내용
    SUPABASE_URL="[Supabase 프로젝트 URL]"
    SUPABASE_ANON_KEY="[Supabase Public Anon Key]"
    SUPABASE_SERVICE_KEY="[Supabase Service Role Key]"
    ```

-----

## 🔒 2. Supabase RLS (Row Level Security) 설정

`q_table_maze_v4` 테이블에 대한 **읽기/쓰기 권한**을 올바르게 설정해야 합니다.

1.  Supabase 대시보드에서 **`q_table_maze_v4`** 테이블을 선택합니다.
2.  **쓰기 권한 (저장):** `train_qlearning.py`는 Service Key를 사용하므로, \*\*RLS 기능을 비활성화(Disable RLS)\*\*하거나 Service Role에 쓰기 권한을 부여해야 합니다. (비활성화 권장)
3.  **읽기 권한 (로드):** `main.py`는 Anon Key를 사용하므로, **SELECT 정책**을 추가하고 `USING expression`을 \*\*`true`\*\*로 설정하여 모든 익명 사용자의 읽기를 허용해야 합니다.

-----

## 🧠 3. AI 학습 및 데이터 저장 (`train_qlearning.py`)

미로 환경에서 AI1을 학습시키고 그 결과를 Supabase에 저장합니다.

### 3.1. 학습 실행

```bash
py -3.11 -m train_qlearning
```

### 3.2. 확인 사항

콘솔에 다음 메시지가 출력되면 성공입니다.

```
[INFO] Q-table saved to Supabase table 'q_table_maze_v4' (XXX entries)
```

-----

## 🎮 4. AI 시뮬레이션 실행 (`main.py`)

학습된 Q-테이블을 로드하고 난이도별 정책을 적용하여 게임을 실행합니다.

### 4.1. 난이도 설정

`main.py` 파일을 열어 전역 변수 `CURRENT_DIFFICULTY`를 변경하여 난이도를 설정할 수 있습니다.

| 난이도 | 효과 | 권장 $\epsilon$ |
| :--- | :--- | :--- |
| **`"easy"`** | 탐험 빈도 높음 (잦은 실수) | $0.7$ |
| **`"normal"`** | 탐험 적당함 (가끔 실수) | $0.3$ |
| **`"hard"`** | 탐험 거의 없음 (최적 행동만 선택) | $0.05$ |

### 4.2. 시뮬레이션 실행

```bash
py -3.11 -m main
```

### 4.3. 확인 사항

화면 우측 사이드바에 \*\*`AI1 Mode: Q-learning`\*\*이 표시되어야 합니다.

-----
