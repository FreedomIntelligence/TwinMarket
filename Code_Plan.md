md
# TwinMarket KR PoC — 코드 수정 상세 계획

---

## Final Implementation Decisions

### 1. DB 기준 파일
실제 repository에 존재하는 `data/sys_1000.db`를 기준 DB로 사용한다.  
KR PoC용 DB는 이를 복사하여 `data/sys_100_kr.db`로 생성한다.  
문서 내 기존 `sys_100.db` 언급은 모두 `sys_1000.db` 기준으로 해석한다.

### 2. Profile row 기준
`created_at = '2023-06-14 00:00:00'` 하드코딩은 사용하지 않는다.  
각 `user_id`별 최신 row를 기준으로 KR profile 초기화를 수행한다.

### 3. STOCK_PROFILE 처리
실제 코드에서 사용되는 `STOCK_PROFILE_DICT`를 삼성전자 단일종목 기준으로 수정한다.  
문서의 `_STOCK_PROFILE_DICT_RAW`는 실제 코드 구조와 다를 경우 무시한다.  
중국 인덱스 설명이 프롬프트에 남지 않도록 관련 profile 참조를 모두 점검한다.

### 4. init_system 처리
`init_system()` 관련 언급은 실제 repository에서 해당 함수가 존재할 경우에만 수정한다.  
존재하지 않거나 관련성이 낮으면 수정하지 않는다.

### 5. Community OFF 범위
`use_community=False`일 때는 다음을 스킵한다:

- forum post 읽기
- forum post 작성
- like / repost / unlike 등 forum action
- post score update
- graph-based neighbor recommendation
- user graph 생성 및 갱신
- social input 기반 belief update

다음은 유지한다:

- 주가/기술지표 조회
- top_user 뉴스 읽기
- 뉴스 기반 belief update
- buy / hold / sell 의사결정
- 주문 매칭
- Profiles, TradingDetails, StockData 업데이트

### 6. Belief fallback
belief similarity 계산 시 최신 forum post나 belief 텍스트가 없으면 `util/belief/belief_100_kr.csv`의 초기 belief를 fallback으로 사용한다.  
그래도 없으면 빈 문자열이 아니라 아래 neutral default belief 문장을 사용한다.


삼성전자와 한국 주식시장에 대해 중립적인 관점을 유지하며, 현재 정보가 부족하여 추가 뉴스와 가격 흐름을 관찰한다.


### 7. 단일종목 정책

KR PoC에서 거래 가능 종목은 항상 `["005930"]`으로 고정한다.
`cur_positions.keys()` 기반으로 거래 종목을 정하지 않는다.
stock selection LLM 호출은 우회한다.

### 8. Hold 및 빈 주문 처리

`hold`는 완전히 유효한 결정이다.
hold 결정 시 표준 출력은 `action: "hold"` 및 `orders: []`로 통일한다.
downstream parser와 validation은 `orders: []`를 정상적인 무거래 결과로 처리해야 한다.

### 9. 수량 정책

거래 단위는 1주다.
매수/매도 수량은 정수로 변환한다.
단, 계산 결과가 1주 미만이면 주문을 강제로 1주로 올리지 않고 해당 주문을 skip한다.
즉, `quantity < 1`이면 주문 없음으로 처리한다.

### 10. 가격제한 및 틱 처리

KRX 가격제한은 PoC 단순화를 위해 전일 종가 기준 ±30%로 구현한다.
호가단위 tick rounding은 이번 PoC에서는 구현하지 않는다.
주문 가격은 기존 코드의 가격 처리 방식을 최대한 유지한다.

### 11. 뉴스 및 InformationDB 경로

기존 중국 뉴스 데이터 `data/sorted_impact_news.pkl`는 유지한다.
KR PoC에서는 사용자가 이미 수집한 원본 뉴스 파일 `data/samsung_mk_news_20260201_20260331.pkl`을 사용한다.
검증/변환 후 최종 런타임 뉴스 파일명은 `data/samsung_news.pkl`로 표준화한다.
KR FAISS/InformationDB는 `data/InformationDB_samsung/`에 생성한다.
런타임에서 뉴스/InformationDB 경로를 CLI argument로 주입할 수 있도록 한다.

### 12. 로그 경로

Smoke test와 본 실험은 명시적으로 `--log_dir`를 지정한다.
Smoke test는 `logs_kr_smoke`, Community OFF는 `logs_kr_off`, Community ON은 `logs_kr_on`을 사용한다.

### 13. 실험 기간

뉴스 데이터 기간에 맞춰 Smoke test는 `2026-02-02`부터 `2026-02-06`까지로 한다.
본 실험은 `2026-02-02`부터 `2026-03-31` 사이의 KRX 거래일을 사용한다.

### 14. LLM 언어 정책

프롬프트의 자연어 설명은 한국어로 전환한다.
단, parser가 의존하는 JSON/YAML key 이름은 기존 코드와 호환되도록 유지한다.

### 15. ini_cash 변환 규칙

KR PoC에서는 재현성과 비교 가능성을 위해 랜덤 범위를 사용하지 않는다.
기존 소액투자자 티어는 `100,000,000`원으로 변환한다.
기존 대형투자자 티어는 `1,000,000,000`원으로 변환한다.
즉, `ini_cash`는 티어별 고정값을 사용한다.

### 16. 뉴스 pkl 표준 포맷

검증/변환 후 최종 런타임 파일명은 `data/samsung_news.pkl`로 표준화한다.

기대 포맷은 다음과 같다:

* `cal_date`: `YYYY-MM-DD` 형식 문자열
* `news`: `list[str]`

원본 pkl에서 `news`가 단일 문자열이면 리스트로 감싼다.
기사 dict 리스트이면 제목과 본문을 결합하여 문자열 리스트로 변환한다.

### 17. 뉴스 없는 거래일 처리

실험 기간 중 뉴스가 없는 KRX 거래일은 `news=[]`로 처리한다.
직전일 뉴스를 carry-forward하지 않는다.

### 18. InformationDB 문서 단위

InformationDB/FAISS 생성 시 임베딩 문서 단위는 “기사 1건 = 문서 1개”로 한다.
하루치 뉴스를 하나로 합치지 않는다.

---

## AI 에이전트 인수인계 지시사항

이 파일은 작업 진행 로그 겸 계획서다.
Usage 한도 또는 세션 중단이 발생할 수 있으므로, 각 Stage 작업 완료 후 반드시 이 파일의 해당 Stage 섹션을 업데이트해야 한다.

### 새 세션 시작 시 해야 할 일

1. 이 파일 전체를 읽고 가장 마지막으로 완료된 Stage를 확인한다.
2. 가장 마지막으로 완료된 Stage의 완료 노트 섹션을 읽어 컨텍스트를 복원한다.
3. 다음 미완료 Stage부터 이어서 진행한다.

### 각 Stage 완료 후 업데이트 규칙

해당 Stage의 섹션 끝에 아래 형식으로 완료 노트를 추가한다.

```md
#### ✅ 완료 노트 (YYYY-MM-DD)
- **수행한 것**: 실제로 어떤 파일을 어떻게 수정/생성했는지
- **핵심 판단**: 계획과 달라진 점이 있다면 왜 그렇게 결정했는지
- **발견한 문제**: 예상과 다른 코드 구조, 에러, 주의사항
- **다음 Stage 준비**: 다음 작업자가 바로 시작하려면 알아야 할 것
```

Stage를 진행하기 전에 이 지시사항을 다시 읽고, 완료 후 반드시 업데이트할 것.

---

## 기준 정보

* 기준 Repository: `ujlee1661/TwinMarket`
* 기준 브랜치: `main`
* 작성일: 2026-05-13
* 최종 업데이트: 2026-05-14

---

## Stage 0. 준비 작업

### Stage 0-1. 작업 디렉토리 구조 확인

```text
TwinMarket/
├── data/
│   ├── sys_1000.db                         ← 원본 기준 DB
│   ├── sorted_impact_news.pkl              ← 기존 중국 뉴스 유지
│   └── [신규]
│       ├── sys_100_kr.db
│       ├── stock_data_kr.csv
│       ├── trading_days_kr.csv
│       ├── samsung_news.pkl
│       └── samsung_mk_news_20260201_20260331.pkl
├── util/belief/
│   ├── belief_100.csv                       ← 원본 보존
│   └── [신규] belief_100_kr.csv
└── script/
    └── [신규] kr/
        ├── prepare_kr_data.py
        ├── build_kr_news.py
        ├── init_kr_profiles.py
        └── init_kr_beliefs.py
```

### Stage 0-2. 원본 DB 백업

```bash
cp data/sys_1000.db data/sys_1000_original_backup.db
```

---

## Stage 1. 데이터 준비 스크립트 작성

### Stage 1-1. 삼성전자 주가 데이터 생성

**대상 파일**: `script/kr/prepare_kr_data.py`

**목적**: KRX 삼성전자 OHLCV + 기술지표를 수집하여 `data/stock_data_kr.csv`를 생성한다.

**필요 라이브러리**:

* `FinanceDataReader`
* `pykrx`
* `pandas`
* `numpy`

**출력 파일**:

```text
data/stock_data_kr.csv
```

**출력 컬럼**:

```python
columns = [
    "stock_id",
    "date",
    "close_price",
    "pre_close",
    "change",
    "pct_chg",
    "pe_ttm",
    "pb",
    "ps_ttm",
    "dv_ttm",
    "vol",
    "vol_5",
    "vol_10",
    "vol_30",
    "ma_hfq_5",
    "ma_hfq_10",
    "ma_hfq_30",
    "elg_amount_net",
]
```

**처리 로직**:

1. `fdr.DataReader("005930", start, end)` 또는 `pykrx.stock.get_market_ohlcv_by_date()`로 삼성전자 OHLCV를 수집한다.
2. `ma_hfq_5`, `ma_hfq_10`, `ma_hfq_30`을 계산한다.
3. `vol_5`, `vol_10`, `vol_30`을 계산한다.
4. `elg_amount_net`는 기관+외국인 순매수로 대체한다.
5. `elg_amount_net`가 없거나 결측이면 0으로 채운다.
6. `pe_ttm`, `pb`는 가능한 경우 `pykrx.stock.get_market_fundamental_by_date()`에서 수집한다.
7. `ps_ttm`, `dv_ttm` 등 결측 가능성이 높은 값은 0으로 채운다.
8. 최종 CSV를 `data/stock_data_kr.csv`로 저장한다.

**기간 정책**:

* 시뮬레이션 시작 최소 30 거래일 전부터 수집한다.
* Smoke test 기준으로는 최소 `2026-01-01` 이전부터 수집하는 것을 권장한다.
* 본 실험 기간은 `2026-02-02`부터 `2026-03-31`까지이다.

---

### Stage 1-2. 한국 거래 캘린더 생성

**대상 파일**: `script/kr/prepare_kr_data.py`

**목적**: KRX 거래일 캘린더를 생성하여 `data/trading_days_kr.csv`로 저장한다.

**출력 파일**:

```text
data/trading_days_kr.csv
```

**출력 컬럼**:

```text
cal_date,is_open,pretrade_date
```

**처리 로직**:

1. 삼성전자 OHLCV 데이터가 존재하는 날짜를 KRX 거래일로 간주한다.
2. `cal_date`는 `YYYY-MM-DD` 문자열로 저장한다.
3. `is_open`은 거래일이면 1로 저장한다.
4. `pretrade_date`는 직전 거래일을 저장한다.
5. 필요 시 비거래일 row는 생성하지 않아도 된다.
6. 기존 코드가 전체 캘린더를 요구할 경우에만 비거래일을 `is_open=0`으로 추가한다.

---

### Stage 1-3. 초기 KR DB 생성

**대상 파일**: `script/kr/init_kr_profiles.py`

**목적**: `data/sys_1000.db`를 복사한 뒤 한국 시장 PoC용 DB `data/sys_100_kr.db`를 생성한다.

**입력 파일**:

```text
data/sys_1000.db
```

**출력 파일**:

```text
data/sys_100_kr.db
```

**처리 순서**:

```python
# Step A: DB 복사
shutil.copy("data/sys_1000.db", "data/sys_100_kr.db")

# Step B: TradingDetails 전체 삭제
conn.execute("DELETE FROM TradingDetails")

# Step C: Profiles 초기화
# created_at 값을 하드코딩하지 않는다.
# 각 user_id별 최신 created_at row를 기준으로 KR profile 초기화를 수행한다.

for user in all_users:
    if original_ini_cash <= 10_000_000:
        new_ini_cash = 100_000_000
    else:
        new_ini_cash = 1_000_000_000

    # UPDATE Profiles SET
    #   ini_cash          = new_ini_cash,
    #   current_cash      = new_ini_cash,
    #   total_value       = new_ini_cash,
    #   total_return      = 0,
    #   return_rate       = 0.0,
    #   cur_positions     = '{}',
    #   initial_positions = '{"005930": 0}',
    #   stock_returns     = '{}',
    #   yest_returns      = '{}',
    #   fol_ind           = '["전기전자", "반도체"]'
    # WHERE user_id = ? AND created_at = latest_created_at_for_user
```

**중요 정책**:

* `ini_cash`는 랜덤 범위를 사용하지 않는다.
* 소액 투자자 티어는 1억원으로 변환한다.
* 대형 투자자 티어는 10억원으로 변환한다.
* 기존 중국 종목 포지션 및 거래 이력은 제거한다.
* 원본 DB `data/sys_1000.db`는 절대 수정하지 않는다.
* `created_at = '2023-06-14 00:00:00'` 같은 하드코딩은 사용하지 않는다.

---

### Stage 1-4. 초기 belief 재생성

**대상 파일**: `script/kr/init_kr_beliefs.py`

**목적**: 한국 시장 삼성전자 단일종목 환경에 맞는 초기 belief 파일 `util/belief/belief_100_kr.csv`를 생성한다.

**출력 파일**:

```text
util/belief/belief_100_kr.csv
```

**핵심 프롬프트 방향**:

```python
SYSTEM_PROMPT = """
당신은 한국 KOSPI 시장에서 삼성전자(005930) 주식에 투자하는 투자자입니다.
삼성전자는 메모리 반도체(DRAM/NAND), 스마트폰, 디스플레이 사업을 영위하는
KOSPI 시가총액 상위 기업입니다.
"""
```

**유지 사항**:

* 기존 belief 구조는 유지한다.
* 시장 트렌드, 시장 밸류에이션, 경제 상황, 시장 심리, 자기 평가 항목을 유지한다.
* attitude 비율은 기존 계획과 동일하게 유지한다.
* LLM 호출 로직, retry, fallback 구조는 가능한 한 유지한다.
* parser가 의존하는 key 이름은 기존 코드와 호환되도록 유지한다.

---

### Stage 1-5. 기존 뉴스 데이터 검증 및 InformationDB 생성

**대상 파일**: `script/kr/build_kr_news.py`

**목적**: 사용자가 이미 수집한 삼성전자 뉴스 파일 `data/samsung_mk_news_20260201_20260331.pkl`을 검증 및 정규화하여 TwinMarket 호환 뉴스 데이터 `data/samsung_news.pkl`과 `data/InformationDB_samsung/` FAISS 인덱스를 생성한다.

**입력 파일**:

```text
data/samsung_mk_news_20260201_20260331.pkl
```

**출력 파일 및 디렉토리**:

```text
data/samsung_news.pkl
data/InformationDB_samsung/
```

**중요 정책**:

* 기존 NAVER Finance 스크래핑 단계는 수행하지 않는다.
* 기존 중국 뉴스 데이터 `data/sorted_impact_news.pkl`는 유지한다.
* 원본 pkl은 직접 덮어쓰지 않는다.
* 변환 결과만 `data/samsung_news.pkl`로 저장한다.

**기대 포맷**:

```python
# DataFrame
# cal_date: "YYYY-MM-DD" 형식 문자열
# news: list[str]
```

**처리 로직**:

1. `data/samsung_mk_news_20260201_20260331.pkl`을 로드한다.
2. `cal_date`와 `news` 컬럼 존재 여부를 확인한다.
3. `cal_date`를 `YYYY-MM-DD` 문자열로 정규화한다.
4. `news`가 이미 `list[str]`이면 그대로 사용한다.
5. `news`가 단일 문자열이면 `[text]` 형태로 변환한다.
6. `news`가 기사 dict 리스트이면 제목과 본문을 결합하여 문자열 리스트로 변환한다.
7. 실험 기간 중 뉴스가 없는 KRX 거래일은 `news=[]`로 처리한다.
8. 직전일 뉴스를 carry-forward하지 않는다.
9. 최종 DataFrame을 `data/samsung_news.pkl`로 저장한다.
10. InformationDB/FAISS 생성 시 임베딩 문서 단위는 “기사 1건 = 문서 1개”로 한다.
11. 하루치 뉴스를 하나로 합치지 않는다.
12. 임베딩 모델은 `paraphrase-multilingual-MiniLM-L12-v2`를 사용한다.
13. 가능하면 기존 `InformationDB.build_database()` 흐름을 활용한다.

---

## Stage 2. 엔진 수정

### Stage 2-1. 종목 프로필 단일화

**대상 파일**: `trader/utility.py`

**수정 대상**: 실제 코드에서 사용되는 `STOCK_PROFILE_DICT`

```python
STOCK_PROFILE_DICT = {
    "005930": (
        "삼성전자(KRX:005930)는 대한민국 대표 전기전자·반도체 기업이다. "
        "KOSPI 주요 종목으로, 메모리 반도체(DRAM/NAND), 스마트폰(갤럭시), "
        "디스플레이 및 가전 사업을 영위한다."
    )
}
```

**주의 사항**:

* 문서의 `_STOCK_PROFILE_DICT_RAW`는 실제 코드 구조와 다를 경우 무시한다.
* 중국 인덱스 설명이 프롬프트나 추천 로직에 재유입되지 않도록 관련 참조를 점검한다.
* `init_system()` 관련 코드는 실제 repository에 해당 함수가 존재하고 관련성이 있을 때만 수정한다.

---

### Stage 2-2. 가격 제한 수정

**대상 파일**: `trader/matching_engine.py`

**수정 내용**:

```python
upper_limit = last_price * 1.3
lower_limit = last_price * 0.7
```

**정책**:

* KRX 가격제한은 PoC 단순화를 위해 전일 종가 기준 ±30%로 구현한다.
* 호가단위 tick rounding은 이번 PoC에서는 구현하지 않는다.
* 기존 ±10% 관련 주석 및 설명도 함께 수정한다.
* 주문 가격 처리 방식은 기존 코드를 최대한 유지한다.

---

### Stage 2-3. 거래 단위, community 토글, 단일종목 고정

**대상 파일**: `trader/trading_agent.py`

#### 2-3-1. 거래 단위 수정

기존 100주 단위 로직을 제거하고, 1주 단위 주문을 허용한다.

```python
quantity = int(quantity)

if quantity < 1:
    continue
```

**정책**:

* `quantity < 1`이면 1주로 강제하지 않고 해당 주문을 skip한다.
* 매수/매도 모두 동일한 수량 정책을 적용한다.
* 잔고보다 많은 매도 수량은 기존 보유 수량 내에서 clamp한다.

#### 2-3-2. community 토글 추가

```python
def __init__(self, ..., use_community: bool = True, ...):
    self.use_community = use_community
```

#### 2-3-3. Community OFF 처리

```python
if self.use_community:
    # forum read/write/reaction/recommendation/social belief update 수행
    ...
else:
    forum_args = None
    post_response_args = None
```

OFF 시 스킵:

* forum post 읽기
* forum post 작성
* like / repost / unlike 등 forum action
* graph-based neighbor recommendation
* social input 기반 belief update

OFF 시 유지:

* 뉴스 읽기
* 가격/기술지표 조회
* 뉴스 기반 belief update
* buy / hold / sell 의사결정

#### 2-3-4. 단일종목 고정

```python
stocks_to_deal = ["005930"]
```

**정책**:

* `cur_positions.keys()` 기반으로 거래 종목을 정하지 않는다.
* stock selection LLM 호출은 우회한다.
* 삼성전자 `005930`만 거래 가능 종목으로 사용한다.

---

### Stage 2-4. 프롬프트 수정

**대상 파일**: `trader/prompts.py`

#### 2-4-1. 강제 거래 문구 제거

기존 중국어 강제 문구 및 유사 문구를 제거한다.

```text
你必须选择至少一个指数进行buy或者sell类型的交易，否则会受到惩罚
```

#### 2-4-2. 대체 문구

```text
hold도 완전히 유효한 결정입니다. 현재 시장 상황과 포트폴리오에 따라 buy / hold / sell 중 가장 적합한 결정을 자유롭게 내리세요.
```

#### 2-4-3. hold 표준 출력 포맷

```python
{
    "action": "hold",
    "orders": []
}
```

#### 2-4-4. 한국어화 대상

* `中国A股市场投资者` → `한국 KOSPI 삼성전자 투자자`
* `元` → `원`
* `指数` → `삼성전자`
* 중국 시장/중국 인덱스 관련 설명 → 한국 시장/삼성전자 설명

단, parser가 의존하는 JSON/YAML key 이름은 기존 코드와 호환되도록 유지한다.

---

### Stage 2-5. 단일종목 소셜 그래프 재설계

**대상 파일**: `util/UserDB.py`

**기존 알고리즘**:

* `TradingDetails.industry` 기반 weighted Jaccard similarity

**신규 알고리즘**:

* 보유비율 유사도
* belief 텍스트 cosine similarity

**신규 함수 예시**:

```python
def build_graph_new_single_stock(
    db_path: str,
    forum_db_path: str,
    current_date: str,
    similarity_threshold: float = 0.2,
    save_name: str = "user_graph",
    save: bool = True,
) -> nx.Graph:
    """
    단일종목 환경을 위한 소셜 그래프 구축.
    연결 강도 = 보유비율 유사도 0.5 + belief 유사도 0.5
    """
```

**처리 로직**:

1. 각 유저의 최신 포지션 비율을 로드한다.

   * `cur_positions["005930"]["ratio"]`
   * 없으면 0으로 처리한다.
2. 각 유저의 최신 belief 텍스트를 로드한다.

   * 최신 forum post 또는 belief 저장 위치에서 가져온다.
   * 없으면 `util/belief/belief_100_kr.csv`의 초기 belief를 fallback으로 사용한다.
   * 그래도 없으면 neutral default belief 문장을 사용한다.
3. belief 텍스트를 동일 embedding model로 벡터화한다.
4. pairwise similarity를 계산한다.

   * `position_sim = 1 - abs(ratio_i - ratio_j) / 100`
   * `belief_sim = cosine_similarity(vec_i, vec_j)`
   * `total_sim = 0.5 * position_sim + 0.5 * belief_sim`
5. `similarity_threshold` 이상인 pair에 edge를 추가한다.
6. 초기에는 TradingDetails가 비어 있으므로 belief similarity 중심으로 그래프를 생성한다.
7. `use_community=False`인 경우 user graph 생성 및 갱신은 스킵한다.

---

### Stage 2-6. 시뮬레이션 엔트리포인트 수정

**대상 파일**: `simulation.py`

#### 2-6-1. 신규 파라미터 추가

```python
parser.add_argument("--use_community", type=lambda x: x.lower() == "true", default=True)
parser.add_argument("--stock_code", type=str, default="005930")
parser.add_argument("--stock_data_path", type=str, default="data/stock_data_kr.csv")
parser.add_argument("--trading_days_path", type=str, default="data/trading_days_kr.csv")
parser.add_argument("--news_path", type=str, default="data/samsung_news.pkl")
parser.add_argument("--log_dir", type=str, default="logs")
```

#### 2-6-2. Community OFF 처리

```python
if use_community:
    asyncio.run(execute_forum_actions(...))
    update_posts_score_by_date_range(...)
else:
    # forum side-effect 없음
    pass
```

#### 2-6-3. 그래프 생성 처리

```python
if use_community:
    current_user_graph = build_graph_new_single_stock(
        db_path=user_db,
        forum_db_path=forum_db,
        current_date=current_date.strftime("%Y-%m-%d"),
        similarity_threshold=similarity_threshold,
    )
else:
    current_user_graph = None
```

**중요 정책**:

* `use_community=False`일 때 forum read/write/action/score update/user graph 갱신이 발생하면 안 된다.
* 단, 거래, 뉴스 읽기, belief update, 주문 매칭, DB 업데이트는 계속 수행한다.
* `process_user_input()` 또는 `PersonalizedStockTrader` 생성 시 `use_community`를 전달한다.

---

## Stage 3. 검증

### Stage 3-1. Smoke Test

초기 smoke test는 뉴스 데이터 기간에 맞춰 `2026-02-02`부터 `2026-02-06`까지 수행한다.
Community OFF 상태에서 10명 규모로 짧은 시뮬레이션을 실행하여 KR 데이터 경로, 단일종목 `005930` 거래, hold 처리, DB 업데이트, 로그 생성이 정상 동작하는지 확인한다.

```bash
python simulation.py \
  --start_date 2026-02-02 --end_date 2026-02-06 \
  --stock_code 005930 \
  --use_community False \
  --node 10 \
  --user_db data/sys_100_kr.db \
  --forum_db data/forum_kr.db \
  --belief_init_path util/belief/belief_100_kr.csv \
  --stock_data_path data/stock_data_kr.csv \
  --trading_days_path data/trading_days_kr.csv \
  --news_path data/samsung_news.pkl \
  --log_dir logs_kr_smoke
```

**체크포인트**:

* [ ] 에러 없이 smoke test 기간 완주
* [ ] `logs_kr_smoke/trading_records/` 생성 확인
* [ ] 거래 로그에 `005930` 포함 확인
* [ ] hold 결정 시 parser 에러 없음 확인
* [ ] 빈 주문 리스트 `orders=[]`가 정상 처리되는지 확인
* [ ] DB `StockData` 테이블에 `005930` 행 추가 확인
* [ ] DB `TradingDetails` 업데이트 정상 확인
* [ ] 포지션/현금 업데이트 정상 확인
* [ ] Community OFF 상태에서 forum write/action/score update 발생하지 않음 확인

---

### Stage 3-2. Community OFF 실험

본 실험은 `2026-02-02`부터 `2026-03-31`까지 진행한다.
Community OFF 조건에서 social input 없이 시장 재현성과 agent behavior를 측정한다.

```bash
python simulation.py \
  --start_date 2026-02-02 --end_date 2026-03-31 \
  --use_community False \
  --node 100 \
  --user_db data/sys_100_kr.db \
  --forum_db data/forum_kr_off.db \
  --stock_data_path data/stock_data_kr.csv \
  --trading_days_path data/trading_days_kr.csv \
  --news_path data/samsung_news.pkl \
  --log_dir logs_kr_off
```

---

### Stage 3-3. Community ON 실험

동일 기간 `2026-02-02`부터 `2026-03-31`까지 Community ON 조건으로 실행한다.
Community ON/OFF agent behavior 차이를 비교한다.

```bash
python simulation.py \
  --start_date 2026-02-02 --end_date 2026-03-31 \
  --use_community True \
  --node 100 \
  --user_db data/sys_100_kr.db \
  --forum_db data/forum_kr_on.db \
  --stock_data_path data/stock_data_kr.csv \
  --trading_days_path data/trading_days_kr.csv \
  --news_path data/samsung_news.pkl \
  --log_dir logs_kr_on
```

---

## 수정 파일 요약

| Stage | 파일                              | 수정 유형 | 핵심 변경                                              |
| ----- | ------------------------------- | ----- | -------------------------------------------------- |
| 1-1   | `script/kr/prepare_kr_data.py`  | 신규    | 삼성전자 OHLCV/기술지표 생성                                 |
| 1-2   | `script/kr/prepare_kr_data.py`  | 신규    | KRX 거래일 캘린더 생성                                     |
| 1-3   | `script/kr/init_kr_profiles.py` | 신규    | `sys_1000.db` 기반 KR DB 생성, 포지션 초기화, ini_cash 고정 변환 |
| 1-4   | `script/kr/init_kr_beliefs.py`  | 신규    | 한국 시장 삼성전자 belief 생성                               |
| 1-5   | `script/kr/build_kr_news.py`    | 신규    | 기존 뉴스 pkl 검증/변환 + FAISS 생성                         |
| 2-1   | `trader/utility.py`             | 기존 수정 | `STOCK_PROFILE_DICT` 삼성전자 단일종목화                    |
| 2-2   | `trader/matching_engine.py`     | 기존 수정 | 가격제한 ±10% → ±30%                                   |
| 2-3   | `trader/trading_agent.py`       | 기존 수정 | 1주 단위, community 토글, 단일종목 고정                       |
| 2-4   | `trader/prompts.py`             | 기존 수정 | 강제 거래 제거, hold 허용, 한국어화                            |
| 2-5   | `util/UserDB.py`                | 기존 수정 | 단일종목용 belief+포지션 그래프 알고리즘                          |
| 2-6   | `simulation.py`                 | 기존 수정 | `use_community`, KR 데이터 경로, community OFF 분기       |

---

## 실행 순서

1. Stage 0 준비 작업 확인
2. Stage 1 데이터 준비 스크립트 구현
3. Stage 1 생성 산출물 검증
4. Stage 2 엔진 수정
5. Stage 3 smoke test 실행
6. Stage 3 Community OFF 실험 실행
7. Stage 3 Community ON 실험 실행
8. ON/OFF 결과 비교 분석

---

*작성: 2026-05-13*
*최종 업데이트: 2026-05-14*

```
```

#### ✅ 완료 노트 (2026-05-14) — Stage 1-1 ~ Stage 1-5
- **수행한 것**:
  - `script/kr/prepare_kr_data.py` 신규 생성: 삼성전자 OHLCV/기초지표 수집, 기술지표 계산, `data/stock_data_kr.csv` 및 `data/trading_days_kr.csv` 생성 로직 구현.
  - `script/kr/init_kr_profiles.py` 신규 생성: `data/sys_1000.db` 복사 후 `data/sys_100_kr.db` 생성, `TradingDetails` 삭제, 사용자별 최신 `Profiles` row 기준 KR 초기화 구현.
  - `script/kr/init_kr_beliefs.py` 신규 생성: `util/belief/belief_100.csv` 기반 `util/belief/belief_100_kr.csv` 생성 및 KR neutral belief 반영.
  - `script/kr/build_kr_news.py` 신규 생성: 삼성 뉴스 pkl 정규화(`cal_date`,`news`), 거래일 merge 기반 뉴스 없는 날짜 `[]` 처리, `data/samsung_news.pkl` 저장 및 `data/InformationDB_samsung/` FAISS/metadata 생성.
- **핵심 판단**:
  - Stage 2 엔진 코드는 요청에 따라 미수정.
  - China-market 기존 파일은 수정하지 않고 KR 전용 스크립트/출력 경로만 추가.
- **발견한 문제**:
  - 원격 브랜치 `kr-samsung-poc`는 기존에 없어 로컬에서 신규 생성.
- **다음 Stage 준비**:
  - Stage 2 진행 시 `--news_pkl`, `--information_db_dir` CLI 주입 경로를 런타임 코드에 연결 필요.
