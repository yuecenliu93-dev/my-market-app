from __future__ import annotations

import base64
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import fitz
from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:  # The app still works without Vision until openai is installed.
    OpenAI = None


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
REPORT_PATTERNS = [
    "daily_context.txt",
    "daily_context_*.txt",
    "daily_context-*.txt",
    "history/daily_context*.txt",
]
SOURCE_PATTERN = re.compile(
    r"--- SOURCE: (?P<source>.*?) ---\n(?P<body>.*?)(?=\n\n--- SOURCE:|\Z)",
    re.DOTALL,
)
UPLOAD_DIR = BASE_DIR / "reports" / "uploads"
CHART_UPLOAD_DIR = BASE_DIR / "reports" / "chart_uploads"
VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini")

SYSTEM_PROMPT = """
You are a top-tier AM/WM investment strategist preparing a student for NYC/HK banking and wealth management interviews.

Output rules:
- Every section and every bullet must be bilingual: English first, then Traditional Chinese.
- Do not over-summarize. Explain causal chains: event -> market pricing -> asset impact -> client implication.
- Focus on clarity, depth, and reasoning. Avoid one-sentence shortcuts unless the point is genuinely simple.
- Tag regions explicitly in asset-class views, for example [US], [HK/China], [Japan], [Global].
- Distinguish what markets already price from where the surprise or divergence may sit.

Required structure:
1. Macro Views (宏觀事件短評)
2. Market Consensus & Divergence (金融市場共識與分歧點)
3. Asset Classes (不同資產類別反應及原因): Equities, Fixed Income, Commodities, Alts
4. Hot Topics (熱點與長期關注話題)
5. Summary for Morning Meeting (晨會小結)
"""

VISION_PROMPT = """
Analyze this macro/market chart for an AM/WM interview dashboard.

Return bilingual English + Traditional Chinese output. Be specific about:
1. What the chart likely measures and the direction/trend.
2. The economic meaning and causal chain.
3. Whether the insight belongs more in Macro Views, Asset Classes, or both.
4. Asset-class implications with explicit region tags such as [US], [HK/China], [Japan], [Global].

Use this exact format:
MACRO_VIEWS:
- EN: ...
  繁中: ...

ASSET_CLASSES:
- EN: ...
  繁中: ...

HOT_TOPIC:
- EN: ...
  繁中: ...
"""


@dataclass(frozen=True)
class ReportFile:
    path: Path
    label: str
    generated_at: datetime | None


def page_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }
        .metric-card {
            border: 1px solid #e6e8ef;
            border-radius: 8px;
            padding: 16px 18px;
            background: #ffffff;
            min-height: 118px;
            box-shadow: 0 1px 3px rgba(16, 24, 40, 0.04);
        }
        .metric-card h3 {
            font-size: 0.86rem;
            color: #475467;
            margin: 0 0 8px 0;
            font-weight: 700;
        }
        .metric-card p {
            color: #101828;
            font-size: 1rem;
            line-height: 1.45;
            margin: 0;
        }
        .section-note {
            color: #475467;
            font-size: 0.95rem;
            line-height: 1.55;
        }
        .flashcard {
            border-left: 5px solid #0f766e;
            background: #f8fbfb;
            padding: 18px 20px;
            border-radius: 8px;
            color: #101828;
            line-height: 1.55;
        }
        .source-box {
            border: 1px solid #e6e8ef;
            border-radius: 8px;
            padding: 14px 16px;
            background: #fcfcfd;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def source_block(source_name: str, body: str) -> str:
    clean_body = body.strip() or "No content collected."
    return f"--- SOURCE: {source_name} ---\n{clean_body}\n"


def parse_generated_at(text: str) -> datetime | None:
    match = re.search(r"Generated:\s*([0-9T:\-.]+)", text)
    if not match:
        return None
    try:
        return datetime.fromisoformat(match.group(1))
    except ValueError:
        return None


def discover_reports() -> list[ReportFile]:
    seen: set[Path] = set()
    reports: list[ReportFile] = []

    for pattern in REPORT_PATTERNS:
        for path in BASE_DIR.glob(pattern):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            text = read_text(path)
            generated_at = parse_generated_at(text)
            if generated_at:
                date_label = generated_at.strftime("%Y-%m-%d %H:%M")
            else:
                date_label = datetime.fromtimestamp(path.stat().st_mtime).strftime(
                    "%Y-%m-%d %H:%M"
                )
            reports.append(ReportFile(path=path, label=f"{date_label} · {path.name}", generated_at=generated_at))

    return sorted(
        reports,
        key=lambda item: item.generated_at or datetime.fromtimestamp(item.path.stat().st_mtime),
        reverse=True,
    )


def parse_sources(text: str) -> dict[str, str]:
    return {
        match.group("source").strip(): match.group("body").strip()
        for match in SOURCE_PATTERN.finditer(text)
    }


def safe_filename(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return clean or "uploaded_report.pdf"


def extract_pdf_text(data: bytes) -> str:
    lines: list[str] = []
    with fitz.open(stream=data, filetype="pdf") as document:
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text("text").strip()
            if page_text:
                lines.append(f"[Page {page_number}]\n{page_text}")
    return "\n\n".join(lines).strip()


def make_manual_source_body(title: str, text: str, submitted_at: datetime) -> str:
    summary = clean_summary(text)[:1800]
    return (
        f"1. {title}\n"
        f"   Published: {submitted_at.isoformat(timespec='minutes')}\n"
        f"   Summary: {summary}\n\n"
        f"Full Text:\n{text.strip()}"
    )


def session_sources() -> dict[str, str]:
    if "custom_sources" not in st.session_state:
        st.session_state.custom_sources = {}
    return st.session_state.custom_sources


def session_chart_insights() -> list[dict[str, str]]:
    if "chart_insights" not in st.session_state:
        st.session_state.chart_insights = []
    return st.session_state.chart_insights


def get_openai_api_key() -> str | None:
    load_dotenv()
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY")
        if secret_key:
            return str(secret_key)
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


def image_data_url(data: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def extract_labeled_section(text: str, label: str) -> str:
    pattern = re.compile(
        rf"{label}:\s*(.*?)(?=\n[A-Z_]+:\s*|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def analyze_chart_with_vision(data: bytes, mime_type: str) -> tuple[bool, str]:
    api_key = get_openai_api_key()
    if OpenAI is None:
        return False, "The openai package is not installed. Add openai to requirements.txt and reinstall dependencies."
    if not api_key:
        return (
            False,
            "OPENAI_API_KEY is not configured. Add it locally in .env or in Streamlit Community Cloud secrets to enable GPT Vision chart analysis.",
        )

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=VISION_MODEL,
        instructions=SYSTEM_PROMPT,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": VISION_PROMPT},
                    {
                        "type": "input_image",
                        "image_url": image_data_url(data, mime_type),
                        "detail": "high",
                    },
                ],
            }
        ],
    )
    return True, response.output_text.strip()


def process_uploaded_charts(uploaded_charts: list) -> None:
    if not uploaded_charts:
        return

    CHART_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    processed_keys = st.session_state.setdefault("processed_chart_keys", set())
    insights = session_chart_insights()

    for chart in uploaded_charts:
        file_key = f"{chart.name}:{chart.size}"
        if file_key in processed_keys:
            continue

        data = chart.getvalue()
        mime_type = chart.type or "image/png"
        saved_path = CHART_UPLOAD_DIR / safe_filename(chart.name)
        saved_path.write_bytes(data)

        with st.sidebar:
            st.image(data, caption=f"Chart uploaded: {chart.name}", use_container_width=True)
            with st.spinner(f"Reading chart with GPT Vision: {chart.name}"):
                ok, result = analyze_chart_with_vision(data, mime_type)

        if ok:
            insight = {
                "name": chart.name,
                "macro": extract_labeled_section(result, "MACRO_VIEWS") or result,
                "asset": extract_labeled_section(result, "ASSET_CLASSES") or result,
                "hot_topic": extract_labeled_section(result, "HOT_TOPIC"),
                "full": result,
            }
            insights.append(insight)
            session_sources()[f"Uploaded Chart: {chart.name}"] = make_manual_source_body(
                f"Uploaded MacroMicro chart: {chart.name}",
                result,
                datetime.now(),
            )
            st.sidebar.success(f"Chart analyzed: {chart.name}")
        else:
            session_sources()[f"Uploaded Chart: {chart.name}"] = (
                f"ERROR: {result}\nSaved image: {saved_path}"
            )
            st.sidebar.warning(result)

        processed_keys.add(file_key)


def process_uploaded_pdfs(uploaded_files: list) -> None:
    if not uploaded_files:
        return

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    custom_sources = session_sources()

    for uploaded_file in uploaded_files:
        file_key = f"{uploaded_file.name}:{uploaded_file.size}"
        processed_keys = st.session_state.setdefault("processed_upload_keys", set())
        if file_key in processed_keys:
            continue

        data = uploaded_file.getvalue()
        saved_path = UPLOAD_DIR / safe_filename(uploaded_file.name)
        saved_path.write_bytes(data)

        try:
            extracted = extract_pdf_text(data)
            if not extracted:
                extracted = "No readable text found in this PDF."
            source_name = f"Uploaded PDF: {uploaded_file.name}"
            custom_sources[source_name] = make_manual_source_body(
                f"Uploaded investment report: {uploaded_file.name}",
                extracted,
                datetime.now(),
            )
            processed_keys.add(file_key)
            st.sidebar.success(f"Parsed PDF: {uploaded_file.name}")
        except Exception as exc:
            custom_sources[f"Uploaded PDF: {uploaded_file.name}"] = (
                f"ERROR: Could not parse uploaded PDF saved at {saved_path}: {exc}"
            )
            processed_keys.add(file_key)
            st.sidebar.error(f"PDF parse failed: {uploaded_file.name}")


def clean_summary(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"Learn more about your ad choices\..*", "", value).strip()
    return value


def parse_items(source: str, body: str) -> list[dict[str, str]]:
    pattern = re.compile(
        r"(?ms)^\s*(?P<num>\d+)\.\s+(?P<title>.*?)\n"
        r"(?:\s+Published:\s*(?P<published>.*?)\n)?"
        r"(?P<rest>.*?)(?=^\s*\d+\.\s+|\Z)"
    )
    items: list[dict[str, str]] = []

    for match in pattern.finditer(body):
        rest = match.group("rest") or ""
        summary_match = re.search(
            r"Summary(?: / transcript note)?:\s*(.*?)(?:\n\s+(?:Link|Audio):|\nFull Text:|\Z)",
            rest,
            re.DOTALL,
        )
        link_match = re.search(r"\n\s+(?:Link|Audio):\s*(\S+)", rest)
        items.append(
            {
                "Source": source,
                "Title": clean_summary(match.group("title")),
                "Published": clean_summary(match.group("published") or ""),
                "Summary": clean_summary(summary_match.group(1) if summary_match else ""),
                "Link": link_match.group(1) if link_match else "",
            }
        )

    return items


def build_news_table(sources: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for source, body in sources.items():
        if source == "Run Metadata":
            continue
        rows.extend(parse_items(source, body))

    if not rows:
        return pd.DataFrame(columns=["Source", "Title", "Published", "Summary", "Link"])
    return pd.DataFrame(rows)


def has_any(text: str, keywords: list[str]) -> bool:
    lower = text.lower()
    return any(keyword.lower() in lower for keyword in keywords)


def bilingual(en: str, zh: str) -> str:
    return f"- **EN:** {en}\n  \n  **繁中：** {zh}"


def build_analysis(
    text: str, chart_insights: list[dict[str, str]] | None = None
) -> dict[str, list[str] | str]:
    macro_views: list[str] = []
    consensus_divergence: list[str] = []
    asset_classes: list[str] = []
    hot_topics: list[str] = []

    if has_any(text, ["u.s.-china", "us-china", "china summit", "tariff", "rare earth"]):
        macro_views.append(
            bilingual(
                "The U.S.-China relationship looks more managed, but not structurally repaired. Dialogue can reduce near-term tail risk, yet tariffs, semiconductor restrictions and rare-earth controls still sit in the background, so markets may price lower event risk without pricing a full geopolitical reset.",
                "美中關係看起來更可控，但並不代表結構性修復。高層對話可以降低短期尾部風險，但關稅、半導體限制與稀土管制仍是底層矛盾，因此市場可能只是在定價事件風險下降，而不是定價地緣政治全面轉向。",
            )
        )
        consensus_divergence.append(
            bilingual(
                "Consensus is that diplomacy supports risk assets by lowering policy volatility. The divergence is that investors may underestimate how quickly sector-specific restrictions can return, especially in tech hardware, advanced chips and strategic materials.",
                "市場共識是外交溝通有利風險資產，因為政策波動下降。分歧點在於，投資人可能低估了針對特定產業的限制重新升溫的速度，尤其是科技硬體、先進晶片與戰略原物料。",
            )
        )
        asset_classes.append(
            bilingual(
                "[US]/[HK/China] Equities: Keep quality exposure to global tech supply chains, but avoid treating the summit as an all-clear signal. The 'why' is that lower diplomatic tension can support multiples, while export controls can still hit earnings visibility.",
                "[美國]/[港股/中國] 股票：可保留高品質科技供應鏈 exposure，但不應把峰會視為全面解除警報。原因是外交緊張下降有利估值，但出口管制仍可能壓低企業盈利能見度。",
            )
        )
        hot_topics.append(
            bilingual(
                "Client talking point: 'Managed competition' is the right phrase. It tells clients why markets can rally on diplomacy, while portfolios still need diversification across regions and supply-chain exposures.",
                "客戶溝通話題：最適合的表述是「可控競爭」。這能解釋為什麼市場會因外交緩和而上漲，同時投資組合仍需要跨區域與供應鏈分散。",
            )
        )

    if has_any(text, ["inflation", "consumer confidence", "gas", "yield", "treasur"]):
        macro_views.append(
            bilingual(
                "Inflation remains the key macro constraint because it affects the Fed's reaction function. If consumer confidence softens but inflation expectations stay sticky, the Fed has less room to cut aggressively, which can keep the 10-year Treasury yield elevated or range-bound rather than decisively lower.",
                "通膨仍是核心宏觀約束，因為它直接影響聯準會反應函數。如果消費者信心轉弱但通膨預期仍具黏性，Fed 就較難大幅降息，10 年期美債殖利率可能維持高位或區間震盪，而不是快速下行。",
            )
        )
        consensus_divergence.append(
            bilingual(
                "The street broadly prices a soft landing with eventual easing. The contrarian risk is a 'no landing' or sticky-inflation path where growth is not weak enough to justify cuts, but inflation is strong enough to pressure duration and equity multiples.",
                "市場普遍定價軟著陸與最終降息。反向風險是「不著陸」或通膨黏性路徑：經濟沒有弱到足以支持降息，但通膨又強到足以壓制久期資產與股票估值。",
            )
        )
        asset_classes.append(
            bilingual(
                "[US] Fixed Income: Selective duration can work if growth slows, but investors should scale in rather than make a one-way rate-cut bet. The 'why' is that sticky inflation can make long-end yields rebound even when activity data cools.",
                "[美國] 固定收益：若成長放緩，選擇性增加久期是合理的，但不應一次性押注降息。原因是即使經濟數據降溫，通膨黏性仍可能讓長端殖利率反彈。",
            )
        )
        asset_classes.append(
            bilingual(
                "[Global] Commodities: Gold and inflation-sensitive assets remain useful hedges when real-rate volatility and geopolitical risk coexist. The 'why' is that they can cushion portfolios if bonds fail to hedge equity drawdowns during inflation scares.",
                "[全球] 商品：當實質利率波動與地緣風險並存時，黃金與通膨敏感資產仍有避險價值。原因是若通膨驚嚇使債券無法有效對沖股票下跌，這些資產可提供緩衝。",
            )
        )

    if has_any(text, ["ai", "semiconductor", "gaming", "data centre", "data center", "memory"]):
        macro_views.append(
            bilingual(
                "AI is moving from a market theme into a capex cycle. The causal chain is: hyperscaler demand rises -> data-center, memory and power investment accelerates -> suppliers gain operating leverage -> markets debate whether productivity gains can justify current valuations.",
                "AI 正從市場題材轉為資本支出週期。因果鏈是：雲端巨頭需求上升 -> 資料中心、記憶體與電力投資加速 -> 供應商取得營運槓桿 -> 市場再評估生產力提升是否足以支撐當前估值。",
            )
        )
        consensus_divergence.append(
            bilingual(
                "Consensus is that AI remains the leading growth engine. The divergence is in monetization: the winners may be firms with distribution, proprietary data, IP and scale, while weaker AI-adjacent companies may see margin pressure from compute cost and competition.",
                "市場共識是 AI 仍是主要成長引擎。分歧點在於商業化：真正贏家可能是具備通路、專有資料、IP 與規模的公司，而較弱的 AI 概念股可能被算力成本與競爭壓縮利潤率。",
            )
        )
        asset_classes.append(
            bilingual(
                "[US]/[Japan]/[Taiwan] Equities: Overweight quality semiconductors, memory, power equipment and infrastructure beneficiaries, but require earnings delivery. The 'why' is that AI capex can lift revenue, while high expectations make disappointments more painful.",
                "[美國]/[日本]/[台灣] 股票：可超配高品質半導體、記憶體、電力設備與基礎建設受益者，但必須要求盈利兌現。原因是 AI 資本支出可推升營收，但高預期也會放大失望時的跌幅。",
            )
        )
        hot_topics.append(
            bilingual(
                "Interview talking point: AI commercialization is no longer only about model capability; it is about cost curves, distribution, energy availability and whether customers can turn automation into measurable ROI.",
                "面試談資：AI 商業化不再只是模型能力，而是成本曲線、通路、能源可得性，以及客戶能否把自動化轉化為可衡量的投資回報。",
            )
        )

    if has_any(text, ["asia", "capex", "industrial", "defense", "energy security"]):
        macro_views.append(
            bilingual(
                "Asia's growth story is broadening from AI alone into a wider industrial capex cycle. AI infrastructure, energy security, defense spending and supply-chain localization can reinforce each other, supporting production, jobs and eventually consumption.",
                "亞洲成長故事正從單一 AI 題材擴展到更廣泛的工業資本支出週期。AI 基礎建設、能源安全、國防支出與供應鏈在地化可以互相強化，進而支撐生產、就業，最後傳導至消費。",
            )
        )
        consensus_divergence.append(
            bilingual(
                "Consensus frames Asia as a semiconductor and AI trade. The divergence is that non-tech capex, energy infrastructure and defense may create a broader cycle, making industrials and commodity-linked markets relevant alongside tech.",
                "市場共識把亞洲視為半導體與 AI 交易。分歧點在於非科技資本支出、能源基建與國防可能形成更廣泛週期，使工業股與商品連動市場也與科技股同樣重要。",
            )
        )
        asset_classes.append(
            bilingual(
                "[HK/China]/[Japan]/[Korea]/[Taiwan] Equities: North Asia industrials, semis and power equipment can benefit from capex depth. The 'why' is that the region supplies both domestic investment demand and global infrastructure demand.",
                "[港股/中國]/[日本]/[韓國]/[台灣] 股票：北亞工業、半導體與電力設備可受惠於資本支出深化。原因是該區域同時供應本地投資需求與全球基礎建設需求。",
            )
        )
        asset_classes.append(
            bilingual(
                "[Australia]/[Indonesia] Commodities: Select commodity exporters may benefit if Asia's industrial production lifts demand for energy and raw materials. The 'why' is operating leverage to physical activity, not just financial liquidity.",
                "[澳洲]/[印尼] 商品：若亞洲工業生產推升能源與原物料需求，部分商品出口國可能受惠。原因是它們對實體活動具營運槓桿，而不只是受金融流動性影響。",
            )
        )

    if has_any(text, ["private equity", "private credit", "alternatives", "complex markets"]):
        macro_views.append(
            bilingual(
                "Private markets are adjusting to a higher-for-longer regime. When financing is no longer cheap, dispersion rises: strong managers can create value through complexity, while weaker deals face refinancing, exit and valuation pressure.",
                "私募市場正在適應利率更高更久的環境。當融資不再便宜，績效分化會上升：優秀管理人可透過複雜交易創造價值，而較弱交易會面臨再融資、退出與估值壓力。",
            )
        )
        consensus_divergence.append(
            bilingual(
                "Consensus is that alternatives remain important for long-term portfolios. The divergence is that illiquidity alone is not a return source; underwriting quality, entry valuation and exit path matter more in a higher-rate world.",
                "市場共識是另類資產仍是長期投資組合的重要部分。分歧點在於，非流動性本身不是報酬來源；在高利率世界中，承銷品質、進場估值與退出路徑更重要。",
            )
        )
        asset_classes.append(
            bilingual(
                "[Global] Alts: Prefer experienced private equity managers, secondaries and disciplined private credit over generic illiquidity exposure. The 'why' is that manager selection becomes the main driver when beta from falling rates is less reliable.",
                "[全球] 另類資產：相較於泛泛配置非流動性資產，更偏好成熟私募股權管理人、二級市場基金與承銷紀律嚴格的私募信貸。原因是當降息 beta 不再可靠，管理人選擇成為核心報酬來源。",
            )
        )
        hot_topics.append(
            bilingual(
                "Client talking point: alternatives are useful, but the first conversation with HNW clients should be liquidity needs and lock-up tolerance, not headline return targets.",
                "客戶溝通話題：另類資產有其價值，但和高淨值客戶的第一個討論應是流動性需求與鎖定期承受度，而不是單看表面目標報酬。",
            )
        )

    chart_insights = chart_insights or []
    for chart in chart_insights:
        if chart.get("macro"):
            macro_views.append(
                f"- **EN:** Chart insight from `{chart['name']}`: {chart['macro']}\n\n  **繁中：** 圖表 `{chart['name']}` 的重點已由 GPT Vision 解讀；請見英文分析中的趨勢、因果鏈與宏觀含意。"
            )
        if chart.get("asset"):
            asset_classes.append(
                f"- **EN:** Chart asset-class read-through from `{chart['name']}`: {chart['asset']}\n\n  **繁中：** 圖表 `{chart['name']}` 的資產類別含意已納入；請見英文分析中的區域標籤與原因。"
            )
        if chart.get("hot_topic"):
            hot_topics.append(
                f"- **EN:** Chart talking point from `{chart['name']}`: {chart['hot_topic']}\n\n  **繁中：** 圖表 `{chart['name']}` 可作為客戶或面試熱點討論，重點已由視覺模型整理。"
            )

    if not macro_views:
        macro_views.append(
            bilingual(
                "Macro signals are mixed rather than dominated by one data point. The right interview framing is to identify whether the day is primarily about rates, growth, inflation, geopolitics or earnings revisions before making an asset-allocation call.",
                "目前宏觀訊號較為混合，並非由單一數據主導。面試時應先判斷當天市場主要交易的是利率、成長、通膨、地緣政治還是盈利修正，再進一步提出資產配置觀點。",
            )
        )
        consensus_divergence.append(
            bilingual(
                "Consensus is cautious waiting for confirmation. The divergence is that even noisy data can create sector rotation if it changes the path of rates or earnings expectations.",
                "市場共識是等待更多確認訊號。分歧點在於，即使資料雜訊較高，只要改變利率路徑或盈利預期，也可能觸發板塊輪動。",
            )
        )

    if not asset_classes:
        asset_classes.extend(
            [
                bilingual(
                    "[Global] Equities: Stay balanced and emphasize quality earnings, strong balance sheets and pricing power. The 'why' is that uncertain macro direction rewards companies that can defend margins.",
                    "[全球] 股票：維持均衡配置，重視高品質盈利、強資產負債表與定價能力。原因是宏觀方向不明時，能守住利潤率的公司更具韌性。",
                ),
                bilingual(
                    "[US] Fixed Income: Keep high-quality bonds as portfolio ballast, while avoiding excessive duration concentration. The 'why' is that bonds help if growth slows, but inflation volatility can still hurt long duration.",
                    "[美國] 固定收益：保留高品質債券作為投資組合穩定器，但避免過度集中長久期。原因是成長放緩時債券有支撐，但通膨波動仍會傷害長久期資產。",
                ),
                bilingual(
                    "[Global] Commodities: Maintain modest gold or commodity hedges if inflation and geopolitics remain unresolved. The 'why' is diversification when stock-bond correlation becomes unstable.",
                    "[全球] 商品：若通膨與地緣政治仍未明朗，可保留適度黃金或商品避險。原因是當股債相關性不穩定時，商品能提供分散效果。",
                ),
                bilingual(
                    "[Global] Alts: Use alternatives selectively, with attention to liquidity and manager quality. The 'why' is that illiquid assets can diversify, but poor underwriting is costly in higher-rate regimes.",
                    "[全球] 另類資產：選擇性使用另類投資，重點放在流動性與管理人品質。原因是非流動性資產可提供分散，但在高利率環境中，承銷品質差的代價更高。",
                ),
            ]
        )

    if not hot_topics:
        hot_topics.append(
            bilingual(
                "Client-ready topic: the investable question is not whether one headline is good or bad, but whether it changes the policy path, earnings path or risk premium. That framing sounds more institutional in AM/WM interviews.",
                "適合客戶溝通的話題：可投資問題不是單一新聞好壞，而是它是否改變政策路徑、盈利路徑或風險溢酬。這種表述在 AM/WM 面試中更具機構投資語感。",
            )
        )

    morning_meeting = (
        "**EN:** Good morning. Today's market setup is best described as a soft-landing base case with several important cross-currents. "
        "Inflation and rates still matter because they limit how quickly central banks can ease; U.S.-China tensions look more managed but not resolved; "
        "AI is becoming a real capex cycle; and Asia's industrial story is broadening beyond semiconductors into energy, defense and supply-chain investment. "
        "In portfolios, I would stay invested but rebalance toward quality equities, selective high-grade fixed income, modest inflation hedges and disciplined alternatives, "
        "while avoiding over-concentration in any single AI or China reopening narrative.\n\n"
        "**繁中：** 各位早。今天的市場可以概括為「軟著陸仍是基準情境，但多重變數正在交錯」。"
        "通膨與利率仍是核心，因為它們限制央行降息速度；美中關係看似更可控，但並未真正解決；"
        "AI 正從題材轉為實際資本支出週期；亞洲工業故事也從半導體擴展到能源、國防與供應鏈投資。"
        "投資組合上，我會維持在場，但再平衡至高品質股票、選擇性高評級固定收益、適度通膨避險與紀律嚴格的另類資產，"
        "同時避免過度集中於單一 AI 或中國復甦敘事。"
    )

    return {
        "macro_views": macro_views[:7],
        "consensus_divergence": consensus_divergence[:7],
        "asset_classes": asset_classes[:10],
        "hot_topics": hot_topics[:7],
        "morning_meeting": morning_meeting,
    }


def render_cards(items: list[str], title_prefix: str) -> None:
    for index, item in enumerate(items, start=1):
        with st.container(border=True):
            st.caption(f"{title_prefix} {index}")
            st.markdown(item)


def refresh_daily_context() -> tuple[bool, str]:
    fetcher = BASE_DIR / "market_fetcher.py"
    if not fetcher.exists():
        return False, "market_fetcher.py was not found."

    result = subprocess.run(
        [sys.executable, str(fetcher)],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    output = "\n".join(part for part in [result.stdout, result.stderr] if part).strip()
    return result.returncode == 0, output or "No output returned."


def main() -> None:
    st.set_page_config(
        page_title="AM/WM Market Sense Dashboard",
        page_icon="📈",
        layout="wide",
    )
    page_style()

    st.sidebar.title("Market Sense")
    st.sidebar.caption("Daily context for AM/WM internship interviews")

    if st.sidebar.button("Refresh latest data", use_container_width=True):
        with st.spinner("Fetching latest market context..."):
            ok, output = refresh_daily_context()
        if ok:
            st.sidebar.success("Refresh complete")
        else:
            st.sidebar.error("Refresh failed")
        with st.sidebar.expander("Fetcher log"):
            st.code(output)

    reports = discover_reports()
    if not reports:
        st.error("No daily_context report found. Run market_fetcher.py first.")
        st.stop()

    selected_label = st.sidebar.selectbox(
        "日期",
        [report.label for report in reports],
        index=0,
    )
    selected_report = next(report for report in reports if report.label == selected_label)
    raw_text = read_text(selected_report.path)
    sources = parse_sources(raw_text)

    st.sidebar.divider()
    st.sidebar.subheader("Add Your Own Inputs")

    uploaded_files = st.sidebar.file_uploader(
        "上傳投行報告 PDF",
        type=["pdf"],
        accept_multiple_files=True,
        help="Drop one or more PDF reports here. The app will extract text and add it to this dashboard session.",
    )
    process_uploaded_pdfs(uploaded_files or [])

    uploaded_charts = st.sidebar.file_uploader(
        "上傳 MacroMicro 圖表截圖",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Upload PNG/JPG chart screenshots. GPT Vision will interpret the chart if OPENAI_API_KEY is configured.",
    )
    process_uploaded_charts(uploaded_charts or [])

    with st.sidebar.form("macromicro_text_form"):
        macro_text = st.text_area(
            "貼上 MacroMicro 文章",
            height=180,
            placeholder="Paste the article text here...",
        )
        submitted = st.form_submit_button("送出分析", use_container_width=True)
        if submitted:
            if macro_text.strip():
                submitted_at = datetime.now()
                session_sources()["MacroMicro pasted article"] = make_manual_source_body(
                    "MacroMicro pasted article",
                    macro_text,
                    submitted_at,
                )
                st.success("MacroMicro text added to the dashboard analysis.")
            else:
                st.warning("Please paste text before submitting.")

    if st.sidebar.button("Clear uploaded / pasted inputs", use_container_width=True):
        st.session_state.custom_sources = {}
        st.session_state.processed_upload_keys = set()
        st.session_state.processed_chart_keys = set()
        st.session_state.chart_insights = []
        st.sidebar.success("Session inputs cleared.")

    custom_sources = session_sources()
    if custom_sources:
        sources.update(custom_sources)
        raw_text = raw_text + "\n\n" + "\n\n".join(
            source_block(source, body) for source, body in custom_sources.items()
        )

    news_df = build_news_table(sources)
    analysis = build_analysis(raw_text, session_chart_insights())

    generated = selected_report.generated_at.strftime("%Y-%m-%d %H:%M") if selected_report.generated_at else "Unknown"
    source_count = max(len(sources) - (1 if "Run Metadata" in sources else 0), 0)

    st.title("AM/WM Market Sense Dashboard")
    st.markdown(
        f"<p class='section-note'>Selected report: <b>{selected_report.path.name}</b> · Generated: <b>{generated}</b></p>",
        unsafe_allow_html=True,
    )

    top_cols = st.columns(3)
    top_cols[0].metric("Sources", source_count)
    top_cols[1].metric("Parsed Items", len(news_df))
    top_cols[2].metric("Report Date", generated.split(" ")[0] if generated != "Unknown" else "Unknown")

    st.divider()

    st.subheader("1. Macro Views（宏觀事件短評）")
    render_cards(analysis["macro_views"], "Macro Views")

    st.subheader("2. Market Consensus & Divergence（金融市場共識與分歧點）")
    render_cards(analysis["consensus_divergence"], "Consensus / Divergence")

    st.subheader("3. Asset Classes（不同資產類別反應及原因）")
    render_cards(analysis["asset_classes"], "Asset Class")

    st.subheader("4. Hot Topics（熱點與長期關注話題）")
    render_cards(analysis["hot_topics"], "Hot Topic")

    st.subheader("5. Summary for Morning Meeting（晨會小結）")
    with st.container(border=True):
        st.markdown(analysis["morning_meeting"])

    st.divider()
    st.subheader("Source Intelligence Table")
    st.caption("Parsed headlines, podcast titles, timestamps, summaries and links.")
    st.dataframe(
        news_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Summary": st.column_config.TextColumn(width="large"),
            "Link": st.column_config.LinkColumn(width="medium"),
        },
    )

    with st.expander("Raw source text"):
        for source, body in sources.items():
            if source == "Run Metadata":
                continue
            st.markdown(f"#### {source}")
            st.text_area(source, body, height=220, label_visibility="collapsed")


if __name__ == "__main__":
    main()
