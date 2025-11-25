# Survey Analytics: Multi-Agent RAG System

**Authors:** Ash Ren, Adithya Kalidindi

-----

## 1\. Problem Statement

### Background

The **Vanderbilt Unity Poll** is a quarterly survey tracking American public opinion on major policy issues—healthcare, immigration, presidential approval, economic policy, and more. Each wave surveys **\~2,000 nationally representative respondents**.

Survey results are typically stored in lengthy PDF reports, academic articles, and static data tables. Researchers and analysts must manually search through numerous reports to find specific statistics, demographic breakdowns, or track trends over time—a process that can take hours for a single analysis. Our project seeks to make survey data more accessible using generative AI capabilities to provide specific survey results in real-time.

### The Challenge

**Goal:** Create a natural language chatbot that answers questions about survey data.

**Problem:** Survey data exists in **heterogeneous structures**:

1.  **Questionnaire data:** Question text, variable names, metadata.
2.  **Topline statistics:** Overall response percentages for each question (e.g., "In the 2024 election, 50% of respondents voted for..., 20% voted for...").
3.  **Crosstabs:** Sociodemographic breakdowns for each answer option of a survey question (e.g., "Out of all respondents who voted for X, 70% are Democrats, 15% are Republicans").

> **Why Simple RAG Fails:**
>
>   * It cannot determine which data source(s) to query from natural language.
>   * It cannot generate multi-step execution plans when queries require multiple sources.
>   * It cannot handle dependencies between retrieval stages.

**Example Complexity:**
*User:* "What percentage of college-educated voters supported Biden in June 2025?"

  * **Step 1:** Find question (Questionnaire)
  * **Step 2:** Get percentages (Toplines)
  * **Step 3:** Break down by education (Crosstabs)

### Our Solution

Use **transformer-based structured generation** to automatically create a "research brief" that routes queries to appropriate sources and generates multi-stage plans when needed.

-----

## 2\. Data Preprocessing: Creating Static Datasets for RAG Agents

### Challenge: Raw Data → Vector Stores

**Input Materials:** Poll questionnaires (PDF/DOCX) and Raw survey responses (SPSS .sav converted to CSV).

### Step 1: Parse Questionnaires

  * **Tool:** `questionnaire_parser.py`
  * **Input:** Questionnaire PDF/DOCX
  * **Process:** GPT-4o extracts structured metadata via prompt engineering.
  * **Output:** Static JSON file with question metadata.
  * **Vectorstore:** `create_questionnaire_vectorstores.py` generates embeddings and uploads to Pinecone.

<!-- end list -->

```json
{
  "question_id": "Vanderbilt_Unity_Poll_2025_June_VAND6A",
  "variable_name": "VAND6A",
  "question_text": "Do you support or oppose deporting individuals...",
  "response_options": ["Strongly support", "Somewhat support", ...],
  "topics": ["immigration"],
  "year": 2025,
  "month": "June"
}
```

### Step 2: Generate Toplines Statistics

  * **Tool:** `toplines_generator.py`
  * **Input:** Raw survey CSV + Questionnaire JSON
  * **Process:** Calculate weighted percentages for the overall sample.
  * **Output:** Static CSV files with response frequencies.
  * **Vectorstore:** `create_toplines_vectorstores.py` generates embeddings and uploads to Pinecone.

<!-- end list -->

```csv
variable,response_label,percent,weighted_n,unweighted_n
VAND6A,Strongly support,35,420,418
VAND6A,Somewhat support,23,276,280
```

### Step 3: Generate Crosstabs

  * **Tool:** `crosstab_generator.py`
  * **Input:** Raw survey CSV + Questionnaire JSON
  * **Process:** Calculate weighted percentages by demographics.
  * **Output:** Static CSV files with demographic breakdowns.
  * **Vectorstore:** `create_crosstab_vectorstore.py` generates embeddings and uploads to Pinecone.

<!-- end list -->

```csv
Answer,PGENDER: Male,PGENDER: Female,PPARTY: Democrat,PPARTY: Republican
Strongly support,32% (156),38% (264),54% (301),12% (45)
Somewhat support,25% (122),21% (154),28% (156),18% (67)
```

-----

## 3\. Architecture Overview

### Multi-Agent System with LangGraph Orchestration

<img width="1116" height="1301" alt="archi drawio" src="https://github.com/user-attachments/assets/7c1b08ad-4878-40f3-89b3-a46a16e5dbe1" />



### Core Components (9 Total)

1.  **ConversationRelevanceChecker (`relevance_checker.py`):**

      * LLM analyzes relationship between current query and history.
      * Classifies: `same_topic_different_demo`, `same_topic_different_time`, `trend_analysis`, `new_topic`.
      * Determines reusable data flags → **reduces API calls by 50-66%**.

2.  **Research Brief Generator (GPT-4o within `survey_agent.py`):**

      * Decides routing strategy based on query + relevance analysis.
      * Generates multi-stage plans when needed and extracts filters.
      * Uses structured output (Pydantic `ResearchBrief` model).

3.  **QuestionnaireRAG (`questionnaire_rag.py`):**

      * Retrieves question metadata from Pinecone via Semantic search + metadata filtering.
      * Outputs: `question_info` (variable names, poll dates, topics).

4.  **ToplinesRAG (`toplines_rag.py`):**

      * Retrieves aggregate response statistics using `question_info` for precise filtering.
      * Outputs: Response percentages with sample sizes.

5.  **CrosstabsRAG (`crosstab_rag.py`):**

      * Retrieves demographic breakdowns using `question_info` + demographic filters.
      * Outputs: Multi-dimensional crosstab data.

6.  **CrosstabSummarizer (LLM-based within `crosstab_rag.py`):**

      * **Problem:** Crosstabs return 50+ chunks per question.
      * **Solution:** GPT-4o condenses chunks → **70-80% reduction**.
      * Combines multi-part documents and extracts only relevant demographics.

7.  **Context Extractor (within `survey_agent.py`):**

      * Extracts `question_info` metadata from stage results.
      * Priority: Recent results \> conversation history. Stores in LangGraph checkpoint.

8.  **Response Synthesizer (GPT-4o within `survey_agent.py`):**

      * Combines retrieved data into a coherent narrative with citations.
      * Token optimization via CrosstabSummarizer integration.

9.  **VisualizationAgent (`viz_agent.py` - Optional):**

      * **Intent Analysis:** LLM determines if chart is appropriate and returns `VizIntent`.
      * **Data Extraction:** Rule-based parsing from stage results (`_extract_from_toplines`, `_extract_from_crosstabs`).
      * **Chart Generation:** Matplotlib rendering (Bar, Grouped Bar, Line, Pie) with auto-layout.

### LangGraph State Management

**State Model:**

```python
class SurveyAnalysisState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_question: str
    research_brief: Optional[ResearchBrief]
    current_stage: int
    stage_results: List[StageResult]  # Preserved across turns!
    final_answer: Optional[str]
    enable_viz_for_query: bool
    visualization_metadata: Optional[Dict]
```

### Routing Logic

| Query Type | Relevance Check | Routing Decision | Example |
| :--- | :--- | :--- | :--- |
| **New question search** | `new_topic` | Single-stage → Questionnaire | "What was asked about immigration?" |
| **Follow-up (same topic)** | `same_topic_different_demo` | Single-stage → Skip questionnaire | "Now show by gender" |
| **Follow-up (time change)** | `same_topic_different_time` | Multi-stage → Re-query all | "Now show June 2024" |
| **Ambiguous statistics** | `new_topic` | Multi-stage → Find Q → Get data | "Support for healthcare reform?" |

-----

## 4\. How Transformers Enable the System

### Three Transformer Models Working Together

| Component | Model | Type | Role |
| :--- | :--- | :--- | :--- |
| **Embeddings** | `text-embedding-3-small` | Encoder-only | Semantic search in vector stores |
| **Planning & Analysis** | `gpt-4o` | Decoder-only | Query classification, routing, relevance checking |
| **Synthesis & Viz** | `gpt-4o` | Decoder-only | Data summarization, answer generation, viz intent |

### Key Transformer Concepts Applied

1.  **Semantic Embeddings (Encoder):** Enables finding semantically related content (e.g., "gun control" matches "firearms policy").
2.  **Structured Output Generation (Decoder):** GPT-4o generates Pydantic models directly (ResearchBrief, RelevanceResult, VizIntent) ensuring reliable coordination.
3.  **In-Context Learning:** System prompts define multi-stage planning and behavior without fine-tuning.
4.  **Retrieval-Augmented Generation (RAG):** Retrieves relevant data to inject into context, preventing hallucination.
5.  **Topic Normalization System:** Maps variations to 12 canonical topics (e.g., "tariffs" → "economy") to prevent semantic drift.

-----

## 5\. Implementation Demo

### Example 1: Single-Stage Questionnaire Search

**Query:** *"What questions were asked about immigration in 2025?"*

1.  **Relevance:** `new_topic`
2.  **Plan:** Single-stage questionnaire query.
3.  **Retrieval:** Semantic search + metadata filter (year=2025, topic="immigration").
4.  **Result:** List of questions with variable names and poll dates.

### Example 2: Multi-Stage Statistics Lookup with Visualization

**Query:** *"What % supported Medicare for All in June 2025?"*

1.  **Relevance:** `new_topic`
2.  **Plan:** Multi-stage (Stage 1: Find Question → Stage 2: Get Toplines using `question_info`).
3.  **Retrieval:** ToplinesRAG returns "58% Support".
4.  **Visualization:**
      * *Intent:* Single-variable statistics → `bar_chart`.
      * *Generation:* Matplotlib figure with styled bars.
5.  **Result:** Accurate statistics + Bar Chart.

### Example 3: Conversation Optimization (Cost Savings)

**Turn 1:** *"How do views on immigration vary by political party?"*

  * **Execution:** QuestionnaireRAG (finds 9 questions) + CrosstabsRAG. **(2 API Calls)**

**Turn 2:** *"Now show me by gender instead."*

1.  **Relevance Check:** `same_topic_different_demo`.
2.  **Optimization:** Reusable data flag set to `questions=true`.
3.  **Plan:** `route_to_sources` (Skips QuestionnaireRAG).
4.  **Execution:** Extracts `question_info` from previous state → Queries CrosstabsRAG only. **(1 API Call)**

<!-- end list -->

  * **Result:** 50% reduction in calls, faster response, cheaper cost.

### Example 4: Visualization Generation

**Query:** *"Show me Biden's approval ratings over time."*

1.  **Retrieval:** Multi-stage retrieval gets percentages from multiple polls.
2.  **Intent Analysis:** Detects time-series data → `should_visualize=true`, `viz_type='line_chart'`.
3.  **Data Extraction:** Parses x-axis (dates) and y-axis (approval %).
4.  **Chart Generation:** Renders line chart with multiple series (Support vs Oppose).

-----

## 6\. Evaluation & Technical Challenges

### Technical Challenges Solved

  * **When to Use Multi-Stage?** GPT-4o dynamically decides via structured Research Briefs.
  * **Context Window Limits:** CrosstabSummarizer condenses large datasets by 70-80% using a two-pass approach (Retrieval → Summarization).
  * **Conversation Continuity:** LangGraph checkpointing + Priority-based context extraction allows state to persist across turns.
  * **Visualization State:** Metadata is stored in state, but figures are generated fresh post-execution to avoid serialization issues.
  * **Topic Drift:** Normalization system maps synonyms to canonical topics for precise metadata filtering.

### Qualitative Assessment

**What Works Well:**

  * Accurate retrieval (\~95% precision for questionnaires).
  * No hallucinated statistics (100% grounded in data).
  * 50-66% API call reduction via conversation optimization.
  * Automatic visualization intent analysis (\~90% accurate).

**Limitations:**

  * Depends entirely on data availability.
  * Visualization is limited to 6 predefined chart types.
  * In-memory checkpointing loses state on restart.

-----

## 7\. Model & Data Cards

### Models Used

| Model | Version | Intended Use | API Calls Per Query |
| :--- | :--- | :--- | :--- |
| **GPT-4 Omni** | `gpt-4o` | Routing, Summarization, Synthesis, Viz Analysis | 3-5 (varies) |
| **OpenAI Embeddings** | `text-embedding-3-small` | Semantic search | 1 per query |
| **Pinecone** | Vector DB | Vector storage & retrieval | 1-3 per query |

**Cost:** $0.04 - $0.06 per typical query (30-50% savings on optimized follow-ups).

### Ethical Considerations

  * **Data Privacy:** System only provides aggregated statistics; no individual-level data is accessible.
  * **Hallucination Risk:** Mitigated by retrieval-first approach; response synthesis explicitly cites sources.
  * **Bias:** LLM routing may reflect training bias; topic mappings are manually curated.

-----

## 8\. Impact & Next Steps

### Impact

  * **Accessibility:** Non-technical researchers can query complex data via natural language (No SQL/Excel needed).
  * **Efficiency:** 100x+ speedup (seconds vs hours).
  * **Scalability:** Architecture is extensible to other datasets (Pew, Gallup).

### Next Steps

1.  **Accuracy Verification:** Benchmark against manual retrieval.
2.  **Dataset Expansion:** Add cross-survey comparisons.
3.  **Visualization Enhancements:** Switch to Plotly for interactivity.
4.  **Production Deployment:** Implement Redis/PostgreSQL backend for persistent checkpointing.
5.  **Advanced Optimizations:** Asynchronous agent execution.

-----

## Repository & Resources

**GitHub:** `https://github.com/vanderbilt-data-science/survey-analytics`

### Project Structure

```text
survey-agent-v2/
├── survey_agent.py              # Main orchestrator with LangGraph
├── relevance_checker.py         # ConversationRelevanceChecker
├── questionnaire_rag.py         # QuestionnaireRAG agent
├── toplines_rag.py              # ToplinesRAG agent
├── crosstab_rag.py              # CrosstabsRAG + CrosstabSummarizer
├── viz_agent.py                 # VisualizationAgent
├── config.py                    # Topic normalization
├── app.py                       # Gradio web interface
└── prompts/
    ├── research_brief_prompt.txt
    ├── crosstab_rag_prompt_system.txt
    └── synthesis_prompt_system.txt
```

-----

## Appendix: Technical Details

### Complete Agent Interaction Flow

1.  **User Query** → `SurveyAnalysisAgent.query()`
2.  **Relevance Checker:** Classifies query and checks for reusable data (e.g., `same_topic_different_demo`).
3.  **Research Brief Generator:** Creates a `ResearchBrief` with specific routing actions (e.g., `route_to_sources`).
4.  **Context Extraction:** Pulls `question_info` from previous stage results (Priority: Recent \> History).
5.  **Stage Execution:** Agents execute. **CrosstabsRAG** includes a sub-step where `CrosstabSummarizer` condenses 50+ chunks.
6.  **Response Synthesizer:** Generates final text answer.
7.  **VisualizationAgent:** Checks `VizIntent` → Extracts Data → Renders Chart.
8.  **Final Output:** Returns Answer + Figure.

### Key Data Structures

**ResearchBrief (Pydantic)**

```python
class ResearchBrief(BaseModel):
    action: Literal["answer", "followup", "route_to_sources", "execute_stages"]
    reasoning: str
    data_sources: List[DataSource]
    stages: List[ResearchStage]
```

**RelevanceResult (Pydantic)**

```python
class RelevanceResult(BaseModel):
    is_related: bool
    relation_type: Literal["same_topic_different_demo", "same_topic_different_time", "trend_analysis", "new_topic"]
    reusable_data: ReusableData
    reasoning: str
```

**VizIntent (Pydantic)**

```python
class VizIntent(BaseModel):
    should_visualize: bool
    viz_type: Optional[Literal["bar_chart", "grouped_bar", "horizontal_bar", "line_chart", "stacked_bar", "pie_chart"]]
    chart_title: Optional[str]
    reasoning: str
```

**ExtractedData (Pydantic)**

```python
class ExtractedData(BaseModel):
    data_type: Literal["toplines", "crosstabs", "trends", "comparison"]
    labels: List[str]
    values: List[float]
    grouped_data: Optional[Dict[str, List[float]]]
    metadata: Dict[str, Any]
```

-----
