# Survey Analytics: Multi-Agent RAG System

**Authors:** Ash Ren, Adithya Kalidindi

-----

## 1\. Problem Statement

### Background

The **Vanderbilt Unity Poll** is a quarterly survey tracking American public opinion on major policy issues—healthcare, immigration, presidential approval, economic policy, and more. Each wave surveys **\~2,000 nationally representative respondents**.

Survey results are typically stored in lengthy PDF reports, academic articles, and static data tables. Researchers and analysts must manually search through numerous reports to find specific statistics, demographic breakdowns, or track trends over time—a process that can take hours for a single analysis. Our project seeks to make survey data more accessible through a RAG-based chatbot to provide survey results in real-time.

### The Technical Challenge
- Survey data exists across three heterogeneous sources: questionnaire metadata, topline statistics, and demographic crosstabs -- each requiring different retrieval strategies
- Complex queries create dependencies between the data sources: crosstab and topline-related queries require structured question metadata (variable names, year, month) from the questionnaire data source
  - For example, "What percentage of ... supporters are college educated?" requires first identifying the candidate approval question from the questionnaire data, then using that question's metadata to retrieve education breakdowns from crosstabs

### Our Solution

A multi-agent RAG-based chatbot that uses **transformer-based structured generation** to automatically create a "research brief" that routes queries to appropriate sources and generates multi-stage plans when needed.

-----

## 2\. Data Preprocessing: Creating Static Datasets for RAG Agents

### Challenge: Raw Data → Vector Stores

**Input Materials:** Poll questionnaires (PDF/DOCX) and Raw survey responses (SPSS .sav converted to CSV).

### Step 1: Parse Questionnaires for

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
  "response_options": ["Strongly support", "Somewhat support", "..."],
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
      * Determines reusable data flags → **reduces API calls by 50%**.

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

## 4\. Transformers Concepts

### 1. **Retrieval-Augmented Generation (RAG)**
This project implements a multi-pipeline RAG system that grounds transformer outputs in actual survey data, preventing hallucinations:
- **Semantic retrieval**: Uses transformer embeddings (`text-embedding-3-small`) to find relevant survey questions, toplines, and crosstabs from vector databases
- **Contextual generation**: Retrieved documents are injected into GPT-4o's context window, allowing the transformer to generate answers based on real data rather than parametric knowledge
- **Multi-source fusion**: Combines Questionnaire, Toplines, and Crosstabs pipelines to provide comprehensive answers

This demonstrates how RAG extends transformer capabilities beyond their training data, enabling them to answer questions about domain-specific information (Vanderbilt Unity Poll data from 2024-2025) that wasn't in the original training corpus.

### 2. **Transformer-Powered Intelligent Routing**
The system uses GPT-4o (a transformer decoder model) to analyze user queries and generate structured research plans:
```python
class ResearchBrief(BaseModel):
    action: Literal["answer", "followup", "route_to_sources", "execute_stages"]
    data_sources: List[DataSource]  # Which pipelines to query
    stages: List[ResearchStage]     # Multi-stage execution plan

# Transformer decides routing strategy
brief_generator = self.llm.with_structured_output(ResearchBrief)
```
The transformer analyzes the query, available data sources, and conversation history to intelligently route between pipelines—or determine if a follow-up question is needed. This showcases how attention mechanisms enable complex decision-making based on context.

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

## 6\. Assessment and Evaluation

### Licenses and Intended Use
| Component | License | Intended Use |
|-----------|---------|--------------|
| GPT-4o | Commercial API (OpenAI) | Research brief generation, query routing, response synthesis |
| text-embedding-3-small | Commercial API (OpenAI) | Semantic search over survey questions and responses |
| Pinecone | Commercial (Serverless) | Vector database for storing and retrieving questionnaire, topline, and crosstab embeddings |
| LangChain/LangGraph | MIT (Open Source) | RAG orchestration and conversation state management |
| Vanderbilt Unity Poll Data | Academic Use (Citation Required) | Source survey data for public opinion analysis |

### Ethical/Bias Considerations

**Privacy Protection:**
- System only provides aggregated statistics; individual survey responses are never accessible
- No personally identifiable information (PII) can be retrieved or reconstructed from the system

**Potential Biases:**
- Survey weighting methodology may not fully capture all demographic groups
- Question wording in original surveys may introduce framing effects
- LLM routing decisions may reflect biases present in GPT-4o's training data

**Limitations on Use:**
- System should only be used for exploratory analysis of aggregated survey data and understanding public opinion trends
- System should not be used for making individual predictions or classifications
- Results are limited to the specific time periods and populations covered by the Vanderbilt Unity Poll
-----

## 7\. Impact & Next Steps

### Impact
- **Accessibility:** Non-technical researchers can query complex survey data via natural language.
- **Scalability:** Architecture is extendable to other survey datasets beyond the Vanderbilt Unity Poll.
- **Innovation:** Multi-agent orchestration automates complex analytical workflows through transformer-based planning, eliminating hardcoded routing logic.

### Next Steps

1.  **Accuracy Verification:** Benchmark outputs against domain expert validation.
2.  **Dataset Expansion:**  Incorporate additional polling organizations beyond the Unity Poll.
3.  **LLM Provider Evaluation**: Test alternative LLM providers (Claude, Gemini, Llama) to compare routing accuracy and query latency against the current GPT-4o implementation.

-----

## Model & Data Cards

### Model Card: GPT-4o

**Model Details:**
- Name: GPT-4 Omni (GPT-4o)
- Developer: OpenAI
- Model Type: Transformer decoder architecture

**Intended Use:**
- Query routing and research brief generation
- Crosstab summarization 
- Response synthesis from multiple data sources
- Visualization intent analysis

**Limitations:**
- Routing decisions depend on prompt engineering quality
- Cannot determine statistical significance or causal relationships
- May reflect biases present in training data

**Ethical Considerations:**
- System only provides aggregated statistics; no individual-level data accessible
- Responses explicitly grounded in retrieved data to prevent misinformation

---

### Model Card: text-embedding-3-small

**Model Details:**
- Name: text-embedding-3-small
- Developer: OpenAI
- Model Type: Transformer-based embedding model
- Dimension: 1536

**Intended Use:**
- Semantic search across questionnaire metadata, toplines, and crosstabs
- Vector representation of survey questions and user queries

**Limitations:**
- Semantic similarity does not guarantee topical relevance
- Performance depends on query phrasing quality

---

### Data Card: Vanderbilt Unity Poll

**Dataset Details:**
- Time Period: March 2023 - June 2025
- Number of Polls: 8
- Sample Size: ~1,000 nationally representative respondents per wave
- Source: Joshua Clinton, Professor of Political Science and Director of the Vanderbilt Poll, Vanderbilt University

**Data Structure:**
- Questionnaires: Question text, variable names, response options, topics (PDF/DOCX → JSON)
- Raw Data: Individual survey responses with demographic variables and question answers (SPSS .sav → CSV)

**Collection & Processing:**
- Nationally representative surveys, weighted to U.S. Census demographics
- Questionnaires parsed via GPT-4o into structured JSON
- Toplines and crosstabs calculated from weighted survey responses
- Text embeddings generated and stored in Pinecone vector database

**Topics Covered:**
Presidential approval, immigration, healthcare, economy, gun control, abortion, election integrity, foreign policy

**Privacy:**
- All data aggregated only; no personally identifiable information (PII) accessible
- Individual responses cannot be reconstructed

**Limitations:**
- Coverage limited to surveyed topics and 2024-2025 time period
- Some demographic subgroups may have small sample sizes
- Question wording may introduce framing effects
- System cannot answer questions about unprocessed polls

----

## Repository & Resources

**GitHub:** `https://github.com/vanderbilt-data-science/survey-analytics`

**Frameworks and Libraries:**
- LangGraph: Framework for building stateful, multi-agent workflows with LLMs https://docs.langchain.com/oss/python/langgraph/overview
- LangChain: Toolkit for developing LLM applications with RAG capabilities https://docs.langchain.com/oss/python/langchain/rag
- Pinecone: Vector database for semantic search and embeddings storage https://www.pinecone.io/
  
**Key Concepts and Techniques:**
- Retrieval-Augmented Generation (RAG) Paper: https://arxiv.org/abs/2005.11401
- GPT-4o: Multimodal language model used for routing, summarization, and synthesis https://openai.com/index/hello-gpt-4o/
- OpenAI Embeddings Documentation: Text-embedding-3-small model https://platform.openai.com/docs/guides/embeddings
- Pydantic: Data validation using Python type annotations for structured outputs https://docs.pydantic.dev/

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
    ├── synthesis_prompt_system.txt
    └── ....

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
