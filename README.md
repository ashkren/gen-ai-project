# Multi-Agent Survey Analytics with Transformer-Based Routing

---

## 1. Problem Statement

### Background

The Vanderbilt Unity Poll is a quarterly survey tracking American public opinion on major policy issues—healthcare, immigration, presidential approval, economic policy, and more. Each wave surveys ~2,000 nationally representative respondents, providing valuable insights for researchers, policymakers, and journalists.

### The Challenge

**Goal**: Create a natural language chatbot that answers questions about survey data.

**Problem**: Survey data exists in heterogeneous structures, each requiring different retrieval strategies:

1. **Questionnaire data**: Question text, variable names, metadata (parsed from PDFs)
2. **Topline statistics**: Overall response percentages for entire sample (calculated from raw data)
3. **Crosstabs**: Response percentages broken down by demographics—gender, age, education, income, etc. (calculated from raw data)

All three are stored in vector stores for semantic search, but their **structural differences** make simple RAG insufficient:

- **Questionnaires** contain metadata only (no response data)
- **Toplines** contain aggregate statistics (e.g., "45% approved")
- **Crosstabs** contain demographic breakdowns (e.g., "Male: 48% approved, Female: 42% approved")

**Simple RAG fails** because it can't:
- Determine which data source(s) to query from natural language alone
- Generate multi-step execution plans when query requires multiple sources
- Handle dependencies between retrieval stages

**Example complexity**:
```
User: "What percentage of college-educated voters supported Biden in June 2025?"

Requires:
1. Find Biden approval questions → questionnaire retrieval
2. Get June 2025 response percentages → topline retrieval  
3. Break down by education level → crosstab retrieval
4. Synthesize across 3 different vector stores
```

### Our Solution

Use **transformer-based structured generation** to automatically create a "research brief" that:
1. **Classifies query intent**: Direct answer? Needs data retrieval? Ambiguous?
2. **Routes to appropriate sources**: Questionnaire, toplines, and/or crosstabs
3. **Generates multi-stage plans**: Stage 1 (get questions) → Stage 2 (get data) with dependencies
4. **Extracts filters**: Time period, demographics, topics from natural language

**Key Innovation**: Instead of hardcoded routing rules, we leverage transformer **self-attention** and **constrained decoding** to learn routing patterns from prompt examples—enabling the system to handle novel query types and orchestrate complex multi-step retrieval across heterogeneous data sources.
---

## 2. Methodology: Transformer Connections

### 2.1 Self-Attention for Contextual Query Understanding

**Course Concept**: Multi-head self-attention builds contextualized representations by attending to all tokens simultaneously.

**Our Application**:
```
Query: "What percentage of people approved of Biden in June?"
        |        |       |    |      |      |     |    |
        └────────┴───────┴────┴──────┴──────┴─────┴────┘
                     ↓ Self-Attention ↓
                     
Learned patterns:
- "percentage" ↔ "people" → frequency data (toplines)
- "Biden" ↔ "approved" → sentiment question  
- "June" ↔ "2025" → temporal filter
```

**Why transformers win**: Unlike RNNs/LSTMs that process sequentially, transformers attend to "June" while understanding its relationship to "Biden" and "approved" simultaneously.

**Mathematical Connection**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V

In our routing:
Q = query embedding ("Biden approval in June")
K = schema examples (routing patterns in prompt)  
V = decision templates (which agents to call)
```

### 2.2 Structured Generation via Constrained Decoding

**Course Concept**: Transformers generate sequences token-by-token using learned probability distributions.

**Our Application**: Generate valid JSON conforming to Pydantic schema:

```python
class ResearchBrief(BaseModel):
    action: Literal["answer", "followup", "route_to_sources", "execute_stages"]
    reasoning: str
    data_sources: List[DataSource]
    stages: List[ResearchStage]

# Transformer learns: P(next_token | context, schema_constraints)
```

**How it works**:
1. System prompt includes schema definition
2. Transformer attention learns valid structure from examples
3. Decoder generates token-by-token maintaining JSON validity
4. Temperature=0 ensures deterministic routing

**Key Insight**: Multi-task learning in one forward pass:
- Intent classification (which action?)
- Named entity recognition (extract time, topic)
- Hierarchical planning (stage ordering)
- Natural language generation (reasoning)

### 2.3 Retrieval-Augmented Generation (RAG)

**Course Concept**: Standard transformers are limited by fixed context windows (O(n²) attention) and static knowledge.

**RAG Solution**: Extend transformer capabilities:
```
Standard: Attend to input tokens only
RAG: Attend to input + dynamically retrieved documents

Query → Embedding (encoder) → Similarity Search → Context + Query → Synthesis (decoder)
```

**Three RAG Pipelines**:

1. **Questionnaire RAG** (Vector Store):
   - OpenAI text-embedding-ada-002 (transformer encoder)
   - 1536-dim embeddings
   - Cosine similarity = dot product attention

2. **Toplines RAG** (Structured Data):
   - Response frequencies
   - Metadata filtering + semantic search

3. **Crosstabs RAG** (Demographics):
   - Multi-stage: questions → breakdowns
   - Compositional reasoning

### 2.4 Tool-Augmented Chain-of-Thought (ReAct)

**Course Concept**: Chain-of-thought prompting improves reasoning through explicit intermediate steps.

**Our Implementation**: Hybrid reasoning - LLM thought + programmatic action:

```
Thought: "Need to find Biden approval questions"
Action: questionnaire_rag.retrieve(topic="Biden approval")
Observation: Found 3 questions

Thought: "Now need June 2025 frequencies"  
Action: toplines_rag.retrieve(questions=[...], filters={"month": "June"})
Observation: 45% approved, 42% disapproved

Thought: "Synthesize answer"
Action: synthesize_response(context)
```

**Why this works**: Transformers handle reasoning (planning, synthesis), deterministic code handles retrieval (no hallucination).

### 2.5 Key Transformer Elements in Our System

**Positional Encoding**:
```python
# We add position markers for temporal ordering
context_parts.append(f"\n=== STAGE {i} ===")
context_parts.append(f"Poll Date: {poll_date}")

# Transformer learns stage dependencies
```

**Multi-Head Attention Analog**:
```python
# Like attention heads learning different patterns:
# Head 1: What questions exist? (questionnaire)
# Head 2: What are frequencies? (toplines)  
# Head 3: How do demographics differ? (crosstabs)

for ds in stage.data_sources:  # Parallel execution
    if ds.source_type == "questionnaire":
        result_q = questionnaire_rag.retrieve(...)
    elif ds.source_type == "toplines":
        result_t = toplines_rag.retrieve(...)
```

**Temperature Control**:
```python
self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
# T→0: Deterministic routing (argmax, no sampling)
# Same query always routes the same way
```

**Context Window Management**:
```python
# GPT-4: 128k token limit, O(n²) attention complexity
# We summarize crosstabs: 10k tokens → 1.5k tokens
def _summarize_crosstab_data(crosstab_docs, query):
    # Use transformer to extract relevant demographics only
    # Prevents context window overflow
```

---

## 3. Implementation & Demo

### System Architecture

```
User Query
    ↓
[Generate Research Brief] ← GPT-4 Transformer
    ↓
{action, reasoning, stages}
    ↓
Route Decision:
├─→ "answer": Direct synthesis
├─→ "followup": Ask clarification
└─→ "execute_stages": Multi-step retrieval
        ↓
[Stage 1: Questionnaire RAG]
        ↓
[Stage 2: Toplines/Crosstabs RAG]
        ↓
[Synthesize Response] ← GPT-4 Transformer
        ↓
Answer + Visualization
```

### Core Transformer Application: Research Brief Generation

```python
def _generate_research_brief(self, state):
    """
    Main transformer application showcasing:
    1. Self-attention for contextual understanding
    2. Structured JSON generation
    3. Multi-task learning
    4. Few-shot learning from prompt examples
    """
    
    # Load system prompt with schema and examples
    system_prompt = _load_prompt_file("research_brief_prompt.txt")
    
    # Configure for structured output (Pydantic schema)
    brief_generator = self.llm.with_structured_output(ResearchBrief)
    
    # Generate routing decision in one forward pass
    brief = brief_generator.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User question: {question}")
    ])
    
    return brief  # {action, reasoning, data_sources, stages}
```

### Demo Examples

**Example 1: Simple Query (Single-Stage)**
```
Query: "Show me questions about healthcare"

Research Brief:
{
  "action": "route_to_sources",
  "reasoning": "User wants questions, not data. Single questionnaire query.",
  "data_sources": [{"source_type": "questionnaire", "query": "healthcare"}]
}

Output: 7 healthcare questions (insurance, Medicare, policy)
```

**Example 2: Multi-Stage Query (Complex)**
```
Query: "What percentage of people approved of Biden in June 2025?"

Research Brief:
{
  "action": "execute_stages",
  "reasoning": "Need Biden questions first, then June 2025 frequencies.",
  "stages": [
    {"stage": 1, "source": "questionnaire", "query": "Biden approval"},
    {"stage": 2, "source": "toplines", "filters": {"month": "June", "year": 2025}}
  ]
}

Execution:
Stage 1 → 3 questions (biden_app_2, biden_job, president_approval)
Stage 2 → June 2025: 45% approve, 42% disapprove, 13% unsure (N=1,847)

Synthesis: "In June 2025, 45% approved of Biden's job performance..."
```

**Example 3: Ambiguity Detection**
```
Query: "Biden approval"

Research Brief:
{
  "action": "followup",
  "reasoning": "Missing time period. Need clarification.",
  "followup_question": "Which time period? We have: Feb 2024, June 2024, 
                        Oct 2024, Feb 2025, June 2025, July 2025."
}

User: "June 2025"
→ System then executes multi-stage retrieval
```

**Example 4: Demographic Analysis**
```
Query: "How does Medicare for All support differ by education?"

Research Brief:
{
  "action": "execute_stages",
  "stages": [
    {"stage": 1, "source": "questionnaire", "query": "Medicare for All"},
    {"stage": 2, "source": "crosstabs", "demographic": "education"}
  ]
}

Output: Bar chart + analysis
- High school: 58% support
- Bachelor's: 52% support
- Graduate: 49% support
```

### Running the System

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys
export OPENAI_API_KEY="your-key"
export PINECONE_API_KEY="your-key"

# Run
python multi_agent_survey.py
```

---

## 4. Evaluation

### Routing Accuracy

Tested on 50 diverse queries across 4 categories:

| Query Type | Count | Correct | Accuracy |
|-----------|-------|---------|----------|
| Direct Answer | 10 | 9 | 90% |
| Single-Stage | 15 | 14 | 93% |
| Multi-Stage | 20 | 17 | 85% |
| Ambiguity | 5 | 5 | 100% |
| **Overall** | **50** | **45** | **90%** |

**Analysis**:
- High accuracy on clear cases
- Most errors in multi-stage planning (stage ordering)
- Zero false positives on ambiguity (never routes unclear queries)

### Error Analysis

**Error Case 1: Complex time comparisons**
```
Query: "How have opinions changed from June to July 2025?"
Expected: 2-stage (June, July)
Actual: 3-stage (questionnaire, June, July) ← redundant

Root Cause: Doesn't track conversation state
Solution: Implemented relevance_checker
```

**Error Case 2: Novel patterns**
```
Query: "Compare questionnaire ordering with response rates"
Expected: Route to questionnaire + toplines
Actual: Tried to answer without data

Root Cause: No training examples of this pattern
Limitation: Constrained by prompt examples
```

### Latency Analysis

| Component | Time | % Total |
|-----------|------|---------|
| Research Brief | 1.2s | 20% |
| Questionnaire | 0.8s | 13% |
| Toplines | 0.9s | 15% |
| Crosstabs | 1.5s | 25% |
| Synthesis | 1.6s | 27% |
| **Total** | **6.0s** | **100%** |

**Bottleneck**: Crosstabs (requires summarization of large documents)

### Rule-Based vs Transformer Routing

| Approach | Flexibility | Accuracy | Maintenance | Novel Queries |
|----------|------------|----------|-------------|---------------|
| Rule-Based | Low | 95%* | High | Requires new rules |
| Transformer | High | 90% | Low | Generalizes |

*Rule-based achieves higher accuracy on known patterns but fails completely on novel queries.

**Trade-off**: 5% accuracy loss for massive flexibility - worth it for research tools with unpredictable query patterns.

---

## 5. Critical Analysis

### What This Reveals

**Finding 1: Transformers Enable Flexible Routing Without Hardcoded Rules**

Traditional systems:
```python
if "percentage" in query and "demographic" in query:
    route_to_crosstabs()
elif "percentage" in query:
    route_to_toplines()
```

Transformer approach:
- Learns routing from prompt examples
- Handles novel query patterns
- Contextual disambiguation
- Multi-step planning with dependencies

**Implication**: Reduces engineering effort for complex decision systems, but introduces some unpredictability.

**Finding 2: RAG Prevents Hallucination But Quality Depends on Retrieval**

Pure generation (no RAG):
```python
llm("What's Biden's approval in June 2025?")
→ Hallucinates numbers (no grounding)
```

RAG system:
```python
retrieved_data = retrieve_from_database(...)
llm("Given this data: ..., answer")
→ Grounded in real data
```

**Challenge**: If retrieval fails (wrong questions, missing data), synthesis will be incomplete even if LLM works correctly.

**Finding 3: Structured Output is Transformer's "Killer App"**

Generates valid JSON from natural language:
- No need for separate NER, intent classification, slot filling
- One model, one forward pass
- Generalizes to new schemas with prompt updates

**Limitation**: Only works for patterns model has seen. Novel structures need fine-tuning.

### Impact

**For Survey Research**:
- Democratizes data access (no SQL knowledge needed)
- Accelerates exploratory analysis (seconds vs hours)
- Enables rapid hypothesis testing

**For NLP/Transformers**:
- Demonstrates practical application of transformer theory
- Shows importance of structured generation
- Illustrates flexibility vs control trade-offs

### Next Steps

**Short-term**:
1. Fine-tune routing model on collected query examples
2. Add interactive visualizations (Plotly charts)
3. Expand to 20+ different surveys

**Long-term**:
1. Multi-lingual support (Spanish surveys)
2. Longitudinal tracking and forecasting
3. Cross-survey comparative analysis

**Broader Implications**:
- Students can explore survey data without SPSS/Stata
- Policymakers can query public opinion in real-time
- Journalists can fact-check claims instantly

---

## 6. Ethical Considerations

**Survey Data Sensitivity**:
- Contains demographic information (race, income, education)
- System emphasizes objective reporting, no stereotypes
- Individual responses never exposed (only aggregates)

**Potential Biases**:
1. **Small sample sizes**: System flags groups <100 respondents
2. **Question framing**: Presents questions verbatim, no editorializing
3. **LLM biases**: Temperature=0 reduces variation, grounding in data limits speculation

**Privacy**: No PII in prompts or responses. All statistics are aggregated.

**Data Limitations**:
- Margins of error: ±2-3% full sample, ±5-10% subgroups
- Online panels may under-represent certain groups
- Quarterly snapshots only, not continuous tracking

---

## 7. Resources & Documentation

### Repository Structure
```
survey-analytics/
├── multi_agent_survey.py          # Main LangGraph agent
├── questionnaire_rag.py            # Vector store RAG
├── toplines_rag.py                 # Summary statistics
├── crosstab_rag.py                 # Demographic analysis
├── relevance_checker.py            # Conversation state
├── prompts/                        # System prompts
└── questionnaire_vectorstores/     # ChromaDB
```

### Key Papers

1. **Attention is All You Need** (Vaswani et al., 2017)  
   https://arxiv.org/abs/1706.03762  
   *Foundation of transformer architecture*

2. **BERT** (Devlin et al., 2018)  
   https://arxiv.org/abs/1810.04805  
   *Encoder for embeddings*

3. **GPT-3** (Brown et al., 2020)  
   https://arxiv.org/abs/2005.14165  
   *Few-shot learning*

4. **RAG** (Lewis et al., 2020)  
   https://arxiv.org/abs/2005.11401  
   *Retrieval-augmented generation*

5. **ReAct** (Yao et al., 2023)  
   https://arxiv.org/abs/2210.03629  
   *Tool-augmented reasoning*

6. **Chain-of-Thought** (Wei et al., 2022)  
   https://arxiv.org/abs/2201.11903  
   *Step-by-step reasoning*

### Code Frameworks

- **LangChain**: https://python.langchain.com
- **LangGraph**: https://langchain-ai.github.io/langgraph
- **ChromaDB**: https://docs.trychroma.com
- **Pydantic**: https://docs.pydantic.dev

---

## Conclusion

This project demonstrates how transformer architecture enables flexible, context-aware decision making in practical applications. By leveraging self-attention, structured generation, and RAG, we built a system that routes complex queries without hardcoded rules.

**Key Contributions**:
1. Practical application of transformer theory to multi-agent routing
2. Structured output generation for decision making
3. Tool-augmented reasoning (ReAct) combining LLM + code
4. RAG architecture preventing hallucination

**Impact**: Democratizes survey data access - from "need SQL/SPSS expertise" to "just ask in plain English."

---

**Repository**: [GitHub link]  
**Demo Video**: [Video link]  
**Contact**: [Your email]@vanderbilt.edu
