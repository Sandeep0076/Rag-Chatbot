VISUALISATION_PROMPT = """For the given context, generate a visualization of the data using smart data optimization
techniques.

**CRITICAL TITLE ACCURACY RULE:**
The chart title MUST accurately reflect the actual data granularity and optimization applied:
- If the query was optimized from "daily" to "weekly" aggregation, the title should say "Weekly" not "Daily"
- If the query was optimized from "daily" to "monthly" aggregation, the title should say "Monthly" not "Daily"
- If categories were limited to top N, the title should reflect this (e.g., "Top 10 Categories")
- The title should match the actual data structure being visualized, not the original user request

**DATA OPTIMIZATION RULES:**
1. **Chart Type Based Limits:**
   - Pie Charts: Maximum 15 categories (merge smaller ones into "Others")
   - Bar Charts: Maximum 50 categories for readability
   - Line Charts: Maximum 100 points for clear visualization (sample large datasets)
   - Scatter Plots: Up to 500 points (sample if more)
   - Heatmaps: Maximum 30x30 grid

2. **Large Dataset Strategies (CRITICAL FOR LARGE DATA):**
   - If data has >150 rows for line charts: Use systematic sampling (every nth row) or weekly aggregation
   - If data has >100 rows for time series: Aggregate by week or month instead of daily
   - If data has >50 categories: Group smaller categories or use top N + "Others"
   - For daily time series >3 months: Convert to weekly aggregates
   - For daily time series >1 year: Convert to monthly aggregates
   - Always prioritize readability over completeness

3. **Time Series Optimization:**
   - Daily data spanning >6 months: Sample every 3-7 days or aggregate weekly
   - Daily data spanning >1 year: Aggregate monthly
   - Use representative sampling: first/last + evenly distributed middle points

4. **Data Quality:**
   - Remove null/empty values before processing
   - For categorical data: Group rare categories (< 1% of total) into "Others"
   - For continuous data: Consider outlier handling if they skew visualization

Return the information in a structured JSON format that can be used to plot the data in a graph.
If no chart type is given in context, then generate appropriate chart type based on context.

CRITICAL RESPONSE RULES:
1. Return ONLY the JSON object. DO NOT wrap it in markdown code blocks (```json). DO NOT include any other text.
2. NEVER respond with "cannot generate chart for this query" - ALWAYS generate a valid chart JSON
3. If the data seems unsuitable for visualization, create a simple bar or line chart with summary statistics
4. For time series data with many points, apply data optimization rules to create a readable chart
5. **TITLE ACCURACY**: The title must reflect the actual data granularity, not the original request

Follow this schema strictly:

{
  "chart_type": "Line Chart | Bar Chart | Pie Chart | Scatter Plot | Histogram | Box Plot |
Heatmap | 3D Scatter Plot | Surface Plot | Bubble Chart",
  "title": "Descriptive chart title that matches actual data granularity",
  "data": {
    // For Line/Bar/Scatter/Box/Bubble Charts:
    "datasets": [
      {
        "label": "Series name",
        "x": [numeric_or_string_values],  // For bar charts with categories, use the categories here
        "y": [numeric_values],            // The actual values to plot
        "z": [numeric_values],            // For 3D plots only
        "size": [numeric_values],         // For bubble charts only
        "color": [numeric_values]         // For color mapping
      }
    ],

    // For Pie Charts and Simple Bar Charts ONLY:
    "values": [numeric_values],           // The numeric values
    "categories": [string_values],        // The category labels

    // For Heatmap:
    "matrix": [[num, num], [num, num]],   // 2D numeric array
    "x_categories": [string_values],      // Column labels
    "y_categories": [string_values]       // Row labels
  },
  "labels": {
    "x": "X-axis label",
    "y": "Y-axis label",
    "z": "Z-axis label"                   // For 3D plots only
  },
  "options": {
    "color_palette": "Viridis",           // Optional
    "stacked": false,                     // For bar charts only
    "data_optimization": {                // Information about data processing
      "original_rows": 1000,              // Original data size
      "processed_rows": 50,               // Processed data size
      "method": "top_n_categories"        // Optimization method used
    }
  }
}

**Critical Rules for Each Chart Type:**

1. Line Chart:
   - MUST use "datasets" format
   - Each dataset MUST have "x" and "y" arrays of equal length
   - "x" can be dates or numbers
   - For large time series (>100 points): Use weekly/monthly aggregation or systematic sampling
   - Example: 194 daily points → aggregate to ~28 weekly points for better readability
   - **TITLE**: If weekly aggregation was applied, title should say "Weekly" not "Daily"

2. Bar Chart:
   - MUST use "datasets" format for multiple series
   - For single series, can use either:
     a) datasets: [{"label": "Data", "x": ["cat1", "cat2"], "y": [val1, val2]}]
     b) simplified: {"values": [val1, val2], "categories": ["cat1", "cat2"]}
   - For >50 categories: Show top categories + "Others"
   - **TITLE**: If limited to top N categories, title should reflect this

3. Pie Chart:
   - MUST use "values" and "categories" format
   - MUST NOT use "datasets"
   - Arrays must be of equal length
   - Maximum 15 slices (merge small ones into "Others")
   - **TITLE**: Should reflect the actual categories shown

4. Scatter/Bubble:
   - MUST use "datasets" format
   - Each dataset MUST have "x" and "y"
   - Bubble charts MUST include "size"
   - For >1000 points: Use systematic sampling
   - **TITLE**: Should reflect the actual data granularity

5. Heatmap:
   - MUST use "matrix" format
   - MUST include "x_categories" and "y_categories"
   - Limit to reasonable grid size (50x50 max)
   - **TITLE**: Should reflect the actual dimensions

6. Box Plot:
   - MUST use "datasets" format
   - Each dataset MUST have "y" values
   - "x" is optional for categories
   - **TITLE**: Should reflect the actual data structure

7. Histogram:
   - MUST use "values" array
   - Categories are auto-generated
   - Use appropriate binning for large datasets
   - **TITLE**: Should reflect the binning strategy if significant

**TITLE EXAMPLES:**
- Original request: "orders per day" → If optimized to weekly: "Weekly Orders Shipped"
- Original request: "sales by region" → If limited to top 10: "Sales by Region (Top 10)"
- Original request: "daily temperature" → If monthly aggregation: "Monthly Temperature Trends"
- Original request: "all customers" → If limited to top 20: "Top 20 Customers by Sales"

Example Bar Chart Response with Optimization:
{
  "chart_type": "Bar Chart",
  "title": "Sales by Region (Top 10)",
  "data": {
    "datasets": [{
      "label": "Sales",
      "x": ["North", "South", "East", "West", "Central", "Others"],
      "y": [100, 150, 120, 180, 90, 45]
    }]
  },
  "labels": {
    "x": "Region",
    "y": "Sales (USD)"
  },
  "options": {
    "color_palette": "Viridis",
    "stacked": false,
    "data_optimization": {
      "original_rows": 25,
      "processed_rows": 6,
      "method": "top_categories_with_others"
    }
  }
}

Remember:
- All numeric values must be real numbers
- All arrays must have matching lengths
- No placeholder values or comments in output
- Use proper JSON format with double quotes
- Use true/false (lowercase) for booleans
- Apply intelligent data optimization based on chart type and data size
- Always include data_optimization info when data was processed
- ALWAYS generate a valid chart JSON - never refuse or say "cannot generate chart"
- For large datasets, use sampling/aggregation to create readable visualizations
- **CRITICAL**: Chart title must accurately reflect the actual data granularity and optimization applied
"""

CHART_DETECTION_PROMPT = """
You are an advanced language model tasked with determining whether a
user's question is requesting the generation of a chart or graph.
Your goal is to return a boolean value: `True` if the question implies a request
 for a chart or graph generation or creation, and `False` otherwise.

Consider the following aspects when making your decision:
- Look for keywords such as "chart," "graph," "plot," "visualize,",
 "diagram," "draw," "trend," "visual representation," etc.
- Consider the context of the question, such as requests for data analysis, trends,
or comparisons that might typically be represented visually.

Here are some examples to guide your understanding:

1. "Can you show me a bar chart of the sales data?" → `True`
2. "What are the sales figures for last quarter?" → `False`
3. "Please plot the"Please plot die temperaturtrends over the past year." → `True`
4. "Explain the process of data normalization." → `False`
5. "Visualize the relationship between age and income." → `True`
6. "What is the average income in New York?" → `False`
7. "Generate a pie chart for the market share of each product." → `True`
8. "Liste die 10 meistverkauften Produkte auf." → False
9. "How many charts are there in the document?" → `False`

Remember : Task is to determine if the user's question is asking for a chart or graph generation or creation.

Based on the user's question, determine if a chart or graph is requested and return `True` or `False`.

**User Question:**

"""

BOILERPLATE_PROMPT = """
Given is the extracted text from a website by the use of Beautiful soup, Selenium, Trafilatura.
Evaluate whether it contains substantive data or is predominantly boilerplate information.
Determine if the extracted text primarily contains valuable data or consists largely of boilerplate content
(e.g., cookie notices, legal disclaimers, navigation elements) and/or irrelevant hyperlinks (hrefs).
If it doesn't contain substantive data, return `False` otherwise return `True`.
"""

TITLE_GENERATION_PROMPT = """
You are a chat title generator for a chatbot that supports both RAG and non-RAG chats,
file uploads (PDF/DOCX/CSV/images) for question answering, and image generation via prompts.
Produce one concise, search-friendly title capturing the main task or topic. The title must be
in the same language as the latest user turn (primarily English or German; if another language
is clearly used, match it). Be concrete, specific, and neutral. Do not include quotes, emojis,
punctuation beyond spaces, model names, file names, user names, or organization names.
Limit to 3–5 words and ≤40 characters. Output ONLY: {"title":"<text>"}.

Conversation format:
- Input is an array of strings alternating between user question and assistant answer:
  ["question 1", "answer 1", "question 2", "answer 2", ...]
- The last user message is the last odd-indexed element (0-based) if the array length is odd;
  otherwise, the last user message is at index length-2.

Reconstruction rules:
- Reconstruct roles internally as:
  index 0 = user, 1 = assistant, 2 = user, 3 = assistant, and so on.
- Determine language from the latest user message; if unclear, fall back to the first user
  message; if still unclear, use English.

Focus & domain rules:
- General Q&A: Summarize the core topic (e.g., "API rate limits", "Neural networks basics").
- CSV/data tasks: Prefer task-oriented phrasing (e.g., "CSV cleaning in Python",
  "Verkaufsanalyse aus CSV").
- RAG over files (PDF/DOCX/CSV/images): Reflect the user's retrieval/QA objective and domain
  (e.g., "Policy RAG QA", "RAG für Handbuch"), but do NOT include specific file names.
- Image understanding (user provides an image): Reflect the core ask (OCR / description /
  classification), e.g., "Image OCR summary", "Bildbeschreibung KI".
- Image generation (user prompts to create an image): Reflect content + generation
  (e.g., "Logo image generation", "Landschaftsbild erstellen").
- Avoid generic/meta titles like "Help me", "General chat", "Questions".
- If multiple topics exist, choose the dominant user intent (prefer the latest user turn).

Edge cases:
- If the topic remains unclear after the first exchange, use:
  - English: "Initial clarification needed"
  - German: "Erste Klärung nötig"

Capitalization:
- Use natural capitalization appropriate to the language (German noun capitalization;
  English sentence-style or natural title case), avoiding ALL CAPS.

Output format:
Return ONLY this JSON object:
{"title":"<3–5 words, ≤40 chars, language of latest user>"}

Self-check (do not print this checklist):
- 3–5 words and ≤40 characters?
- Matches latest user language (EN/DE preferred; match other language if clearly used)?
- Specific and task-focused (Q&A/CSV/RAG/Image-understanding/Image-generation)?
- No quotes/emojis/punctuation beyond spaces; no file/model/user/org names.
If any check fails, revise internally and output only the corrected JSON.
"""
