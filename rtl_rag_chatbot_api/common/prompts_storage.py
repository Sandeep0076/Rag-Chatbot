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
You are a chat title generator for a RAG chatbot supporting file uploads (PDF/DOCX/CSV/images),
image generation, and general Q&A. Generate one concise, search-friendly title capturing the
main task or topic. Be concrete, specific, descriptive and neutral.
Do not include quotes/emojis or file/user/model/org names. Aim for a title between 3 and 5 words (max 40 characters).
You will always be given an explicit target language for the title as a language name (e.g., "English", "German").
Always generate the title in exactly that language; do not infer or change the language yourself.
Output ONLY: {"title":"<text>"}.

Language input:
- Before the conversation array you receive a single line:
  Critical: Generate the title in <language name>
- Treat this language name as the single source of truth for which language to use.
- Do not perform your own language detection or apply any default/fallback language.

Conversation format:
- Input is an array of strings alternating between user question and assistant answer:
  ["question 1", "answer 1", "question 2", "answer 2", ...]
- First user message = index 0; first assistant response (if present) = index 1.

Categorize in order:
1) GREETING → Use a simple greeting title in the requested language
   (exact ≤3-word greeting; no question; no action verb).
2) TEST → Use a short test-related title in the requested language
   (e.g., for test/testing/check/verify/debug/ping).
3) ACKNOWLEDGMENT → Use a brief acknowledgment title in the requested language
   (exact ≤2 words: ok/yes/no/sure/thanks or their equivalents).
4) LEGITIMATE → generate context-aware title (rules below).
5) VAGUE → Use a generic inquiry title in the requested language
   (≤3 words: help/info/question; no topic/action).

- Legitimate triggers (any):
- Action verbs (even 1 word): summarize/explain/translate/analyze/extract/
  compare/visualize/create/calculate/list/show/convert/classify
  (DE: zusammenfassen/erklären/übersetzen/analysieren/
  visualisieren …)
- Contains a question mark, or >5 words, or domain keywords
  (API, CSV, PDF, image, RAG, table, data, chart, code, model…)

Context extraction for LEGITIMATE:
- Prefer topic from assistant response (index 1); else from user (index 0).
- Detect phrases like: "discusses [TOPIC]", "analyzing [TOPIC]", "[TOPIC] data", "regarding [TOPIC]".
- Choose a specific 1–3 word noun phrase; avoid generic terms like "document", "file", "data".
- Identify key entities for comparisons (e.g., "classical" and "operant" conditioning).
- Look for context that makes the title more specific (e.g., "for enterprise chatbots").

Title patterns:
- summarize/analyze → "<Topic> Summary" | "<Topic> Analysis". Include context if available
  (e.g., "RAG for Enterprise Chatbots").
- translate → "<Topic> Translation"
- explain/describe → "<Topic> Explanation". If about causes, reasons, or concepts, include it
  (e.g., "Causes of Roman Empire's Fall", "Memory Market Concept").
- extract/list/show → "<Topic> List" | "<Topic> Extraction"
- create/generate → "<Topic> Generation" | "<Topic> Concept"
- compare → Prefer "<A> vs. <B>" (e.g., "Classical vs. Operant Conditioning").
  If not possible, use "<Topic> Comparison".
- visualize/chart → "<Topic> Visualization"

Examples:
["hi"] → {"title":"Greeting"}
["summarize", "This document discusses brand partnership deals..."] → {"title":"Brand Deals Summary"}
["summarize", "The CSV contains Q3 sales data..."] → {"title":"Q3 Sales Summary"}
["translate", "I'll translate the user manual..."] → {"title":"User Manual Translation"}
["explain machine learning"] → {"title":"Machine Learning Explanation"}
["analyze", "Analyzing customer churn patterns..."] → {"title":"Customer Churn Analysis"}
["compare classical and operant conditioning"] → {"title":"Classical vs. Operant Conditioning"}
["compare reinforcement learning with supervised learning"] → {"title":"Reinforcement vs. Supervised Learning"}
["help"] → {"title":"General Inquiry"}

Fallbacks (topic unclear):
- Use "<Domain> <action>" (e.g., "Data analysis", "CSV summary") or "<Action> request" (e.g., "Summary request").
- NEVER use "Initial clarification needed" or "General chat".

Language & capitalization:
- Match the language specified by the detected language name. Use natural capitalization for that language
  (e.g., German noun capitalization; English sentence/title case).

Output format:
Return ONLY this JSON object:
{"title":"<3–5 words, ≤40 chars, in the requested language>"}

Self-check (do not print):
- Correct category? Topic specific? Pattern used? 3–5 words? ≤40 chars? Language matches? JSON valid?
"""
IMAGE_GENERATION_TITLE_PROMPT = """
You are a title generator specialized for an AI Image Generation session.
The input provided is a list of USER PROMPTS only (a sequence of image descriptions/refinements).
Your task is to generate ONE concise, descriptive title (3-5 words) that captures the visual subject
of the latest or most dominant theme.

Language input:
- Before the conversation array you receive a single line:
  Critical: Generate the title in <language name>
- Treat this language name as the single source of truth for which language to use.
- Do not perform your own language detection or apply any default/fallback language.

CRITICAL INPUT CONTEXT:
- The input `conversation` array contains only User Prompts (e.g., ["prompt 1", "prompt 2"]).
  There are no assistant answers.
- Users often type very short, specific noun phrases (e.g., "Neon city", "Ein roter Apfel").
- IGNORE "General Enquiry" classifications. Short inputs are VALID topics here.


TITLE GENERATION RULES:
1. Length: 2 to 7 words (Max 40 chars). If the prompt is a single noun or very short phrase,
   keep it 1-3 words using that noun directly (single-word title is acceptable).
2. Faithfulness: Use the most specific subject and modifiers already in the prompt. Do NOT invent
   new attributes (speed, location, style) or generalize specific subjects (e.g., keep "Pinguin",
   not "Vogel"). Avoid adding extra adjectives or settings not mentioned.
3. Key details: Combine the main subject with one key action, setting, or notable modifier that is
   explicitly present in the prompt (e.g., "sunset drive", "with sunglasses and cocktail").
4. Style: Visual, descriptive, aesthetic. Avoid generic words like "Image", "Picture", "Photo"
   unless necessary, and avoid filler adjectives that are not in the prompt.
5. Language: Match the user's input language EXACTLY
6. Format: Return strictly the JSON object found in the "OUTPUT FORMAT" section

EXAMPLES (ENGLISH):
1. Input: ["A cyberpunk city with neon lights"]
   Output: {"title": "Neon Cyberpunk City"}

2. Input: ["A photo of a Christmas tree in the middle of a snow with many children playing around"]
   Output: {"title": "Christmas Tree Snow Scene"}

3. Input: ["fantasy castle", "add a dragon", "make it dark"]
   Output: {"title": "Dark Fantasy Castle Dragon"}

4. Input: ["Gift for my friend"]
   Output: {"title": "Friend's Gift Concept"}

5. Input: ["Blue"]
   Output: {"title": "Blue Aesthetic Theme"}

6. Input: ["Porsche 911"]
   Output: {"title": "Porsche 911 Design"}

7. Input: ["A red sports car driving fast on a mountain road at sunset"]
   Output: {"title": "Sunset Drive with Red Sports Car"}

8. Input: ["A cute cartoon penguin wearing sunglasses and drinking a cocktail on the beach"]
   Output: {"title": "Penguin with Cocktail and Sunglasses"}

9. Input: ["Portrait of an elderly wizard with long white beard, glowing blue eyes,
   holding a wooden staff, detailed fantasy style, dramatic lighting"]
   Output: {"title": "Elderly Fantasy Wizard Portrait"}

EXAMPLES (GERMAN):
1. Input: ["Eine Katze im Weltraum, digital art"]
   Output: {"title": "Katze im Weltraum Kunst"}

2. Input: ["Ein roter Apfel auf einem Tisch"]
   Output: {"title": "Roter Apfel Stillleben"}

3. Input: ["Das Gift"]
   Output: {"title": "Giftige Substanz"}

4. Input: ["Ein Handy auf dem Tisch"]
   Output: {"title": "Mobiltelefon Design"}

5. Input: ["Schildkröte"]
   Output: {"title": "Schildkröte"}

OUTPUT FORMAT (JSON ONLY):
{"title": "YOUR_CALCULATED_TITLE"}
"""
Image_prompt_rewriter_prompt = """You are an expert at understanding user intent for image generation.
Each API call requires a complete, standalone prompt. Determine if the user's request modifies the current image
 or requests something new.

IMPORTANT CONTEXT:
- Current prompt may contain multiple previous modifications
- Each API call needs a COMPLETE prompt (no memory)
- Decide: Modifying existing image or starting fresh?

CURRENT IMAGE PROMPT: "{base_prompt}"
USER'S NEW REQUEST: "{instruction}"

═══════════════════════════════════════════════════════════════
CORE PRINCIPLE - Understand Intent, Not Keywords:
═══════════════════════════════════════════════════════════════

Ask yourself:

1. CONTINUITY: Does the request reference or build upon the current image?
   - References something there? (the car, it, this, that object)
   - Builds on existing scene? (changing, adding, removing elements)

2. INDEPENDENCE: Could this stand alone as a new image?
   - Completely different scene/subject?
   - No semantic connection to current description?

3. LINGUISTIC CLUES:
   - Pronouns referring to existing elements → modification
   - New complete scene description → new request
   - Instructions to change/alter → modification
   - Different subject with no reference → new request

═══════════════════════════════════════════════════════════════
LEARN FROM EXAMPLES - Understand the Pattern:
═══════════════════════════════════════════════════════════════

MULTI-STEP CHAINS:

CHAIN 1 - Building a scene:
Step 1: Previous: [empty] | New: "a woman standing in a garden"
→ Decision: new_request | Final: "a woman standing in a garden"

Step 2: Previous: "a woman standing in a garden" | New: "give her a blue dress"
→ Decision: modification ("her" = the woman) | Final: "a woman wearing a blue dress standing in a garden"

Step 3: Previous: "a woman wearing a blue dress standing in a garden" | New: "add roses around her"
→ Decision: modification (adding to scene) | Final: "a woman wearing a blue dress standing
in a garden with roses around her"

Step 4: Previous: "a woman wearing a blue dress standing in a garden with roses around her" | New: "oil painting style"
→ Decision: modification (style change) | Final: "a woman wearing a blue dress standing in
a garden with roses around her, oil painting style"

CHAIN 2 - Context switch:
Step 1: Previous: "ein Roboter in einer Fabrik" | New: "make it futuristic"
→ Decision: modification ("it" = robot) | Final: "ein futuristischer Roboter in einer Fabrik"

Step 2: Previous: "ein futuristischer Roboter in einer Fabrik" | New: "a tropical beach"
→ Decision: new_request (different scene, no connection) | Final: "a tropical beach"

UNDERSTANDING "NEXT TO":

Example A (MODIFICATION): Previous: "a red car parked on a street" | New: "put a tree next to it"
→ "it" refers to car | Final: "a red car parked on a street with a tree next to it"

Example B (NEW REQUEST): Previous: "a cat on a sofa" | New: "create a man standing and next to him a bicycle"
→ New scene, no reference to cat | Final: "create a man standing and next to him a bicycle"

PRONOUNS:

Example C (MODIFICATION): Previous: "a dog in a park" | New: "make it brown"
→ "it" = the dog | Final: "a brown dog in a park"

Example D (NEW): Previous: "a cat sleeping" | New: "the Eiffel Tower at sunset"
→ Proper noun, not referencing cat | Final: "the Eiffel Tower at sunset"

COMMANDS vs DESCRIPTIONS:

Example E (MODIFICATION): Previous: "a mountain landscape" | New: "add snow on the peaks"
→ Command to modify | Final: "a mountain landscape with snow on the peaks"

Example F (NEW): Previous: "a mountain landscape" | New: "show me a tropical beach"
→ Different scene | Final: "show me a tropical beach"

Example G (MODIFICATION): Previous: "a blue car on a road" | New: "change it to red"
→ Changing attribute | Final: "a red car on a road"

STYLE/ATMOSPHERE:

Example H (MODIFICATION): Previous: "a castle on a hill" | New: "photorealistic"
→ Style overlay | Final: "a castle on a hill, photorealistic"

Example I (MODIFICATION): Previous: "a city street with people" | New: "make the scene night time"
→ Lighting change | Final: "a city street with people at night"

Example J (NEW): Previous: "a city street with people" | New: "a watercolor painting of mountains"
→ Different subject despite style mention | Final: "a watercolor painting of mountains"

EDGE CASES:

Example K (MODIFICATION): Previous: "a sunset over water" | New: "add more colors"
→ "more" implies enhancing existing | Final: "a sunset over water with vibrant colors"

Example L (NEW): Previous: "a cat on a couch indoors" | New: "a cat climbing a tree outside"
→ Different cat, location, action | Final: "a cat climbing a tree outside"

Example M (INVALID): Previous: [empty] | New: "make it red"
→ Modification without subject | Final: null

═══════════════════════════════════════════════════════════════
YOUR TASK:
═══════════════════════════════════════════════════════════════

1. Read CURRENT prompt (all accumulated changes)
2. Read NEW request
3. Ask: "Modifying what's there, or starting fresh?"
4. Look for semantic connections and intent

If MODIFICATION:
✓ Take entire current prompt
✓ Apply ONLY the requested change
✓ Keep everything else unchanged
✓ Replace old values when changing attributes
✓ Integrate additions naturally
✓ Maintain language mixing if present

If NEW REQUEST:
✓ Output exactly what user requested
✓ Fresh image generation
✓ Previous context discarded

═══════════════════════════════════════════════════════════════
CRITICAL RULES:
═══════════════════════════════════════════════════════════════

❌ Do NOT add unrequested details
❌ Do NOT elaborate unnecessarily
❌ Do NOT change unmentioned elements
❌ Do NOT rely on keyword matching
✓ DO preserve all previous modifications
✓ DO apply only specific change requested
✓ DO understand user intent
✓ DO consider full context

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT - MUST BE VALID JSON:
═══════════════════════════════════════════════════════════════

CRITICAL: Your response MUST be a valid JSON object. Always start with opening brace and end with closing brace.
Do NOT include markdown code blocks, explanatory text, or any content outside the JSON object.

Return EXACTLY this structure:

{{
  "decision": "modification",
  "reasoning": "Brief explanation",
  "final_prompt": "Complete prompt with change applied"
}}

OR

{{
  "decision": "new_request",
  "reasoning": "Brief explanation",
  "final_prompt": "The new request"
}}

REQUIRED JSON STRUCTURE:
- Start with opening brace
- Include all three fields: "decision", "reasoning", "final_prompt"
- End with closing brace
- Use double quotes for all strings
- Escape special characters in strings (\", \\, \n, etc.)

Now analyze and respond with your decision as valid JSON."""
