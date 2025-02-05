VISUALISATION_PROMPT = """For the given context, generate a visualization of the data. If the context is list,
dictionary, json or tabularform and have more than 50 rows or columns, then take first 50 rows or columns only.
Return the information in a structured JSON format that can be used to plot the data in a graph.
If no chart type is given in context, then generate appropriate chart type based on context.

CRITICAL: Return ONLY the JSON object. DO NOT wrap it in markdown code blocks (```json). DO NOT include any other text.

Follow this schema strictly:

{
  "chart_type": "Line Chart | Bar Chart | Pie Chart | Scatter Plot | Histogram | Box Plot |
Heatmap | 3D Scatter Plot | Surface Plot | Bubble Chart",
  "title": "Descriptive chart title",
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
    "stacked": false                      // For bar charts only
  }
}

**Critical Rules for Each Chart Type:**

1. Line Chart:
   - MUST use "datasets" format
   - Each dataset MUST have "x" and "y" arrays of equal length
   - "x" can be dates or numbers

2. Bar Chart:
   - MUST use "datasets" format for multiple series
   - For single series, can use either:
     a) datasets: [{"label": "Data", "x": ["cat1", "cat2"], "y": [val1, val2]}]
     b) simplified: {"values": [val1, val2], "categories": ["cat1", "cat2"]}

3. Pie Chart:
   - MUST use "values" and "categories" format
   - MUST NOT use "datasets"
   - Arrays must be of equal length

4. Scatter/Bubble:
   - MUST use "datasets" format
   - Each dataset MUST have "x" and "y"
   - Bubble charts MUST include "size"

5. Heatmap:
   - MUST use "matrix" format
   - MUST include "x_categories" and "y_categories"

6. Box Plot:
   - MUST use "datasets" format
   - Each dataset MUST have "y" values
   - "x" is optional for categories

7. Histogram:
   - MUST use "values" array
   - Categories are auto-generated

Example Bar Chart Response:
{
  "chart_type": "Bar Chart",
  "title": "Sales by Region",
  "data": {
    "datasets": [{
      "label": "Sales",
      "x": ["North", "South", "East", "West"],
      "y": [100, 150, 120, 180]
    }]
  },
  "labels": {
    "x": "Region",
    "y": "Sales (USD)"
  },
  "options": {
    "color_palette": "Viridis",
    "stacked": false
  }
}

Remember:
- All numeric values must be real numbers
- All arrays must have matching lengths
- No placeholder values or comments in output
- Use proper JSON format with double quotes
- Use true/false (lowercase) for booleans
- If the context is list, dictionary, json or tabularform and have more than 50 rows or columns,
then take first 50 rows or columns only.
"""
