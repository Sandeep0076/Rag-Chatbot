VISUALISATION_PROMPT = """For the given context, generate a visualization of the data.
Return the information in a structured JSON format that can be used to plot the data in a graph.
If not chart type is given in context, then generate appropriate chart type based on context.
Follow this schema strictly:

{
  "chart_type": "Line Chart | Bar Chart | Pie Chart | Scatter Plot | Histogram | Box Plot | \
Heatmap | 3D Scatter Plot | Surface Plot | Bubble Chart",
  "title": "Descriptive chart title",
  "data": {
    // STRUCTURE DEPENDS ON CHART TYPE:
    // For Line/Bar/Scatter/Box:
    "datasets": [
      {
        "label": "Series 1 (required for multi-series)",
        "x": [numeric_or_string_values], // REQUIRED
        "y": [numeric_values],           // REQUIRED
        "z": [numeric_values],           // For 3D only
        "size": [numeric_values],        // For bubble charts
        "color": [numeric_values]        // For color mapping
      }
    ],

    // For Pie/Histogram:
    "values": [numeric_values],          // REQUIRED
    "categories": ["label1", "label2"],  // REQUIRED

    // For Heatmap:
    "matrix": [[num, num], [num, num]],  // 2D numeric array REQUIRED
    "x_categories": ["x1", "x2"],        // Column labels
    "y_categories": ["y1", "y2"]         // Row labels
  },
  "labels": {
    "x": "X-axis label (required)",
    "y": "Y-axis label (required)",
    "z": "Z-axis label (for 3D)"         // Optional
  },
  "options": {
    "color_palette": "Viridis | Plotly | ...",
    "stacked": True/False                // For bar charts
  }
}

**Critical Rules:**
1. **Data Types:**
   - All values must be real numbers (no placeholders like val1/Date1)
   - Dates must be ISO strings ("2023-01-01"), not "Date1"

2. **Chart-Specific Requirements:**
   - Pie Charts: MUST use "values" and "categories" (no datasets)
   - 3D Plots: MUST include "z" values in datasets
   - Heatmaps: MUST include "matrix" with numeric 2D array

3. **Error Handling:**
   - If insufficient data, return: {"error": "Insufficient data to plot"}
   - Never include placeholder comments (e.g., // Replace with...)

4. **Output Control:**
   - No markdown formatting (only pure JSON)
   - No trailing commas
   - Return pure JSON only
   - Use double quotes for strings
   - Use True/False (uppercase) for booleans
   - No trailing commas
   - No comments in output

**Examples:**
1. Line Chart:
```json
{
  "chart_type": "Line Chart",
  "title": "Apple Stock Prices",
  "data": {
    "datasets": [{
      "x": ["2023-01-01", "2023-02-01"],
      "y": [150.2, 165.7]
    }]
  },
  "labels": {"x": "Date", "y": "Price (USD)"}
}"""
