VISUALISATION_PROMPT = """For the given context, generate a visualization of the data.
Return the information in a structured JSON format that can be used to plot the data in a graph.
If not chart type is given in context, then generate appropriate chart type based on data.
Follow this schema strictly:

{
  "chart_type": "Line Chart | Bar Chart | Pie Chart | Scatter Plot | Histogram | Box Plot | Heatmap",
  "title": "Descriptive title for the chart",
  "data": {
    // CHOOSE ONE STRUCTURE BASED ON CHART TYPE:
    "datasets": [  // For line/bar/scatter
      {
        "label": "Series 1",
        "x": [val1, val2, ...],
        "y": [val1, val2, ...]
      }
    ],
    "values": [num1, num2, ...],  // For pie/histogram
    "categories": ["Cat1", "Cat2", ...],  // For bar/pie labels
    "matrix": [[...], [...]]       // For heatmaps only
  },
  "labels": {
    "x": "X-axis label (e.g., Time)",
    "y": "Y-axis label (e.g., Revenue)"
  },
  "options": {
    "color_palette": "viridis",  // Optional
    "stacked": true/false        // For bar charts
  }
}

**Rules:**
1. For Pie Charts:
   - Use "values" and "categories" instead of datasets
   - Example: {"chart_type":"Pie Chart","data":{"values":[30,50,20],"categories":["A","B","C"]}}

2. For Heatmaps:
   - Include "matrix" (2D array) and x/y categories
   - Example: {"chart_type":"Heatmap","data":{"matrix":[[1,2],[3,4]],
   "x_categories":["X1","X2"],"y_categories":["Y1","Y2"]}}

3. For Box Plots:
   - Use "datasets" with raw values arrays
   - Example: {"chart_type":"Box Plot","data":{"datasets":[{"label":"Group A","values":[10,20,30]}]}}

4. If insufficient data, return: {"error": "Insufficient data to plot"}

Return **only valid JSON** (no markdown formatting). Use double quotes.
Replace placeholder values with actual data from the context."""
