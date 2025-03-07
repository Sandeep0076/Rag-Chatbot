def detect_visualization_need(query_text: str) -> bool:
    """
    Analyze a user query to determine if it likely requires visualization.

    Args:
        query_text: The user's question or prompt

    Returns:
        bool: True if the query likely requires visualization, False otherwise
    """
    # Keywords and phrases that suggest visualization needs
    visualization_keywords = [
        "chart",
        "diagramm",
        "graph",
        "grafik",
        "plot",
        "visualize",
        "visualisieren",
        "visualization",
        "visualisierung",
        "display data",
        "daten anzeigen",
        "show data",
        "daten zeigen",
        "show me a",
        "zeig mir ein",
        "generate a chart",
        "ein diagramm erstellen",
        "generate a graph",
        "eine grafik erstellen",
        "histogram",
        "histogramm",
        "pie chart",
        "kreisdiagramm",
        "bar chart",
        "balkendiagramm",
        "line graph",
        "liniendiagramm",
        "scatter plot",
        "punktdiagramm",
        "trend",
        "trends",
        "distribution",
        "verteilung",
        "compare visually",
        "visuell vergleichen",
        "visual representation",
        "visuelle darstellung",
        "visualize the data",
        "die daten visualisieren",
        "create a graph",
        "eine grafik erstellen",
        "create a chart",
        "ein diagramm erstellen",
        "draw a graph",
        "eine grafik zeichnen",
        "illustrate",
        "illustrieren",
        "represent graphically",
        "grafisch darstellen",
        "visual analysis",
        "visuelle analyse",
    ]

    # Case-insensitive check for keywords
    query_lower = query_text.lower()

    # Check if any visualization keyword is in the query
    for keyword in visualization_keywords:
        if keyword.lower() in query_lower:
            return True

    return False
