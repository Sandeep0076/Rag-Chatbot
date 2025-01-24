import base64
from io import BytesIO
from typing import Any, Dict

import plotly.express as px
import plotly.graph_objects as go


class UniversalChartGenerator:
    """
    Unified chart generator supporting 2D/3D visualizations with backward compatibility

    Features:
    - Handles all common chart types (line, bar, pie, scatter, heatmap, 3D, etc.)
    - Maintains compatibility with original 2D JSON format
    - Supports advanced features (animations, facets, bubble charts)
    - Professional styling with customization options
    - Multiple output formats (HTML, PNG, JPEG, SVG, Base64)

    Version: 2.0 (Backward compatible with v1.0)
    """

    def __init__(self, chart_config: Dict[str, Any]):
        self.config = self._validate_config(chart_config)
        self.fig = self._create_figure()
        self._apply_styling()

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced validation supporting both old and new formats"""
        # Version compatibility layer
        if "datasets" not in config.get("data", {}) and "values" in config.get(
            "data", {}
        ):
            config["data"] = {"datasets": [config["data"]]}

        # Chart requirements mapping
        REQUIREMENTS = {
            "line chart": [["x", "y"]],
            "bar chart": [["x", "y"], ["categories", "values"]],
            "pie chart": [["categories", "values"]],
            "scatter plot": [["x", "y"]],
            "histogram": [["values"]],
            "box plot": [["values"]],
            "heatmap": [["matrix"]],
            "3d scatter plot": [["x", "y", "z"]],
            "surface plot": [["z"]],
            "bubble chart": [["x", "y", "size"]],
        }
        charts_allowed = list(REQUIREMENTS.keys())
        chart_type = config["chart_type"].lower()
        if chart_type not in REQUIREMENTS:
            raise ValueError(
                f"Unsupported chart type: {chart_type}, please choose one of {charts_allowed}"
            )

        data = config["data"]
        required = REQUIREMENTS[chart_type]
        if not any(
            all(d in data.get("datasets", [{}])[0] or d in data for d in group)
            for group in required
        ):
            raise ValueError(f"Missing required fields for {chart_type}: {required}")

        # Set defaults
        config.setdefault("labels", {})
        config.setdefault("options", {})
        return config

    def _create_figure(self):
        """Smart figure creation handling both 2D and 3D"""
        chart_type = self.config["chart_type"].lower()
        data = self.config["data"]
        options = self.config["options"]

        # Get primary dataset
        dataset = data.get("datasets", [{}])[0]

        # 3D and Special Cases
        if chart_type == "3d scatter plot":
            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=dataset.get("x"),
                        y=dataset.get("y"),
                        z=dataset.get("z"),
                        mode="markers",
                        marker=dict(
                            size=dataset.get("size", 10),
                            color=dataset.get("color"),
                            colorscale=options.get("color_scale", "Viridis"),
                        ),
                    )
                ]
            )
        elif chart_type == "surface plot":
            fig = go.Figure(data=[go.Surface(z=data.get("z"))])
        elif chart_type == "bubble chart":
            fig = px.scatter(
                x=dataset.get("x"),
                y=dataset.get("y"),
                size=dataset.get("size"),
                color=dataset.get("color"),
                size_max=options.get("size_factor", 50),
            )
        else:
            # Handle 2D charts using original logic
            fig = self._create_2d_figure(chart_type, data, dataset)

        return fig

    def _create_2d_figure(self, chart_type: str, data: Dict, dataset: Dict):
        """Backward-compatible 2D figure creation"""
        if chart_type == "line chart":
            fig = go.Figure()
            for ds in data.get("datasets", []):
                fig.add_trace(
                    go.Scatter(
                        x=ds.get("x"),
                        y=ds.get("y"),
                        mode="lines+markers",
                        name=ds.get("label", ""),
                    )
                )
        elif chart_type == "bar chart":
            if "datasets" in data:
                fig = go.Figure()
                for ds in data["datasets"]:
                    fig.add_trace(
                        go.Bar(x=ds.get("x"), y=ds.get("y"), name=ds.get("label", ""))
                    )
                if self.config["options"].get("stacked"):
                    fig.update_layout(barmode="stack")
            else:
                fig = px.bar(x=data.get("categories"), y=data.get("values"))
        elif chart_type == "pie chart":
            fig = px.pie(
                names=data.get("categories"),
                values=data.get("values"),
                hole=self.config["options"].get("hole_ratio", 0),
            )
        elif chart_type == "heatmap":
            fig = go.Figure(
                go.Heatmap(
                    z=data.get("matrix"),
                    x=data.get("x_categories"),
                    y=data.get("y_categories"),
                )
            )
        else:
            fig = getattr(px, chart_type.split()[0])(**dataset)

        return fig

    def _apply_styling(self):
        """Unified styling for all chart types"""
        self.fig.update_layout(
            title=self.config.get("title", ""),
            template="plotly_white",
            font=dict(family="Arial", size=12),
            margin=dict(autoexpand=True),
            **self._get_axis_config(),
            **self._get_legend_config(),
            **self._get_color_config(),
        )

    def _get_axis_config(self) -> Dict:
        """Correct axis title configuration for Plotly 5.18.0+"""
        config = {}

        # Handle 2D axes
        for axis in ["x", "y"]:
            if axis in self.config["labels"]:
                config[f"{axis}axis_title"] = self.config["labels"][axis]

        # Handle 3D scene axes
        if "z" in self.config["labels"]:
            config.setdefault("scene", {})
            config["scene"]["zaxis_title"] = self.config["labels"]["z"]

        return config

    def _get_legend_config(self) -> Dict:
        return (
            {
                "legend": dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                )
            }
            if len(self.config["data"].get("datasets", [])) > 1
            else {}
        )

    def _get_color_config(self) -> Dict:
        """Color scheme management"""
        options = self.config["options"]
        if "color_palette" in options:
            return {
                "colorway": getattr(
                    px.colors.qualitative, options["color_palette"], None
                )
            }
        return {}

    # Output Methods (Same for all versions)
    def show(self):
        """Display in notebook or browser"""
        self.fig.show()

    def save_plot(self, filename: str, format: str = "html"):
        """Save plot to file (supports html, png, jpeg, svg)"""
        if format == "html":
            self.fig.write_html(filename)
        else:
            self.fig.write_image(filename, engine="kaleido")

    def get_base64(self, format: str = "png") -> str:
        """Return base64 encoded image for web applications"""
        buffer = BytesIO()
        self.fig.write_image(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode()

    def get_html(self) -> str:
        """Return raw HTML string for embedding"""
        return self.fig.to_html(full_html=False)


# Example Usage for Both Versions
if __name__ == "__main__":
    # Original 2D Format (v1.0 compatible)
    v1_config = {
        "chart_type": "Line Chart",
        "title": "Average Mercury Levels in Fish Consistent with Tolerable Dose Levels",
        "data": {
            "datasets": [
                {
                    "label": "Mercury Level (ppm)",
                    "x": ["100g", "200g", "300g", "400g", "500g", "1kg"],
                    "y": [0.1, 0.05, 0.033, 0.025, 0.02, 0.01],
                }
            ]
        },
        "labels": {
            "x": "Fish Consumption (grams per week)",
            "y": "Average Mercury Level (ppm)",
        },
        "options": {"color_palette": "Viridis", "stacked": False},
    }

    # New 3D Format (v2.0)
    # v2_config = {
    #     "chart_type": "3D Scatter Plot",
    #     "data": {
    #         "datasets": [{
    #             "x": [1,2,3],
    #             "y": [4,5,6],
    #             "z": [7,8,9],
    #             "color": [0.1,0.5,0.9]
    #         }]
    #     },
    #     "labels": {"x": "X", "y": "Y", "z": "Z"},
    #     "options": {"color_scale": "Plasma"}
    # }

    # Generate both charts
    v1_chart = UniversalChartGenerator(v1_config)
    v1_chart.save_plot("local_data/v1_chart.html")

    # v2_chart = UniversalChartGenerator(v2_config)
    # v2_chart.save_plot("local_data/v2_chart.jpg")
