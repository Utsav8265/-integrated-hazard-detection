"""
Integrated Hazard Detection for Long-Distance Drivers
------------------------------------------------------
Features:
1. AI-based road damage detection from dashcam videos.
2. Route analysis with weather forecasts and geographical context.
3. Segment-wise road risk assessment and interactive mapping.

Run:
    python -m streamlit run app.py
"""

from pathlib import Path
import logging

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import detector
import risk_engine
import route_engine
import weather_service


# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parent
OUTPUT_DIR = BASE / "outputs"
MODEL_DIR = BASE / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = MODEL_DIR / "road_damage.pt"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Integrated Hazard Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

BAND_COLOR = {
    "Low": "#16a34a",
    "Medium": "#f59e0b",
    "High": "#dc2626",
}

BAND_ICON = {
    "Low": "🟢",
    "Medium": "🟠",
    "High": "🔴",
}

AREAS = list(risk_engine.AREA.keys())
TRAFFIC_LEVELS = list(risk_engine.TRAFFIC)

VIDEO_TYPES = ["mp4", "avi", "mov", "mkv"]


# ============================================================
# PAGE STYLING
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 1.5rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #0f172a, #1d4ed8);
        color: white;
        margin-bottom: 1.2rem;
    }

    .hero h1 {
        color: white;
        margin-bottom: 0.5rem;
        font-size: 2rem;
    }

    .hero p {
        color: #dbeafe;
        margin-bottom: 0;
    }

    .section-heading {
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    .notice {
        padding: 0.8rem 1rem;
        border-radius: 10px;
        background-color: #eff6ff;
        border-left: 4px solid #2563eb;
        margin-bottom: 1rem;
    }

div[data-testid="stMetric"] {
    background-color: #1e293b;
    border: 1px solid #334155;
    padding: 1rem;
    border-radius: 12px;
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] label p {
    color: #cbd5e1 !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
}

div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #94a3b8 !important;
}

    div[data-testid="stTabs"] button {
        font-weight: 600;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "video_res" not in st.session_state:
    st.session_state["video_res"] = None

if "route" not in st.session_state:
    st.session_state["route"] = None

if "last_video_name" not in st.session_state:
    st.session_state["last_video_name"] = None


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner="Loading road-damage detection model...")
def get_model(model_path_string):
    """
    Load and cache the AI model.

    The cache key includes the model path.
    """
    return detector.load_model(Path(model_path_string))


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_filename(filename):
    """Create a safe filename for storing uploaded videos."""
    name = Path(filename).name
    cleaned = "".join(
        char if char.isalnum() or char in "._-" else "_"
        for char in name
    )
    return cleaned or "uploaded_video.mp4"


def render_section_title(title, description=None):
    """Render a reusable section heading."""
    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True,
    )

    if description:
        st.caption(description)


def show_risk_badge(band):
    """Return the configured icon for a risk band."""
    return BAND_ICON.get(band, "⚪")


def dataframe_download_button(
    label,
    dataframe,
    filename,
    key,
):
    """Display a CSV download button."""
    csv_data = dataframe.to_csv(index=False).encode("utf-8")

    st.download_button(
        label=label,
        data=csv_data,
        file_name=filename,
        mime="text/csv",
        key=key,
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🚗 Hazard Detection")

    st.caption("Long-distance driving safety dashboard")

    st.divider()

    st.subheader("AI Model Settings")

    model_path = st.text_input(
        "Model weights path",
        value=str(DEFAULT_MODEL),
        help="Path to your trained road-damage model.",
    )

    conf = st.slider(
        "Detection confidence",
        min_value=0.05,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Higher confidence can reduce false detections.",
    )

    stride = st.slider(
        "Process every Nth frame",
        min_value=1,
        max_value=6,
        value=2,
        help="Higher values generally reduce processing time.",
    )

    st.caption(
        "A higher frame stride can miss short-lived hazards "
        "and small road defects."
    )

    st.divider()

    st.subheader("Project Information")

    st.write("**AI:** Road-damage detection")
    st.write("**Weather:** Forecast-based context")
    st.write("**Mapping:** Interactive route visualization")
    st.write("**Risk:** Prototype heuristic scoring")

    st.info(
        "Risk scores are experimental indicators. "
        "They are not validated accident probabilities "
        "and should not replace road signs or official warnings."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🚗 Integrated Hazard Detection</h1>
        <p>
            AI-powered road inspection, route weather analysis,
            and risk assessment for long-distance drivers.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="notice">
        <strong>Safety note:</strong>
        This prototype assists with hazard awareness.
        It does not provide certified road-safety predictions.
        Always follow traffic laws and official instructions.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MAIN TABS
# ============================================================

tab_video, tab_route, tab_risk = st.tabs(
    [
        "🎥 Road Video AI",
        "🗺️ Route & Weather",
        "📊 Risk Dashboard",
    ]
)


# ============================================================
# TAB 1 — ROAD VIDEO AI
# ============================================================

with tab_video:

    render_section_title(
        "Dashcam Road Damage Detection",
        "Upload a video to identify potential potholes and road cracks.",
    )

    uploaded_video = st.file_uploader(
        "Upload dashcam video",
        type=VIDEO_TYPES,
        help="Supported formats: MP4, AVI, MOV, MKV.",
        key="dashcam_uploader",
    )

    col_info1, col_info2, col_info3 = st.columns(3)

    col_info1.info("🎯 AI-based hazard detection")
    col_info2.info("📈 Detection statistics")
    col_info3.info("📥 Downloadable CSV report")

    run_detection = st.button(
        "▶ Run Road Damage Detection",
        type="primary",
        disabled=uploaded_video is None,
        use_container_width=True,
        key="run_detection",
    )

    if run_detection and uploaded_video is not None:

        model_file = Path(model_path).expanduser()

        if not model_file.is_file():
            st.error(
                f"Model file not found:\n\n`{model_file}`\n\n"
                "Place your trained model at the configured path "
                "or use the model-download procedure described "
                "in your README."
            )

        else:
            input_name = safe_filename(uploaded_video.name)
            source_path = OUTPUT_DIR / f"input_{input_name}"
            output_path = OUTPUT_DIR / "annotated.mp4"

            progress = st.progress(
                0,
                text="Preparing video...",
            )

            try:
                uploaded_video.seek(0)

                with source_path.open("wb") as output_file:
                    output_file.write(uploaded_video.getbuffer())

                progress.progress(
                    0.05,
                    text="Loading AI model...",
                )

                model = get_model(str(model_file.resolve()))

                progress.progress(
                    0.10,
                    text="Analyzing video frames...",
                )

                result = detector.analyze_video(
                    model,
                    source_path,
                    output_path,
                    conf=conf,
                    frame_stride=stride,
                    progress_cb=lambda value: progress.progress(
                        max(0.0, min(float(value), 1.0)),
                        text="Analyzing video frames...",
                    ),
                )

                if not isinstance(result, dict):
                    raise ValueError(
                        "The detector returned an unexpected result."
                    )

                if "summary" not in result:
                    raise ValueError(
                        "The detector result does not contain a summary."
                    )

                if "detections" not in result:
                    raise ValueError(
                        "The detector result does not contain detections."
                    )

                st.session_state["video_res"] = result
                st.session_state["last_video_name"] = uploaded_video.name

                st.success("Video analysis completed successfully.")

            except Exception as exc:
                logger.exception("Road-damage video analysis failed.")

                st.error(
                    "Video analysis failed. Check the terminal logs "
                    "for technical details."
                )

                st.caption(f"Error: {exc}")

            finally:
                progress.empty()

    video_result = st.session_state.get("video_res")

    if video_result:

        summary = video_result["summary"]

        st.divider()

        render_section_title("Detection Summary")

        metric_columns = st.columns(5)

        metric_columns[0].metric(
            "Potholes",
            summary.get("potholes", 0),
        )

        metric_columns[1].metric(
            "Cracks",
            summary.get("cracks", 0),
        )

        metric_columns[2].metric(
            "Frames Analyzed",
            summary.get("frames_processed", 0),
        )

        metric_columns[3].metric(
            "Inference Time",
            f'{summary.get("avg_inference_ms", 0)} ms/frame',
        )

        metric_columns[4].metric(
            "Pipeline FPS",
            summary.get("pipeline_fps", 0),
        )

        st.caption(
            "Count method: "
            + str(summary.get("count_method", "Not specified"))
        )

        st.caption(
            "Uploaded video: "
            + str(st.session_state.get("last_video_name") or "Unknown")
        )

        video_file = video_result.get("video")

        if video_file is not None:
            try:
                video_bytes = (
                    video_file.read_bytes()
                    if isinstance(video_file, Path)
                    else bytes(video_file)
                )

                if video_bytes:
                    render_section_title("Annotated Video")
                    st.video(video_bytes)

            except (OSError, TypeError, ValueError) as exc:
                logger.warning("Could not display annotated video: %s", exc)
                st.warning(
                    "The annotated video could not be displayed. "
                    "Check whether video processing produced a valid file."
                )

        snapshots = video_result.get("snapshots", [])

        if snapshots:
            render_section_title("Highest-Confidence Detections")

            image_columns = st.columns(3)

            for index, image_data in enumerate(snapshots):
                with image_columns[index % 3]:
                    st.image(
                        image_data,
                        caption=f"Detection {index + 1}",
                        width="stretch",
                    )

        detections = video_result["detections"]

        if isinstance(detections, pd.DataFrame) and not detections.empty:

            render_section_title("Detection Details")

            st.dataframe(
                detections,
                width="stretch",
                height=300,
                hide_index=True,
            )

            dataframe_download_button(
                "⬇ Download Detection CSV",
                detections,
                "hazard_detections.csv",
                "download_detections",
            )

        else:
            st.info(
                "No road damage was detected above the selected "
                "confidence threshold."
            )

    else:
        st.info(
            "Upload a dashcam video and run detection to see "
            "AI-generated results here."
        )


# ============================================================
# ROUTE BUILDER
# ============================================================

def build_route(start, destination, segment_count, demo=False):
    """
    Build either a live route or an offline demonstration route.

    Live routing depends on the existing route_engine and
    weather_service implementations.
    """

    if demo:

        route = route_engine.demo_route(segment_count)

        route["source"] = "DEMO — Offline Synthetic Data"

        points = route_engine.sample_points(
            route["coords"],
            segment_count,
            route["duration_h"],
        )

        demo_weather = weather_service.DEMO_WEATHER

        for index, point in enumerate(points):
            weather = demo_weather[index % len(demo_weather)]
            point.update(weather)
            point["area"] = "Unknown"

        if len(points) > 1:
            points[1]["area"] = "Dense populated area"

        if len(points) > 2:
            points[2]["area"] = "Forest / low-service area"

        route["start"] = "Delhi (Demo)"
        route["dest"] = "Jaipur (Demo)"

    else:

        if not start.strip() or not destination.strip():
            raise ValueError(
                "Please enter both a starting location "
                "and a destination."
            )

        start_location = route_engine.geocode(start.strip())
        end_location = route_engine.geocode(destination.strip())

        route = route_engine.get_route(
            start_location,
            end_location,
        )

        route["source"] = (
            "LIVE — OSRM + Open-Meteo + OpenStreetMap"
        )

        route["start"] = start_location["name"]
        route["dest"] = end_location["name"]

        points = route_engine.sample_points(
            route["coords"],
            segment_count,
            route["duration_h"],
        )

        for point in points:

            forecast = weather_service.get_forecast(
                point["lat"],
                point["lon"],
                point["eta_h"],
            )

            point.update(forecast)

            point["area"] = route_engine.area_context(
                point["lat"],
                point["lon"],
            )

    if not points:
        raise ValueError(
            "No route segments were generated. "
            "Try another route or use the demo route."
        )

    route["segments"] = points

    return route


# ============================================================
# TAB 2 — ROUTE & WEATHER
# ============================================================

with tab_route:

    render_section_title(
        "Route & Weather Analysis",
        "Analyze road segments using route information, weather forecasts, "
        "and geographical context.",
    )

    with st.form("route_form"):

        input_col1, input_col2, input_col3 = st.columns(
            [2, 2, 1]
        )

        start_location = input_col1.text_input(
            "Starting location",
            value="Delhi",
        )

        destination = input_col2.text_input(
            "Destination",
            value="Jaipur",
        )

        segment_count = input_col3.slider(
            "Segments",
            min_value=3,
            max_value=8,
            value=5,
        )

        live_route_button = st.form_submit_button(
            "🌐 Analyze Live Route",
            type="primary",
            width="stretch",
        )

        demo_route_button = st.form_submit_button(
            "🧪 Load Offline Demo",
            width="stretch",
        )

    if live_route_button or demo_route_button:

        is_demo = demo_route_button

        try:
            with st.spinner(
                "Preparing route and collecting route context..."
            ):

                new_route = build_route(
                    start_location,
                    destination,
                    segment_count,
                    demo=is_demo,
                )

            st.session_state["route"] = new_route

            st.success("Route analysis completed.")

        except Exception as exc:

            logger.exception("Route analysis failed.")

            st.error(
                "Could not complete route analysis. "
                "Check your network connection and route-service "
                "configuration, or try the offline demo."
            )

            st.caption(f"Error: {exc}")

    route = st.session_state.get("route")

    if route:

        st.divider()

        st.subheader(
            f'{route["start"]} → {route["dest"]}'
        )

        overview = st.columns(4)

        overview[0].metric(
            "Distance",
            f'{route["distance_km"]:.1f} km',
        )

        overview[1].metric(
            "Estimated Duration",
            f'{route["duration_h"]:.1f} hours',
        )

        overview[2].metric(
            "Segments",
            len(route["segments"]),
        )

        overview[3].metric(
            "Route Mode",
            "Offline Demo" if "DEMO" in route["source"] else "Live",
        )

        st.caption(f'Source: {route["source"]}')

        segment_dataframe = pd.DataFrame(route["segments"])

        desired_columns = [
            "segment",
            "km",
            "eta_h",
            "forecast_time",
            "rain_prob",
            "precip_mm",
            "visibility_m",
            "wind_kmh",
            "area",
        ]

        available_columns = [
            column
            for column in desired_columns
            if column in segment_dataframe.columns
        ]

        render_section_title("Route Segment Details")

        st.dataframe(
            segment_dataframe[available_columns],
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "Weather values represent forecast information "
            "for each segment's estimated arrival time. "
            "Availability depends on the weather-service response."
        )

        dataframe_download_button(
            "⬇ Download Route Segments CSV",
            segment_dataframe,
            "route_segments.csv",
            "download_route_segments",
        )

    else:

        st.info(
            "Enter your route details and choose live analysis "
            "or load the offline demonstration route."
        )


# ============================================================
# TAB 3 — RISK DASHBOARD
# ============================================================

with tab_risk:

    route = st.session_state.get("route")

    if not route:

        st.info(
            "Analyze a route in the Route & Weather tab before "
            "opening the risk dashboard."
        )

    else:

        segments = route["segments"]
        video_result = st.session_state.get("video_res")

        video_summary = (
            video_result.get("summary", {})
            if video_result
            else {}
        )

        video_potholes = video_summary.get("potholes", 0)
        video_cracks = video_summary.get("cracks", 0)

        st.subheader("Route Risk Assessment")

        st.caption(
            "Adjust traffic and area classifications to explore "
            "how the prototype risk score changes. Traffic is a "
            "manual input, not a live traffic feed."
        )

        default_edit_data = pd.DataFrame(
            {
                "segment": [
                    segment["segment"]
                    for segment in segments
                ],
                "traffic": ["Low"] * len(segments),
                "area": [
                    segment.get("area", "Unknown")
                    for segment in segments
                ],
            }
        )

        # Preserve user edits across reruns where possible.
        if (
            "risk_editor_data" not in st.session_state
            or len(st.session_state["risk_editor_data"]) != len(segments)
            or st.session_state.get("risk_editor_route") != (
                route["start"],
                route["dest"],
            )
        ):

            st.session_state["risk_editor_data"] = default_edit_data
            st.session_state["risk_editor_route"] = (
                route["start"],
                route["dest"],
            )

        render_section_title("Adjust Risk Inputs")

        edited_data = st.data_editor(
            st.session_state["risk_editor_data"],
            width="stretch",
            hide_index=True,
            disabled=["segment"],
            key="risk_editor",
            column_config={
                "traffic": st.column_config.SelectboxColumn(
                    "Traffic Level",
                    options=TRAFFIC_LEVELS,
                    required=True,
                ),
                "area": st.column_config.SelectboxColumn(
                    "Area Type",
                    options=AREAS,
                    required=True,
                ),
            },
        )

        st.session_state["risk_editor_data"] = edited_data

        segment_options = [
            segment["segment"]
            for segment in segments
        ]

        video_segment = st.selectbox(
            "Segment where the uploaded video was recorded",
            options=segment_options,
            help=(
                "The uploaded video does not automatically provide "
                "GPS location. Choose the segment to which its "
                "detections should be assigned."
            ),
        )

        st.caption(
            f"Video detections: {video_potholes} potholes and "
            f"{video_cracks} cracks."
        )

        results = []

        for segment, row in zip(
            segments,
            edited_data.itertuples(index=False),
        ):

            video_applies = (
                segment["segment"] == video_segment
            )

            risk = risk_engine.score_segment(
                potholes=(
                    video_potholes
                    if video_applies
                    else 0
                ),
                cracks=(
                    video_cracks
                    if video_applies
                    else 0
                ),
                rain_prob=segment["rain_prob"],
                precip_mm=segment["precip_mm"],
                visibility_m=segment["visibility_m"],
                traffic=row.traffic,
                area=row.area,
            )

            results.append(
                {
                    **segment,
                    **risk,
                    "traffic": row.traffic,
                    "area": row.area,
                }
            )

        if not results:

            st.warning("No risk results are available.")

        else:

            worst_segment = max(
                results,
                key=lambda result: result["score"],
            )

            high_risk_count = sum(
                result["band"] == "High"
                for result in results
            )

            medium_risk_count = sum(
                result["band"] == "Medium"
                for result in results
            )

            low_risk_count = sum(
                result["band"] == "Low"
                for result in results
            )

            st.divider()

            render_section_title("Risk Overview")

            risk_metrics = st.columns(5)

            risk_metrics[0].metric(
                "Highest Risk",
                f'{worst_segment["score"]}/100',
                worst_segment["band"],
            )

            risk_metrics[1].metric(
                "Riskiest Segment",
                f'#{worst_segment["segment"]}',
            )

            risk_metrics[2].metric(
                "High Risk",
                high_risk_count,
            )

            risk_metrics[3].metric(
                "Medium Risk",
                medium_risk_count,
            )

            risk_metrics[4].metric(
                "Low Risk",
                low_risk_count,
            )

            st.caption(
                "These categories use the thresholds implemented "
                "in risk_engine.py."
            )

            # ------------------------------------------------
            # Risk Chart
            # ------------------------------------------------

            render_section_title("Risk Score by Segment")

            chart_dataframe = pd.DataFrame(
                {
                    "Segment": [
                        f'Segment {result["segment"]}'
                        for result in results
                    ],
                    "Risk Score": [
                        result["score"]
                        for result in results
                    ],
                }
            ).set_index("Segment")

            st.bar_chart(
                chart_dataframe,
                y="Risk Score",
            )

            # ------------------------------------------------
            # Interactive Map
            # ------------------------------------------------

            render_section_title(
                "Interactive Route Risk Map",
                "Select a marker to inspect its risk classification "
                "and associated warnings.",
            )

            middle_segment = results[len(results) // 2]

            route_map = folium.Map(
                location=[
                    middle_segment["lat"],
                    middle_segment["lon"],
                ],
                zoom_start=7,
                control_scale=True,
            )

            folium.PolyLine(
                route["coords"],
                color="#2563eb",
                weight=5,
                opacity=0.8,
                tooltip="Planned route",
            ).add_to(route_map)

            for result in results:

                band = result["band"]
                color = BAND_COLOR.get(band, "#64748b")

                warning_list = result.get("warnings", [])

                warning_text = (
                    "<br>".join(
                        str(warning)
                        for warning in warning_list
                    )
                    if warning_list
                    else "No major hazards flagged by this prototype"
                )

                popup_html = (
                    f'<b>Segment {result["segment"]}</b><br>'
                    f'Risk Score: {result["score"]}/100<br>'
                    f'Risk Level: {band}<br>'
                    f'Estimated Distance: {result["km"]} km<br>'
                    f'{warning_text}'
                )

                folium.CircleMarker(
                    location=[
                        result["lat"],
                        result["lon"],
                    ],
                    radius=10,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.85,
                    tooltip=(
                        f'Segment {result["segment"]}: '
                        f'{band} ({result["score"]}/100)'
                    ),
                    popup=folium.Popup(
                        popup_html,
                        max_width=300,
                    ),
                ).add_to(route_map)

            st_folium(
                route_map,
                height=480,
                width=None,
                returned_objects=[],
            )

            st.caption(
                "Green = Low Risk | Orange = Medium Risk | "
                "Red = High Risk"
            )

            # ------------------------------------------------
            # Driver Alerts
            # ------------------------------------------------

            render_section_title("Driver Alerts")

            for result in results:

                band = result["band"]
                icon = show_risk_badge(band)

                with st.expander(
                    f'{icon} Segment {result["segment"]} — '
                    f'km {result["km"]} — '
                    f'{band} Risk ({result["score"]}/100)',
                    expanded=(band == "High"),
                ):

                    warnings = result.get("warnings", [])

                    if warnings:
                        for warning in warnings:
                            st.warning(str(warning))
                    else:
                        st.success(
                            "No major hazards flagged by this prototype."
                        )

                    parts = result.get("parts", {})

                    if parts:
                        st.caption(
                            "Score breakdown: "
                            + ", ".join(
                                f"{key}: {value}"
                                for key, value in parts.items()
                            )
                        )

            # ------------------------------------------------
            # Risk Report
            # ------------------------------------------------

            report_columns = [
                "segment",
                "km",
                "eta_h",
                "lat",
                "lon",
                "rain_prob",
                "precip_mm",
                "visibility_m",
                "wind_kmh",
                "traffic",
                "area",
                "score",
                "band",
            ]

            report = pd.DataFrame(results)

            available_report_columns = [
                column
                for column in report_columns
                if column in report.columns
            ]

            report = report[available_report_columns]

            render_section_title("Route Risk Report")

            st.dataframe(
                report,
                width="stretch",
                hide_index=True,
            )

            dataframe_download_button(
                "⬇ Download Route Risk Report",
                report,
                "route_risk_report.csv",
                "download_risk_report",
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Integrated Hazard Detection for Long-Distance Drivers | "
    "Academic prototype | AI-assisted road awareness"
)