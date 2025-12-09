import base64
import cv2
import numpy as np

from dash import Dash, html, dcc, Input, Output, no_update
import dash_bootstrap_components as dbc


# ================== IMAGE ANALYSIS LOGIC ==================

def analyze_rice_field_discoloration(
    img: np.ndarray,
    min_region_area: int = 500,
    field_area_m2: float = None,  # total field area in m² (optional)
    gsd_m: float = None           # ground sampling distance in meters/pixel (optional)
):
    if img is None:
        raise ValueError("Input image is None")

    # Optional: downscale very large images for faster processing
    max_dim = 1500
    h, w = img.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    # Convert to HSV (better for color segmentation)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Healthy vegetation: strong green
    healthy_lower = np.array([25, 40, 40])     # H, S, V
    healthy_upper = np.array([90, 255, 255])

    # Stressed vegetation: yellowish / brownish
    stressed_yellow_lower = np.array([15, 40, 40])
    stressed_yellow_upper = np.array([35, 255, 255])

    stressed_brown_lower = np.array([5, 30, 30])
    stressed_brown_upper = np.array([20, 255, 200])

    # Create masks
    healthy_mask = cv2.inRange(hsv, healthy_lower, healthy_upper)
    stressed_yellow_mask = cv2.inRange(hsv, stressed_yellow_lower, stressed_yellow_upper)
    stressed_brown_mask = cv2.inRange(hsv, stressed_brown_lower, stressed_brown_upper)
    stressed_mask = cv2.bitwise_or(stressed_yellow_mask, stressed_brown_mask)

    # Vegetation = healthy + stressed
    vegetation_mask = cv2.bitwise_or(healthy_mask, stressed_mask)

    # Empty (soil/water/road) = NOT vegetation
    empty_mask = cv2.bitwise_not(vegetation_mask)

    # Clean masks (remove noise)
    kernel = np.ones((5, 5), np.uint8)
    healthy_mask = cv2.morphologyEx(healthy_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    stressed_mask = cv2.morphologyEx(stressed_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    empty_mask = cv2.morphologyEx(empty_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Compute pixel statistics
    total_pixels = h * w
    healthy_pixels = cv2.countNonZero(healthy_mask)
    stressed_pixels = cv2.countNonZero(stressed_mask)
    empty_pixels = cv2.countNonZero(empty_mask)

    # Normalize overlapping pixels (if any)
    overlap = cv2.bitwise_and(healthy_mask, stressed_mask)
    if cv2.countNonZero(overlap) > 0:
        stressed_mask = cv2.bitwise_and(stressed_mask, cv2.bitwise_not(overlap))
        stressed_pixels = cv2.countNonZero(stressed_mask)

    vegetation_mask = cv2.bitwise_or(healthy_mask, stressed_mask)
    empty_mask = cv2.bitwise_not(vegetation_mask)
    empty_pixels = cv2.countNonZero(empty_mask)

    # Basic stats (pixels + percentages)
    stats = {
        "total_pixels": total_pixels,
        "healthy_pixels": healthy_pixels,
        "stressed_pixels": stressed_pixels,
        "empty_pixels": empty_pixels,
        "healthy_percent": 100 * healthy_pixels / total_pixels if total_pixels > 0 else 0,
        "stressed_percent": 100 * stressed_pixels / total_pixels if total_pixels > 0 else 0,
        "empty_percent": 100 * empty_pixels / total_pixels if total_pixels > 0 else 0,
    }

    # Compute areas if possible
    if gsd_m is not None:
        pixel_area_m2 = gsd_m * gsd_m
        stats["healthy_area_m2"] = healthy_pixels * pixel_area_m2
        stats["stressed_area_m2"] = stressed_pixels * pixel_area_m2
        stats["empty_area_m2"] = empty_pixels * pixel_area_m2

    elif field_area_m2 is not None:
        stats["healthy_area_m2"] = (healthy_pixels / total_pixels) * field_area_m2
        stats["stressed_area_m2"] = (stressed_pixels / total_pixels) * field_area_m2
        stats["empty_area_m2"] = (empty_pixels / total_pixels) * field_area_m2

    # Visualization overlay
    color_healthy = (0, 255, 0)   # green
    color_stressed = (0, 0, 255)  # red
    color_empty = (255, 0, 0)     # blue

    class_img = np.zeros_like(img)
    class_img[healthy_mask > 0] = color_healthy
    class_img[stressed_mask > 0] = color_stressed
    class_img[empty_mask > 0] = color_empty

    overlay = cv2.addWeighted(img, 0.6, class_img, 0.4, 0)

    # Draw contours
    def draw_contours(mask, draw_img, color, label):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < min_region_area:
                continue
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            cv2.rectangle(draw_img, (x, y), (x + w_box, y + h_box), color, 2)
            cv2.putText(draw_img, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    draw_contours(stressed_mask, overlay, (0, 0, 255), "STRESSED")
    draw_contours(empty_mask, overlay, (255, 0, 0), "EMPTY")

    return stats, overlay, img


# ================== UTILS: ENCODE / DECODE IMAGES ==================

def np_to_base64_img(img: np.ndarray) -> str:
    """Convert BGR np.ndarray to base64 data URL (JPEG)."""
    success, buffer = cv2.imencode(".jpg", img)
    if not success:
        raise RuntimeError("Failed to encode image")
    img_bytes = buffer.tobytes()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def base64_to_np(data_url: str) -> np.ndarray:
    """Convert Dash upload data URL to BGR np.ndarray."""
    _, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


# ================== DASH APP ==================

DEFAULT_FIELD_AREA_M2 = None
DEFAULT_GSD_M = None

app = Dash(__name__, external_stylesheets=[dbc.themes.COSMO])
server = app.server

app.layout = html.Div(
    style={
        "minHeight": "100vh",
        "background": "linear-gradient(135deg, #1e3c72 0%, #2a5298 40%, #34a853 100%)",
        "padding": "30px 10px",
    },
    children=[
        dbc.Container(
            [
                dbc.Card(
                    [
                        dbc.CardHeader(
                            html.Div(
                                [
                                    html.H2(
                                        "🌾FarmEasy - Smart Rice Field Discoloration Analyzer",
                                        className="mb-0",
                                        style={"fontWeight": "700"},
                                    ),
                                    html.Div(
                                        "Powered by Computer Vision + Dash",
                                        className="text-light small",
                                    ),
                                ],
                                className="text-center",
                            ),
                            style={
                                "background": "linear-gradient(90deg, #ff6a00, #ee0979)",
                                "color": "white",
                                "borderRadius": "1rem 1rem 0 0",
                            },
                        ),

                        dbc.CardBody(
                            [
                                # Controls row (only upload now)
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dcc.Upload(
                                                id="upload-image",
                                                children=html.Div(
                                                    "📤 Upload Field Image",
                                                    style={"fontWeight": "600"},
                                                ),
                                                multiple=False,
                                                className="btn btn-light shadow-sm",
                                                style={
                                                    "cursor": "pointer",
                                                    "display": "inline-block",
                                                    "padding": "0.6rem 1.5rem",
                                                    "borderRadius": "999px",
                                                    "border": "2px solid #ff6a00",
                                                },
                                            ),
                                            width="auto",
                                            className="d-flex justify-content-center",
                                        ),
                                    ],
                                    className="mb-4 g-3 justify-content-center",
                                ),

                                dcc.Loading(
                                    type="circle",
                                    children=[
                                        # Images row
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Card(
                                                        [
                                                            dbc.CardHeader(
                                                                "Input Image",
                                                                className="fw-bold",
                                                                style={
                                                                    "background": "linear-gradient(90deg,#4facfe,#00f2fe)",
                                                                    "color": "white",
                                                                },
                                                            ),
                                                            dbc.CardBody(
                                                                html.Img(
                                                                    id="input-image-display",
                                                                    style={
                                                                        "width": "100%",
                                                                        "height": "auto",
                                                                        "borderRadius": "0.75rem",
                                                                        "objectFit": "contain",
                                                                        "boxShadow": "0 8px 20px rgba(0,0,0,0.15)",
                                                                    },
                                                                )
                                                            ),
                                                        ],
                                                        className="shadow-lg",
                                                        style={"borderRadius": "1rem"},
                                                    ),
                                                    md=6,
                                                    className="mb-4",
                                                ),
                                                dbc.Col(
                                                    dbc.Card(
                                                        [
                                                            dbc.CardHeader(
                                                                "Processed Output",
                                                                className="fw-bold",
                                                                style={
                                                                    "background": "linear-gradient(90deg,#11998e,#38ef7d)",
                                                                    "color": "white",
                                                                },
                                                            ),
                                                            dbc.CardBody(
                                                                html.Img(
                                                                    id="output-image-display",
                                                                    style={
                                                                        "width": "100%",
                                                                        "height": "auto",
                                                                        "borderRadius": "0.75rem",
                                                                        "objectFit": "contain",
                                                                        "boxShadow": "0 8px 20px rgba(0,0,0,0.15)",
                                                                    },
                                                                )
                                                            ),
                                                        ],
                                                        className="shadow-lg",
                                                        style={"borderRadius": "1rem"},
                                                    ),
                                                    md=6,
                                                    className="mb-4",
                                                ),
                                            ],
                                        ),

                                        # Stats section
                                        dbc.Row(
                                            dbc.Col(
                                                dbc.Card(
                                                    [
                                                        dbc.CardHeader(
                                                            "📊 Analysis Summary",
                                                            className="fw-bold",
                                                            style={
                                                                "background": "linear-gradient(90deg,#ff6a00,#ee0979)",
                                                                "color": "white",
                                                            },
                                                        ),
                                                        dbc.CardBody(
                                                            id="stats-display",
                                                            # 👇 centered stat cards
                                                            className="d-flex flex-wrap justify-content-center gap-3",
                                                        ),
                                                    ],
                                                    className="shadow-lg",
                                                    style={"borderRadius": "1rem"},
                                                ),
                                                md=12,
                                            ),
                                            className="mt-2",
                                        ),
                                    ],
                                ),
                            ]
                        ),
                    ],
                    style={
                        "borderRadius": "1rem",
                        "overflow": "hidden",
                        "boxShadow": "0 15px 35px rgba(0,0,0,0.3)",
                        "backdropFilter": "blur(8px)",
                    },
                )
            ],
            fluid=True,
        )
    ],
)


def make_stat_cards(stats: dict):
    """Create colorful mini-cards for stats."""
    items = []

    def mini_card(title, value, subtitle=None, gradient=""):
        return dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.Div(title, className="fw-bold small text-light"),
                        html.H5(value, className="mb-0 text-white"),
                        html.Div(subtitle, className="text-light small mt-1")
                        if subtitle
                        else None,
                    ]
                )
            ],
            className="shadow-sm",
            style={
                "minWidth": "190px",
                "borderRadius": "0.9rem",
                "background": gradient,
                "color": "white",
                "border": "none",
            },
        )

    items.append(
        mini_card(
            "Total Pixels",
            f"{stats.get('total_pixels', 0):,}",
            gradient="linear-gradient(135deg,#667eea,#764ba2)",
        )
    )
    items.append(
        mini_card(
            "Healthy Area",
            f"{stats.get('healthy_pixels', 0):,} px",
            f"{stats.get('healthy_percent', 0):.2f} %",
            gradient="linear-gradient(135deg,#11998e,#38ef7d)",
        )
    )
    items.append(
        mini_card(
            "Stressed Area",
            f"{stats.get('stressed_pixels', 0):,} px",
            f"{stats.get('stressed_percent', 0):.2f} %",
            gradient="linear-gradient(135deg,#ff6a00,#ee0979)",
        )
    )
    items.append(
        mini_card(
            "Empty / Non-crop",
            f"{stats.get('empty_pixels', 0):,} px",
            f"{stats.get('empty_percent', 0):.2f} %",
            gradient="linear-gradient(135deg,#00c6ff,#0072ff)",
        )
    )

    if "healthy_area_m2" in stats:
        items.append(
            mini_card(
                "Healthy Area (m²)",
                f"{stats['healthy_area_m2']:.2f} m²",
                gradient="linear-gradient(135deg,#1d976c,#93f9b9)",
            )
        )
        items.append(
            mini_card(
                "Stressed Area (m²)",
                f"{stats['stressed_area_m2']:.2f} m²",
                gradient="linear-gradient(135deg,#e52d27,#b31217)",
            )
        )
        items.append(
            mini_card(
                "Empty / Non-crop (m²)",
                f"{stats['empty_area_m2']:.2f} m²",
                gradient="linear-gradient(135deg,#3a1c71,#d76d77)",
            )
        )

    return items


# ============== CALLBACK: HANDLE UPLOAD ONLY ==================

@app.callback(
    Output("input-image-display", "src"),
    Output("output-image-display", "src"),
    Output("stats-display", "children"),
    Input("upload-image", "contents"),
    prevent_initial_call=True,
)
def handle_image(upload_contents):
    if upload_contents is None:
        return no_update, no_update, no_update

    img_bgr = base64_to_np(upload_contents)

    # Run analysis on this image
    stats, overlay_img, resized_input = analyze_rice_field_discoloration(
        img_bgr,
        field_area_m2=DEFAULT_FIELD_AREA_M2,
        gsd_m=DEFAULT_GSD_M,
    )

    input_display_src = np_to_base64_img(resized_input)
    output_display_src = np_to_base64_img(overlay_img)
    stat_cards = make_stat_cards(stats)

    return input_display_src, output_display_src, stat_cards


if __name__ == "__main__":
    app.run_server(debug=True)
