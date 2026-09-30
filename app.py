import io
import streamlit as st

from api import analyze_satellite, buildings_nearby, geocode_places, reverse_geocode, satellite_snapshot
from feasibility_checker import check_feasibility, load_models


st.set_page_config(
    page_title="CivicLens | Pan-India Satellite & Feasibility",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700;800&display=swap');
    :root { --ink:#17252f; --muted:#657680; --line:#dbe4e6; --paper:#f5f8f7; --teal:#087f8c; --teal-dark:#075966; --coral:#e87558; }
    .stApp { background:var(--paper); color:var(--ink); }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stSidebar"] { background:#0e242d; }
    [data-testid="stSidebar"] * { color:#e8f2f0; }
    [data-testid="stSidebar"] .stButton button { border-color:#3e6269; color:#e8f2f0; }
    h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; color:var(--ink); }
    p,label,.stMarkdown,.stCaption { font-family:'DM Sans',sans-serif; }
    .hero { padding:1.8rem 0 1.2rem; }
    .eyebrow { color:var(--teal); font:700 .75rem 'DM Sans'; letter-spacing:.14em; text-transform:uppercase; }
    .hero h1 { font-size:clamp(2.1rem,4vw,3.6rem); line-height:1.06; margin:.35rem 0 .8rem; }
    .hero p { color:var(--muted); font-size:1.05rem; max-width:720px; }
    .panel { background:white; border:1px solid var(--line); border-radius:14px; padding:1.35rem 1.5rem .8rem; margin-bottom:1rem; box-shadow:0 8px 30px rgba(23,37,47,.04); }
    .panel-title { color:var(--teal-dark); font:700 1.1rem 'Space Grotesk'; margin-bottom:.25rem; }
    .panel-subtitle { color:var(--muted); font-size:.88rem; margin-bottom:.8rem; }
    .result-good { background:#e2f4ed; border-left:5px solid #26956d; border-radius:12px; padding:1.2rem 1.35rem; }
    .result-review { background:#fff0e9; border-left:5px solid var(--coral); border-radius:12px; padding:1.2rem 1.35rem; }
    .result-label { color:var(--muted); font-size:.75rem; letter-spacing:.1em; text-transform:uppercase; }
    .result-value { font:700 1.6rem 'Space Grotesk'; margin-top:.3rem; }
    .side-brand { font:800 1.45rem 'Space Grotesk'; letter-spacing:-.03em; margin:.7rem 0 1.5rem; }
    .side-brand span { color:#00f0ff; }
    .side-step { border-left:2px solid #38616a; margin:1rem 0; padding-left:.85rem; }
    .side-step strong { display:block; font-size:.9rem; }
    .side-step small { color:#a9c5c5; }
    .stButton > button[kind="primary"] { background:var(--teal); border:0; border-radius:9px; font-weight:700; padding:.75rem; }
    .stButton > button[kind="primary"]:hover { background:var(--teal-dark); }
    div[data-testid="stMetric"] { background:white; border:1px solid var(--line); border-radius:12px; padding:.9rem; }
    .intel-box { background:#ffffff; border:1px solid #dbe4e6; border-radius:12px; padding:1.2rem; margin-top:1rem; box-shadow:0 4px 16px rgba(0,0,0,0.03); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_models():
    return load_models()


try:
    models = get_models()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

CITIES_DATA = {
    "Delhi NCR": (28.6139, 77.2090),
    "Mumbai": (18.9220, 72.8347),
    "Bengaluru": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Jaipur": (26.9124, 75.7873),
    "Bhopal": (23.2599, 77.4126),
    "Indore": (22.7196, 75.8577),
    "Kochi": (9.9312, 76.2673),
}

cities = list(CITIES_DATA.keys())
zones = ["Residential", "Commercial", "Mixed-Use", "Industrial", "Institutional"]

with st.sidebar:
    st.markdown('<div class="side-brand">Civic<span>Lens</span></div>', unsafe_allow_html=True)
    st.markdown("**PAN-INDIA GIS PLATFORM**")
    st.markdown(
        '<div class="side-step"><strong>🛰️ 01 &nbsp;Satellite inspection</strong><small>High-res Esri &amp; Overpass footprints</small></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="side-step"><strong>🏗️ 02 &nbsp;Feasibility screening</strong><small>Compliance &amp; utility risk ML</small></div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("✅ 100% Free Geocoding (Nominatim)\n\n✅ Free Satellite Imagery (ArcGIS)\n\n✅ Overpass Cadastral Extraction")

st.markdown(
    '<div class="hero"><div class="eyebrow">Pan-India Urban Development Intelligence</div>'
    '<h1>Know the risk<br>before you build.</h1>'
    '<p>Screen a proposed project against compliance signals and underground utility risk, with high-resolution satellite inspection anywhere in India.</p></div>',
    unsafe_allow_html=True,
)

tab_sat, tab_form = st.tabs(["🛰️ Pan-India Satellite Inspection", "🏗️ Project Feasibility Assessment"])

# ── TAB 1: SATELLITE INSPECTION ──────────────────────────────────────────────
with tab_sat:
    st.markdown("### 🛰️ Satellite Site Inspection (Anywhere in India)")
    st.write("Inspect any address, coordinates, or urban center across India with satellite imagery and building footprints.")

    col_search, col_city = st.columns([1.5, 1])
    with col_search:
        search_query = st.text_input("Search Indian Address / Landmark / PIN Code", placeholder="e.g. Connaught Place, Whitefield, Bandra, 560001")
    with col_city:
        selected_city_preset = st.selectbox("Quick Jump City Preset", cities)

    # Initialize session state coordinates
    if "sat_lat" not in st.session_state:
        st.session_state.sat_lat = CITIES_DATA["Delhi NCR"][0]
        st.session_state.sat_lon = CITIES_DATA["Delhi NCR"][1]
        st.session_state.sat_label = "Delhi NCR, India"

    # Handle search
    if search_query:
        if st.button("Search Location", type="primary"):
            try:
                results = geocode_places(search_query, limit=1)
                if results:
                    top = results[0]
                    st.session_state.sat_lat = top["lat"]
                    st.session_state.sat_lon = top["lon"]
                    st.session_state.sat_label = top["label"]
                    st.success(f"Found: {top['label']}")
                else:
                    st.warning("No matching Indian address found. Try another query or use coordinates.")
            except Exception as e:
                st.error(f"Search failed: {e}")
    elif selected_city_preset:
        c_lat, c_lon = CITIES_DATA[selected_city_preset]
        if st.button(f"Go to {selected_city_preset}"):
            st.session_state.sat_lat = c_lat
            st.session_state.sat_lon = c_lon
            st.session_state.sat_label = f"{selected_city_preset}, India"

    col_lat, col_lon = st.columns(2)
    with col_lat:
        target_lat = st.number_input("Latitude", value=float(st.session_state.sat_lat), format="%.5f")
    with col_lon:
        target_lon = st.number_input("Longitude", value=float(st.session_state.sat_lon), format="%.5f")

    st.markdown(f"**Target Site:** `{target_lat:.5f}°N, {target_lon:.5f}°E` &mdash; {st.session_state.sat_label}")

    col_img, col_info = st.columns([1.2, 1])

    with col_img:
        st.markdown("#### 📸 Satellite Imagery Snapshot (Esri World Imagery)")
        try:
            img_bytes, content_type, meta = satellite_snapshot(target_lat, target_lon)
            st.image(img_bytes, caption=f"250m inspection radius around {target_lat:.4f}, {target_lon:.4f}", use_container_width=True)
        except Exception as e:
            st.error(f"Could not load satellite snapshot: {e}")

    with col_info:
        st.markdown("#### 🏢 Cadastral Building Footprints (Overpass)")
        try:
            b_data = buildings_nearby(target_lat, target_lon, 300)
            features = b_data.get("features", [])
            st.metric("Mapped Buildings within 300m", len(features))
            if features:
                types = {}
                for f in features:
                    t = f.get("properties", {}).get("building", "unspecified")
                    types[t] = types.get(t, 0) + 1
                st.write("**Surrounding Typologies:**")
                for k, v in sorted(types.items(), key=lambda x: -x[1])[:4]:
                    st.write(f"- `{k}`: {v} structures")
            else:
                st.info("No OpenStreetMap building polygons mapped within 300m buffer.")
        except Exception as e:
            st.warning(f"Building data query: {e}")

        st.markdown("---")
        if st.button("⚡ Run Satellite Spatial & Bylaw Analysis", type="primary", use_container_width=True):
            with st.spinner("Generating spatial morphology and NBC 2016 screening report..."):
                try:
                    analysis_res = analyze_satellite(target_lat, target_lon, st.session_state.sat_label)
                    st.markdown(f"**Engine:** `{analysis_res.get('model', 'Spatial Engine')}`")
                    st.markdown(analysis_res["analysis"])
                    st.caption(analysis_res.get("disclaimer", ""))
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

# ── TAB 2: FEASIBILITY ASSESSMENT ───────────────────────────────────────────
with tab_form:
    with st.form("proposal_form"):
        st.markdown('<div class="panel"><div class="panel-title">01 / Project profile</div><div class="panel-subtitle">Tell us where the project sits and how large the site is.</div>', unsafe_allow_html=True)
        location, site = st.columns(2)
        with location:
            city = st.selectbox("City", cities)
            zone_type = st.selectbox("Zone type", zones)
        with site:
            plot_size_sqm = st.number_input("Plot size (sq m)", min_value=20.0, value=500.0, step=10.0)
            road_width_m = st.number_input("Road width (m)", min_value=1.0, value=12.0, step=0.5)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel"><div class="panel-title">02 / Building proposal</div><div class="panel-subtitle">Use the intended values from your concept or planning submission.</div>', unsafe_allow_html=True)
        building, setbacks = st.columns(2)
        with building:
            far_proposed = st.number_input("Proposed FAR", min_value=0.1, value=1.5, step=0.1)
            height_proposed_m = st.number_input("Proposed height (m)", min_value=1.0, value=18.0, step=0.5)
            floors = st.number_input("Number of floors", min_value=1, value=5, step=1)
            ground_coverage_pct = st.number_input("Ground coverage (%)", min_value=1.0, max_value=100.0, value=45.0, step=1.0)
        with setbacks:
            setback_front_m = st.number_input("Front setback (m)", min_value=0.0, value=6.0, step=0.5)
            setback_rear_m = st.number_input("Rear setback (m)", min_value=0.0, value=4.5, step=0.5)
            setback_side_m = st.number_input("Side setback (m)", min_value=0.0, value=3.0, step=0.5)
            parking_provided_ecs = st.number_input("Parking provided (ECS)", min_value=0.0, value=25.0, step=1.0)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel"><div class="panel-title">03 / Underground utilities</div><div class="panel-subtitle">These signals help estimate possible conflicts below the site.</div>', unsafe_allow_html=True)
        utility, foundation, confidence = st.columns(3)
        with utility:
            basement_proposed = st.checkbox("Basement proposed")
        with foundation:
            foundation_depth_m = st.number_input("Foundation depth (m)", min_value=0.1, value=2.0, step=0.1)
        with confidence:
            gis_record_confidence = st.slider("GIS record confidence", 0.0, 1.0, 0.80, 0.05)
        st.markdown('</div>', unsafe_allow_html=True)

        submitted = st.form_submit_button("Run feasibility assessment  ->", type="primary", use_container_width=True)

    if submitted:
        inputs = {
            "city": city, "zone_type": zone_type, "plot_size_sqm": plot_size_sqm,
            "road_width_m": road_width_m, "far_proposed": far_proposed,
            "height_proposed_m": height_proposed_m, "setback_front_m": setback_front_m,
            "setback_rear_m": setback_rear_m, "setback_side_m": setback_side_m,
            "ground_coverage_pct": ground_coverage_pct, "floors": floors,
            "parking_provided_ecs": parking_provided_ecs,
            "basement_proposed": basement_proposed, "foundation_depth_m": foundation_depth_m,
            "gis_record_confidence": gis_record_confidence,
        }
        result = check_feasibility(inputs, models)
        st.markdown("## Assessment result")
        status_class = "result-good" if result["feasible"] else "result-review"
        st.markdown(
            f'<div class="{status_class}"><div class="result-label">Screening outcome</div>'
            f'<div class="result-value">{result["decision"]}</div></div>',
            unsafe_allow_html=True,
        )
        st.write("")
        risk_col, violations_col, area_col = st.columns(3)
        with risk_col:
            st.metric("Utility conflict risk", f"{result['utility_conflict_probability']:.0%}")
        with violations_col:
            st.metric("Predicted violations", sum(result["violations"].values()))
        with area_col:
            st.metric("Estimated built-up area", f"{result['proposal']['built_up_area_sqm']:,.0f} sq m")

        explanation, values = st.columns([1.35, 1])
        with explanation:
            st.markdown("### What informed this result")
            for reason in result["reasons"]:
                st.markdown(f"- {reason}")
        with values:
            st.markdown("### Calculated benchmarks")
            calculated = result["proposal"]
            st.dataframe({
                "Benchmark": ["Permissible FAR", "Permissible height", "Footprint", "Parking required"],
                "Value": [
                    calculated["far_permissible"],
                    f"{calculated['height_permissible_m']} m",
                    f"{calculated['footprint_sqm']:,.1f} sq m",
                    f"{calculated['parking_required_ecs']} ECS",
                ],
            }, hide_index=True, use_container_width=True)

    st.caption("AI-assisted screening only. This does not replace statutory approval or professional review.")
