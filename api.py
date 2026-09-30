"""Dependency-free HTTP API for the trained CivicLens models.

Run with::

    python api.py

The API loads the persisted models once at first request and exposes the
existing frontend contract at ``POST /api/check``. ``POST /api/predict`` is
provided as a clearer API-oriented alias for integrations.
"""

import base64
import json
import math
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlencode, urlparse
from urllib.request import Request, urlopen

from feasibility_checker import check_feasibility, load_models
from generate_dataset import CITIES, ZONES


PROJECT_DIR = Path(__file__).resolve().parent
FRONTEND_FILE = PROJECT_DIR / "frontend" / "index.html"
MODELS = None


def _load_dotenv(path):
    """Load simple KEY=value settings without adding a dotenv dependency."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(PROJECT_DIR / ".env")
GEOAPIFY_API_KEY = os.environ.get("GEOAPIFY_API_KEY", "").strip()
GEOAPIFY_URL = "https://api.geoapify.com/v1/geocode/search"
GEOCODING_COUNTRY_CODE = os.environ.get("CIVICLENS_COUNTRY_CODE", "in").strip().lower()
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_NOMINATIM_LOCK = Lock()
_NOMINATIM_LAST_REQUEST = 0.0
_NOMINATIM_RESPONSE_CACHE = {}
NOMINATIM_DETAILS_URL = "https://nominatim.openstreetmap.org/details"
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", "")
).strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash").strip()
GEMINI_GENERATE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent"
)
SATELLITE_EXPORT_URL = os.environ.get(
    "CIVICLENS_SATELLITE_EXPORT_URL",
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export",
)
SATELLITE_RADIUS_M = 250
MAX_SATELLITE_IMAGE_BYTES = 4_000_000
OVERPASS_URLS = tuple(
    item.strip()
    for item in os.environ.get(
        "CIVICLENS_OVERPASS_URLS",
        "https://overpass-api.de/api/interpreter,"
        "https://overpass.kumi.systems/api/interpreter,"
        "https://overpass.private.coffee/api/interpreter",
    ).split(",")
    if item.strip()
)
MAX_BUILDING_RADIUS_M = 1500

REQUIRED_FIELDS = {
    "city",
    "zone_type",
    "plot_size_sqm",
    "road_width_m",
    "far_proposed",
    "height_proposed_m",
    "floors",
    "ground_coverage_pct",
    "setback_front_m",
    "setback_rear_m",
    "setback_side_m",
    "parking_provided_ecs",
    "basement_proposed",
    "foundation_depth_m",
    "gis_record_confidence",
}

NUMERIC_FIELDS = {
    "plot_size_sqm": (0, None),
    "road_width_m": (0, None),
    "far_proposed": (0, None),
    "height_proposed_m": (0, None),
    "floors": (1, None),
    "ground_coverage_pct": (0, 100),
    "setback_front_m": (0, None),
    "setback_rear_m": (0, None),
    "setback_side_m": (0, None),
    "parking_provided_ecs": (0, None),
    "foundation_depth_m": (0, None),
    "gis_record_confidence": (0, 1),
}
POSITIVE_FIELDS = {
    "plot_size_sqm",
    "road_width_m",
    "far_proposed",
    "height_proposed_m",
    "foundation_depth_m",
}


def _get_models():
    global MODELS
    if MODELS is None:
        MODELS = load_models()
    return MODELS


class ExternalServiceError(RuntimeError):
    """An upstream location, imagery, or AI service could not be reached."""


def _fetch_json(url, data=None, content_type=None, extra_headers=None):
    headers = {"User-Agent": "CivicLens/1.0 building-inspection prototype"}
    if content_type:
        headers["Content-Type"] = content_type
    if extra_headers:
        headers.update(extra_headers)
    request = Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urlopen(request, timeout=25) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as error:
        raise ExternalServiceError(f"External service request failed: {error}") from error


def _fetch_image(url):
    """Download a bounded image response for the satellite preview/AI review."""
    request = Request(
        url,
        headers={"User-Agent": "CivicLens/1.0 satellite-inspection prototype"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            content_type = response.headers.get_content_type()
            image_bytes = response.read(MAX_SATELLITE_IMAGE_BYTES + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        raise ExternalServiceError(f"Satellite imagery request failed: {error}") from error
    if len(image_bytes) > MAX_SATELLITE_IMAGE_BYTES:
        raise ExternalServiceError("Satellite image exceeded the 4 MB safety limit.")
    if not content_type.startswith("image/"):
        raise ExternalServiceError("Satellite service returned a non-image response.")
    return image_bytes, content_type


def _geocode_nominatim(query, limit=5):
    """Free OpenStreetMap search with a cautious fallback for long addresses."""
    global _NOMINATIM_LAST_REQUEST
    query = " ".join(str(query).split())
    if not query:
        return []
    if len(query) > 500:
        raise ValueError("Address is too long. Keep it under 500 characters.")

    # Unit, floor, and building details at the start of a comma-separated
    # address can prevent a useful locality match. Try the full address first,
    # then one or two progressively broader trailing portions.
    components = [part.strip() for part in query.replace(";", ",").split(",") if part.strip()]
    query_variants = [query]
    if len(components) >= 4:
        for start in range(1, min(3, len(components) - 1)):
            variant = ", ".join(components[start:])
            if variant and variant not in query_variants:
                query_variants.append(variant)

    results = []
    request_limit = max(1, min(int(limit), 10))
    for search_text in query_variants:
        cache_key = (search_text.casefold(), request_limit, GEOCODING_COUNTRY_CODE)
        items = None
        with _NOMINATIM_LOCK:
            cached = _NOMINATIM_RESPONSE_CACHE.get(cache_key)
            if cached and cached[0] > time.monotonic():
                items = cached[1]
            else:
                wait_seconds = 1.05 - (time.monotonic() - _NOMINATIM_LAST_REQUEST)
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
                params = urlencode(
                    {
                        "q": search_text,
                        "format": "jsonv2",
                        "limit": request_limit,
                        "countrycodes": GEOCODING_COUNTRY_CODE,
                        "addressdetails": 1,
                    }
                )
                request = Request(
                    f"{NOMINATIM_URL}?{params}",
                    headers={
                        "User-Agent": "CivicLens/1.0 (address lookup)",
                        "Accept-Language": "en",
                    },
                )
                try:
                    with urlopen(request, timeout=10) as response:
                        items = json.loads(response.read().decode("utf-8"))
                except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as error:
                    _NOMINATIM_LAST_REQUEST = time.monotonic()
                    raise ExternalServiceError(f"Nominatim geocoding failed: {error}") from error
                _NOMINATIM_LAST_REQUEST = time.monotonic()
                _NOMINATIM_RESPONSE_CACHE[cache_key] = (time.monotonic() + 600, items)

        for item in items:
            lat = item.get("lat")
            lon = item.get("lon")
            if lat is None or lon is None:
                continue
            addr = item.get("address") or {}
            city = (
                addr.get("city")
                or addr.get("town")
                or addr.get("village")
                or addr.get("county")
            )
            label = item.get("display_name") or search_text
            results.append(
                {
                    "label": label,
                    "lat": float(lat),
                    "lon": float(lon),
                    "bbox": item.get("boundingbox"),
                    "result_type": item.get("type") or item.get("class"),
                    "city": city,
                    "state": addr.get("state"),
                    "postcode": addr.get("postcode"),
                    "match_mode": "broader" if search_text != query else "full",
                    "matched_query": search_text,
                }
            )
        if results:
            break
    return results


def reverse_geocode(latitude, longitude):
    """Reverse geocode coordinates to an Indian address using Nominatim (free)."""
    params = urlencode(
        {
            "lat": latitude,
            "lon": longitude,
            "format": "jsonv2",
            "addressdetails": 1,
            "zoom": 18,
        }
    )
    request = Request(
        f"{NOMINATIM_REVERSE_URL}?{params}",
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CivicLensUrbanPlanning/1.0",
            "Accept-Language": "en",
        },
    )
    try:
        with urlopen(request, timeout=10) as response:
            item = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as error:
        raise ExternalServiceError(f"Reverse geocoding failed: {error}") from error

    addr = item.get("address") or {}
    city = (
        addr.get("city")
        or addr.get("town")
        or addr.get("village")
        or addr.get("county")
        or addr.get("municipality")
        or addr.get("state_district")
        or ""
    )
    label = item.get("display_name") or f"Location at {latitude:.5f}, {longitude:.5f}"
    return {
        "label": label,
        "lat": float(latitude),
        "lon": float(longitude),
        "city": city,
        "state": addr.get("state") or "",
        "postcode": addr.get("postcode") or "",
        "suburb": addr.get("suburb") or addr.get("neighbourhood") or "",
        "road": addr.get("road") or "",
        "country": addr.get("country") or "India",
    }


def _geocode_geoapify(query, limit=5):
    """Geocoding via Geoapify (requires GEOAPIFY_API_KEY in .env)."""
    params = urlencode(
        {
            "text": query,
            "format": "json",
            "limit": max(1, min(int(limit), 10)),
            "filter": f"countrycode:{GEOCODING_COUNTRY_CODE}",
            "apiKey": GEOAPIFY_API_KEY,
        }
    )
    payload = _fetch_json(f"{GEOAPIFY_URL}?{params}")
    results = []
    for item in payload.get("results", []):
        if "lat" not in item or "lon" not in item:
            continue
        results.append(
            {
                "label": item.get("formatted") or item.get("address_line1") or query,
                "lat": float(item["lat"]),
                "lon": float(item["lon"]),
                "bbox": item.get("bbox"),
                "result_type": item.get("result_type"),
                "city": item.get("city") or item.get("town") or item.get("village"),
                "state": item.get("state"),
                "postcode": item.get("postcode"),
            }
        )
    return results


def geocode_places(query, limit=5):
    """Search places. Uses Geoapify if configured, otherwise Nominatim (free)."""
    query = str(query or "").strip()
    if len(query) > 500:
        raise ValueError("Address is too long. Keep it under 500 characters.")
    if GEOAPIFY_API_KEY:
        try:
            results = _geocode_geoapify(query, limit)
            if results:
                return results
        except ExternalServiceError:
            # Keep address lookup usable when the optional provider is down or
            # cannot resolve a detailed address; Nominatim remains the fallback.
            pass
    return _geocode_nominatim(query, limit)


def address_check(query, limit=5):
    """Resolve an Indian address for the live address-checking interface."""
    results = geocode_places(query, limit)
    return {
        "query": query,
        "country_code": GEOCODING_COUNTRY_CODE,
        "matched": bool(results),
        "results": results,
    }


def geocoder_status():
    """Return geocoding provider status. Nominatim works with no configuration."""
    if GEOAPIFY_API_KEY:
        try:
            _geocode_geoapify("India", limit=1)
            return {
                "configured": True,
                "valid": True,
                "provider": "Geoapify",
                "country_code": GEOCODING_COUNTRY_CODE,
            }
        except ExternalServiceError as error:
            return {
                "configured": True,
                "valid": False,
                "provider": "Geoapify",
                "country_code": GEOCODING_COUNTRY_CODE,
                "error": str(error),
            }
    # Nominatim is always available — no key needed.
    return {
        "configured": True,
        "valid": True,
        "provider": "Nominatim (OpenStreetMap)",
        "country_code": GEOCODING_COUNTRY_CODE,
        "note": "Free tier — no API key required.",
    }


def gemini_status():
    """Return key configuration only; never expose the secret itself."""
    return {
        "configured": bool(GEMINI_API_KEY),
        "provider": "Google Gemini",
        "model": GEMINI_MODEL,
    }


def satellite_snapshot(latitude, longitude):
    """Fetch a small, current satellite/aerial view around a site coordinate."""
    latitude_delta = SATELLITE_RADIUS_M / 111_320
    longitude_delta = SATELLITE_RADIUS_M / (
        111_320 * max(math.cos(math.radians(latitude)), 0.1)
    )
    bbox = (
        longitude - longitude_delta,
        latitude - latitude_delta,
        longitude + longitude_delta,
        latitude + latitude_delta,
    )
    params = urlencode(
        {
            "bbox": ",".join(f"{value:.7f}" for value in bbox),
            "bboxSR": 4326,
            "imageSR": 4326,
            "size": "1024,1024",
            "format": "jpg",
            "f": "image",
        }
    )
    image_bytes, content_type = _fetch_image(f"{SATELLITE_EXPORT_URL}?{params}")
    return image_bytes, content_type, {
        "source": "Esri World Imagery",
        "bbox": bbox,
        "radius_m": SATELLITE_RADIUS_M,
    }


def _gemini_response_text(response):
    candidates = response.get("candidates") or []
    if not candidates:
        detail = response.get("promptFeedback") or response.get("error") or {}
        raise ExternalServiceError(f"Gemini returned no analysis: {detail}")
    parts = candidates[0].get("content", {}).get("parts", [])
    analysis = "\n".join(part.get("text", "") for part in parts if part.get("text")).strip()
    if not analysis:
        raise ExternalServiceError("Gemini returned no text analysis for the satellite image.")
    return analysis


def _generate_spatial_heuristic_analysis(latitude, longitude, address, imagery, proposal_context=None):
    """Describe mapped context and supplied proposal values without claiming legal compliance."""
    building_count = 0
    building_types = {}
    level_values = []
    try:
        mapped = buildings_nearby(latitude, longitude, 300)
        features = mapped.get("features", [])
        building_count = len(features)
        for feature in features:
            tags = feature.get("properties") or {}
            kind = tags.get("building", "unclassified")
            building_types[kind] = building_types.get(kind, 0) + 1
            try:
                if tags.get("building:levels"):
                    level_values.append(float(tags["building:levels"]))
            except (ValueError, TypeError):
                pass
    except Exception:
        pass

    mapped_types = ", ".join(
        f"{kind} ({count})"
        for kind, count in sorted(building_types.items(), key=lambda item: -item[1])[:3]
    ) or "No building type tags were available."
    average_levels = round(sum(level_values) / len(level_values), 1) if level_values else None
    levels_note = (
        f"Mapped building level tags average {average_levels} levels."
        if average_levels is not None
        else "No mapped building level tags were available."
    )
    context = proposal_context or {}
    proposal_lines = []
    for label, key, suffix in [
        ("City / zone", "city", ""),
        ("Plot area", "plot_size_sqm", " m²"),
        ("Proposed FAR", "far_proposed", ""),
        ("Proposed height", "height_proposed_m", " m"),
        ("Floors", "floors", ""),
        ("Road width", "road_width_m", " m"),
        ("Ground coverage", "ground_coverage_pct", "%"),
    ]:
        if key in context:
            value = context[key]
            if key == "city":
                value = f"{value} / {context.get('zone_type', 'zone not supplied')}"
            proposal_lines.append(f"• {label}: {value}{suffix}")
    if not proposal_lines:
        proposal_lines.append("• No project dimensions were provided to the imagery review.")

    return (
        "**1. Satellite and mapped context**\n"
        f"• Selected location: {latitude:.5f}, {longitude:.5f} ({address or 'selected site'}).\n"
        f"• Satellite preview: Esri World Imagery, approximately {imagery.get('radius_m', SATELLITE_RADIUS_M)}m around the selected point; capture date may vary.\n"
        f"• OpenStreetMap lists {building_count} building footprint(s) within about 300m. Coverage may be incomplete or out of date.\n"
        f"• Mapped building types: {mapped_types} {levels_note}\n\n"
        "**2. User-entered project context**\n"
        + "\n".join(proposal_lines)
        + "\nThese values come from the proposal form. This fallback does not interpret the image or determine compliance; review the model flags below separately.\n\n"
        "**3. Site checks**\n"
        "• Verify plot boundaries and road measurements with a licensed survey.\n"
        "• Confirm the current zoning, approved plan, and development rules with the local planning authority.\n"
        "• Obtain utility-owner records and a field utility survey before excavation."
    )
def analyze_satellite(latitude, longitude, address="", proposal_context=None):
    """Review imagery with optional proposal context, without making legal findings."""
    image_bytes, content_type, imagery = satellite_snapshot(latitude, longitude)

    if GEMINI_API_KEY:
        try:
            location_hint = f"The selected address is: {address}. " if address else ""
            project_hint = (
                "User-entered proposal data (unverified): "
                + json.dumps(proposal_context, ensure_ascii=False, sort_keys=True)
                + ". Treat these as user-supplied context, not facts visible in the image. "
                if proposal_context
                else ""
            )
            prompt = (
                "Review this satellite/aerial image for an early-stage planning screen. "
                f"{location_hint}"
                f"{project_hint}"
                "Write three short sections: Visible observations, Proposal context to verify, "
                "and Image limitations. Describe only clearly visible site conditions. "
                "Compare the user-entered proposal context only to identify questions for a human reviewer; "
                "do not infer a violation from the image, decide legal compliance, name an offender, or "
                "recommend criminal enforcement. "
                "You may mention visible built form, access roads, open land, vegetation, surface water, "
                "and immediately adjacent construction. Do not identify people, infer ownership or occupancy, "
                "claim statutory compliance, estimate measurements as facts, or make claims about underground utilities. "
                "Treat the image as potentially dated and state uncertainty where appropriate."
            )
            request_body = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": content_type,
                                    "data": base64.b64encode(image_bytes).decode("ascii"),
                                }
                            },
                        ]
                    }
                ],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 600},
            }
            endpoint = GEMINI_GENERATE_URL.format(model=quote(GEMINI_MODEL, safe="-_."))
            response = _fetch_json(
                endpoint,
                data=json.dumps(request_body).encode("utf-8"),
                content_type="application/json; charset=utf-8",
                extra_headers={"x-goog-api-key": GEMINI_API_KEY},
            )
            return {
                "analysis": _gemini_response_text(response),
                "model": f"Google Gemini ({GEMINI_MODEL})",
                "imagery": imagery,
                "proposal_context": proposal_context or {},
                "disclaimer": (
                    "AI-generated image observations are not a legal, survey, or statutory compliance determination."
                ),
            }
        except Exception:
            # Fall back to heuristic spatial engine if Gemini fails or quota exceeded
            pass

    heuristic_analysis = _generate_spatial_heuristic_analysis(
        latitude, longitude, address, imagery, proposal_context
    )
    return {
        "analysis": heuristic_analysis,
        "model": "CivicLens Spatial Context (Esri imagery + OpenStreetMap)",
        "imagery": imagery,
        "proposal_context": proposal_context or {},
        "disclaimer": (
            "The satellite image is visual context only. Automated notes use OpenStreetMap data and do not interpret the image or determine legal compliance."
        ),
    }


def buildings_nearby(latitude, longitude, radius):
    """Return OpenStreetMap building ways around a coordinate as GeoJSON."""
    latitude_delta = radius / 111_320
    longitude_delta = radius / (
        111_320 * max(math.cos(math.radians(latitude)), 0.1)
    )
    south = latitude - latitude_delta
    west = longitude - longitude_delta
    north = latitude + latitude_delta
    east = longitude + longitude_delta
    query = (
        "[out:json][timeout:15];"
        f'way["building"]({south},{west},{north},{east});'
        "out body geom qt;"
    )
    request_data = urlencode({"data": query}).encode("utf-8")
    payload = None
    last_error = None
    for endpoint in OVERPASS_URLS:
        try:
            payload = _fetch_json(
                endpoint,
                data=request_data,
                content_type="application/x-www-form-urlencoded; charset=UTF-8",
            )
            break
        except ExternalServiceError as error:
            last_error = error
    if payload is None:
        raise last_error or ExternalServiceError("No Overpass endpoint is configured.")

    features = []
    for element in payload.get("elements", []):
        geometry = element.get("geometry") or []
        coordinates = [
            [float(point["lon"]), float(point["lat"])]
            for point in geometry
            if "lat" in point and "lon" in point
        ]
        if len(coordinates) < 3:
            continue
        if coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])
        properties = {
            "osm_id": element.get("id"),
            "osm_type": element.get("type"),
            **(element.get("tags") or {}),
        }
        features.append(
            {
                "type": "Feature",
                "id": f"{element.get('type', 'way')}/{element.get('id')}",
                "properties": properties,
                "geometry": {"type": "Polygon", "coordinates": [coordinates]},
            }
        )
    return {
        "type": "FeatureCollection",
        "features": features,
        "source": "OpenStreetMap",
        "center": {"lat": latitude, "lon": longitude},
        "radius_m": radius,
    }


def _parse_coordinate(value, field):
    try:
        coordinate = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a number.") from error
    if not math.isfinite(coordinate):
        raise ValueError(f"{field} must be finite.")
    return coordinate


def validate_payload(payload):
    """Validate and normalize a JSON request before it reaches the models."""
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    missing = sorted(REQUIRED_FIELDS.difference(payload))
    if missing:
        raise ValueError("Missing required fields: " + ", ".join(missing))

    city = payload["city"]
    if not isinstance(city, str) or not city.strip() or len(city.strip()) > 120:
        raise ValueError("city must be a non-empty name no longer than 120 characters.")
    if payload["zone_type"] not in ZONES:
        raise ValueError("zone_type must be one of: " + ", ".join(ZONES))
    if not isinstance(payload["basement_proposed"], bool):
        raise ValueError("basement_proposed must be a boolean.")

    normalized = dict(payload)
    normalized["city"] = city.strip()
    for field, (minimum, maximum) in NUMERIC_FIELDS.items():
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} must be a number.")
        if not math.isfinite(float(value)):
            raise ValueError(f"{field} must be finite.")
        lower_bound_failed = (
            value <= minimum if field in POSITIVE_FIELDS else value < minimum
        )
        if lower_bound_failed or (maximum is not None and value > maximum):
            raise ValueError(
                f"{field} must be between {minimum} and {maximum}."
                if maximum is not None
                else (
                    f"{field} must be greater than {minimum}."
                    if field in POSITIVE_FIELDS
                    else f"{field} must be at least {minimum}."
                )
            )
        normalized[field] = int(value) if field == "floors" else float(value)

    if normalized["floors"] != payload["floors"]:
        raise ValueError("floors must be a whole number.")
    return normalized


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "CivicLensAPI/1.0"

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "private, max-age=300")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_frontend(self):
        body = FRONTEND_FILE.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0 or content_length > 1_000_000:
            raise ValueError("Request body must be between 1 byte and 1 MB.")
        payload = json.loads(self.rfile.read(content_length))
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def do_OPTIONS(self):
        self._send_json({}, status=204)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path in {"/", "/index.html", "/frontend/index.html"}:
            self._send_frontend()
            return
        if path == "/api/health":
            try:
                _get_models()
                self._send_json({"status": "ok", "model": "online"})
            except FileNotFoundError as error:
                self._send_json(
                    {"status": "error", "model": "offline", "error": str(error)},
                    503,
                )
            return
        if path == "/api/model-info":
            self._send_json(
                {
                    "name": "CivicLens feasibility model",
                    "version": "1.0",
                    "endpoints": [
                        "POST /api/check",
                        "POST /api/predict",
                        "POST /api/satellite-analysis",
                        "POST /api/site-assessment",
                    ],
                    "required_fields": sorted(REQUIRED_FIELDS),
                    "map_endpoints": [
                        "GET /api/geocode?q=<city or address>",
                        "GET /api/address-check?q=<Indian address>",
                        "GET /api/buildings?lat=<lat>&lon=<lon>&radius=<metres>",
                        "GET /api/satellite-preview?lat=<lat>&lon=<lon>",
                    ],
                    "geocoding_configured": bool(GEOAPIFY_API_KEY),
                    "geocoding_country_code": GEOCODING_COUNTRY_CODE,
                    "gemini_configured": bool(GEMINI_API_KEY),
                    "gemini_model": GEMINI_MODEL,
                }
            )
            return
        if path == "/api/key-status":
            try:
                self._send_json(geocoder_status())
            except ExternalServiceError as error:
                self._send_json(
                    {
                        "configured": True,
                        "valid": False,
                        "provider": "Geoapify",
                        "country_code": GEOCODING_COUNTRY_CODE,
                        "error": str(error),
                    },
                    503,
                )
            return
        if path == "/api/gemini-status":
            self._send_json(gemini_status())
            return
        if path in {"/api/geocode", "/api/address-check"}:
            search_text = (query.get("q") or [""])[0].strip()
            if len(search_text) < 2:
                self._send_json({"error": "q must contain at least 2 characters."}, 400)
                return
            try:
                limit = int((query.get("limit") or [5])[0])
                if path == "/api/address-check":
                    self._send_json(address_check(search_text, limit))
                else:
                    self._send_json({"results": geocode_places(search_text, limit)})
            except (ValueError, TypeError) as error:
                self._send_json({"error": str(error)}, 400)
            except ExternalServiceError as error:
                self._send_json({"error": str(error)}, 503)
            return
        if path == "/api/satellite-preview":
            try:
                latitude = _parse_coordinate((query.get("lat") or [None])[0], "lat")
                longitude = _parse_coordinate((query.get("lon") or [None])[0], "lon")
                if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
                    raise ValueError("lat/lon are outside valid geographic ranges.")
                image_bytes, content_type, _ = satellite_snapshot(latitude, longitude)
                self._send_bytes(image_bytes, content_type)
            except (ValueError, TypeError) as error:
                self._send_json({"error": str(error)}, 400)
            except ExternalServiceError as error:
                self._send_json({"error": str(error)}, 503)
            return
        if path == "/api/buildings":
            try:
                latitude = _parse_coordinate((query.get("lat") or [None])[0], "lat")
                longitude = _parse_coordinate((query.get("lon") or [None])[0], "lon")
                radius = int((query.get("radius") or [500])[0])
                if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
                    raise ValueError("lat/lon are outside valid geographic ranges.")
                if not 50 <= radius <= MAX_BUILDING_RADIUS_M:
                    raise ValueError(
                        f"radius must be between 50 and {MAX_BUILDING_RADIUS_M} metres."
                    )
                self._send_json(buildings_nearby(latitude, longitude, radius))
            except (ValueError, TypeError) as error:
                self._send_json({"error": str(error)}, 400)
            except ExternalServiceError as error:
                self._send_json({"error": str(error)}, 503)
            return
        if path == "/api/reverse-geocode":
            try:
                latitude = _parse_coordinate((query.get("lat") or [None])[0], "lat")
                longitude = _parse_coordinate((query.get("lon") or [None])[0], "lon")
                if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
                    raise ValueError("lat/lon are outside valid geographic ranges.")
                self._send_json(reverse_geocode(latitude, longitude))
            except (ValueError, TypeError) as error:
                self._send_json({"error": str(error)}, 400)
            except ExternalServiceError as error:
                self._send_json({"error": str(error)}, 503)
            return
        self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/site-assessment":
            try:
                payload = self._read_json_body()
                inputs = validate_payload(payload)
                latitude = _parse_coordinate(payload.get("lat"), "lat")
                longitude = _parse_coordinate(payload.get("lon"), "lon")
                if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
                    raise ValueError("lat/lon are outside valid geographic ranges.")
                address = str(payload.get("address") or "").strip()[:500]
                assessment = check_feasibility(inputs, _get_models())
                try:
                    satellite = analyze_satellite(
                        latitude,
                        longitude,
                        address,
                        proposal_context=assessment["proposal"],
                    )
                except ExternalServiceError as error:
                    satellite = {"error": str(error), "proposal_context": assessment["proposal"]}
                self._send_json(
                    {
                        "site": {"lat": latitude, "lon": longitude, "address": address},
                        "assessment": assessment,
                        "satellite": satellite,
                    }
                )
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                self._send_json({"error": f"Invalid site assessment: {error}"}, 400)
            except FileNotFoundError as error:
                self._send_json({"error": str(error)}, 503)
            except Exception as error:
                self._send_json({"error": "Site assessment failed: " + str(error)}, 500)
            return

        if path == "/api/satellite-analysis":
            try:
                payload = self._read_json_body()
                latitude = _parse_coordinate(payload.get("lat"), "lat")
                longitude = _parse_coordinate(payload.get("lon"), "lon")
                if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
                    raise ValueError("lat/lon are outside valid geographic ranges.")
                address = str(payload.get("address") or "").strip()[:500]
                self._send_json(analyze_satellite(latitude, longitude, address))
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self._send_json({"error": f"Invalid satellite analysis request: {error}"}, 400)
            except ExternalServiceError as error:
                self._send_json({"error": str(error)}, 503)
            return

        if path not in {"/api/check", "/api/predict"}:
            self._send_json({"error": "Not found"}, 404)
            return

        try:
            payload = self._read_json_body()
            inputs = validate_payload(payload)
            result = check_feasibility(inputs, _get_models())
            self._send_json(result)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            self._send_json({"error": f"Invalid proposal: {error}"}, 400)
        except FileNotFoundError as error:
            self._send_json({"error": str(error)}, 503)
        except Exception as error:
            self._send_json({"error": "Model inference failed: " + str(error)}, 500)

    def log_message(self, format_string, *args):
        print(f"{self.address_string()} - {format_string % args}")


def main():
    host = os.environ.get("CIVICLENS_HOST", "127.0.0.1")
    port = int(os.environ.get("CIVICLENS_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), RequestHandler)
    print(f"CivicLens API is running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping CivicLens API")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
