# CivicLens site review setup

## Deploy on Vercel

Import this repository in Vercel, keep the project root at the repository root, and deploy with the default build settings. The Python WSGI entrypoint serves the frontend and the `/api/*` endpoints from the same origin. Runtime dependencies are listed in `requirements.txt`.

In the Vercel project settings, add `GEMINI_API_KEY` and/or `GEOAPIFY_API_KEY` under **Environment Variables** if you use those services. `CIVICLENS_COUNTRY_CODE` and `GEMINI_MODEL` are optional overrides. Redeploy after changing environment variables. Do not upload `.env` or commit API keys.

You can also deploy from the project directory with the Vercel CLI:

```powershell
npx vercel
npx vercel --prod
```

The first command links the local project and creates a preview deployment; the second publishes to production. The model artifacts in `artifacts/` are required at runtime and are included in the function bundle.

## Start the app

Copy `.env.example` to `.env` in the project folder, add any keys you want to use, then start the server:

```powershell
Copy-Item .env.example .env
python api.py
```

The API keys are read by the server and are never placed in the browser. When Gemini is configured, the selected satellite image and entered proposal values are sent to Google for the image review.

- `GEMINI_API_KEY` is optional. It enables Gemini to review satellite imagery alongside the project values entered in the form. Without it, CivicLens still returns the model-based planning and utility screening plus available map context.
- `GEOAPIFY_API_KEY` is optional. It enables Geoapify address search. OpenStreetMap Nominatim remains the fallback.
- `CIVICLENS_COUNTRY_CODE` limits address lookup; the default is `in`.

Detailed Indian addresses are searched as entered, then retried with leading flat, floor, or building components removed when the search service finds no match. A broader match can point to a locality rather than an exact parcel; verify the plot point and boundary on the map or with a survey.

The app shows whether Gemini is configured without revealing the key. No actual API key was included in this project, so enter your own in the local `.env` file if you want Gemini image review.

## Combined site assessment

Add one or more plot cards by searching each address or selecting a point on the map for the first plot. Choose the city jurisdiction separately for each plot, enter the building proposal dimensions once, and choose **Run Combined Site Assessment**. CivicLens runs a separate `POST /api/site-assessment` for every populated plot card and displays each plot's satellite observations, screening output, conditional BNS references, and next steps in its own result block. Building proposal values are shared across the plots.

The review is a preliminary planning screen. A model flag does not establish a statutory breach or criminal offence. Confirm requirements against current city and state rules, an accurate survey, and approval records. BNS references are conditional and are not a finding, charge, or legal opinion.

## Other endpoints

- `POST /api/check` and `POST /api/predict` run the project-data screening only.
- `POST /api/satellite-analysis` reviews an image without proposal context.
- `GET /api/gemini-status` reports whether a Gemini key is configured, without returning it.
- `GET /api/geocode?q=Bhopal` searches an address or place.
- `GET /api/buildings?lat=23.2599&lon=77.4126&radius=500` returns mapped OpenStreetMap building footprints.

Satellite/aerial imagery is not live and may be dated. Building footprints and utility risk estimates can be incomplete; use field surveys and utility-owner records before relying on them.
