import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from upload import router as upload_router, search_router

app = FastAPI(
    title="AI Video Annotator",
    description="SaaS product for AI-powered video annotation and object detection",
    version="1.0.0"
)

# CORS configuration for frontend integration (React/HTML)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict this to your actual frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the outputs directory statically so users can view/download files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# Include the API routers
app.include_router(upload_router, prefix="/api", tags=["Upload"])
app.include_router(search_router, tags=["Search"])

@app.get("/", response_class=HTMLResponse)
async def homepage():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>AI Video Annotator</title>
    <style>
        :root {
            --bg: #f3f7fb;
            --panel: #ffffff;
            --panel-soft: #f8fbff;
            --text: #10233e;
            --muted: #5b6b82;
            --line: #d7e3f1;
            --accent: #0f62fe;
            --accent-dark: #0848be;
            --success: #0f9d58;
            --shadow: 0 18px 44px rgba(16, 35, 62, 0.10);
        }
        body {
            font-family: Arial, sans-serif;
            background:
                radial-gradient(circle at top left, rgba(15, 98, 254, 0.10), transparent 28%),
                linear-gradient(180deg, #f7fbff 0%, var(--bg) 100%);
            color: var(--text);
            margin: 0;
            padding: 32px 20px 48px;
        }
        .container {
            max-width: 1080px;
            margin: 0 auto;
        }
        .hero {
            background: linear-gradient(135deg, #0f62fe 0%, #1b82ff 55%, #76b5ff 100%);
            color: white;
            border-radius: 24px;
            padding: 32px;
            box-shadow: var(--shadow);
            margin-bottom: 24px;
        }
        .hero h1 {
            margin-top: 0;
            margin-bottom: 10px;
            font-size: 38px;
        }
        .hero p {
            margin: 0;
            max-width: 720px;
            line-height: 1.6;
            color: rgba(255, 255, 255, 0.92);
        }
        .grid {
            display: grid;
            grid-template-columns: 1.1fr 0.9fr;
            gap: 24px;
        }
        .panel {
            background: var(--panel);
            border-radius: 20px;
            box-shadow: var(--shadow);
            padding: 24px;
            border: 1px solid rgba(215, 227, 241, 0.65);
        }
        .panel h2,
        .panel h3 {
            margin-top: 0;
        }
        .subtle {
            color: var(--muted);
            line-height: 1.6;
        }
        form {
            display: grid;
            gap: 14px;
            margin: 22px 0 12px;
        }
        input[type="file"] {
            width: 100%;
            padding: 12px;
            border: 1px solid var(--line);
            border-radius: 12px;
            background: #fff;
            box-sizing: border-box;
        }
        .search-row {
            display: flex;
            gap: 12px;
            margin-top: 14px;
        }
        .search-row input {
            flex: 1;
            padding: 12px 14px;
            border: 1px solid var(--line);
            border-radius: 12px;
            box-sizing: border-box;
        }
        button {
            background: var(--accent);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 12px 18px;
            cursor: pointer;
            font-weight: 700;
        }
        button:disabled {
            background: #9fc1ff;
            cursor: not-allowed;
        }
        .status {
            margin: 8px 0 0;
            font-weight: 600;
        }
        .status.success {
            color: var(--success);
        }
        .status.error {
            color: #c62828;
        }
        .hidden {
            display: none;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin: 18px 0;
        }
        .stat {
            background: var(--panel-soft);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 16px;
        }
        .stat-label {
            font-size: 12px;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .stat-value {
            font-size: 24px;
            font-weight: 700;
            margin-top: 8px;
        }
        .meta-list {
            display: grid;
            gap: 10px;
            margin: 18px 0;
        }
        .meta-item {
            display: flex;
            justify-content: space-between;
            gap: 18px;
            padding: 12px 14px;
            background: var(--panel-soft);
            border: 1px solid var(--line);
            border-radius: 12px;
        }
        .meta-item span:last-child {
            text-align: right;
            word-break: break-all;
        }
        video {
            width: 100%;
            max-height: 540px;
            border-radius: 18px;
            background: black;
            margin-top: 8px;
        }
        a {
            color: var(--accent-dark);
        }
        pre {
            white-space: pre-wrap;
            background: #eef4fb;
            padding: 14px;
            border-radius: 12px;
            border: 1px solid var(--line);
            max-height: 240px;
            overflow: auto;
        }
        .link-row {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin: 16px 0 4px;
        }
        .link-row a {
            display: inline-block;
            padding: 10px 14px;
            background: #eef4ff;
            border-radius: 12px;
            text-decoration: none;
            border: 1px solid #d4e2ff;
        }
        .results-list {
            display: grid;
            gap: 12px;
            margin-top: 16px;
        }
        .result-chip {
            padding: 14px;
            border: 1px solid var(--line);
            border-radius: 12px;
            background: var(--panel-soft);
        }
        .small {
            color: var(--muted);
            font-size: 14px;
        }
        @media (max-width: 900px) {
            .grid {
                grid-template-columns: 1fr;
            }
            .stats {
                grid-template-columns: 1fr;
            }
            .search-row {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <section class="hero">
            <h1>AI Video Annotator</h1>
            <p>Upload a video, track objects across frames, review the processed MP4 in the browser, search detections by label, and download machine-readable JSON from one place.</p>
        </section>

        <div class="grid">
            <section class="panel">
                <h2>Upload Video</h2>
                <p class="subtle">Supported formats: MP4, MOV, AVI, MKV, WEBM. The app will process the video, generate tracked detections, and return browser-ready output links.</p>

                <form id="uploadForm">
                    <input id="videoFile" name="file" type="file" accept="video/*" required />
                    <button id="submitBtn" type="submit">Upload and Process</button>
                </form>

                <div id="status" class="status">Choose a video to begin.</div>

                <div id="statsSection" class="stats hidden">
                    <div class="stat">
                        <div class="stat-label">Frames</div>
                        <div id="framesValue" class="stat-value">0</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">FPS</div>
                        <div id="fpsValue" class="stat-value">0</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Tracked Labels</div>
                        <div id="labelsValue" class="stat-value">0</div>
                    </div>
                </div>

                <div id="resultCard" class="hidden">
                    <div class="link-row">
                        <a id="videoLink" target="_blank" rel="noopener noreferrer">Open video in browser</a>
                        <a id="jsonLink" target="_blank" rel="noopener noreferrer">Download JSON output</a>
                        <a id="transcriptLink" target="_blank" rel="noopener noreferrer">Open transcript</a>
                    </div>

                    <div class="meta-list">
                        <div class="meta-item"><span>Source video</span><span id="sourceVideoValue">-</span></div>
                        <div class="meta-item"><span>Annotated video</span><span id="annotatedVideoValue">-</span></div>
                    </div>

                    <h3>Annotated Preview</h3>
                    <video id="resultVideo" controls preload="metadata"></video>
                </div>
            </section>

            <section class="panel">
                <h2>Analysis Workspace</h2>
                <p class="subtle">Review the latest transcript and search detections by label from the most recent processed result.</p>

                <h3>Transcript</h3>
                <pre id="transcriptText">No transcript yet.</pre>

                <h3>Search Detections</h3>
                <div class="search-row">
                    <input id="searchInput" type="text" placeholder="Enter label like person, car, truck" />
                    <button id="searchBtn" type="button">Search</button>
                </div>
                <div id="searchStatus" class="status">Search will use the latest processed result.</div>
                <div id="searchResults" class="results-list"></div>
            </section>
        </div>
    </div>

    <script>
        const form = document.getElementById("uploadForm");
        const fileInput = document.getElementById("videoFile");
        const submitBtn = document.getElementById("submitBtn");
        const statusEl = document.getElementById("status");
        const resultCard = document.getElementById("resultCard");
        const resultVideo = document.getElementById("resultVideo");
        const videoLink = document.getElementById("videoLink");
        const jsonLink = document.getElementById("jsonLink");
        const transcriptLink = document.getElementById("transcriptLink");
        const transcriptText = document.getElementById("transcriptText");
        const sourceVideoValue = document.getElementById("sourceVideoValue");
        const annotatedVideoValue = document.getElementById("annotatedVideoValue");
        const statsSection = document.getElementById("statsSection");
        const framesValue = document.getElementById("framesValue");
        const fpsValue = document.getElementById("fpsValue");
        const labelsValue = document.getElementById("labelsValue");
        const searchInput = document.getElementById("searchInput");
        const searchBtn = document.getElementById("searchBtn");
        const searchStatus = document.getElementById("searchStatus");
        const searchResults = document.getElementById("searchResults");

        function formatStatus(message, type = "") {
            statusEl.textContent = message;
            statusEl.className = "status" + (type ? " " + type : "");
        }

        function updateDashboard(data) {
            const labelSet = new Set();
            (data.annotations || []).forEach((frame) => {
                (frame.detections || []).forEach((detection) => labelSet.add(detection.label));
            });

            framesValue.textContent = data.frames_processed ?? 0;
            fpsValue.textContent = data.fps ? Number(data.fps).toFixed(2) : "0";
            labelsValue.textContent = labelSet.size;
            sourceVideoValue.textContent = data.source_video || data.input_file || "-";
            annotatedVideoValue.textContent = data.annotated_video || "-";
            transcriptText.textContent = data.transcript || "No transcript detected.";

            if (data.video_url) {
                resultVideo.pause();
                resultVideo.removeAttribute("src");
                resultVideo.load();
                resultVideo.src = data.video_url;
                resultVideo.load();
                videoLink.href = data.video_url;
                videoLink.textContent = "Open annotated video";
            }
            if (data.json_url) {
                jsonLink.href = data.json_url;
                jsonLink.textContent = "Download JSON output";
            }
            if (data.transcript_url) {
                transcriptLink.href = data.transcript_url;
                transcriptLink.textContent = "Open transcript";
            }

            statsSection.classList.remove("hidden");
            resultCard.classList.remove("hidden");
        }

        function renderSearchResults(data) {
            searchResults.innerHTML = "";

            if (!data.results || !data.results.length) {
                searchResults.innerHTML = '<div class="result-chip">No matches found for this label.</div>';
                return;
            }

            data.results.forEach((item) => {
                const div = document.createElement("div");
                div.className = "result-chip";
                const matches = item.matches.map((match) =>
                    `${match.label} | ID: ${match.track_id ?? "N/A"} | confidence: ${match.confidence}`
                ).join("<br>");

                div.innerHTML = `
                    <strong>Timestamp:</strong> ${item.timestamp}s<br>
                    <span class="small">Frame ${item.frame}</span><br><br>
                    ${matches}
                `;
                searchResults.appendChild(div);
            });
        }

        async function loadLatestResult() {
            try {
                const response = await fetch("/api/latest-result");
                if (!response.ok) {
                    return;
                }
                const data = await response.json();
                updateDashboard(data);
                formatStatus("Latest processed result loaded.", "success");
            } catch {
                // Keep the page quiet if there is no prior result yet.
            }
        }

        form.addEventListener("submit", async (event) => {
            event.preventDefault();

            if (!fileInput.files.length) {
                formatStatus("Please choose a video file.", "error");
                return;
            }

            const formData = new FormData();
            formData.append("file", fileInput.files[0]);

            submitBtn.disabled = true;
            formatStatus("Uploading and processing video. This may take a little while...");
            resultCard.classList.add("hidden");

            try {
                const response = await fetch("/api/upload", {
                    method: "POST",
                    body: formData
                });

                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.detail || "Upload failed.");
                }

                const latestResponse = await fetch("/api/latest-result");
                const latestData = latestResponse.ok ? await latestResponse.json() : data;
                updateDashboard(latestData);
                formatStatus("Processing complete.", "success");
            } catch (error) {
                formatStatus(error.message || "Something went wrong.", "error");
            } finally {
                submitBtn.disabled = false;
            }
        });

        searchBtn.addEventListener("click", async () => {
            const label = searchInput.value.trim();
            if (!label) {
                searchStatus.textContent = "Enter a label before searching.";
                return;
            }

            searchStatus.textContent = "Searching latest result...";
            searchResults.innerHTML = "";

            try {
                const response = await fetch(`/search?label=${encodeURIComponent(label)}`);
                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.detail || "Search failed.");
                }
                searchStatus.textContent = `Found ${data.total_timestamps} matching timestamp(s) for "${data.label}".`;
                renderSearchResults(data);
            } catch (error) {
                searchStatus.textContent = error.message || "Search failed.";
            }
        });

        loadLatestResult();
    </script>
</body>
</html>
"""
