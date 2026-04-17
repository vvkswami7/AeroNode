import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json

# Ensure environment variables are loaded securely
load_dotenv()

# Configure the Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    print("WARNING: GEMINI_API_KEY not found in environment or .env file.")

# Global in-memory state
venue_state = {}
latest_ai_command = None

class TelemetryPayload(BaseModel):
    node_id: str
    node_type: str
    acoustic_density: int
    rf_attenuation: float
    timestamp: str

async def venue_monitor_task():
    """Background task to run every 10 seconds and check for surges."""
    global latest_ai_command
    while True:
        await asyncio.sleep(10)
        
        surge_detected = False
        surging_nodes = []
        
        # Scan all connected nodes
        for node_id, data in venue_state.items():
            if data["acoustic_density"] > 90 and data["rf_attenuation"] > 0.85:
                surge_detected = True
                surging_nodes.append(node_id)
                
        if surge_detected:
            print(f"Surge detected critically at nodes: {surging_nodes}. Contacting AI Orchestrator...")
            try:
                # Establish the "Venue Traffic Controller" role
                model = genai.GenerativeModel(
                    'gemini-1.5-flash',
                    system_instruction="You are a Venue Traffic Controller for a massive sporting event. "
                                       "Analyze the given venue telemetry. Determine the best operational "
                                       "routing command to resolve crowd surges. You must output strictly valid JSON "
                                       "with exactly three keys: 'alert', 'action', and 'dispatch'.",
                    generation_config={"response_mime_type": "application/json"}
                )
                
                prompt = (f"Current Venue State: {json.dumps(venue_state)}. "
                          f"Nodes experiencing a severe surge: {surging_nodes}. "
                          f"Please provide your JSON routing command immediately.")
                
                # Fetch AI insights asynchronously so we don't block the FastAPI server
                response = await model.generate_content_async(prompt)
                
                try:
                    # Extract the JSON command from the model's text response
                    ai_response = json.loads(response.text)
                    latest_ai_command = ai_response
                    print(f"AI Orchestrator New Command: {json.dumps(ai_response, indent=2)}")
                except json.JSONDecodeError:
                    print("Failed to decode AI response:", response.text)
                    
            except Exception as e:
                print(f"Error calling Gemini API: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup phase: start the proactive monitoring task
    task = asyncio.create_task(venue_monitor_task())
    yield
    # Shutdown phase: cancel the task
    task.cancel()

app = FastAPI(lifespan=lifespan)

@app.post("/api/telemetry")
async def receive_telemetry(payload: TelemetryPayload):
    """Endpoint for edge physical hardware simulators to POST logic metrics."""
    # Storing directly in our global dictionary
    venue_state[payload.node_id] = payload.model_dump()
    return {"status": "success"}

@app.get("/api/state")
async def get_state():
    """Endpoint to return the full global backend state."""
    active_command_str = None
    if latest_ai_command:
        active_command_str = f"ALERT: {latest_ai_command.get('alert', 'N/A')} | ACTION: {latest_ai_command.get('action', 'N/A')} | DISPATCH: {latest_ai_command.get('dispatch', 'N/A')}"
        
    return {
        "nodes": venue_state,
        "active_command": active_command_str
    }

@app.get("/")
async def serve_dashboard():
    """Serve the telemetry dashboard."""
    return FileResponse("index.html")
    
if __name__ == "__main__":
    import uvicorn
    # Allow running directly via "python backend.py"
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
