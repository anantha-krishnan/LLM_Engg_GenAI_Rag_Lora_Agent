# server.py (Corrected)
import asyncio
import websockets
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BridgeServer")

EXECUTOR_WEBSOCKET = None
JOB_LOCK = asyncio.Lock()

async def register_executor(websocket):
    """Handles the connection from the user's remote execution client."""
    global EXECUTOR_WEBSOCKET
    if EXECUTOR_WEBSOCKET:
        logger.warning("An existing executor client was disconnected by a new one.")
    
    EXECUTOR_WEBSOCKET = websocket
    logger.info(f"✅ Remote Executor Client connected from {websocket.remote_address} and is now active.")
    try:
        # This keeps the connection alive, waiting for it to close.
        await websocket.wait_closed()
    finally:
        logger.info(f"❌ Remote Executor Client from {websocket.remote_address} disconnected.")
        if EXECUTOR_WEBSOCKET is websocket:
            EXECUTOR_WEBSOCKET = None

async def run_job(websocket):
    """Handles a temporary connection from the agent's tool."""
    global EXECUTOR_WEBSOCKET

    async with JOB_LOCK:
        if not EXECUTOR_WEBSOCKET :
            logger.error("Agent tool tried to run a job, but no executor client is connected.")
            await websocket.send(json.dumps({
                "status": "error",
                "message": "Execution failed: No remote executor client is available to run the job."
            }))
            return
        
        try:
            # <<< FIX 2: Moved the print statement after the recv() call >>>
            job_payload = await websocket.recv()
            logger.info(f"Received job from agent, forwarding to executor: {job_payload[:100]}...") # Log snippet
            
            await EXECUTOR_WEBSOCKET.send(job_payload)
            logger.info("Waiting for result from executor...")
            
            result_payload_str = await EXECUTOR_WEBSOCKET.recv()
            logger.info(f"Received result from executor, sending back to agent: {result_payload_str}")
            
            await websocket.send(result_payload_str)

        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"A connection closed unexpectedly: {e}")
            await websocket.send(json.dumps({
                "status": "error",
                "message": f"Connection with executor client was lost during job: {e}"
            }))
        except Exception as e:
            err_msg = f"An error occurred in the job bridge: {e}"
            logger.error(err_msg, exc_info=True)
            await websocket.send(json.dumps({
                "status": "error",
                "message": err_msg
            }))

async def main_handler(websocket):
    """Routes incoming connections to the correct handler function."""
    path = websocket.request.path
    logger.info(f"Incoming connection to path: {path}")

    if path == "/register_executor":
        # <<< FIX 1: Added 'await' to keep the connection alive >>>
        await register_executor(websocket)
    elif path == "/run_job":
        # <<< FIX 1: Added 'await' to process the job fully >>>
        await run_job(websocket)
    else:
        logger.warning(f"Connection attempt to unknown path: {path}")
        await websocket.close(1012, "Unknown path")

async def start_server():
    """The main function that starts the WebSocket server."""
    host = "0.0.0.0" # Use localhost for local testing
    port = 8765
    logger.info(f"Starting WebSocket server on ws://{host}:{port}")
    async with websockets.serve(main_handler, host, port):
        await asyncio.Future()  # run forever

if __name__ == '__main__':
    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        logger.info("\nServer shut down by user.")