# server.py
# A standalone WebSocket server for debugging connections.
# This version corrects the handler function signature.

import asyncio
import websockets
import json
import time

async def job_handler(websocket):
    """
    Handles a connection from a single client.
    This function now has the correct signature: (websocket, path).
    """
    client_address = websocket.remote_address
    print(f"\n✅ Client connected from: {client_address}")
    #print(f"   Request path: '{path}' (This is the second argument from the library)")

    try:
        # 1. Define a simple, predictable job to send to the client.
        job_id = f"job_{int(time.time())}"
        job_payload = {
            "job_id": job_id,
            "task": "make_uppercase",
            "data": "This is a test message from the server."
        }

        # 2. Send the job as a JSON string.
        print(f"-> Sending job '{job_id}' to {client_address}...")
        await websocket.send(json.dumps(job_payload))
        print("   Waiting for a result from the client...")

        # 3. Wait for the client to send the result back.
        result_json = await websocket.recv()
        print(f"<- Received a message from {client_address}: {result_json}")
        result_data = json.loads(result_json)
        
        # 4. Process and display the result.
        print(f"   Successfully parsed result for job '{result_data['job_id']}'")
        print(f"   Client's result data: {result_data['result']}")
        print("   Job complete for this client.")

    except websockets.exceptions.ConnectionClosedOK:
        print(f"ℹ️ Client {client_address} disconnected normally.")
    
    except json.JSONDecodeError as e:
        print(f"!!!!!!!!!!!!! JSON DECODE ERROR !!!!!!!!!!!!!")
        print(f"Could not parse message from client {client_address}. Error: {e}")
        print(f"The invalid message was: {result_json}")

    except Exception as e:
        print(f"!!!!!!!!!!!!! UNEXPECTED SERVER ERROR !!!!!!!!!!!!!")
        print(f"An error occurred while handling client {client_address}.")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
    
    finally:
        print(f"🏁 Connection handler for client {client_address} has finished.")


async def main():
    """Starts the WebSocket server and keeps it running."""
    host = "0.0.0.0" 
    port = 8765 

    print("=====================================================")
    print("      Standalone WebSocket Debugging Server (v2)")
    print("=====================================================")
    print(f"🚀 Starting server on ws://{host}:{port}")
    print("   Run the client.py script from another terminal to connect.")
    print("   Press Ctrl+C to stop the server.")
    
    async with websockets.serve(job_handler, host, port):
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer is shutting down.")