# File: 2_websocket_bridge.py

import asyncio
import websockets
import json
import traceback

async def run_job_handler(websocket, path):
    """Handles incoming requests to run a Python function."""
    print("Bridge: Client connected.")
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                function_code = data["function_code"]
                kwargs = data["kwargs"]
                
                print(f"Bridge: Received job. Function: {function_code.splitlines()[0]}")
                print(f"Bridge: With arguments: {kwargs}")

                local_scope = {}
                exec(function_code, globals(), local_scope)
                
                func_name = function_code.split('def ')[1].split('(')[0].strip()
                target_function = local_scope[func_name]
                
                result = target_function(**kwargs)
                response = {"status": "ok", "result": result}
                print(f"Bridge: Execution successful. Result: {result}")

            except Exception as e:
                error_message = f"An error occurred: {str(e)}\n{traceback.format_exc()}"
                response = {"status": "error", "message": error_message}
                print(f"Bridge: Execution failed. Error: {error_message}")
            
            await websocket.send(json.dumps(response))

    except websockets.exceptions.ConnectionClosed:
        print("Bridge: Client disconnected.")

async def main():
    host = "localhost"
    port = 8765
    print(f"Starting WebSocket bridge server on ws://{host}:{port}")
    async with websockets.serve(run_job_handler, host, port):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())