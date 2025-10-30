# client.py (Corrected with non-blocking execution)
import asyncio
import websockets
import json
import traceback

# --- CHANGE 1: This is now a regular, synchronous function. ---
# It does blocking work, so it should not be an 'async def'.
def execute_job(function_code, kwargs, job):
    """
    Dynamically executes a function from a string and returns the result.
    This function is run in a separate thread to avoid blocking the event loop.
    """
    try:
        # Create a local scope for the execution
        local_scope = {}
        # If there are any helper functions, they should be defined in the local scope first
        helper_functions_code = job.get("helper_functions", [])
        for helper_code in helper_functions_code:
            exec(helper_code, local_scope) # Execute helpers in the same scope

        # Define the main function within the same local scope
        exec(function_code, local_scope)
        
        # A more robust way to find the function name than splitting the string
        # It finds the last callable object defined in the scope
        function_name = None
        for name, obj in local_scope.items():
            if callable(obj):
                function_name = name
        
        if not function_name:
             raise NameError("Could not find a callable function defined in the provided code.")
        
        func_to_run = local_scope[function_name]
        
        print(f"   [Local Executor] Running function '{function_name}' with args: {kwargs}")
        result = func_to_run(**kwargs)
        
        # The key in the response should match what the executor client template uses
        return {"status": "success", "result": result}
    
    except Exception as e:
        print(f"   [Local Executor] ERROR executing code: {e}")
        # Return a detailed error message, including the traceback
        return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}


async def connect_and_work():
    """Connects to the server, registers as an executor, and waits for jobs."""
    uri = "ws://10.75.32.62:8765/register_executor"
    
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                print(f"✅ Connected to server at {uri}. Registered as executor. Waiting for jobs...")
                
                async for message in websocket:
                    print(f"\n<- Received job payload from server.")
                    job = json.loads(message)
                    
                    function_code = job.get("function_code")
                    kwargs = job.get("kwargs")
                    if not function_code or kwargs is None:
                        response_payload = {"status": "error", "message": "Invalid job payload format."}
                    else:
                        # --- CHANGE 2: Run the blocking function in a thread pool ---
                        loop = asyncio.get_running_loop()
                        
                        # await tells the event loop to pause this function, run the
                        # execute_job in another thread, and work on other tasks
                        # (like responding to pings) in the meantime.
                        response_payload = await loop.run_in_executor(
                            None,  # Use the default thread pool
                            execute_job,
                            function_code,
                            kwargs,
                            job
                        )

                    print(f"-> Sending result back to server: {json.dumps(response_payload)[:200]}...")
                    await websocket.send(json.dumps(response_payload))
                    print("   Result sent. Waiting for next job...")
                    
        except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError, OSError) as e:
            print(f"❌ Connection to server failed: {e}. Retrying in 5 seconds...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(connect_and_work())
    except KeyboardInterrupt:
        print("\nClient is shutting down.")