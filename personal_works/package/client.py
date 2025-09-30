# client.py (Corrected)
import asyncio
import websockets
import json
import traceback

async def execute_job(function_code, kwargs, job):
    """
    Dynamically executes a function from a string and returns the result.
    """
    try:
        # Create a local scope for the execution
        local_scope = {}
        # If there are any helper functions, they should be defined in the local scope first
        if 'helper_functions' in job:
            helper_functions_code = job.get("helper_functions", [])
            for helper_code in helper_functions_code:
                exec(helper_code, local_scope)
        # Define the function within the local scope
        exec(function_code, local_scope)
        
        # The function name is the first word after 'def'
        function_name = function_code.split('def ')[1].split('(')[0].strip()
        
        # Get the actual function object from the local scope
        func_to_run = local_scope.get(function_name)
        
        if not func_to_run:
            raise NameError(f"Function '{function_name}' was not defined correctly by exec().")

        print(f"   [Local Executor] Running function '{function_name}' with args: {kwargs}")
        result = func_to_run(**kwargs)
        
        return {"status": "success", "result": result}
    
    except Exception as e:
        print(f"   [Local Executor] ERROR executing code: {e}")
        # Return a detailed error message, including the traceback
        return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}


async def connect_and_work():
    """Connects to the server, registers as an executor, and waits for jobs."""
    # Use localhost if running on the same machine
    uri = "ws://localhost:8765/register_executor"
    
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                print(f"✅ Connected to server at {uri}. Registered as executor. Waiting for jobs...")
                
                async for message in websocket:
                    print(f"\n<- Received job payload from server.")
                    job = json.loads(message)
                    
                    # Get the code and arguments from the payload
                    function_code = job.get("function_code")
                    kwargs = job.get("kwargs")
                    if not function_code or kwargs is None:
                        response_payload = {"status": "error", "message": "Invalid job payload format."}
                    else:
                        # Execute the job and get the result
                        response_payload = await execute_job(function_code, kwargs, job)

                    print(f"-> Sending result back to server: {response_payload}")
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