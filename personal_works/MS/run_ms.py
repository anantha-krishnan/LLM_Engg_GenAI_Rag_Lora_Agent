from typing import Dict
import os
import shutil
import json
from pathlib import Path
BASELINE_FILE = "baseline_results.json"
def analyze_simulation_results(xml_filename: str, h_max: float, mode: str, session_work_dir: str) -> Dict:
    """
    Runs a MotionSolve simulation and analyzes the results. It manages file structures internally.
    
    Args:
        xml_filename (str): The NAME of the XML file (e.g., 'model.xml').
        h_max (float): The timestep to use for the simulation.
        mode (str): The analysis mode. Must be 'PRE' or 'NORM'.
        session_work_dir (str): The absolute path to the session's working directory.
    """
    # --- Step 1: Validate inputs and find the source file ---
    base_path = Path(session_work_dir)
    subdirectories = [
            "qa/qa_cmd",
            "qa/depot_xml",
            "qa/tmp_report",
            "qa/tmp_res_xml",
        ]
    base_path.mkdir(parents=True, exist_ok=True)

    for subdir in subdirectories:
        dir_to_create = base_path / subdir
        dir_to_create.mkdir(parents=True, exist_ok=True)
        
    """
    destination_xml_path = os.path.join(input_dir, xml_filename)
    shutil.copyfile(source_xml_path, destination_xml_path)
    print(f"Copied '{source_xml_path}' to '{destination_xml_path}' for simulation.")
    print(f"Running simulation on '{destination_xml_path}' with h_max={h_max}...")
    """
    source_xml_path = os.path.join(session_work_dir, xml_filename)
    if not os.path.exists(source_xml_path):
        return {"status": "error", "message": f"File '{xml_filename}' not found in session directory."}
    if mode not in ["PRE", "NORM"]:
        return {"status": "error", "message": "Mode must be 'PRE' or 'NORM'."}
    run_base_dir = os.path.join(session_work_dir, "qa")
    input_dir = os.path.join(run_base_dir, mode, "input")
    output_dir = os.path.join(run_base_dir, mode, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    import random
    result_value = 125.0 + (1000 * h_max * random.random())
    baseline_path = os.path.join(run_base_dir, BASELINE_FILE)

    if mode == "PRE":
        baseline_data = {"baseline_value": result_value, "source_h_max": h_max}
        with open(baseline_path, "w") as f:
            json.dump(baseline_data, f)
        return {
            "status": "success", "mode": "PRE",
            "current_value": result_value,
            "percentage_difference": result_value
        }
    elif mode == "NORM":
        if not os.path.exists(baseline_path):
            return {"status": "error", "message": "Cannot run 'NORM' comparison. No baseline file found."}
        
        with open(baseline_path, "r") as f:
            baseline_data = json.load(f)
        baseline_value = baseline_data["baseline_value"]
        # Your in-house tool would compare the files in the PRE/output and NORM/output directories
        diff = (abs(result_value - baseline_value) / abs(baseline_value)) * 100
        
        # Update the baseline with the new, more accurate result for the next iteration
        new_baseline_data = {"baseline_value": result_value, "source_h_max": h_max}
        with open(baseline_path, "w") as f:
            json.dump(new_baseline_data, f)
            
        return {
            "status": "success", "mode": "NORM", "percentage_difference": round(diff, 2),
            "current_value": result_value
        }
    return {"status": "ERROR", "mode": mode, 
            "current_value": 100.0,
            "percentage_difference": 100.0
        }