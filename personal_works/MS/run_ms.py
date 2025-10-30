

def print_error(message: str):
    """output error message."""
    #DEBUG_FILE="C:\\Users\\anantk\\Downloads\\qa\\debug_log.txt"

    # with open(DEBUG_FILE, "a") as debug_file:
        # debug_file.write(f"DEBUG: {message}\n")
    print(f"DEBUG: {message}")
    
import subprocess
from typing import Dict
import os
import shutil
import json
from pathlib import Path
from lxml import etree
from typing import Dict
from pathlib import Path

def get_model_from_xml(xml_filename: str, qa_folder: Path) -> dict:
    """
    Extracts the model name from the given XML filename.
    Assumes the model name is the filename without the extension.
    
    Args:
        xml_filename (str): The XML filename (e.g., 'model.xml').
    Returns:
        dict: A dictionary with 'status', 'model_path' and 'message'
    """
    ret_value = {"status": "success", "model_path": "", "message": "", "model_id": ""}
    try:
        qa_cmd_folder = qa_folder / "qa_cmd"
        if not qa_cmd_folder.exists():
            ret_value["status"] = "error"
            ret_value["message"] = f"QA command folder '{qa_cmd_folder}' does not exist."
            return ret_value
        parser = etree.XMLParser(remove_blank_text=False)
        for file in qa_cmd_folder.glob("*.xml"):
            tree=etree.parse(file, parser)
            root = tree.getroot()

            # Step 3: Find the specific element using an XPath expression
            NuQA_Global = root.find('.//NuQA_Global')
            if NuQA_Global is not None:
                model_dir = Path(NuQA_Global.get('model_basedir'))
                # from model_dir get the path after 'qa' folder
                try:
                    # find the index of 'qa' in the parts
                    qa_index = model_dir.parts.index("qa")
                    # get the relative path from 'qa'
                    model_dir = Path(*model_dir.parts[qa_index + 1:])                    
                except Exception as e:
                    print_error(f"Could not find 'qa' in the model_basedir path: {model_dir}. Error: {e}")
                    ret_value["status"] = "error"
                    ret_value["message"] = f"Could not find 'qa' in the model_basedir path: {model_dir}. Error: {e}"
                    return ret_value
            # loop through all NuQA_Model tags
            for NuQA_Model in root.findall('NuQA_Model'):
                if xml_filename in NuQA_Model.get('model_file'):
                    model_path = model_dir / Path(NuQA_Model.get('model_file'))
                    ret_value["model_path"] = model_path
                    ret_value["model_id"] = NuQA_Model.get('id')
                    ret_value["status"] = "success"
                    ret_value["message"] = f"Found model path: {model_path}"
                    return ret_value
        ret_value["status"] = "error"
        ret_value["message"] = f"Could not find model for XML filename '{xml_filename}' in any command file."
        return ret_value
    except Exception as e:
        ret_value["status"] = "error"
        ret_value["message"] = f"An unexpected error occurred: {e}"
        print_error(f"An unexpected error occurred: {e}")
        return ret_value

import subprocess
from typing import Dict
import os
import shutil
import json
from pathlib import Path
from lxml import etree
from typing import Dict
from pathlib import Path

def update_h_max_preserving_format(xml_file_path: str, new_h_max_value: float, output_file_path: str) -> Dict:
    """
    Parses an XML file using lxml, finds the h_max attribute in the 
    Param_Transient tag, updates its value, and saves the file while
    preserving the original formatting.

    Args:
        xml_file_path (str): The path to the input/output XML file.
        new_h_max_value (float): The new float value for h_max.
    """
    # sample return dict
    ret_value = {"status": "success", "message": "Updated h_max successfully.", "new_value": new_h_max_value}
    print_error(f"Updating h_max in '{xml_file_path}' to {new_h_max_value} (format preserving)...")
    try:
        # Step 1: Create a parser that preserves comments and whitespace
        # This is the crucial step for maintaining format.
        parser = etree.XMLParser(remove_blank_text=False)
        
        # Step 2: Parse the XML file with our special parser
        tree = etree.parse(xml_file_path, parser)
        root = tree.getroot()

        # Step 3: Find the specific element using an XPath expression
        param_transient_element = root.find('.//Param_Transient')

        if param_transient_element is not None:
            # Step 4: Get the old value (optional, for logging)
            old_value = param_transient_element.get('h_max')
            
            # Step 5: Set the new value for the 'h_max' attribute
            new_value_str = str(new_h_max_value)
            param_transient_element.set('h_max', new_value_str)
            
            # Step 6: Write the changes back to the file
            # This will overwrite the original file with the updated content.
            # The original formatting, comments, and processing instructions
            # will be preserved.
            if output_file_path:
                xml_file_path = output_file_path
            tree.write(xml_file_path, encoding='UTF-8', xml_declaration=True)
            print_error(f"Successfully updated h_max in '{xml_file_path}' (format preserved).")
            print_error(f"Old value: {old_value}")
            print_error(f"New value: {new_value_str}")
            ret_value["status"] = "success"

            return ret_value
        else:
            print_error("Error: Could not find the '<Param_Transient>' element in the XML file.")
            ret_value["status"] = "error"
            ret_value["message"] = "Param_Transient element not found."
            return ret_value

    except Exception as e:
        ret_value["status"] = "error"
        ret_value["message"] = f"An unexpected error occurred: {e}"
        print_error(f"An unexpected error occurred: {e}")
        return ret_value

from typing import Dict
import os
import shutil
import json
from pathlib import Path
from lxml import etree
from typing import Dict
from pathlib import Path

def get_run_diff_nuqa(report_file: Path, model_name: str) -> Dict:
    """
    Parses the NUQA report file to extract the current value and percentage difference.
    Sample report format:

    CHECK REPORT GENERATED ON 30-SEP-2025 11:11:54

    MODEL_ID     TOLERANCE       MAX_CHK_VALUE STATUS                           MODEL  COMMENT
    =============================================================================================
       491     1.0000000          41.5524483 FAILED                    c11x001m.xml  Single Wheel, Lateral velocity input, Vx = 20m/s, Vy = 2 m/s, Vz=0 m/s, Wx =0 rad/s, Wy=65.2 rad/s, Wz= 0 rad/s

    Args:
        report_file (Path): Path to the NUQA report file.
        model_name (str): The model name to search for in the report.
    """
    print_error(f"Parsing NUQA report file '{report_file}' for model '{model_name}'...")
    ret_dict={
                "status": "success",
                "mode": "NORM",
                "percentage_difference": 100.0,
                "message": ""
    }
    # read the report file and extract MAX_CHK_VALUE using the sample format above
    if not report_file.exists():
        return {"status": "error", "mode": "NORM", "message": f"Report file '{report_file}' does not exist.", "percentage_difference": 100.0}
    
    with open(report_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if model_name in line:
                parts = line.split()
                try:
                    current_value = float(parts[2]) # MAX_CHK_VALUE directly provides the difference value
                    ret_dict["status"]  = "success"                    
                    ret_dict["percentage_difference"] = current_value
                    ret_dict["message"] = "Found the difference in results and updated the key 'percentage_difference' with the corresponding value."
                except (IndexError, ValueError):
                    ret_dict["status"] = "error"
                    ret_dict["message"] = "Could not find MAX_CHK_VALUE from the report line."
                return ret_dict
        # If we reach here, the model name was not found
        ret_dict["status"] = "error"
        ret_dict["message"] = f"Model name '{model_name}' not found in the report."
        return ret_dict

from typing import Dict
import os
import shutil
import json
from pathlib import Path
from lxml import etree
from typing import Dict
from pathlib import Path
import re, subprocess

BASELINE_FILE = "baseline_results.json"
                
def analyze_simulation_results(xml_filename: str, h_max: float, mode: str) -> Dict:
    """
    Runs a MotionSolve simulation and analyzes the results. It manages file structures internally.
    
    Args:
        xml_filename (str): The NAME of the XML file (e.g., 'model.xml').
        h_max (float): The timestep to use for the simulation.
        mode (str): The analysis mode. Must be 'PRE' or 'NORM'.        
    """
    session_work_dir = "C:\\Users\\anantk\\Downloads"  # hardcoded for now, replace with actual session directory path
    # return dict with status, mode, current_value, percentage_difference
    ret_value = {"status": "success", "mode": mode, "percentage_difference": 100.0, "message": ""}

    # --- Step 1: Validate inputs and find the source file ---
    base_path = Path(session_work_dir)
    #zipfile_path = base_path / "qa.zip"
    # unzip if not already done
    #if zipfile_path.exists() and not (base_path / "qa").exists():
    #    shutil.unpack_archive(zipfile_path, base_path)
    qa_folder = base_path / "qa"
    model_info = get_model_from_xml(xml_filename, qa_folder)
    if model_info["status"] == "error":
        return model_info
    xml_file_path = qa_folder / model_info["model_path"]
    if not xml_file_path.exists():
        print_error(f"File '{xml_file_path}' not found in session directory.")        
        ret_value["status"] = "error"
        ret_value["message"] = f"File '{xml_file_path}' not found in session directory."
        return ret_value
    
    # update h_max in the xml file
    update_h_max_preserving_format(str(xml_file_path), h_max, None)

    # call a bat file to set environment variables for MotionSolve and wait for it to finish
    # This assumes you have a batch file named 'setup_and_run_nuqa_env.bat' in the current directory. pass the mode as an argument
    nuqa_run_bat = qa_folder / "setup_and_run_nuqa_env.bat"
    if not nuqa_run_bat.exists():
        ret_value["status"] = "error"
        ret_value["message"] = f"Batch file '{nuqa_run_bat}' not found."
        return ret_value
    # use sub process to call the bat file and wait for it to finish
    print_error(f"Running MotionSolve simulation in mode '{mode}' with h_max={h_max}...")
    try:    
        result = subprocess.run(
            [str(nuqa_run_bat), mode, str(qa_folder), model_info["model_id"]], 
            check=True, 
            capture_output=True, 
            text=True,
            shell=True  # Recommended for running .bat files on Windows
        )
        print_error(f"MotionSolve simulation finished successfully {result.returncode}. STDOUT: {result.stdout} STDERR: {result.stderr}\n")

    except subprocess.CalledProcessError as e:
        print_error(f"MotionSolve simulation failed with exit code {e.returncode}. STDOUT: {e.stdout} STDERR: {e.stderr}")
        ret_value["status"] = "error"
        ret_value["message"] = f"The simulation script failed with exit code {e.returncode}. STDOUT: {e.stdout} STDERR: {e.stderr}"
        return ret_value
    
    if mode == "PRE":
        ret_value['status'] = "success"
        ret_value["message"] = "PRE run completed. Baseline established. Now run with mode='NORM'."
        return ret_value
    # --- Step 3: Get Results from tmp_report ---
    qa_report_file = qa_folder / "tmp_report" / "chk_report_xml_tire.txt"
    if not qa_report_file.exists():
        print_error(f"Report file '{qa_report_file}' not found after simulation.")
        ret_value["status"] = "error"
        ret_value["message"] = f"Report file '{qa_report_file}' not found after simulation."
        return ret_value
    diff_report = get_run_diff_nuqa(qa_report_file, xml_filename)
    if diff_report["status"] == "error":
        print_error(f"Error parsing report file: {diff_report.get('message', '')}")
        ret_value["status"] = "error"
        ret_value["message"] = diff_report.get("message", "Unknown error while parsing report.")
        return ret_value
    
    ret_value["percentage_difference"] = diff_report["percentage_difference"]
    ret_value["message"] = diff_report.get("message", "")

    return ret_value
    # --- dummy set up for testing without MotionSolve ---
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

        
# write test case for analyze_simulation_results
if __name__ == "__main__":
    # test update_h_max_preserving_format
    test_xml_name = "c11x003m.xml"
    session_dir = "C:\\Users\\anantk\\Downloads"
    h_max = 0.001
    mode="PRE"
    analyze_simulation_results(test_xml_name, h_max, mode)
    
    
    mode="NORM"
    h_max = 0.0015
    analyze_simulation_results(test_xml_name, h_max, mode)
    
    #get_model_from_xml(test_xml_name, Path(session_dir) / "qa")

from typing import Dict
