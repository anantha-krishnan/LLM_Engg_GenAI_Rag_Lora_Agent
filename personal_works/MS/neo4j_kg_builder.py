# main_neo4j_importer.py (Corrected)

from neo4j import GraphDatabase
from lxml import etree
from pathlib import Path
from typing import List, TypedDict, Generator, Optional, Any
import pandas as pd
import re
import global_vars
# --- 1. NEO4J CONNECTION DETAILS ---
NEO4J_URI = global_vars.NEO4J_URI
NEO4J_USER = global_vars.NEO4J_USER
NEO4J_PASSWORD = global_vars.NEO4J_PASSWORD

class Neo4jConnector:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def query(self, query: str, parameters: Optional[dict] = None) -> List[Any]:
        """Runs a query and returns the results."""
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]

    def format_results_to_text(self, records: List[Any]) -> str:
        """
        [POC VERSION] Converts raw Neo4j query results into a clean string.
        This version is designed for a Proof of Concept and will output the
        FULL time-series data for OutputComponent nodes.
        
        WARNING: This can generate very large outputs for long simulations and may
        exceed LLM token limits. Use with small datasets for initial testing.
        """
        if not records:
            return "No information found in the knowledge graph for the specified component."

        lines = []
        for record in records:
            node = record.get("n", record.get("oc"))
            neighbor = record.get("neighbor")
            relationship = record.get("r")

            if node:
                # --- POC CHANGE: Output FULL data for OutputComponent nodes ---
                if "OutputComponent" in node.labels:
                    component_name = node.get('component', 'N/A')
                    lines.append(f"Output Component '{component_name}' (Type: OutputComponent):")
                    
                    time_vals = node.get('time_values', [])
                    output_vals = node.get('output_values', [])
                    
                    if time_vals and output_vals:
                        # Directly embed the full lists as strings into the context
                        lines.append(f"  - Number of Data Points: {len(time_vals)}")
                        lines.append(f"  - Time Values: {str(time_vals)}")
                        lines.append(f"  - Output Values: {str(output_vals)}")
                    else:
                        lines.append("  - No time series data found.")

                # --- Existing logic for other nodes (unchanged) ---
                elif 'name' in node:
                    node_type = next(iter(node.labels - {'Node'}), "Component")
                    lines.append(f"Component '{node['name']}' (Type: {node_type}):")
                    properties_to_print = {k: v for k, v in dict(node).items() if not isinstance(v, list)}
                    lines.append(f"  - Properties: {properties_to_print}")

            if neighbor and relationship:
                neighbor_name = neighbor.get('name', neighbor.get('component', 'Unnamed Component'))
                neighbor_type = next(iter(neighbor.labels - {'Node'}), "Component")
                lines.append(f"  - Is connected via '{relationship.type}' to '{neighbor_name}' (Type: {neighbor_type})")

        # Remove duplicates while preserving order
        unique_lines = list(dict.fromkeys(lines))
        return "\n".join(unique_lines)
    def get_full_context_for_output(self, request_name: str) -> str:
        """
        [MARKER AWARE VERSION]
        Performs a multi-hop query that respects the central role of Reference_Markers
        to gather a complete "dossier" for root cause analysis.
        """
        
        # This single, powerful query follows the true causal chain via marker IDs.
        cypher_query = """
        // 1. Find the PostRequest of interest by its name
        MATCH (pr:PostRequest {name: $request_name})

        // 2. Get its numerical output data (The Symptom)
        OPTIONAL MATCH (pr)-[:HAS_COMPONENT]->(output:OutputComponent)

        // 3. Find the Body that is being measured by tracing through the PostRequest's i_marker_id property
        // This is the key step that follows the marker logic.
        WITH pr, collect(DISTINCT output) as symptom_data
        MATCH (body_of_interest:Body)<-[:HAS_MARKER]-(marker:Reference_Marker {ms_id: pr.measures_marker})

        // 4. Now that we have the Body, find ALL joints connected to it.
        // A joint is connected if it references ANY marker on that body.
        WITH pr, body_of_interest, symptom_data
        MATCH (body_of_interest)<-[:HAS_MARKER]-(any_marker_on_body:Reference_Marker)
        MATCH (joint:Joint)
        WHERE joint.i_marker_id = any_marker_on_body.ms_id OR joint.j_marker_id = any_marker_on_body.ms_id

        // 5. Find any motions applied to THOSE joints (The Potential Cause)
        OPTIONAL MATCH (motion:Motion)-[:APPLIED_TO]->(joint)

        // Return all pieces of the puzzle for the dossier
        RETURN pr,
            body_of_interest,
            symptom_data,
            collect(DISTINCT joint) as connected_joints,
            collect(DISTINCT motion) as joint_drivers
        """
        
        # Before this query can work, your graph schema needs a small but vital change.
        # The original ingestion script did not create :Reference_Marker nodes or the
        # [:HAS_MARKER] relationship. Let's add a temporary helper here to show how to fix it.
        # NOTE: This should ideally be in your main_neo4j_importer.py script!
        
        # --- TEMPORARY SCHEMA FIX (for this to work) ---
        # You MUST update your main_neo4j_importer.py to include this logic.
        # I'm adding a check here to make it runnable for you.
        with self.driver.session() as session:
            check = session.run("MATCH (m:Reference_Marker) RETURN count(m) as count").single()
            if not check or check['count'] == 0:
                print("\n!!! WARNING: :Reference_Marker nodes not found. Your ingestion script needs an update.")
                print("This query will fail. Please update main_neo4j_importer.py with the Marker creation logic.\n")
                return "ERROR: Graph schema is missing critical :Reference_Marker nodes. Please update the ingestion script."
        # --- END SCHEMA FIX NOTE ---

        with self.driver.session() as session:
            result = session.run(cypher_query, request_name=request_name).single()

            if not result or not result["body_of_interest"]:
                return f"No complete causal chain found for PostRequest '{request_name}'."
            
            # Assemble the dossier from the rich query result
            pr_node = result["pr"]
            body_node = result["body_of_interest"]
            symptom_nodes = result["symptom_data"]
            joint_nodes = result["connected_joints"]
            driver_nodes = result["joint_drivers"]
            
            pr_context = self.format_results_to_text([{'n': pr_node}])
            body_context = self.format_results_to_text([{'n': body_node}])
            symptom_context = self.format_results_to_text([{'n': node} for node in symptom_nodes])
            joint_context = self.format_results_to_text([{'n': node} for node in joint_nodes])
            driver_context = self.format_results_to_text([{'n': node} for node in driver_nodes])

            full_dossier = f"""
            ---
            **Investigation Dossier for '{request_name}'**
            ---
            **0. ANALYSIS CONTEXT: The Output Request**
            {pr_context}
            ---
            **1. SYMPTOM: Numerical Output Data**
            {symptom_context if symptom_nodes else "No numerical output data found."}
            ---
            **2. PRIMARY COMPONENT: Measured Body**
            This is the body exhibiting the behavior, identified via its marker.
            {body_context}
            ---
            **3. STRUCTURAL PATH: Connected Joints**
            These joints are connected to the Measured Body via one of its markers.
            {joint_context if joint_nodes else "No joints found connected to this body."}
            ---
            **4. POTENTIAL DRIVERS: Motions/Forces on Joints**
            These motions control the behavior of the connected joints. The root cause is likely an expression here.
            {driver_context if driver_nodes else "No motions found applied to the connected joints."}
            ---
            """
            return full_dossier
class Neo4jUploader:
    """
    Handles the connection to Neo4j and the uploading of graph data.
    """
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.force_comps = ['FX', 'FY', 'FZ', 'TX', 'TY', 'TZ', 'FM', 'TM']
        self.disp_comps = ['DX', 'DY', 'DZ', 'RX', 'RY', 'RZ', 'DM', 'RM', 'YAW', 'PITCH', 'ROLL']

    def close(self):
        self.driver.close()

    def clear_database(self):
        """Wipes the entire database. Use with caution!"""
        with self.driver.session() as session:
            print("Clearing the entire database...")
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared.")

    def create_constraints(self):
        """
        Creates uniqueness constraints on node IDs to prevent duplicates and
        speed up lookups. This is a crucial best practice.
        """
        with self.driver.session() as session:
            print("Creating uniqueness constraints for each node type...")
            
            # Drop the old generic constraint if it still exists (for safety)
            constraint_name_result = session.run("""
                SHOW CONSTRAINTS YIELD name, labelsOrTypes, properties
                WHERE labelsOrTypes = ['Node'] AND properties = ['ms_id']
                RETURN name
            """)
            
            # .single() will return the first record or None if no records are found.
            record = constraint_name_result.single()
            if record:
                constraint_name = record["name"]
                # 2. Drop the constraint by its fetched name.
                session.run(f"DROP CONSTRAINT {constraint_name} IF EXISTS")
                print(f"  - Dropped old generic constraint: {constraint_name}")
            else:
                print("  - No old generic constraint on :Node(ms_id) found to drop.")
            
            labels = ["Simulation", "Body", "Joint", "Motion", "PostRequest"]
            for label in labels:
                # Note the syntax change in the CREATE CONSTRAINT command
                query = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.ms_id IS UNIQUE"
                session.run(query)
                print(f"  - Ensured constraint exists for :{label}")
            print("Constraints created.")

    def upload_graph_from_xml(self, xml_file_path):
        """
        Parses the MotionSolve XML file and populates the Neo4j database.
        """
        xml_file_path = str(xml_file_path)
        parser = etree.XMLParser(recover=True)
        tree = etree.parse(xml_file_path, parser)
        root = tree.getroot()

        with self.driver.session() as session:
            print(f"\n--- Starting Import for {xml_file_path.split('/')[-1]} ---")

            # --- 1. Find the Simulation Command ---
            sim_command = root.find('.//Command/Simulate')
            if sim_command is not None:
                sim_props = {
                    'type': sim_command.get('analysis_type'),
                    'end_time': float(sim_command.get('end_time')),
                    'source_file': xml_file_path.split('/')[-1]
                }
                session.run("""
                    MERGE (s:Simulation:Node {ms_id: 'Simulation_Run'})
                    SET s += $props
                    """, props=sim_props)
                print("  + Created Simulation node.")

            # --- 2. Extract All Rigid Bodies ---
            bodies = root.findall('.//Model/Body_Rigid')
            for body in bodies:
                body_props = {
                    'ms_id': body.get('id'),
                    'name': body.get('label'),
                    'mass': float(body.get('mass', 0.0)),
                    'inertia_xx': float(body.get('inertia_xx', 0.0)),
                    'inertia_yy': float(body.get('inertia_yy', 0.0)),
                    'inertia_zz': float(body.get('inertia_zz', 0.0)),
                    'is_ground': body.get('IsGround', 'FALSE') == 'TRUE'
                }
                
                session.run("""
                    MERGE (b:Body:Node {ms_id: $ms_id})
                    SET b += $props
                    """, ms_id=body_props['ms_id'], props=body_props)
            print(f"  + Processed {len(bodies)} Body nodes.")

            # --- NEW & ESSENTIAL: Create Reference_Marker Nodes ---
            markers = root.findall('.//Model/Reference_Marker')
            for marker in markers:
                marker_props = {
                    'ms_id': marker.get('id'),
                    'name': marker.get('label'),
                }
                body_id = marker.get('body_id')

                session.run("""
                    // Create the marker itself
                    MERGE (m:Reference_Marker:Node {ms_id: $ms_id})
                    SET m += $props
                    
                    // Find its parent body and create the relationship
                    WITH m
                    MATCH (b:Body {ms_id: $body_id})
                    MERGE (b)<-[:HAS_MARKER]-(m)
                    """, ms_id=marker_props['ms_id'], props=marker_props, body_id=body_id)
            print(f"  + Processed {len(markers)} Reference_Marker nodes and their connections to Bodies.")

            # Helper dict to map markers to bodies
            marker_to_body = {m.get('id'): m.get('body_id') for m in root.findall('.//Model/Reference_Marker')}

            # --- 3. Extract All Joints ---
            joints = root.findall('.//Model/Constraint_Joint')
            for joint in joints:
                joint_props = {
                    'ms_id': joint.get('id'),
                    'name': joint.get('label'),
                    'type': joint.get('type'),
                    'i_marker_id': joint.get('i_marker_id'),
                    'j_marker_id': joint.get('j_marker_id')
                }
                # ***** FIX APPLIED HERE *****
                session.run("""
                    MERGE (j:Joint:Node {ms_id: $ms_id})
                    SET j += $props
                    """, ms_id=joint_props['ms_id'], props=joint_props)

                # Create relationships to Bodies
                body1_id = marker_to_body.get(joint.get('i_marker_id'))
                body2_id = marker_to_body.get(joint.get('j_marker_id'))

                if body1_id and body2_id:
                    session.run("""
                        MATCH (b1:Body {ms_id: $body1_id})
                        MATCH (j:Joint {ms_id: $joint_id})
                        MATCH (b2:Body {ms_id: $body2_id})
                        MERGE (b1)-[:CONNECTS_TO]->(j)
                        MERGE (j)-[:CONNECTS_TO]->(b2)
                        """, body1_id=body1_id, joint_id=joint.get('id'), body2_id=body2_id)
            print(f"  + Processed {len(joints)} Joint nodes and their connections.")


            # --- 4. Extract Motions and link them to Joints ---
            motions = root.findall('.//Model/Motion_Joint')
            for motion in motions:
                motion_props = {
                    'ms_id': motion.get('id'),
                    'name': motion.get('label'),
                    'expression': motion.get('expr'),
                    'type': motion.get('type'),
                    'value_type': motion.get('val_type')
                }
                target_joint_id = motion.get('joint_id')

                # ***** FIX APPLIED HERE *****
                session.run("""
                    MERGE (m:Motion:Node {ms_id: $ms_id})
                    SET m += $props
                    """, ms_id=motion_props['ms_id'], props=motion_props)

                if target_joint_id:
                    session.run("""
                        MATCH (m:Motion {ms_id: $motion_id})
                        MATCH (j:Joint {ms_id: $joint_id})
                        MERGE (m)-[:APPLIED_TO]->(j)
                        """, motion_id=motion.get('id'), joint_id=target_joint_id)
            print(f"  + Processed {len(motions)} Motion nodes and their connections.")

            # --- 5: Extract Output Requests with Full Context ---
            requests = root.findall('.//Model/Post_Request')
            for req in requests:
                req_props = {
                    'ms_id': req.get('id'),
                    'name': req.get('label'),
                    'measurement': req.get('type'),
                    'measures_marker': req.get('i_marker_id'),
                    'relative_to_marker': req.get('j_marker_id'),
                    'in_frame_of_marker': req.get('ref_marker_id')
                }
                
                session.run("""
                    MERGE (pr:PostRequest:Node {ms_id: $ms_id})
                    SET pr += $props
                    """, ms_id=req_props['ms_id'], props=req_props)

                body_i = marker_to_body.get(req.get('i_marker_id'))
                body_j = marker_to_body.get(req.get('j_marker_id'))
                body_ref = marker_to_body.get(req.get('ref_marker_id'))

                if body_i:
                    session.run("""
                        MATCH (pr:PostRequest {ms_id: $req_id})
                        MATCH (b:Body {ms_id: $body_id})
                        MERGE (pr)-[:MEASURES {type: $req_type}]->(b)
                        """, req_id=req.get('id'), body_id=body_i, req_type=req.get('type'))
                if body_j:
                    session.run("""
                        MATCH (pr:PostRequest {ms_id: $req_id})
                        MATCH (b:Body {ms_id: $body_id})
                        MERGE (pr)-[:RELATIVE_TO]->(b)
                        """, req_id=req.get('id'), body_id=body_j)
                if body_ref:
                    session.run("""
                        MATCH (pr:PostRequest {ms_id: $req_id})
                        MATCH (b:Body {ms_id: $body_id})
                        MERGE (pr)-[:IN_FRAME_OF]->(b)
                        """, req_id=req.get('id'), body_id=body_ref)
                
            print(f"  + Processed {len(requests)} PostRequest nodes and their connections.")
            print("--- Import Finished ---")

    def upload_simulation_results(self, results_directory: Path):
        """
        Scans for PostRequest nodes, finds corresponding CSVs in the given
        directory, and uploads all components in a single batch per file.
        """
        print(f"\n--- Starting Batched Multi-Component Results Import ---")
        
        # First, get a list of all PostRequest nodes from the graph
        query_requests = "MATCH (pr:PostRequest) RETURN pr.name AS name, pr.ms_id AS ms_id"
        
        with self.driver.session() as session:
            request_nodes = session.run(query_requests)
            
            for record in request_nodes:
                request_name = record['name']
                request_id = record['ms_id']
                print(f"\n-> Checking results for PostRequest: '{request_name}'")

                # Construct the expected CSV filename
                results_file = results_directory / (request_name + ".csv")
                
                if not results_file.exists():
                    print(f"  - No results file found at '{results_file.name}'. Skipping.")
                    continue

                try:
                    df = pd.read_csv(results_file)
                    if 'Time' not in df.columns:
                        print(f"  - ERROR: 'Time' column not found in '{results_file.name}'. Skipping.")
                        continue
                    time_vector = df['Time'].tolist()
                except Exception as e:
                    print(f"  - ERROR: Could not read or parse CSV '{results_file.name}': {e}. Skipping.")
                    continue

                # Prepare a list of data maps, one for each component to be uploaded.
                components_batch = []
                all_known_components = self.force_comps + self.disp_comps

                for column_name in all_known_components:
                    if column_name not in df.columns:
                        continue  # Skip if this component is not in the results file

                    output_vector = df[column_name].tolist()
                    component_type = f'rotational {column_name}' if column_name in ['RX', 'RY', 'RZ', 'YAW', 'PITCH', 'ROLL', 'MX', 'MY', 'MZ'] else f'translational {column_name}'

                    # Add all necessary info for this component to our batch list
                    components_batch.append({
                        'req_id': request_id,
                        'comp_name': column_name,
                        'comp_type': component_type,
                        'time_data': time_vector,
                        'output_data': output_vector
                    })
                
                # <<< REFINEMENT 4: EFFICIENT BATCH UPLOAD with UNWIND >>>
                # If we found components, upload them all in one go for this file.
                if components_batch:
                    print(f"  - Found {len(components_batch)} components. Uploading as a single batch...")
                    
                    # This single query iterates over our list and creates/updates all nodes/relationships
                    session.run("""
                        UNWIND $batch AS component_data
                        MATCH (pr:PostRequest {ms_id: component_data.req_id})
                        MERGE (oc:OutputComponent {parent_id: pr.ms_id, component: component_data.comp_name})
                        SET oc.type = component_data.comp_type,
                            oc.time_values = component_data.time_data,
                            oc.output_values = component_data.output_data
                        MERGE (pr)-[:HAS_COMPONENT]->(oc)
                        """,
                        batch=components_batch
                    )
                    print(f"  - Successfully uploaded batch for '{request_name}'.")
                else:
                    print(f"  - No known components found in '{results_file.name}'.")

        print("--- Multi-Component Results Import Finished ---")

if __name__ == "__main__":
    # Same as before...
    script_dir = Path(__file__).parent
    data_dir = script_dir / ".."/"../" / "Pdata"

    XML_FILE = data_dir / "pairs_model.xml" 
    RESULTS_FILE = data_dir / "mrf_disp_export.csv"
    

    if not XML_FILE.exists():
        print(f"Error: XML file not found at {XML_FILE}")
        print("Please make sure the file exists and the path is correct.")
    else:
        uploader = Neo4jUploader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        uploader.clear_database()
        uploader.create_constraints()
        uploader.upload_graph_from_xml(XML_FILE)
        uploader.upload_simulation_results(data_dir)
        uploader.close()
        print("\nData successfully uploaded to Neo4j.")
        print("You can now explore the graph in the Neo4j Browser.")