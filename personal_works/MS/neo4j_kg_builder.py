# Neo4j Knowledge Graph Builder for MotionSolve XML Data

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
    
    def get_node_properties(self, entity_name: str) -> Optional[dict]:
        """Retrieves the full property dictionary for a single node by name."""
        with self.driver.session() as session:
            result = session.run("MATCH (n {name: $name}) RETURN n", name=entity_name).single()
            if result and result["n"]:
                # Combine properties and labels for a complete picture
                node_data = dict(result["n"])
                node_data['_labels'] = list(result["n"].labels)
                return node_data
            return None
        
    def entity_exists(self, entity_name: str) -> bool:
        """A simple, fast query to check if a node with the given name exists."""
        with self.driver.session() as session:
            result = session.run("MATCH (n {name: $name}) RETURN n LIMIT 1", name=entity_name)
            return result.single() is not None
    def get_nodes_by_type(self, node_type: str) -> list[dict]:
        """
        Retrieves all nodes of a given type (label) from the graph.

        Args:
            node_type (str): The label/type of the nodes to retrieve.

        Returns:
            A list of dictionaries, each representing a node's properties.
        """
        with self.driver.session() as session:
            query = f"""
            MATCH (n:{node_type})
            RETURN n
            ORDER BY n.name
            """
            result = session.run(query)
            nodes = []
            for record in result:
                node = record["n"]
                node_data = dict(node)
                node_data['_labels'] = list(node.labels)
                nodes.append(node_data)
            return nodes
    def get_all_nodes_with_primary_type(self) -> list[dict]:
        """
        Retrieves a list of all nodes, returning each node's name and its primary type.
        The primary type is the first label that is not 'Node'.

        Returns:
            A list of dictionaries, e.g., [{'name': 'Body_1', 'type': 'Body'}]
        """
        with self.driver.session() as session:
            query = """
            MATCH (n)
            WHERE n.name IS NOT NULL
            RETURN 
                n.name AS name, 
                [label IN labels(n) WHERE label <> 'Node'][0] AS type
            ORDER BY type, name
            """
            result = session.run(query)            
            return [{"name": record["name"], "type": record["type"]} for record in result]

    def get_graph_schema(self) -> str:
        """
        Retrieves the schema of the graph, showing node labels and their relationships.
        This version correctly handles the neo4j.graph.Node and Relationship objects.
        """
        print("\n--- CONNECTOR: Getting graph schema ---")
        try:
            query = "CALL db.schema.visualization()"
            
            with self.driver.session() as session:
                result = session.run(query).single()
                if not result:
                    return "Could not retrieve schema. The database might be empty or the procedure failed."

                nodes = result.get("nodes", [])
                relationships = result.get("relationships", [])

                schema_lines = ["--- Knowledge Graph Schema ---"]
                
                schema_lines.append("\n**Node Types (Labels):**")
                for node_obj in nodes:
                    # <<< FIX 1 & 2: Correctly handle the frozenset and filter 'Node' >>>
                    primary_label = next((label for label in node_obj.labels if label != 'Node'), "Component")
                    schema_lines.append(f"- {primary_label}")

                schema_lines.append("\n**Relationship Types:**")
                for rel_obj in relationships:
                    # The result is a Relationship object with .start_node, .end_node, and .type
                    start_node_labels = rel_obj.start_node.labels
                    end_node_labels = rel_obj.end_node.labels
                    
                    start_label = next((label for label in start_node_labels if label != 'Node'), "Component")
                    end_label = next((label for label in end_node_labels if label != 'Node'), "Component")
                    rel_type = rel_obj.type
                    
                    schema_lines.append(f"- ({start_label})-[:{rel_type}]->({end_label})")
                
                formatted_schema = "\n".join(schema_lines)
                print(formatted_schema)
                return formatted_schema
        except Exception as e:
            print(f"Error getting schema: {e}")
            return "An error occurred while fetching the graph schema. It may not be supported by your Neo4j version (requires 5.x+)."
            
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
                    component_name = node.get('name', 'N/A')
                    lines.append(f"Output Component '{component_name}':")
                    
                    time_vals = node.get('time_values', [])
                    output_vals = node.get('output_values', [])
                    
                    if time_vals and output_vals:
                        # Directly embed the full lists as strings into the context
                        lines.append(f"  - Number of Data Points: {len(time_vals)}")
                        lines.append(f"  - Time Values: {str(time_vals)}")
                        lines.append(f"  - {component_name} Values: {str(output_vals)}")
                    else:
                        lines.append("  - No time series data found.")

                # --- logic for other nodes ---
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
        # --- TEMPORARY SCHEMA FIX (for this to work) ---
        with self.driver.session() as session:
            check = session.run("MATCH (m:Reference_Marker) RETURN count(m) as count").single()
            if not check or check['count'] == 0:
                print("\n!!! WARNING: :Reference_Marker nodes not found. Your ingestion script needs an update.")
                print("This query will fail. Please update main_neo4j_importer.py with the Marker creation logic.\n")
                return "ERROR: Graph schema is missing critical :Reference_Marker nodes. Please update the ingestion script."
        # --- END SCHEMA FIX ---

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

    def get_dossier_for_any_entity(self, entity_name: str) -> str:
        """
        Generates a context-rich "dossier" for ANY given entity by name.
        It uses a single, generic query that leverages the enriched graph model
        to find all relevant context, INCLUDING RELATIONSHIP PROPERTIES.
        """
        cypher_query = """
        MATCH (n {name: $name})
        OPTIONAL MATCH (n)-[r]-(neighbor)
        OPTIONAL MATCH (n)-[:HAS_COMPONENT]->(output:OutputComponent)
        RETURN n, 
            collect(DISTINCT {rel: r, end_node: neighbor}) as neighbors,
            collect(DISTINCT output) as outputs
        """
        with self.driver.session() as session:
            result = session.run(cypher_query, name=entity_name).single()

            if not result or not result["n"]:
                return f"--- Dossier for '{entity_name}' ---\nEntity not found in the knowledge graph."

            node_data = result["n"]
            neighbor_data = result["neighbors"]
            output_data = result["outputs"]

            dossier_parts = [
                self.format_results_to_text([{'n': node_data}]),
                self.format_results_to_text([{'n': out} for out in output_data])
            ]
            
            neighbor_lines = ["\n--- Connections & Influences ---"]
            if not neighbor_data:
                neighbor_lines.append("No direct connections found.")
            else:
                for item in neighbor_data:
                    rel = item['rel']
                    end_node = item['end_node']
                    if rel!=None and end_node!=None:
                        rel_properties = dict(rel.items()) # Get properties as a dictionary
                        
                        # Format the relationship with its properties included.
                        line = f"- Is connected via '{rel.type}' to '{end_node.get('name', 'Unnamed')}' (Type: {list(end_node.labels)[0]})"
                        if rel_properties:
                            line += f" (Details: {rel_properties})" # Append the properties
                        
                        neighbor_lines.append(line)

            dossier_parts.append("\n".join(neighbor_lines))
            
            full_dossier = "\n".join(filter(None, dossier_parts))
            
            return f"--- Dossier for '{entity_name}' ---\n{full_dossier}"
    # In your Neo4jConnector class

    def get_complete_schema_definition(self) -> str:
        """
        Returns a manually defined, complete schema string representing ALL possible
        nodes and relationships the importer can create. This is more robust than
        relying on db.schema.visualization() which only shows existing data.
        """
        schema_string = """
        --- Complete Knowledge Graph Schema ---

        **Node Types (Labels):**
        - Simulation: Contains global simulation settings.
        - SolverSettings: Holds numerical solver parameters.
        - Body: Represents a rigid body in the model.
        - Reference_Marker: A coordinate system attached to a body.
        - Joint: A constraint between two bodies.
        - Motion: A prescribed motion applied to a joint.
        - PostRequest: A request to output a specific measurement.
        - OutputComponent: The numerical time-series results of a PostRequest.
        - Force: A force element acting between bodies.
        - AutoTireSystem: A special node representing a 'black-box' tire model.
        - TirePropertyFile: Represents a .tpf file.
        - RoadPropertyFile: Represents a .rdf file.
        - StateEquation: The core logic/equations for a subsystem like a tire.
        - StateVariable: An input variable to a StateEquation.

        **Relationship Types:**
        - (Reference_Marker)-[:HAS_MARKER]->(Body)
        - (Body)-[:CONNECTS_TO]->(Joint)
        - (Joint)-[:CONNECTS_TO]->(Body)
        - (Motion)-[:APPLIED_TO]->(Joint)
        - (PostRequest)-[:MEASURES]->(Body)
        - (PostRequest)-[:RELATIVE_TO]->(Body)
        - (PostRequest)-[:IN_FRAME_OF]->(Body)
        - (PostRequest)-[:HAS_COMPONENT]->(OutputComponent)
        - (PostRequest)-[:MEASURES_AUTOTIRE]->(AutoTireSystem)
        - (Force)-[:APPLIES_TO]->(Body)
        - (Force)-[:HAS_REACTION_ON]->(Body)
        - (AutoTireSystem)-[:APPLIES_FORCE_VIA]->(Force)
        - (AutoTireSystem)-[:DEFINED_BY]->(TirePropertyFile)
        - (AutoTireSystem)-[:DEFINED_BY]->(RoadPropertyFile)
        - (AutoTireSystem)-[:GOVERNED_BY]->(StateEquation)
        - (StateEquation)-[:USES_INPUT]->(StateVariable)
        - (StateVariable)-[:MEASURES_KINEMATICS_OF]->(Reference_Marker)
        - (Component)-[:SOLVED_WITH]->(SolverSettings) [Note: 'Component' can be Body, Joint, Force]
        - (Motion)-[:INFLUENCES]->(PostRequest)
        - (StateVariable)-[:INFLUENCES]->(PostRequest)
        - (AutoTireSystem)-[:INFLUENCES]->(PostRequest)
        """
        return schema_string.strip()
class Neo4jUploader:
    """
    Handles the connection to Neo4j and the uploading of graph data.
    """
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.force_comps = ['FX', 'FY', 'FZ', 'TX', 'TY', 'TZ', 'FM', 'TM']
        self.disp_comps = ['X','Y','Z','DX', 'DY', 'DZ', 'RX', 'RY', 'RZ', 'DM', 'RM', 'YAW', 'PITCH', 'ROLL']
        self.reqsub_cols=['f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8']
        self.rad_omegaact_omega = {'f2':'Radius',
                                   'f3':'OmegaActual',
                                   'f4':'OmegaFree'}
        self.slip_inc = {'f2': 'lon slip',
                         'f3': 'lat angle',
                         'f4': 'inc angle'}
        self.cp_forces = {'f2':'longitudinal force',
                          'f3':'lateral force',
                          'f4':'vertical force',
                          'f6':'residual overturning moment',
                          'f7':'rolling resistance moment',
                          'f8':'aligning moment'}
        self.cp_locations = {'f2':'road contact point x location',
                             'f3':'road contact point y location',
                             'f4':'road contact point z location',
                             'f6': 'tire radial penetration into the road surface'}
        self.pr={'Radius OmegaActual OmegaFree':self.rad_omegaact_omega,
                 'LonSlip LatSlip IncAngle (W-Axis system)':self.slip_inc,
                 'Tire CP Forces (W-Axis system)':self.cp_forces,
                 'Contact Patch Locations':self.cp_locations}

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
            # --- Find the Param_Transient Command ---
            param_transient_command = root.find('.//Command/Param_Transient')
            if param_transient_command is not None:                  
                transient_props = {
                    'integrator_type': param_transient_command.get('integrator_type'),
                    'integr_tol': float(param_transient_command.get('integr_tol', 0.0)),
                    'h_max': float(param_transient_command.get('h_max', 0.0)),
                    'h0_max': float(param_transient_command.get('h0_max', 0.0)),
                    'h_min': float(param_transient_command.get('h_min', 0.0)),
                    'max_order': int(param_transient_command.get('max_order', 0)),
                    'vel_tol_factor': float(param_transient_command.get('vel_tol_factor', 0.0)),
                    'dae_constr_tol': float(param_transient_command.get('dae_constr_tol', 0.0)),
                    'dae_corrector_maxit': int(param_transient_command.get('dae_corrector_maxit', 0)),
                    'dae_corrector_minit': int(param_transient_command.get('dae_corrector_minit', 0)),
                    'dae_index': int(param_transient_command.get('dae_index', 0)),
                    'dae_vel_ctrl': param_transient_command.get('dae_vel_ctrl', 'FALSE') == 'TRUE'  
                }
            # --- 1. Find the Simulation Command ---
            sim_command = root.find('.//Command/Simulate')
            if sim_command is not None:
                sim_props = {
                    'type': sim_command.get('analysis_type'),
                    'name': 'Model Solver Settings',
                    'end_time': float(sim_command.get('end_time')),
                    'source_file': xml_file_path.split('/')[-1]
                }
                if param_transient_command is not None:        
                    sim_props.update(transient_props)

                session.run("""
                    MERGE (s:Simulation:Node {ms_id: 'Simulation_Settings'})
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
                    'is_ground': body.get('IsGround', 'FALSE') == 'TRUE',
                    'v_ic_x': float(body.get('v_ic_x', 0.0)),
                    'v_ic_y': float(body.get('v_ic_y', 0.0)),
                    'v_ic_z': float(body.get('v_ic_z', 0.0)),
                    'w_ic_x': float(body.get('w_ic_x', 0.0)),
                    'w_ic_y': float(body.get('w_ic_y', 0.0)),
                    'w_ic_z': float(body.get('w_ic_z', 0.0)),
                    'v_ic_x_flag': body.get('v_ic_x_flag', "FALSE"),
                    'v_ic_y_flag': body.get('v_ic_y_flag', "FALSE"),
                    'v_ic_z_flag': body.get('v_ic_z_flag', "FALSE"),
                    'w_ic_x_flag': body.get('w_ic_x_flag', "FALSE"),
                    'w_ic_y_flag': body.get('w_ic_y_flag', "FALSE"),
                    'w_ic_z_flag': body.get('w_ic_z_flag', "FALSE")
                }
                
                session.run("""
                    MERGE (b:Body:Node {ms_id: $ms_id})
                    SET b += $props
                    """, ms_id=body_props['ms_id'], props=body_props)
            print(f"  + Processed {len(bodies)} Body nodes.")

            # --- Create Reference_Marker Nodes ---
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

            # --- Read Force_Vector_TwoBody with dependencies ---            
            force_vector_twobody_items = root.findall('.//Model/Force_Vector_TwoBody')
            tire_system_nodes = {}
            for force_element in force_vector_twobody_items:
                param_string = force_element.get('usrsub_param_string', '')
                match = re.search(r'USER\(\d+,\s*(\d+),', param_string)
                if not match:
                    continue # Skip if the format is unexpected
                
                system_id = match.group(1)
                if system_id not in tire_system_nodes:
                    tire_system_props = {
                        'ms_id': system_id,
                        'name': f'AutoTireSystem_{system_id}',
                    }
                # Find the associated TPF and RDF files using the unique string labels from the XML
                tpf_element = root.find('.//Model/Reference_String[@label="tire property file string"]')
                rdf_element = root.find('.//Model/Reference_String[@label="road property file string"]')
                if tpf_element is not None:
                    tire_system_props['tire_file_path'] = tpf_element.get('string')
                if rdf_element is not None:
                    tire_system_props['road_file_path'] = rdf_element.get('string')
                    
                    tire_system_nodes[system_id] = tire_system_props

                # Create the AutoTireSystem node
                session.run("""
                    MERGE (ats:AutoTireSystem:Node {ms_id: $ms_id})
                    SET ats += $props
                """, ms_id=system_id, props=tire_system_props)

                # Create the Force node and link it to the AutoTireSystem
                fvtb_props = {
                    'ms_id': force_element.get('id'),
                    'name': force_element.get('label')+f"_{force_element.get('id')}",
                    'type': force_element.get('type'),
                    'i_marker_id': force_element.get('i_marker_id'),
                    'j_floating_marker_id': force_element.get('j_floating_marker_id'),
                    'ref_marker_id': force_element.get('ref_marker_id'),
                }
                session.run("""
                    MERGE (f:Force:Node {ms_id: $ms_id})
                    SET f += $props
                    // Link it to the parent AutoTireSystem
                    WITH f
                    MATCH (ats:AutoTireSystem {ms_id: $ats_id})
                    MERGE (ats)-[:APPLIES_FORCE_VIA]->(f)
                """, ms_id=fvtb_props.get('ms_id'), props=fvtb_props, ats_id=system_id)

                # Link the Force to the bodies it acts upon
                body_i_id = marker_to_body.get(fvtb_props.get('i_marker_id'))
                body_j_id = marker_to_body.get(fvtb_props.get('j_floating_marker_id'))
                ref_body_id = marker_to_body.get(fvtb_props.get('ref_marker_id'))
                if body_i_id:
                    session.run("""
                        MATCH (f:Force {ms_id: $force_id})
                        MATCH (b:Body {ms_id: $body_id})
                        MERGE (f)-[:APPLIES_FORCE_TO]->(b)
                        """, force_id=fvtb_props.get('ms_id'), body_id=body_i_id)
                if ref_body_id:
                    session.run("""
                        MATCH (f:Force {ms_id: $force_id})
                        MATCH (b:Body {ms_id: $body_id})
                        MERGE (f)-[:IN_FRAME_OF]->(b)
                        """, force_id=fvtb_props.get('ms_id'), body_id=ref_body_id)
            
            # --- Create the StateEquation node and link it to the AutoTireSystem
            state_eqns = root.findall('.//Model/Control_StateEqn')
            for se in state_eqns:
                param_string = se.get('usrsub_param_string', '')
                # get the second USER(...) parameter which is the AutoTireSystem ID
                match = re.search(r'USER\(\d+,\s*(\d+),', param_string)
                if not match:
                    continue
                system_id = match.group(1)
                se_props = {
                    'ms_id': se.get('id'),
                    'name': f"State Equation for Tire {system_id}",
                    'type': se.get('type'),
                    'usrsub_param_string': se.get('usrsub_param_string'),
                    'u_solver_array_id': se.get('u_solver_array_id')
                }
                session.run("""
                    MERGE (cse:StateEquation:Node {ms_id: $ms_id})
                    SET cse += $props
                    // Link it to the parent AutoTireSystem
                    WITH cse
                    MATCH (ats:AutoTireSystem {ms_id: $ats_id})
                    MERGE (ats)-[:GOVERNED_BY]->(cse)
                """, ms_id=se_props.get('ms_id'), props=se_props, ats_id=system_id)
                # --- 8. Process State Variables and link them to the State Equation ---
                input_array_id = se_props['u_solver_array_id']
                input_array_element = root.find(f'.//Model/Reference_Array[@id="{input_array_id}"]')
                if input_array_element is not None:
                    var_ids = input_array_element.text.strip().split()
                    for var_id in var_ids:
                        var_element = root.find(f'.//Model/Reference_Variable[@id="{var_id}"]')
                        if var_element is not None:
                            var_props = {
                                'ms_id': var_element.get('id'),
                                'name': var_element.get('label'),
                                'type': var_element.get('type'),                                
                            }
                            if var_props['type'] == 'EXPRESSION':
                                var_props['expr'] = var_element.get('expr')
                            elif var_props['type'] == 'USERSUB':
                                var_props['usrsub_param_string'] = var_element.get('usrsub_param_string')

                            session.run("""
                                // Create the variable
                                MERGE (sv:StateVariable:Node {ms_id: $ms_id})
                                SET sv += $props
                                
                                // Link it to the state equation that uses it
                                WITH sv
                                MATCH (c:StateEquation {ms_id: $eqn_id})
                                MERGE (c)-[:USES_INPUT]->(sv)
                                """, ms_id=var_id, props=var_props, eqn_id=se_props['ms_id'])
                                
                            if var_props['type'] == 'USERSUB':
                                marker_ids_in_expr = re.findall(r',(\d{8,})', var_props['usrsub_param_string'])
                                for marker_id in marker_ids_in_expr:
                                    session.run("""
                                        MATCH (sv:StateVariable {ms_id: $var_id})
                                        MATCH (m:Reference_Marker {ms_id: $marker_id})
                                        MERGE (sv)-[:MEASURES_KINEMATICS_OF]->(m)
                                    """, var_id=var_id, marker_id=marker_id)
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
                if req.get('type') == 'USERSUB':
                    # use regex to extract the third param usersub_param_string. It is in the format USER(x, y, z,...)
                    req_props['measures_autotire'] = re.findall(r'USER\((.*?)\)', req.get('usrsub_param_string'))[0].split(',')[2].strip()
                    session.run("""
                        MATCH (pr:PostRequest {ms_id: $ms_id})
                        MATCH (ats:AutoTireSystem {ms_id: $req_id})
                        MERGE (pr)-[:MEASURES_AUTOTIRE]->(ats)
                    """, ms_id=req_props['ms_id'], req_id=req_props['measures_autotire'])
                else:
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

                
                # --- REFINEMENT 3: HANDLE GENERAL CASES WHERE RESULTS ARE NAMED as in self.reqsub_cols ---
                # check all the keys in self.pr to see if any of them are substrings of request_name
                sub_key = None
                for key in self.pr.keys():
                    if key in request_name:
                        sub_key = key
                        break

                if sub_key:
                    for column_name, col_val in self.pr[sub_key].items():
                            if column_name not in df.columns:
                                continue  # Skip if this component is not in the results file

                            output_vector = df[column_name].tolist()
                            component_type = col_val

                            components_batch.append({
                                'req_id': request_id,
                                'comp_name': component_type,
                                'comp_type': column_name,
                                'time_data': time_vector,
                                'output_data': output_vector
                            })
                else:
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
                        MERGE (oc:OutputComponent {parent_id: pr.ms_id, name: component_data.comp_name})
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

    def create_summary_relationships(self):
        """
        Creates high-level "shortcut" relationships like [:INFLUENCES]
        for both Motion-driven and Tire-driven systems, enriching them
        with properties describing the causal path.
        """
        with self.driver.session() as session:
            print("\n--- Creating Enriched Summary Relationships for Analysis ---")

            # --- Query 1: For standard Motion -> Joint -> Body -> PostRequest paths (Unchanged) ---
            motion_query = """
            MATCH (motion:Motion)-[:APPLIED_TO]->(joint:Joint)-[:CONNECTS_TO*1..2]->(body:Body)
            MATCH (pr:PostRequest)-[:MEASURES]->(body)
            WITH motion, pr,
                collect(DISTINCT joint.name) as via_joint_names,
                collect(DISTINCT body.name) as on_body_names
            MERGE (motion)-[r:INFLUENCES]->(pr)
            SET r.via_joint_names = via_joint_names,
                r.on_body_names = on_body_names,
                r.reason = "Motion influences this PostRequest via the listed joints and bodies."
            """
            result_motion = session.run(motion_query).consume()
            print(f"  + Created/Updated {result_motion.counters.relationships_created} Motion-driven [:INFLUENCES] relationships.")

            # --- Query 2: <<< ENHANCED FOR TIRE SYSTEM PATH DETAILS >>> ---
            tire_query = """
            // 1. Start from the input StateVariable and find its full chain
            MATCH (sv:StateVariable)<-[:USES_INPUT]-(se:StateEquation)<-[:GOVERNED_BY]-(ats:AutoTireSystem)
            MATCH (ats)-[:APPLIES_FORCE_VIA]->(force:Force)-[:APPLIES_FORCE_TO]->(tireforcebody:Body)
            
            // 2. Find the PostRequests that are influenced by this Tire System
            MATCH (ats)<-[:MEASURES_AUTOTIRE]-(pr:PostRequest)

            // 3. Find the body being measured by the State Variable
            MATCH (sv)-[:MEASURES_KINEMATICS_OF]->(:Reference_Marker)-[:HAS_MARKER]->(body:Body)

            // 4. Group by the start and end points (sv, pr) and collect path details
            WITH sv, pr,
                collect(DISTINCT body.name) as measured_bodies,
                collect(DISTINCT se.name) as via_state_equations,
                collect(DISTINCT ats.name) as via_tire_systems,
                collect(DISTINCT force.name) as resulting_in_forces,
                collect(DISTINCT tireforcebody.name) as affected_bodies

            // 5. Create the enriched INFLUENCES relationship
            MERGE (sv)-[r:INFLUENCES]->(pr)
            SET r.reason = "This kinematic variable is an input to a tire state equation. The resulting tire force influences this post-request output.",
                r.measured_bodies = measured_bodies,
                r.via_state_equations = via_state_equations,
                r.via_tire_systems = via_tire_systems,
                r.resulting_in_forces = resulting_in_forces,
                r.affected_bodies = affected_bodies
            """
            result_tire = session.run(tire_query).consume()
            print(f"  + Created/Updated {result_tire.counters.relationships_created} Tire input-driven [:INFLUENCES] relationships with full path details.")

    def create_tire_force_summary(self):
        """
        Creates high-level "shortcut" relationships for Tire Forces.
        """
        with self.driver.session() as session:
            print("\n--- Creating Tire Force Summary Relationships ---")

            query = """
            MATCH (ats:AutoTireSystem)-[:APPLIES_FORCE_VIA]->(force:Force)-[:APPLIES_FORCE_TO]->(body:Body)
            MATCH (pr:PostRequest)-[:MEASURES_AUTOTIRE]->(ats)
            MATCH (ats)-[:GOVERNED_BY]->(se:StateEquation)
            WITH ats, pr, se, force, body
            // Create the summary relationship
            MERGE (ats)-[r:INFLUENCES]->(pr)
            SET r.reason = "The effects of the Force/Moment from AutoTireSystem is measured by this PostRequest.",
                r.governed_by_state_equation = se.ms_id,
                r.tire_force_id= force.ms_id,
                r.applies_to_body_id = body.ms_id
            """
            result = session.run(query)
            summary = result.consume()
            print(f"  + Created/Updated {summary.counters.relationships_created} [:APPLIES_TIRE_FORCE_TO] relationships.")
  
if __name__ == "__main__":
    script_dir = Path(__file__).parent
    data_dir = script_dir / ".."/"../" / "Pdata"
    data_dir_sub = data_dir / "TestRig"
    XML_FILE = data_dir_sub / "SingleTire_Run.xml"
    create_db = False

    if not XML_FILE.exists():
        print(f"Error: XML file not found at {XML_FILE}")
        print("Please make sure the file exists and the path is correct.")
    else:
        if create_db:
            uploader = Neo4jUploader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
            uploader.clear_database()
            uploader.create_constraints()
            uploader.upload_graph_from_xml(XML_FILE)
            uploader.upload_simulation_results(data_dir_sub)
            uploader.create_summary_relationships() 
        connector = Neo4jConnector(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

        #all_nodes_with_types = connector.get_dossier_for_any_entity('OmegaActual')
        #all_nodes_with_types = connector.get_all_nodes_with_primary_type()
        s = connector.get_nodes_by_type('PostRequest')
        uploader.close()
        print("\nData successfully uploaded to Neo4j.")
        print("You can now explore the graph in the Neo4j Browser.")