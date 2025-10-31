# main_neo4j_importer.py (Corrected)

from neo4j import GraphDatabase
from lxml import etree
from pathlib import Path
from typing import List, TypedDict, Generator, Optional, Any

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

    @staticmethod
    def format_results_to_text(records: List[Any]) -> str:
        """Converts the raw Neo4j query results into a clean string for the LLM."""
        if not records:
            return "No information found in the knowledge graph for the specified component."

        lines = []
        for record in records:
            node = record.get("n")
            neighbor = record.get("neighbor")
            relationship = record.get("r")

            if node and 'name' in node:
                lines.append(f"Component '{node['name']}' (Type: {list(node.labels)[0]}):")
                lines.append(f"  - Properties: {dict(node)}")
            
            if neighbor and relationship:
                neighbor_name = neighbor.get('name', 'Unnamed Component')
                lines.append(f"  - Is connected via '{relationship.type}' to '{neighbor_name}' (Type: {list(neighbor.labels)[0]})")
        
        # Remove duplicates while preserving order
        unique_lines = []
        for line in lines:
            if line not in unique_lines:
                unique_lines.append(line)
        
        return "\n".join(unique_lines)
    
class Neo4jUploader:
    """
    Handles the connection to Neo4j and the uploading of graph data.
    """
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

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
                # ***** FIX APPLIED HERE *****
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


if __name__ == "__main__":
    # Same as before...
    script_dir = Path(__file__).parent
    data_dir = script_dir / ".."/"../" / "Pdata"

    XML_FILE = data_dir / "pairs_model.xml" 
    
    if not XML_FILE.exists():
        print(f"Error: XML file not found at {XML_FILE}")
        print("Please make sure the file exists and the path is correct.")
    else:
        uploader = Neo4jUploader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        uploader.clear_database()
        uploader.create_constraints()
        uploader.upload_graph_from_xml(XML_FILE)
        uploader.close()
        print("\nData successfully uploaded to Neo4j.")
        print("You can now explore the graph in the Neo4j Browser.")