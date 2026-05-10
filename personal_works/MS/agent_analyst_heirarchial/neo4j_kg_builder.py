# Neo4j Knowledge Graph Builder for MotionSolve XML Data

from neo4j import GraphDatabase
from neo4j.graph import Node, Relationship, Path as Neo4jPath
from lxml import etree
from pathlib import Path as pathlib_Path
from typing import List, TypedDict, Generator, Optional, Any
import pandas as pd
import re
import global_vars
import openai

# --- 1. NEO4J CONNECTION DETAILS ---
NEO4J_URI = global_vars.NEO4J_URI
NEO4J_USER = global_vars.NEO4J_USER
NEO4J_PASSWORD = global_vars.NEO4J_PASSWORD

class Neo4jConnector:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password),notifications_min_severity="OFF")
        self.blacklist = {'embedding'}

    def close(self):
        self.driver.close()
    
    def get_node_properties(self, entity_name: str) -> Optional[dict]:
        """Retrieves the full property dictionary for a single node by name."""
        with self.driver.session() as session:
            result = session.run("MATCH (n {name: $name}) RETURN n", name=entity_name).single()
            if result and result["n"]:
                # Combine properties and labels for a complete picture
                node_data = dict(result["n"])
                clean_data = {
                    k: v for k, v in node_data.items() 
                    if k not in self.blacklist
                }
                clean_data['_labels'] = list(result["n"].labels)
                return clean_data
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

    def query(self, query: str, parameters: Optional[dict] = None) -> List[Any]:
        """Runs a query and returns the results."""
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]

    def format_results_to_text(self, records: List[Any]) -> str:
        if not records:
            return "No results found in the knowledge graph."
        # Helper to format node
        def fmt_node(n):
            label = next((l for l in n.labels if l != 'Node'), "Entity")
            name = n.get('name', 'Unnamed')
            # Filter out heavy properties (embeddings/time-series) for the LLM
            props = {k: v for k, v in dict(n).items() 
                    if k not in ['embedding', 'output_values', 'time_values', 'parent_id'] 
                    and not isinstance(v, list)}
            return f"[{label}] '{name}'"
        # Format relationship
        def get_id(obj):
            if hasattr(obj, 'element_id'): return str(obj.element_id)
            if hasattr(obj, 'id'): return str(obj.id)
            return str(obj)
        
        def process_item(value):
            # --- 1. HANDLE PATHS (The "Story" logic) ---
            if isinstance(value, Neo4jPath):
                path_text = []
                nodes = list(value.nodes)
                rels = list(value.relationships)

                for i in range(len(rels)):
                    current_node  = nodes[i]
                    rel = rels[i]
                    next_node = nodes[i+1]

                    
                    if get_id(rel.start_node) == get_id(current_node):
                        arrow = f" --[:{rel.type}]--> "
                    else:
                        arrow = f" <--[:{rel.type}]-- "
                    if i == 0:
                        path_text.append(fmt_node(current_node))
                    path_text.append(arrow)
                    path_text.append(fmt_node(next_node))

                output_strings.append("CHAIN: " + "".join(path_text))

            # --- 2. HANDLE SINGLE NODES ---
            elif isinstance(value, Node):
                label = next((l for l in value.labels if l != 'Node'), "Entity")
                name = value.get('name', 'Unnamed')
                props = {k: v for k, v in dict(value).items() if k not in ['embedding', 'output_values', 'time_values']}
                output_strings.append(f"NODE: [{label}] {name}")

            # --- 3. HANDLE EVERYTHING ELSE ---
            else:
                output_strings.append(f"{value}")
        output_strings = []

        # --- MAIN LOOP ---
        for record in records:
            if isinstance(record, Neo4jPath):
                output_strings.append(process_item(record))
            
            # Scenario B: record is a Node object
            elif isinstance(record, Node):
                output_strings.append(f"NODE: {fmt_node(record)}")

            # Scenario C: record is a standard Neo4j Record (dict-like)
            elif hasattr(record, 'keys'):
                for key in record.keys():
                    val = record[key]
                    if isinstance(val, Neo4jPath):
                        output_strings.append(process_item(val))
                    elif isinstance(val, Node):
                        output_strings.append(f"NODE: {fmt_node(val)}")
                    else:
                        output_strings.append(f"{key}: {val}")
            
            # Scenario D: record is a simple value
            else:
                output_strings.append(str(record))

        # Remove duplicates while preserving order
        unique_output = list(dict.fromkeys(output_strings))
        return "\n\n".join(unique_output)

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
    
    def get_complete_schema_definition(self) -> str:
        """
        Dynamically fetches specific labels, properties, and relationships 
        while filtering out the generic 'Node' noise.
        """
        node_descriptions = {
        "Simulation": "Contains global simulation settings.",
        "SolverSettings": "Holds numerical solver parameters.",
        "Body": "A physical part representing a rigid body in the model.",
        "Joint": "A physical part applying constraints on degrees of freedom between two physical parts.",
        "Motion": "A prescribed motion applied to a joint.",
        "PostRequest": "PostRequest & OutputComponent Exist as pairs only. Measure Physical quantities from any node storing them as time series data.",
        "OutputComponent": "PostRequest & OutputComponent Exist as pairs only. Measure Physical quantities from any node storing them as time series data.",
        "Force": "Nodes representing a interface system to the calling Motion Solve solver in MBD model.",        
        "StateEquation": "Control state equations for a subsystem like a tire that calculates force and moment based on inputs from the connected Body."}
        # Query 1: Clean Properties (Filtering out the generic 'Node' label)
        prop_query = """
        CALL db.schema.nodeTypeProperties() 
        YIELD nodeLabels, propertyName
        WITH [lbl IN nodeLabels WHERE lbl <> 'Node'][0] AS label, propertyName
        WHERE label IS NOT NULL AND NOT propertyName IN ['time_values', 'output_values']
        RETURN label, collect(propertyName) AS props
        ORDER BY label
        """

        # Query 2: Clean Relationships (How nodes are connected)
        rel_query = """
        CALL db.schema.visualization() YIELD relationships
        UNWIND relationships AS rel
        WITH startNode(rel) AS s, endNode(rel) AS e, type(rel) AS relType
        RETURN DISTINCT 
            [lbl IN labels(s) WHERE lbl <> 'Node'][0] AS source,
            relType,
            [lbl IN labels(e) WHERE lbl <> 'Node'][0] AS target
        """

        schema_output = ["--- DYNAMIC MBD GRAPH SCHEMA ---"]

        with self.driver.session() as session:
            # Process Properties
            prop_results = session.run(prop_query)
            schema_output.append("\n**Nodes/Entities and Attributes:**")
            for rec in prop_results:
                label = rec['label']
                props = rec['props']
                # Get description from our static map, or use a default
                desc = node_descriptions.get(label, " ")
                schema_output.append(f"- {label}: {desc} (Attributes: {props})")

            # Process Relationships
            rel_results = session.run(rel_query)
            schema_output.append("\n**Model Connectivity (Edges):**")
            for rec in rel_results:
                schema_output.append(f"- ({rec['source']})-[:{rec['relType']}]->({rec['target']})")
        return "\n".join(schema_output)

        # --- THE 'MBD Knowledge graph LAWS' ---
        

        schema_output.append("**1. Basic Kinematic Topology**"
        "Bodies are connected to other bodies via Joints. Joints can also be connected directly to other Joints to form complex kinematic chains."
        "   **Sample Query Logic:** To find connected bodies and their joints:"
        "`MATCH (b1:Body)-[:CONNECTS_TO]->(j:Joint)-[:CONNECTS_TO]->(b2:Body) RETURN b1.name, j.name, j.type, b2.name`")

        schema_output.append("**2. Tire System Core**"
        "The `StateEquation` node represents the physical tire entity. It contains the differential equations defining the tire's states."
        "   **Sample Query Logic:** To find the governing logic of a tire:"
        "`MATCH (se:StateEquation) RETURN se.name, se.input_variable_details`")

        schema_output.append("**3. Kinematic Inputs (State Variables)**"
        "`StateEquation` nodes receive physical inputs (displacement, velocity, etc.) via kinematic measurements of a Body. StateVariables define these inputs from a measured Body. The graph links `StateEquation` directly to the measured `Body`. Technical details (IDs and call commands) are stored in the `input_variable_details` property."
        "   **Sample Query Logic:** To find what kinematics drive a tire:"
        "`MATCH (se:StateEquation)-[:MEASURES_INPUT]->(b:Body) RETURN se.name, se.input_variable_details, b.name`")

        schema_output.append("**4. Tire Force Application**"
        "The `StateEquation` finally applies physical forces/moments to a `Body` through a `Force` node."
        "   **Sample Query Logic:** To trace where tire forces are acting:"
        "`MATCH (se:StateEquation)-[:APPLIES_FORCE_VIA]->(f:Force)-[:APPLIES_FORCE_TO]->(b:Body) RETURN se.name, f.name, b.name`")

        schema_output.append("**5. Output Analysis (PostRequests)**"
        "`PostRequest` nodes measure the physical states of a `Body`. They contain `OutputComponent` nodes. There can be 1 to 6 components per request, each representing a specific kinematic or force/moment component ( Translational and Rotational)."
        " Get all the components and check their names to find out which one you need"
        "*   **Sample Query Logic:** To find vertical force (FZ) at a specific part:"
        "`MATCH (pr:PostRequest)-[:HAS_COMPONENT]->(oc:OutputComponent) WHERE pr.name CONTAINS 'hub' RETURN pr.name, oc.name, oc.output_values`")

        schema_output.append("**6. Fixed Joint Surrogate Logic (Expert Rule)**"
        "If the target Body (B) lacks a direct `PostRequest`, search for a `Joint` of type **\"FIXED\"** connecting Body (B) to another Body (D). In MBD, measuring Body (D) is indirectly equivalent to measuring Body (B) because they move as a single rigid unit."
        "*   **Query Logic:** To find a surrogate measurement point for a 'Hub':"
        "`MATCH (b1:Body {name: 'Hub'})-[:CONNECTS_TO]-(j:Joint {type: 'FIXED'})-[:CONNECTS_TO]-(neighbor:Body) RETURN neighbor.name AS surrogate_body`")
        return "\n".join(schema_output)
    def get_full_graph(self)-> str:
        """
        Retrieves the entire graph as a list of Paths.
        Each Path contains nodes and relationships.
        """
        with self.driver.session() as session:
            query = """
            MATCH p=()-[r]->()
            RETURN p
            """
            result = session.run(query)
            paths = [record["p"] for record in result]
            details = []
            for path in paths:
                rel= path.relationships[0]
                start_node = rel.start_node.get('name')
                end_node = rel.end_node.get('name')
                details.append(f"RELATIONSHIP: ({start_node})-[:{rel.type}]->({end_node})")
            return details
    def find_chains_between_nodes(self, node_names: list[str]) -> str:
        query = """
        MATCH (n) WHERE n.name IN $node_list
        WITH collect(n) as nodes
        UNWIND nodes as n1
        UNWIND nodes as n2
        WITH n1, n2 WHERE id(n1) < id(n2)
        MATCH p = shortestPath((n1)-[*..15]-(n2))
        RETURN p
        """
        with self.driver.session() as session:
            records = session.run(query, node_list=node_names)
            
            # Step 1: Collect every unique relationship from ALL paths
            unique_rels = {}
            for record in records:
                path = record['p']
                for rel in path.relationships:
                    # Use element_id or id as the unique key for the edge
                    rel_id = getattr(rel, 'element_id', getattr(rel, 'id', None))
                    if rel_id not in unique_rels:
                        unique_rels[rel_id] = rel

            if not unique_rels:
                return "No connections found between these items."

            # Step 2: Format these unique connections as a single "Master Model Structure"
            output = ["--- MASTER MODEL SUBGRAPH ---"]
            for rel in unique_rels.values():
                s = rel.start_node
                e = rel.end_node
                s_fmt = f"[{list(s.labels)[0] if s.labels else 'Node'}] '{s.get('name')}'"
                e_fmt = f"[{list(e.labels)[0] if e.labels else 'Node'}] '{e.get('name')}'"
                output.append(f"{s_fmt} --[:{rel.type}]--> {e_fmt}")

            return "\n".join(output),unique_rels

    def generate_mermaid_topology(self, unique_rels: dict) -> str:
        mermaid_code = ["graph TD"]
        
        # 1. DEFINE ALL STYLES (Make sure every class you use is defined here)
        mermaid_code.append("classDef body fill:#f9f,stroke:#333,stroke-width:2px;")
        mermaid_code.append("classDef post fill:#bbf,stroke:#333,stroke-width:2px;")
        mermaid_code.append("classDef output fill:#dfd,stroke:#333,stroke-width:1px;")
        mermaid_code.append("classDef force fill:#ffcc00,stroke:#333,stroke-width:2px;")
        mermaid_code.append("classDef controlstate fill:#e1f5fe,stroke:#01579b,stroke-width:2px;")
        mermaid_code.append("classDef joint fill:#eeeeee,stroke:#333,stroke-dasharray: 5 5;")

        # Helper to map Labels to CSS Classes
        style_map = {
            "Body": "body",
            "PostRequest": "post",
            "OutputComponent": "output",
            "Force": "force",
            "StateEquation": "controlstate",
            "Joint": "joint"
        }

        def safe_id(name):
            return re.sub(r'[^a-zA-Z0-9_]', '_', name)

        # 2. TRACK NODES TO AVOID DUPLICATE DEFINITIONS
        processed_nodes = set()

        for rel in unique_rels.values():
            nodes_to_format = [rel.start_node, rel.end_node]
            formatted_nodes = []

            for node in nodes_to_format:
                n_id = safe_id(node.get('name'))
                n_label = node.get('name')
                n_type = list(node.labels)[0] if node.labels else ""
                n_style = style_map.get(n_type, "")

                # If we haven't seen this node, define it with its style
                # Syntax: id["label"]:::style
                if n_id not in processed_nodes:
                    style_suffix = f":::{n_style}" if n_style else ""
                    formatted_nodes.append(f'{n_id}["{n_label}"]{style_suffix}')
                    processed_nodes.add(n_id)
                else:
                    formatted_nodes.append(n_id)

            # 3. ADD THE RELATIONSHIP
            line = f'{formatted_nodes[0]} -->|{rel.type}| {formatted_nodes[1]}'
            mermaid_code.append(line)

        return "\n".join(mermaid_code)
    def find_nodes_hybrid(self, keyword: str, mode="hybrid", top_k=3, openai_api_key=None):
        """
        Modes: 
        - 'text': Uses Lucene Full-Text (Best for IDs/Technical names)
        - 'vector': Uses OpenAI Embeddings (Best for synonyms/concepts)
        - 'hybrid': Tries text first, falls back to vector if score is low
        """
        results = []
        with self.driver.session() as session:
            label_res = session.run("CALL db.labels()")
            all_labels = [r["label"] for r in label_res if r["label"] != "Node"]
        # --- MODE: FULL-TEXT SEARCH ---
        if mode in ["text", "hybrid"]:
            with self.driver.session() as session:
                # The ~ provides fuzzy matching for typos
                res = session.run("""
                    CALL db.index.fulltext.queryNodes("node_names", $term) 
                    YIELD node, score
                    RETURN node.name as name, score, 'text' as method
                    LIMIT $k
                """, term=f"{keyword}~", k=top_k)
                results.extend([dict(r) for r in res])

        # --- MODE: VECTOR SEARCH ---
        if mode in ["vector", "hybrid"]:
            
            client = openai.OpenAI()
            embedding = client.embeddings.create(input=[keyword], model="text-embedding-3-small").data[0].embedding
            for label in all_labels:
                index_name = f"vector_{label.lower()}"
                with self.driver.session() as session:
                    res = session.run("""
                        CALL db.index.vector.queryNodes($index_name, $k, $vec)
                        YIELD node, score
                        RETURN node.name as name, score, 'vector' as method
                    """, index_name=index_name, k=top_k, vec=embedding)
                    results.extend([dict(r) for r in res])

        # Sort by score and return unique names
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        return sorted_results[:2] if sorted_results else None

 
class Neo4jUploader:
    """
    Handles the connection to Neo4j and the uploading of graph data.
    """
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password),notifications_min_severity="OFF")
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
                 'Contact Patch Locations':self.cp_locations,
                 'Tire Hub Forces (C-Axis System)': self.cp_forces}

    def close(self):
        self.driver.close()

    def clear_database(self):
        """Wipes the entire database, including data, constraints, and indexes."""
        with self.driver.session() as session:
            print("Clearing all data (Nodes/Relationships)...")
            session.run("MATCH (n) DETACH DELETE n")

            # 1. Drop all Constraints first
            # Dropping a constraint automatically drops its associated index.
            print("Dropping all constraints...")
            constraints = session.run("SHOW CONSTRAINTS YIELD name")
            for record in constraints:
                name = record["name"]
                session.run(f"DROP CONSTRAINT {name} IF EXISTS")
                print(f"  - Dropped constraint: {name}")

            # 2. Drop remaining Indexes
            # We only drop indexes that are NOT managed by constraints 
            # (like your Full-Text and Vector indexes)
            print("Dropping remaining indexes...")
            indexes = session.run("SHOW INDEXES YIELD name, type")
            for record in indexes:
                name = record["name"]
                idx_type = record["type"]
                
                # Skip built-in LOOKUP indexes (they start with 'index_' usually and are type 'LOOKUP')
                if idx_type == 'LOOKUP':
                    continue
                    
                try:
                    session.run(f"DROP INDEX {name} IF EXISTS")
                    print(f"  - Dropped index: {name}")
                except Exception as e:
                    # This captures cases where an index might be a system index we missed
                    print(f"  - Could not drop index {name} (likely system-owned): {e}")

            print("Database, constraints, and custom indexes cleared.")

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
            
            labels = ["Simulation", "Body", "Joint", "Motion", "PostRequest","OutputComponent", "Force", "StateEquation"]
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
                    MERGE (s:Simulation {ms_id: 'Simulation_Settings'})
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
                    MERGE (b:Body {ms_id: $ms_id})
                    SET b += $props
                    """, ms_id=body_props['ms_id'], props=body_props)
            print(f"  + Processed {len(bodies)} Body nodes.")

            # --- Create Reference_Marker Nodes ---
            markers = root.findall('.//Model/Reference_Marker')
            if (0):
                for marker in markers:
                    marker_props = {
                        'ms_id': marker.get('id'),
                        'name': marker.get('label'),
                    }
                    body_id = marker.get('body_id')

                    session.run("""
                        // Create the marker itself
                        MERGE (m:Reference_Marker {ms_id: $ms_id})
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
                    MERGE (j:Joint {ms_id: $ms_id})
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
                    MERGE (m:Motion {ms_id: $ms_id})
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
                    tire_system_nodes[system_id] = force_element.get('id')
                        
                # Find the associated TPF and RDF files using the unique string labels from the XML
                """
                tpf_element = root.find('.//Model/Reference_String[@label="tire property file string"]')
                rdf_element = root.find('.//Model/Reference_String[@label="road property file string"]')
                if tpf_element is not None:
                    tire_system_props['tire_file_path'] = tpf_element.get('string')
                if rdf_element is not None:
                    tire_system_props['road_file_path'] = rdf_element.get('string')
                    
                    tire_system_nodes[system_id] = tire_system_props
                """
                # Create the AutoTireSystem node
                #session.run("""
                #    MERGE (ats:AutoTireSystem {ms_id: $ms_id})
                #    SET ats += $props
                #""", ms_id=system_id, props=tire_system_props)

                # Create the Force node and link it to the AutoTireSystem
                fvtb_props = {
                    'ms_id': force_element.get('id'),
                    'name': force_element.get('label'),
                    'type': force_element.get('type'),
                    'i_marker_id': force_element.get('i_marker_id'),
                    'j_floating_marker_id': force_element.get('j_floating_marker_id'),
                    'ref_marker_id': force_element.get('ref_marker_id'),
                }
                session.run("""
                    MERGE (f:Force {ms_id: $ms_id})
                    SET f += $props
                """, ms_id=fvtb_props.get('ms_id'), props=fvtb_props)#, ats_id=system_id
                
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
                match = re.search(r'USER\(\d+,\s*(\d+),', param_string)
                if not match:
                    continue
                system_id = match.group(1)
                
                # Track unique bodies to avoid redundant relationship calls
                unique_body_ids = set()
                # Track metadata for the LLM to read as a property
                input_metadata = []

                input_array_id = se.get('u_solver_array_id')
                input_array_element = root.find(f'.//Model/Reference_Array[@id="{input_array_id}"]')
                
                if input_array_element is not None:
                    var_ids = input_array_element.text.strip().split()
                    for var_index, var_id in enumerate(var_ids):
                        var_element = root.find(f'.//Model/Reference_Variable[@id="{var_id}"]')
                        if var_element is not None:
                            expr = var_element.get('expr') or var_element.get('usrsub_param_string')
                            m = re.search(r'(\w+)\((.*)\)', expr)
                            if not m: continue
                            
                            call_cmd = m.group(1)
                            call_args = m.group(2).split(',')
                            
                            m_id = None
                            if var_element.get('type') == 'EXPRESSION':
                                m_id = call_args[0].strip()
                            else: # USERSUB
                                dll = var_element.get('usrsub_dll_name')
                                if dll == 'msautoutils': m_id = call_args[1].strip()
                                elif dll == 'mbdtire': m_id = call_args[2].strip()
                            
                            if m_id:
                                b_id = marker_to_body.get(m_id)
                                if b_id:
                                    unique_body_ids.add(b_id)
                                    # get body name from body id b_id
                                    b_id_name = session.run("""                                    
                                    MATCH (b:Body {ms_id: $body_id})                                    
                                    RETURN b.name AS body_name
                                    """, body_id=b_id)
                                    b_id_name_record = b_id_name.single()
                                    if b_id_name_record:
                                        b_id_name_record = b_id_name_record['body_name']
                                    else:
                                        b_id_name_record = b_id
                                    input_metadata.append(f"Input Var {var_index}: {var_element.get('label')} measures Body {b_id_name_record}")

                # 1. MERGE the StateEquation and link to AutoTireSystem
                # We store the detailed variable list as a property here.
                se_props = {
                    'ms_id': system_id,
                    'name': f"Control State Equation for Tire {system_id}",
                    'input_variable_details': input_metadata, 
                    'usrsub_param_string': se.get('usrsub_param_string'),
                }
                
                session.run("""
                    MERGE (cse:StateEquation {ms_id: $ms_id})
                    SET cse += $props                    
                """, ms_id=se_props['ms_id'], props=se_props)

                # 2. Create direct relationships to the Bodies measured
                # We use UNWIND for a clean batch update of relationships
                if unique_body_ids:
                    session.run("""
                        MATCH (cse:StateEquation {ms_id: $eqn_id})
                        MATCH (fv:Force {ms_id: $fv_id})
                        SET cse.output_variable_details = "Forces and Moments; applied via node '" + toLower(fv.name) + "' of type 'Force'"
                        //Link the StateEquation to the Force_Vector_TwoBody
                        MERGE (cse)-[:APPLIES_FORCE_VIA]->(fv)
                        WITH cse
                        UNWIND $body_list AS b_id
                        MATCH (b:Body {ms_id: b_id})
                        MERGE (cse)-[:MEASURES_INPUT]->(b)                        
                    """, eqn_id=se_props['ms_id'], body_list=list(unique_body_ids),fv_id=tire_system_nodes[se_props['ms_id']])
                
                
               
                
            # --- 5: Extract Output Requests with Full Context ---
            requests = root.findall('.//Model/Post_Request')
            for req in requests:
                req_props = {
                    'ms_id': req.get('id'),
                    'name': req.get('label'),
                    'measurement': req.get('type'),
                    #'measures_body_ms_id': marker_to_body.get(req.get('i_marker_id')),                    
                }
                
                session.run("""
                    MERGE (pr:PostRequest {ms_id: $ms_id})
                    SET pr += $props
                    """, ms_id=req_props['ms_id'], props=req_props)
                if req.get('type') == 'USERSUB':
                    # use regex to extract the third param usersub_param_string. It is in the format USER(x, y, z,...)
                    req_props['measures_autotire'] = re.findall(r'USER\((.*?)\)', req.get('usrsub_param_string'))[0].split(',')[2].strip()
                    session.run("""
                        MATCH (pr:PostRequest {ms_id: $ms_id})
                        MATCH (cse:StateEquation {ms_id: $req_id})
                        MERGE (pr)-[:MEASURES_OUTPUT {type: 'AutoTireSystem'}]->(cse)
                    """, ms_id=req_props['ms_id'], req_id=req_props['measures_autotire'])
                else:
                    body_i = marker_to_body.get(req.get('i_marker_id'))
                    body_j = marker_to_body.get(req.get('j_marker_id'))
                    body_ref = marker_to_body.get(req.get('ref_marker_id'))

                    if body_i:
                        session.run("""
                            MATCH (pr:PostRequest {ms_id: $req_id})
                            MATCH (b:Body {ms_id: $body_id})
                            MERGE (pr)-[:MEASURES_OUTPUT {type: $req_type}]->(b)
                            """, req_id=req.get('id'), body_id=body_i, req_type=req.get('type'))
                    if 0 and body_j:
                        session.run("""
                            MATCH (pr:PostRequest {ms_id: $req_id})
                            MATCH (b:Body {ms_id: $body_id})
                            MERGE (pr)-[:RELATIVE_TO]->(b)
                            """, req_id=req.get('id'), body_id=body_j)
                    if 0 and body_ref:
                        session.run("""
                            MATCH (pr:PostRequest {ms_id: $req_id})
                            MATCH (b:Body {ms_id: $body_id})
                            MERGE (pr)-[:IN_FRAME_OF]->(b)
                            """, req_id=req.get('id'), body_id=body_ref)
    
            print(f"  + Processed {len(requests)} PostRequest nodes and their connections.")
            print("--- Import Finished ---")

    def upload_simulation_results(self, results_directory: pathlib_Path):
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

    def create_hybrid_indexes(self):
        self.client = openai.OpenAI()
        
        with self.driver.session() as session:
            # 1. Get all unique labels in the graph dynamically
            # We filter out 'Node' because it's your generic base label
            label_result = session.run("CALL db.labels()")
            all_labels = [record["label"] for record in label_result if record["label"] != "Node"]
            
            # 2. Create Dynamic Full-Text Index (supports all labels in one)
            labels_pipe = "|".join(all_labels)
            session.run(f"""
                CREATE FULLTEXT INDEX node_names FOR (n:{labels_pipe}) 
                ON EACH [n.name]
            """)

            # 3. Create Dynamic Vector Indexes (one per label)
            for label in all_labels:
                index_name = f"vector_{label.lower()}"
                session.run(f"""
                    CREATE VECTOR INDEX {index_name}
                    FOR (n:{label}) ON (n.embedding)
                    OPTIONS {{indexConfig: {{
                    `vector.dimensions`: 1536,
                    `vector.similarity_function`: 'cosine'
                    }}}}
                """)
            

            # 4. Vectorize nodes (same as before)
            nodes = session.run("MATCH (n) WHERE n.embedding IS NULL AND n.name IS NOT NULL RETURN n.name as name, id(n) as id")
            for record in nodes:
                name = record['name']
                node_id = record['id']
                embedding = self.client.embeddings.create(input=[name], model="text-embedding-3-small").data[0].embedding
                session.run("MATCH (n) WHERE id(n) = $id SET n.embedding = $embedding", id=node_id, embedding=embedding)
    def normalize_node_names(self):
        """Converts all node 'name' properties to lowercase."""
        query = """
        MATCH (n)
        WHERE n.name IS NOT NULL
        SET n.name = toLower(n.name)
        """
        with self.driver.session() as session:
            print("Normalizing all node names to lowercase...")
            session.run(query)
            print("Normalization complete.")
if __name__ == "__main__":
    script_dir = pathlib_Path(__file__).parent
    data_dir = script_dir / ".."/"../" /"../"/ "Pdata"
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
            uploader.create_hybrid_indexes()
            uploader.normalize_node_names()
            uploader.close()
            #uploader.create_summary_relationships() 
        connector = Neo4jConnector(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        #print(connector.get_full_graph())

        all_nodes_with_types = connector.get_dossier_for_any_entity('OmegaActual')
        all_nodes_with_types = connector.get_all_nodes_with_primary_type()
        s = connector.get_nodes_by_type('PostRequest')
        connector.close()

        
        print("\nData successfully uploaded to Neo4j.")