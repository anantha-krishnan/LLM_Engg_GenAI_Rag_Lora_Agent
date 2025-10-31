import networkx as nx
from lxml import etree
from pathlib import Path
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import write_dot
import textwrap

def export_graph_to_dot(G, output_filename:Path="graph.dot"):
    """
    Exports the NetworkX graph to a .dot file that Graphviz can read,
    handling attribute name conflicts.
    """
    H = G.copy()

    # Prepare node attributes for Graphviz
    for node, data in H.nodes(data=True):
        name = data.get('name', node)
        if data.get('label') == 'Simulation':
            name = f"Simulation\n({data.get('type')})"
        data['label'] = '\n'.join(textwrap.wrap(name, width=15))

        color_map = {
            'Body': '#88CCEE', 'Joint': '#DDCC77', 'Motion': '#CC6677',
            'OutputRequest': '#44AA99', 'Simulation': '#AAAAAA'
        }
        data['fillcolor'] = color_map.get(data.get('label', 'Body'), '#808080')
        data['style'] = 'filled'
        data['shape'] = 'box'
        data['fontname'] = 'Helvetica'
        
        if 'name' in data:
            del data['name']

    # Prepare edge attributes for Graphviz
    for u, v, data in H.edges(data=True):
        data['fontcolor'] = 'firebrick'
        data['fontname'] = 'Helvetica'

    # Write the MultiDiGraph directly
    write_dot(H, output_filename)
    
    print(f"\nGraph structure exported to: {output_filename}")
    print(f"Run the following command in your terminal to generate the image:")
    print(f"dot -Tpng {output_filename} -o {output_filename.parent}/knowledge_graph_professional.png")

def parse_xml_to_graph(xml_file_path):
    """
    Parses the MotionSolve XML file and builds a NetworkX graph.
    """
    G = nx.MultiDiGraph()

    xml_file_path = str(xml_file_path)
    # Use a robust parser
    parser = etree.XMLParser(recover=True)
    tree = etree.parse(xml_file_path, parser)
    root = tree.getroot()

    # --- 1. Find the Simulation Command ---
    sim_command = root.find('.//Command/Simulate')
    if sim_command is not None:
        sim_type = sim_command.get('analysis_type')
        end_time = sim_command.get('end_time')
        G.add_node('Simulation_Run', 
                   label='Simulation',
                   type=sim_type,
                   end_time=float(end_time),
                   source_file=xml_file_path.split('/')[-1])

    # --- 2. Extract All Rigid Bodies ---
    bodies = root.findall('.//Model/Body_Rigid')
    for body in bodies:
        body_id = body.get('id')
        body_label = body.get('label')
        is_ground = body.get('IsGround', 'FALSE') == 'TRUE'
        
        G.add_node(body_id, 
                   label='Body',
                   name=body_label,
                   mass=float(body.get('mass', 0.0)),
                   inertia_xx=float(body.get('inertia_xx', 0.0)),
                   inertia_yy=float(body.get('inertia_yy', 0.0)),
                   inertia_zz=float(body.get('inertia_zz', 0.0)),
                   is_ground=is_ground)

    # --- 3. Extract All Joints and Create Connections ---
    joints = root.findall('.//Model/Constraint_Joint')
    
    # Helper dict to map markers to bodies
    marker_to_body = {m.get('id'): m.get('body_id') for m in root.findall('.//Model/Reference_Marker')}
    
    for joint in joints:
        joint_id = joint.get('id')
        joint_label = joint.get('label')
        joint_type = joint.get('type')
        
        i_marker_id = joint.get('i_marker_id')
        j_marker_id = joint.get('j_marker_id')
        
        body1_id = marker_to_body.get(i_marker_id)
        body2_id = marker_to_body.get(j_marker_id)

        # Add the joint as a node itself
        G.add_node(joint_id, 
                   label='Joint',
                   name=joint_label,
                   type=joint_type,
                   i_marker_id=i_marker_id,
                   j_marker_id=j_marker_id)
        
        # Add edges representing the physical connection
        if body1_id and body2_id:
            G.add_edge(body1_id, joint_id, label='CONNECTS_TO')
            G.add_edge(joint_id, body2_id, label='CONNECTS_TO')

    # --- 4. Extract Motions and link them to Joints ---
    motions = root.findall('.//Model/Motion_Joint')
    for motion in motions:
        motion_id = motion.get('id')
        motion_label = motion.get('label')
        motion_expr = motion.get('expr')
        target_joint_id = motion.get('joint_id')
        motion_type = motion.get('type')
        motion_val_type = motion.get('val_type')

        G.add_node(motion_id,
                   label='Motion',
                   name=motion_label,
                   expression=motion_expr,
                   type=motion_type,
                   value_type=motion_val_type)
        
        if target_joint_id:
             G.add_edge(motion_id, target_joint_id, label='APPLIED_TO')

    # --- 5: Extract Output Requests with Full Context ---
    requests = root.findall('.//Model/Post_Request')
    for req in requests:
        req_id = req.get('id')
        req_label = req.get('label')
        req_type = req.get('type')
        
        # Get all relevant marker IDs
        i_marker_id = req.get('i_marker_id')
        j_marker_id = req.get('j_marker_id')
        ref_marker_id = req.get('ref_marker_id')
        
        # Add the request node with raw marker IDs as attributes for reference
        G.add_node(req_id,
                   label='Post_Request',
                   name=req_label,
                   measurement=req_type,
                   measures_marker=i_marker_id,
                   relative_to_marker=j_marker_id,
                   in_frame_of_marker=ref_marker_id)
        
        # Find the corresponding bodies for each marker
        body_i = marker_to_body.get(i_marker_id)
        body_j = marker_to_body.get(j_marker_id)
        body_ref = marker_to_body.get(ref_marker_id)

        # Create the graph relationships
        if body_i:
            # This edge answers "What is being measured?"
            G.add_edge(req_id, body_i, label=f'MEASURES {req_type}')
        
        if body_j:
            # This edge answers "What is it measured relative to?"
            G.add_edge(req_id, body_j, label='RELATIVE_TO')
        
        if body_ref:
            # This edge answers "In what coordinate system are the results expressed?"
            G.add_edge(req_id, body_ref, label='IN_FRAME_OF')
    return G

def enhance_graph_with_mdl(G, mdl_file_path):
    """
    Reads the MDL file to extract more info
    and add them to the existing graph.
    """
    return G  
    with open(mdl_file_path, 'r') as f:
        for line in f:
            # Simple parsing for Output requests
            if line.strip().startswith('*Output('):
                parts = [p.strip() for p in line.split(',')]
                output_var = parts[0].split('(')[1] # o_0
                output_name = parts[1].strip('"')   # "Output 0"
                output_type = parts[2]              # DISP
                target_type = parts[3]              # BODY
                target_name = parts[4].split('.')[0] # b_1 from b_1.l
                
                # Find the corresponding body node in the graph
                # Note: This is a simple mapping. A real system needs a robust
                # way to map MDL names (b_1) to XML IDs (10103). For now, we search.
                target_node_id = None
                for node, data in G.nodes(data=True):
                    if data.get('label') == 'Body' and target_name in data.get('name', ''):
                         target_node_id = node
                         break
                
                if target_node_id:
                    G.add_node(output_var, 
                               label='OutputRequest',
                               name=output_name,
                               measurement=output_type)
                    G.add_edge(output_var, target_node_id, label='MEASURES')
    return G


if __name__ == "__main__":
    # Example usage
    
    # --- File Paths ---
    # Assuming the script is in a folder, and Pdata is two levels up
    # Adjust this path if your file structure is different.
    # For this example, let's assume Pdata is in the same directory.
    script_dir = Path(__file__).parent
    data_dir = script_dir / ".."/"../" / "Pdata"
    XML_FILE = data_dir / "pairs_model.xml" 
    MDL_FILE = data_dir / "pairs_model.mdl"

    # --- Build the Graph ---
    # 1. Start with the structured XML data
    knowledge_graph = parse_xml_to_graph(XML_FILE)

    # 2. Enhance with high-level MDL data
    knowledge_graph = enhance_graph_with_mdl(knowledge_graph, MDL_FILE)

    # --- Print and Visualize ---
    print(f"Graph created with {knowledge_graph.number_of_nodes()} nodes and {knowledge_graph.number_of_edges()} edges.")
    print("\nNodes:")
    for node, data in knowledge_graph.nodes(data=True):
        print(f"- {node}: {data}")

    # --- Visualization with Human-Readable Names ---
    plt.figure(figsize=(16, 16)) # Increased size for better readability
    pos = nx.spring_layout(knowledge_graph, k=1.5, iterations=50) # Adjusted layout parameters

    # *** THE KEY CHANGE IS HERE ***
    # 1. Create a dictionary that maps node IDs to their 'name' attribute for labeling.
    #    We use data.get('name', node) as a fallback in case a node doesn't have a 'name'.
    node_labels = {node: data.get('name', node) for node, data in knowledge_graph.nodes(data=True)}

    # For the 'Simulation_Run' node, which has no 'name', let's make its label clearer.
    if 'Simulation_Run' in node_labels:
        sim_node_data = knowledge_graph.nodes['Simulation_Run']
        node_labels['Simulation_Run'] = f"{sim_node_data.get('label')}\n({sim_node_data.get('type')})"


    # 2. Pass this dictionary to the 'labels' argument in nx.draw().
    nx.draw(knowledge_graph, pos, 
            labels=node_labels, 
            with_labels=True,  # Still need this to activate drawing of labels
            node_size=3000, 
            node_color='skyblue', 
            font_size=9, 
            font_weight='bold',
            edge_color='gray')

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(knowledge_graph, 'label')
    nx.draw_networkx_edge_labels(knowledge_graph, pos, edge_labels=edge_labels, font_color='red')

    plt.title("MotionSolve Knowledge Graph")
    
    output_filename = data_dir / "knowledge_graph.png"
    plt.savefig(output_filename, format="PNG", dpi=300, bbox_inches='tight')
    print(f"\nGraph image saved to: {output_filename}")

    # It's good practice to close the plot to free up memory
    plt.close()
    export_graph_to_dot(knowledge_graph, data_dir / "my_graph.dot")

