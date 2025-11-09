from flask import Flask, jsonify, request, g
from neo4j import GraphDatabase
from flask_cors import CORS
import sys
import os
import requests
import logging
import time
import json
from neo4j.graph import Node, Relationship, Path
from neo4j.time import Date, DateTime, Time, Duration
from neo4j.spatial import Point

# --- START: LOGGING CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# --- END: LOGGING CONFIGURATION ---

# Load credentials from Environment Variables
#NEO4J_URI = os.getenv("NEO4J_URI", "bolt://3.210.202.54:7687")
#NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
#NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "truck-question-distribution")
#GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

NEO4J_URI = "bolt://98.92.215.12:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "respiration-carbon-tack"
GEMINI_API_KEY = "AIzaSyC9qqyhaCGRM_dIR_s-8HddLgWExTgW85E" # Replace with your key if needed
app = Flask(__name__)
CORS(app)

schema_cache = None

# --- START: REQUEST/RESPONSE LOGGING ---
@app.before_request
def before_request_logging():
    g.start_time = time.time()
    logging.info(f"--> Incoming Request: {request.method} {request.path}")
    if request.data and request.is_json:
        try:
            logging.info(f"Request Body: {json.dumps(request.get_json(), indent=2)}")
        except Exception:
            logging.info("Request Body: (Non-JSON data)")

@app.after_request
def after_request_logging(response):
    duration = time.time() - g.start_time
    response_data = response.get_data(as_text=True)
    logging.info(f"<-- Outgoing Response: {request.method} {request.path} - Status {response.status_code} ({duration:.4f}s)")
    
    # Truncate long responses for cleaner logs
    if len(response_data) > 1000:
         logging.info(f"Response Body (truncated): {response_data[:1000]}...")
    else:
         logging.info(f"Response Body: {response_data}")
    return response
# --- END: REQUEST/RESPONSE LOGGING ---

try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    logging.info("✅ Successfully connected to Neo4j!")
except Exception as e:
    logging.error(f"❌ Failed to connect to Neo4j. Error: {e}")
    sys.exit(1)

# --- START: NEW SAVE ONTOLOGY ENDPOINT ---
@app.route('/api/save-ontology', methods=['POST'])
def save_ontology():
    """
    Receives ontology data and a filename, saves it to a local 'ontologies' folder.
    NOTE: For a cloud deployment (like Cloud Run), you would replace the local file
    saving logic with code that uploads to a Google Cloud Storage bucket.
    """
    try:
        data = request.get_json()
        filename = data.get('filename')
        ontology_data = data.get('ontologyData')

        if not filename or not ontology_data:
            return jsonify({"error": "Filename and ontologyData are required."}), 400

        # Sanitize filename to prevent directory traversal attacks
        safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_')).rstrip()
        if not safe_filename.endswith('.json'):
            safe_filename += '.json'
            
        # *** FIX: Save relative to the script's directory for reliability ***
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(script_dir, 'ontologies')
        os.makedirs(save_dir, exist_ok=True)
        
        file_path = os.path.join(save_dir, safe_filename)
        
        with open(file_path, 'w') as f:
            json.dump(ontology_data, f, indent=2)
            
        logging.info(f"✅ Ontology saved successfully to {file_path}")
        
        # Here you could also update a central manifest file if needed
        
        return jsonify({"message": f"Ontology saved successfully as {safe_filename}"}), 200

    except Exception as e:
        logging.error(f"❌ Error in /api/save-ontology: {e}")
        return jsonify({"error": "An error occurred while saving the ontology."}), 500
# --- END: NEW SAVE ONTOLOGY ENDPOINT ---

def serialize_properties(props):
    """
    Converts Neo4j specific data types (Date, Point, etc.) to JSON serializable strings.
    """
    serialized = {}
    for key, value in props.items():
        if isinstance(value, (Date, DateTime, Time, Duration)):
            serialized[key] = str(value)
        elif isinstance(value, Point):
            serialized[key] = f"Point(srid={value.srid}, x={value.x}, y={value.y})"
        else:
            serialized[key] = value
    return serialized

def get_detailed_schema(tx):
    """
    Generates a detailed schema of the graph, including node properties and relationship connections.
    """
    schema_str = "This is the graph schema:\n\n"
    nodes_properties = {}
    labels_result = tx.run("CALL db.labels()")
    for record in labels_result:
        label = record["label"]
        properties_result = tx.run(f"MATCH (n:`{label}`) WITH n LIMIT 1 UNWIND keys(n) as key RETURN collect(distinct key) as props").single()
        if properties_result and properties_result["props"]:
            nodes_properties[label] = properties_result["props"]
    schema_str += "Node Properties:\n"
    for label, props in nodes_properties.items():
        schema_str += f"- {label}: {props}\n"
    schema_str += "\n"
    
    rels_result = tx.run("""
        MATCH (a)-[r]->(b)
        UNWIND labels(a) AS start_label
        UNWIND labels(b) AS end_label
        RETURN DISTINCT start_label, type(r) AS rel_type, end_label
    """)
    rel_set = set()
    for record in rels_result:
        rel_set.add(f"- (:{record['start_label']})-[:{record['rel_type']}]->(:{record['end_label']})")

    schema_str += "Relationship Details (Connections):\n"
    for rel_string in sorted(list(rel_set)):
        schema_str += f"{rel_string}\n"
    return schema_str

def get_schema_graph(tx):
    """
    Generates a graph representation of the schema for visualization.
    Nodes are labels, edges are relationship types.
    This version uses a direct query for higher reliability.
    """
    logging.info("--- Building Schema Graph ---")
    nodes_map = {}
    edges = []
    
    # 1. Get all node labels and their properties first. This ensures even isolated nodes are included.
    all_labels_result = tx.run("CALL db.labels() YIELD label")
    for record in all_labels_result:
        label = record["label"]
        if label not in nodes_map:
            logging.info(f"Found node label: {label}")
            nodes_map[label] = {"id": label, "label": label, "group": label, "description": "No properties found.", "properties": {}}
            # Get properties for this node type
            properties_result = tx.run(f"MATCH (n:`{label}`) WITH n LIMIT 100 UNWIND keys(n) as key RETURN collect(distinct key) as props").single()
            if properties_result and properties_result["props"]:
                props_list = sorted(properties_result["props"])
                nodes_map[label]["description"] = "Properties: " + ", ".join(props_list)
                logging.info(f" -> Properties for {label}: {props_list}")

    # 2. Get all unique relationship connections between labels
    rels_query = """
    MATCH (a)-[r]->(b)
    WHERE a <> b
    UNWIND labels(a) AS start_label
    UNWIND labels(b) AS end_label
    RETURN DISTINCT start_label, type(r) AS rel_type, end_label
    """
    rels_result = tx.run(rels_query)
    
    rel_count = 0
    for record in rels_result:
        start, rel_type, end = record["start_label"], record["rel_type"], record["end_label"]
        logging.info(f"Found relationship pattern: ({start})-[:{rel_type}]->({end})")
        edges.append({
            "from": start,
            "to": end,
            "label": rel_type,
            "id": f"{start}-{rel_type}-{end}"
        })
        rel_count += 1
    
    logging.info(f"Found {len(nodes_map)} node types and {rel_count} unique relationship patterns.")
    
    final_graph = {"nodes": list(nodes_map.values()), "edges": edges}
    
    return final_graph

def call_gemini(prompt):
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    # NEW CODE (from your AI Studio)
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.0}}
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        if result.get("candidates"):
            return result["candidates"][0]["content"]["parts"][0]["text"].strip()
        return f"Error: The AI model's response was empty or blocked. Response: {result}"
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Gemini API request error: {e}")
        return f"Error communicating with the AI model: {e}"

@app.route('/api/ask', methods=['POST'])
def ask_ai():
    global schema_cache
    question = request.json.get('question')
    if not question:
        return jsonify({"error": "No question provided."}), 400
    try:
        with driver.session() as session:
            if not schema_cache:
                schema_cache = session.execute_read(get_detailed_schema)
            
            cypher_prompt = (
                f"You are a Neo4j expert who translates user questions into Cypher queries. You MUST follow these rules:\n"
                f"1. You MUST ONLY use the node labels, properties, and relationship types defined in the schema below.\n"
                f"2. Your response MUST be ONLY the Cypher query and nothing else.\n"
                f"3. **Prioritize Performance:** Always write the most performant query possible.\n"
                f"4. **Use `DISTINCT`:** When returning nodes from a path match, use `RETURN DISTINCT` to avoid duplicates.\n"
                f"5. **Prefer Anonymous Paths:** For variable-length paths, use anonymous relationships like `[*..5]` instead of named ones like `[r*..5]` unless the relationships in the path are explicitly asked for.\n"
                f"6. **Limit Path Depth:** For general 'find relationship' queries, use a reasonable maximum depth, such as 5, not 10 or more.\n\n"
                f"7. **Handle Ontology Labels:** When the user's question mentions a specific ontology (e.g., 'in the Movies_Ontology', 'from the Hollywood_Ontology'), that ontology name MUST be used as an additional node label in the query. For example, 'movies in Movies_Ontology' must be queried as `(n:Movie:Movies_Ontology)`.\n\n"
                f"8. **Use Integers for Numerical Comparisons:** When filtering on properties that are clearly numerical (like `born`, `year`, `age`, `count`), always use integer values. Do NOT put quotes around numbers in a `WHERE` clause (e.g., use `p.born < 1950`, NOT `p.born < '1950'`).\n\n"
                f"--- SCHEMA ---\n{schema_cache}\n--- END SCHEMA ---\n\n"
                f"--- EXAMPLES OF CORRECT, PERFORMANT QUERIES ---\n"
                f"# Example 1: How to query by date.\n"
                f"USER QUESTION: 'Which loans started in 2024?'\n"
                f"CYPHER QUERY: MATCH (l:Loan)-[:HAS_START_DATE]->(d:Date) WHERE d.label STARTS WITH '2024' RETURN l.label\n\n"
                f"# Example 2: How to query for parties by risk rating and find related product types.\n"
                f"USER QUESTION: 'What are the product types for loans given to parties with a medium risk rating?'\n"
                f"CYPHER QUERY: MATCH (p:Party)-[:PARTICIPATES_IN]->(l:Loan)-[:IS_PRODUCT_TYPE]->(pt:ProductType) WHERE p.description CONTAINS 'Risk Rating: Medium' RETURN pt.label\n\n"
                f"# Example 3: How to perform calculations on string properties.\n"
                f"USER QUESTION: 'What is the total position quantity of all loans?'\n"
                f"CYPHER QUERY: MATCH (:Loan)-[:HAS_POSITION]->(p:Position) RETURN sum(toInteger(trim(split(p.description, ':')[1])))\n\n"
                f"# Example 4: How to find the relationship between two entities.\n"
                f"USER QUESTION: 'How is Singapore Logistics related to Bob Williams?'\n"
                f"CYPHER QUERY: MATCH (a:Party {{label: 'Singapore Logistics'}}), (b:Party {{label: 'Bob Williams'}}), p = shortestPath((a)-[*..5]-(b)) RETURN p\n\n"
                f"# Example 5: How to write a performant query to find connected nodes.\n"
                f"USER QUESTION: 'which corporate entities have any relationship to events type Tariff?'\n"
                f"CYPHER QUERY: MATCH (c:ClientCorporate)-[*..5]-(e:Event {{type:\"Tariff\"}}) RETURN DISTINCT c\n\n"
                f"# Example 6: How to count nodes within a specific ontology.\n"
                f"USER QUESTION: 'how many movies are in the Movies_Ontology?'\n"
                f"CYPHER QUERY: MATCH (m:Movie:Movies_Ontology) RETURN count(m)\n\n"
                f"# Example 7: How to list nodes from a specific ontology.\n"
                f"USER QUESTION: 'List all people from the Hollywood_Ontology.'\n"
                f"CYPHER QUERY: MATCH (p:Person:Hollywood_Ontology) RETURN p.name\n\n"
                f"# Example 8: How to filter on a numerical property (integer).\n"
                f"USER QUESTION: 'Which people were born before 1950?'\n"
                f"CYPHER QUERY: MATCH (p:Person) WHERE p.born < 1950 RETURN p.name\n\n"
                f"--- END EXAMPLES ---\n\n"
                f"--- CURRENT TASK ---\n"
                f"USER QUESTION: \"{question}\"\n"
                f"CYPHER QUERY:"
            )

            generated_cypher_response = call_gemini(cypher_prompt)
            cypher_query = ""
            if "```" in generated_cypher_response:
                parts = generated_cypher_response.split("```")
                if len(parts) > 1:
                    cypher_query = parts[1]
                    if cypher_query.lower().lstrip().startswith('cypher'):
                        cypher_query = cypher_query.lstrip()[6:].strip()
            else:
                cypher_query = generated_cypher_response
            cypher_query = cypher_query.strip()
            logging.info(f"🤖 Parsed Cypher: {cypher_query}")
            if not cypher_query.upper().startswith("MATCH"):
                 return jsonify({"answer": "Sorry, I can only perform read-only queries. The AI did not return a valid query."})
            
            result = session.run(cypher_query).data()
            
            final_answer = ""
            if result:
                answer_prompt = (
                    f"You are a helpful assistant. A user asked the following question: '{question}'.\n"
                    f"The database returned this data as the answer: {result}.\n"
                    f"Please summarize this data into a clear, natural language sentence. Do not mention the user's original question. "
                    f"For example, if the data is a list of loans, say 'The following loans were found: L001, L002.' "
                    f"If the data is a single count, say 'The total count is X.'."
                )
                final_answer = call_gemini(answer_prompt)
            else:
                final_answer = "No results were found in the database for your query."

            return jsonify({"answer": final_answer})

    except Exception as e:
        logging.error(f"❌ Error in /api/ask: {e}")
        return jsonify({"error": "An error occurred while processing your request."}), 500

def get_graph_data(tx):
    """
    Fetches all nodes and relationships from the database, ensuring all node labels are included.
    """
    nodes = {}
    edges = []
    nodes_result = tx.run("MATCH (n) RETURN n")
    for record in nodes_result:
        node = record["n"]
        node_id = str(node.element_id)
        if node_id not in nodes:
            node_labels = list(node.labels)
            props = serialize_properties(dict(node))
            nodes[node_id] = {
                "id": node_id,
                "label": props.get('label', props.get('name', 'Unnamed')),
                "group": node_labels[0] if node_labels else "Default",
                "labels": node_labels,
                "properties": props
            }
    edges_result = tx.run("MATCH ()-[r]->() RETURN r")
    for record in edges_result:
        rel = record["r"]
        edges.append({
            "from": str(rel.start_node.element_id),
            "to": str(rel.end_node.element_id),
            "label": rel.type,
            "properties": serialize_properties(dict(rel))
        })
    return {"nodes": list(nodes.values()), "edges": edges}

@app.route('/api/graph', methods=['POST'])
def fetch_graph():
    try:
        with driver.session() as session:
            return jsonify(session.execute_read(get_graph_data))
    except Exception as e:
        logging.error(f"❌ Error querying Neo4j: {e}")
        return "Error connecting to the database.", 500

@app.route('/api/schema', methods=['GET'])
def fetch_schema():
    try:
        with driver.session() as session:
            schema_data = session.execute_read(get_schema_graph)
            return jsonify(schema_data)
    except Exception as e:
        logging.error(f"❌ Error fetching schema graph: {e}")
        return jsonify({"error": "Error connecting to the database to fetch schema."}), 500
        
def parse_neo4j_graph_objects(records):
    """
    Parses records containing Neo4j graph objects (Nodes, Relationships, Paths)
    into a format suitable for the frontend viewer.
    """
    nodes = {}
    edges = []
    
    def add_node(node):
        node_id = str(node.element_id)
        if node_id not in nodes:
            node_labels = list(node.labels)
            props = serialize_properties(dict(node))
            nodes[node_id] = {
                "id": node_id,
                "label": props.get('label', props.get('name', 'Unnamed')),
                "group": node_labels[0] if node_labels else "Default",
                "labels": node_labels,
                "properties": props
            }

    def add_edge(rel):
        start_id = str(rel.start_node.element_id)
        end_id = str(rel.end_node.element_id)
        edge = {
            "id": str(rel.element_id),
            "from": start_id,
            "to": end_id,
            "label": rel.type,
            "properties": serialize_properties(dict(rel))
        }
        # Avoid duplicate edges
        if not any(e['id'] == edge['id'] for e in edges):
            edges.append(edge)

    for record in records:
        for value in record.values():
            if isinstance(value, Path):
                for node in value.nodes:
                    add_node(node)
                for rel in value.relationships:
                    add_edge(rel)
            elif isinstance(value, Node):
                add_node(value)
            elif isinstance(value, Relationship):
                add_node(value.start_node)
                add_node(value.end_node)
                add_edge(value)
            elif isinstance(value, list):
                 for item in value:
                    if isinstance(item, Node):
                        add_node(item)
                    elif isinstance(item, Relationship):
                        add_node(item.start_node)
                        add_node(item.end_node)
                        add_edge(item)

    return {"nodes": list(nodes.values()), "edges": edges}

def process_results_for_table(keys, records):
    """
    Processes Neo4j results into a structure with ordered headers and rows.
    """
    if not records:
        return {"headers": [], "rows": []}

    # Step 1: Generate the ordered list of headers from the first record.
    headers = []
    header_counts = {}
    first_record = records[0]
    for key in keys:
        value = first_record[key]
        header_name = key

        if isinstance(value, Node):
            header_name = list(value.labels)[0] if value.labels else "Node"
        elif isinstance(value, Relationship):
            header_name = "Relationship"
        
        if header_name in header_counts:
            header_counts[header_name] += 1
            final_header = f"{header_name}_{header_counts[header_name]}"
        else:
            header_counts[header_name] = 1
            final_header = header_name
        headers.append(final_header)

    # Step 2: Generate the rows as lists of lists, ensuring order is preserved.
    rows = []
    for record in records:
        row_data = []
        for key in keys: # Iterate in the original key order (e.g., a, b, c)
            value = record[key]
            cell_data = ""

            if isinstance(value, Node):
                props = dict(value)
                cell_data = props.get('label', props.get('name', 'Unnamed Node'))
            elif isinstance(value, Relationship):
                cell_data = value.type
            else:
                serialized = serialize_properties({key: value})
                cell_data = serialized.get(key)
            
            row_data.append(cell_data)
        rows.append(row_data)
        
    return {"headers": headers, "rows": rows}


@app.route('/api/query', methods=['POST'])
def run_query():
    query = request.json.get('query')
    if not query:
        return jsonify({"error": "No query provided."}), 400
    
    try:
        with driver.session() as session:
            logging.info(f"Executing Cypher: {query}")
            result_cursor = session.run(query)
            keys = result_cursor.keys()
            # This is the critical change: we pass the raw records, not .data()
            results = list(result_cursor) 

            if not results:
                return jsonify({"type": "table", "data": {"headers": [], "rows": []}})

            # Heuristic: If the result contains a path, treat it as a graph. Otherwise, a table.
            is_path_query = False
            for val in results[0].values():
                if isinstance(val, Path):
                    is_path_query = True
                    break

            if is_path_query:
                # We need to convert to dicts here for the graph parser
                graph_records = [r.data() for r in results]
                graph_data = parse_neo4j_graph_objects(graph_records)
                return jsonify({"type": "graph", "data": graph_data})
            else:
                table_data_obj = process_results_for_table(keys, results)
                return jsonify({"type": "table", "data": table_data_obj})

    except Exception as e:
        logging.error(f"❌ Error in /api/query: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        logging.critical("❌ FATAL: GEMINI_API_KEY is not set or is still the placeholder value.")
        sys.exit(1)
    app.run(port=3000)

