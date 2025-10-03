import os
import os.path as osp
from datasets import load_from_disk
import networkx as nx
from pyvis.network import Network
from typing import Literal
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


NodeType = Literal["directory", "file", "python file", "class def", "func def"]
node_types = NodeType.__args__


type_colors = {
    "directory": "#cccccc",   
    "file": "#ffffff",        
    "python file": "#99ccff", 
    "class def": "#ffcc99",   
    "func def": "#ffff99"     
}


edge_colors = {
    "contain": "blue",
    "call": "green",
    "superclasses": "red",
    "previous": "#ffcc00"  
}

def visualize_temporal_graph(repo_name, output_dir="visualizations"):
    
    os.makedirs(output_dir, exist_ok=True)
    
    
    data_path = osp.join("savedata/repos", repo_name)
    try:
        dataset = load_from_disk(data_path)
        print(f"Loaded dataset with {len(dataset)} nodes")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    
    G = nx.DiGraph()
    
    
    commit_times = set()
    
    
    for node in dataset:
        node_id = node['id']
        type_index = node['type']
        type_name = node_types[type_index]
        color = type_colors.get(type_name, "#ff99cc")
        
        
        label = f"{node_id}:{type_name}"
        if node['name']:
            label += f":{node['name'].split('.')[-1]}"
        elif type_name in ["file", "python file", "directory"]:
            label += f":{osp.basename(node['path'])}"
        
        
        G.add_node(node_id, 
                  label=label, 
                  color=color,
                  title=f"ID: {node_id}\nType: {type_name}\nPath: {node['path']}\nName: {node['name']}\nStart: {node['start_commit']}\nEnd: {node['end_commit']}",
                  shape="dot",
                  size=20,
                  start_commit=node['start_commit'],
                  end_commit=node['end_commit'])
        
        
        commit_times.add(node['start_commit'])
        if node['end_commit'] != "none":
            commit_times.add(node['end_commit'])

    
    for node in dataset:
        source = node['id']
        
        
        for target in node['contain']:
            if source != target and target in G.nodes:
                G.add_edge(source, target, color=edge_colors["contain"], title="contain")
        
        
        for target in node['call']:
            if source != target and target in G.nodes:
                G.add_edge(source, target, color=edge_colors["call"], title="call")
        
        
        for target in node['superclasses']:
            if source != target and target in G.nodes:
                G.add_edge(source, target, color=edge_colors["superclasses"], title="superclass")
        
        
        for target in node['previous']:
            if source != target and target in G.nodes:
                G.add_edge(source, target, color=edge_colors["previous"], title="previous", dashes=True)

    
    net = Network(
        notebook=False, 
        directed=True, 
        height="750px", 
        width="100%", 
        cdn_resources='in_line'
    )
    
    
    for node_id, node_data in G.nodes(data=True):
        net.add_node(node_id, **node_data)
    
    for edge in G.edges(data=True):
        net.add_edge(edge[0], edge[1], **edge[2])
    
    
    commits = sorted(list(commit_times))
    commit_positions = {commit: idx for idx, commit in enumerate(commits)}
    
    print(f"Found {len(commits)} unique commits")
    
    
    node_positions = {}
    level_nodes = defaultdict(list)
    
    
    for node in net.nodes:
        start_commit = node.get('start_commit', commits[0])
        if start_commit in commit_positions:
            level = commit_positions[start_commit]
            level_nodes[level].append(node['id'])
    
    
    max_nodes_per_level = max(len(nodes) for nodes in level_nodes.values()) if level_nodes else 1
    vertical_spacing = 150 
    level_offsets = {}
    
    
    for node in net.nodes:
        node_id = node['id']
        start_commit = node.get('start_commit', commits[0])
        end_commit = node.get('end_commit', "none")
        
        start_level = commit_positions.get(start_commit, 0)
        end_level = commit_positions.get(end_commit, len(commits) - 1) if end_commit != "none" else len(commits) - 1
        
       
        x = (start_level + end_level) / 2.0 * 150  
        
        
        if start_level not in level_offsets:
            level_offsets[start_level] = 0
        else:
            level_offsets[start_level] += 1
        
        
        y_base = level_offsets[start_level] / max(1, len(level_nodes.get(start_level, []))) * vertical_spacing
        y = y_base - vertical_spacing/2 + (node_id % 3) * 20  
        
        
        node['x'] = x
        node['y'] = y
        node['physics'] = False
        
        
        node_positions[node_id] = (x, y)
    
    
    timeline_options = {
        "timeline": {
            "enabled": True,
            "style": "range",  
            "minHeight": "120px",
            "maxHeight": "160px",
            "showCurrentTime": True,
            "format": {
                "minorLabels": {
                    "minute": 'h:mma',
                    "hour": 'hA',
                    "day": 'MMM Do',
                    "month": 'YYYY MMM',
                    "year": 'YYYY'
                }
            },
            "zoomable": True,
            "type": "point"
        }
    }
    
    
    physics_options = {
        "physics": {
            "enabled": False
        }
    }
    
   
    options = {
        "manipulation": {
            "enabled": True  
        },
        "nodes": {
            "borderWidth": 2,
            "borderWidthSelected": 4,
            "size": 25, 
            "font": {
                "size": 12,
                "face": "Arial"
            }
        },
        "edges": {
            "arrows": {
                "to": {
                    "enabled": True,
                    "scaleFactor": 0.5
                }
            },
            "smooth": {
                "type": "cubicBezier",
                "forceDirection": "horizontal",
                "roundness": 0.4
            },
            "color": {
                "inherit": False
            },
            "width": 1.5
        },
        "interaction": {
            "hover": True,
            "hoverConnectedEdges": True,
            "tooltipDelay": 100,
            "zoomView": True,
            "dragView": True,
            "navigationButtons": True
        },
        "configure": {
            "enabled": True,
            "filter": "physics,nodes,edges",
            "showButton": True
        }
    }
    
    
    options.update(physics_options)
    
   
    for i, commit in enumerate(commits):
        for node in net.nodes:
            if node.get('start_commit') == commit:
                node['start'] = i
            if node.get('end_commit') == commit:
                node['end'] = i
    
    
    net.set_options(json.dumps(options))
    
    
    output_file = osp.join(output_dir, f"{repo_name}_temporal_graph.html")
    
    try:
        net.save_graph(output_file)
        print(f"Temporal graph visualization saved to: {output_file}")
        
        
        with open(output_file, 'r') as f:
            content = f.read()
        
        
        custom_js = """
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            
            var network = document.querySelector('.vis-network').parentElement;
            var visData = network.nodes;
            
            
            var controls = document.createElement('div');
            controls.style = 'position: absolute; top: 10px; left: 10px; z-index: 999; background: rgba(255,255,255,0.8); padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);';
            controls.innerHTML = `
                <div style="font-weight: bold; margin-bottom: 10px;">Code Evolution Visualization</div>
                <div>
                    <label for="commit-selector">commit-selector:</label>
                    <select id="commit-selector" style="margin-left: 5px; padding: 3px;">
                        ${COMMITS_OPTIONS}
                    </select>
                </div>
                <div style="margin-top: 10px; display: flex; justify-content: space-between;">
                    <button id="prev-commit" style="padding: 5px 10px;">prev-commit</button>
                    <button id="next-commit" style="padding: 5px 10px; margin-left: 5px;">next-commit</button>
                </div>
                <div style="margin-top: 10px;">
                    <input type="checkbox" id="show-all-nodes" checked>
                    <label for="show-all-nodes">show-all-nodes</label>
                </div>
            `;
            
            document.body.appendChild(controls);
            
           
            var commits = COMMITS_LIST;
            var commitsOptions = '';
            commits.forEach((commit, index) => {
                commitsOptions += `<option value="${index}">${commit.slice(0, 7)}</option>`;
            });
            controls.querySelector('#commit-selector').innerHTML = commitsOptions;
            
           
            var networkInstance = null;
            var allNodes = null;
            var allEdges = null;
            
            
            setTimeout(() => {
                try {
                    
                    var container = document.getElementById('mynetwork');
                    networkInstance = container.vis.network;
                    allNodes = container.vis.data.nodes.get();
                    allEdges = container.vis.data.edges.get();
                    
                    
                    updateVisibleNodes(0);
                } catch (e) {
                    console.error("Unable to obtain network instance:", e);
                }
            }, 1000);
            
            
            var currentCommitIndex = 0;
            
            function updateVisibleNodes(commitIndex) {
                if (!networkInstance || !allNodes) return;
                
                currentCommitIndex = commitIndex;
                var selectedCommit = commits[commitIndex];
                var showAll = document.getElementById('show-all-nodes').checked;
                
                
                var updatedNodes = [];
                allNodes.forEach(node => {
                    var nodeStartCommit = node.start_commit;
                    var nodeEndCommit = node.end_commit;
                    
                    
                    var isVisible = showAll || 
                                    (commits.indexOf(nodeStartCommit) <= commitIndex && 
                                     (nodeEndCommit === "none" || commits.indexOf(nodeEndCommit) >= commitIndex));
                    
                    if (node.hidden !== !isVisible) {
                        node.hidden = !isVisible;
                        updatedNodes.push(node);
                    }
                });
                
                
                var updatedEdges = [];
                allEdges.forEach(edge => {
                    var fromNode = allNodes.find(n => n.id === edge.from);
                    var toNode = allNodes.find(n => n.id === edge.to);
                    
                    var isVisible = fromNode && toNode && !fromNode.hidden && !toNode.hidden;
                    
                    if (edge.hidden !== !isVisible) {
                        edge.hidden = !isVisible;
                        updatedEdges.push(edge);
                    }
                });
                
                
                if (updatedNodes.length > 0) {
                    container.vis.data.nodes.update(updatedNodes);
                }
                if (updatedEdges.length > 0) {
                    container.vis.data.edges.update(updatedEdges);
                }
                
                
                controls.querySelector('#commit-selector').value = commitIndex;
                controls.querySelector('#prev-commit').disabled = commitIndex === 0;
                controls.querySelector('#next-commit').disabled = commitIndex === commits.length - 1;
                
               
                var commitInfo = document.createElement('div');
                commitInfo.style = 'position: absolute; bottom: 10px; left: 10px; background: rgba(255,255,255,0.8); padding: 5px; border-radius: 3px;';
                commitInfo.innerHTML = `current commit: ${selectedCommit.slice(0, 7)}`;
                
                var oldInfo = document.getElementById('commit-info');
                if (oldInfo) oldInfo.remove();
                
                commitInfo.id = 'commit-info';
                document.body.appendChild(commitInfo);
            }
            
            controls.querySelector('#commit-selector').addEventListener('change', function(e) {
                updateVisibleNodes(parseInt(e.target.value));
            });
            
            controls.querySelector('#prev-commit').addEventListener('click', function() {
                if (currentCommitIndex > 0) {
                    updateVisibleNodes(currentCommitIndex - 1);
                }
            });
            
            controls.querySelector('#next-commit').addEventListener('click', function() {
                if (currentCommitIndex < commits.length - 1) {
                    updateVisibleNodes(currentCommitIndex + 1);
                }
            });
            
            controls.querySelector('#show-all-nodes').addEventListener('change', function() {
                updateVisibleNodes(currentCommitIndex);
            });
            
            
            document.addEventListener('keydown', function(e) {
                if (e.key === 'ArrowLeft') {
                    if (currentCommitIndex > 0) {
                        updateVisibleNodes(currentCommitIndex - 1);
                    }
                } else if (e.key === 'ArrowRight') {
                    if (currentCommitIndex < commits.length - 1) {
                        updateVisibleNodes(currentCommitIndex + 1);
                    }
                }
            });
        });
        </script>
        """.replace('COMMITS_OPTIONS', '').replace('COMMITS_LIST', json.dumps(commits))
        
        
        content = content.replace('</body>', f'{custom_js}\n</body>')
        
        with open(output_file, 'w') as f:
            f.write(content)
            
    except Exception as e:
        print(f"Error saving graph: {e}")
            
    
    fig, ax = plt.subplots(figsize=(15, 3))
    for i, commit in enumerate(commits):
        ax.text(i, 0.5, commit[:7], ha='center', va='center', fontsize=10, rotation=45)
        ax.plot([i, i], [0.3, 0.7], 'k-', lw=1)
    
    
    ax.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
    
   
    ax.set_xlim(-0.5, len(commits)-0.5)
    ax.set_ylim(0, 1)
    ax.set_title('Commit Timeline', fontsize=14)
    ax.axis('off')
    
    
    timeline_file = osp.join(output_dir, f"{repo_name}_commit_timeline.png")
    plt.savefig(timeline_file, bbox_inches='tight', dpi=200)
    print(f"Commit timeline saved to: {timeline_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize temporal code element graph")
    parser.add_argument("repo_name", type=str, help="Name of the repository to visualize")
    args = parser.parse_args()
    
    visualize_temporal_graph(args.repo_name)