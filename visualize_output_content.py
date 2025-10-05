#!/usr/bin/env python3
"""
Script to map AskDocs.jsonl with Provided_medscore_output.jsonl and create HTML visualization.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Any

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def map_data(askdocs_data: List[Dict], medscore_data: List[Dict]) -> Dict[str, Dict]:
    """Map the data from both files using id as the key."""
    # Create a mapping from id to askdocs data
    askdocs_map = {item['id']: item for item in askdocs_data}
    
    # Create a mapping from id to medscore data
    medscore_map = {item['id']: item for item in medscore_data}
    
    # Combine the data
    combined_data = {}
    for id_key in askdocs_map:
        if id_key in medscore_map:
            combined_data[id_key] = {
                'askdocs': askdocs_map[id_key],
                'medscore': medscore_map[id_key]
            }
    
    return combined_data

def group_claims_by_sentence(claims: List[Dict]) -> Dict[str, List[Dict]]:
    """Group claims by sentence."""
    sentence_groups = defaultdict(list)
    for claim in claims:
        sentence = claim['sentence']
        sentence_groups[sentence].append(claim)
    return dict(sentence_groups)

def generate_html(combined_data: Dict[str, Dict], output_path: str):
    """Generate HTML visualization with hierarchical structure."""
    
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MedScore Visualization - Response Analysis</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }
        
        .content {
            padding: 30px;
        }
        
        .response-card {
            background: #fff;
            border: 1px solid #e1e5e9;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            overflow: hidden;
        }
        
        .response-header {
            background: #f8f9fa;
            padding: 20px;
            border-bottom: 1px solid #e1e5e9;
            cursor: pointer;
            user-select: none;
            transition: background-color 0.3s ease;
        }
        
        .response-header:hover {
            background: #e9ecef;
        }
        
        .response-header::after {
            content: '▼';
            float: right;
            transition: transform 0.3s ease;
        }
        
        .response-header.collapsed::after {
            transform: rotate(-90deg);
        }
        
        .response-content {
            transition: max-height 0.3s ease;
            overflow: hidden;
        }
        
        .response-content.collapsed {
            max-height: 0;
        }
        
        .response-id {
            font-size: 0.9em;
            color: #6c757d;
            margin-bottom: 10px;
        }
        
        .response-question {
            font-size: 1.1em;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 15px;
        }
        
        .response-text {
            background: #e8f4fd;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
            font-style: italic;
            color: #2c3e50;
        }
        
        .sentence-group {
            margin: 20px 0;
            border: 1px solid #e1e5e9;
            border-radius: 6px;
            overflow: hidden;
        }
        
        .sentence-header {
            background: #f1f3f4;
            padding: 15px;
            border-bottom: 1px solid #e1e5e9;
            font-weight: 600;
            color: #495057;
            cursor: pointer;
            user-select: none;
            transition: background-color 0.3s ease;
        }
        
        .sentence-header:hover {
            background: #e9ecef;
        }
        
        .sentence-header::after {
            content: '▼';
            float: right;
            transition: transform 0.3s ease;
        }
        
        .sentence-header.collapsed::after {
            transform: rotate(-90deg);
        }
        
        .sentence-content {
            transition: max-height 0.3s ease;
            overflow: hidden;
        }
        
        .sentence-content.collapsed {
            max-height: 0;
        }
        
        .sentence-text {
            background: #fff;
            padding: 15px;
            border-bottom: 1px solid #e1e5e9;
            font-style: italic;
            color: #6c757d;
        }
        
        .claims-container {
            background: #f8f9fa;
        }
        
        .claim-item {
            padding: 15px;
            border-bottom: 1px solid #e1e5e9;
            background: white;
            margin: 5px;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .claim-item:last-child {
            border-bottom: none;
        }
        
        .claim-text {
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .claim-details {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 10px;
        }
        
        .claim-detail {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            border-left: 3px solid #3498db;
        }
        
        .claim-detail.evidence {
            border-left-color: #e74c3c;
        }
        
        .claim-detail.raw {
            border-left-color: #f39c12;
        }
        
        .claim-detail.score {
            border-left-color: #27ae60;
        }
        
        .claim-detail-label {
            font-size: 0.8em;
            font-weight: 600;
            color: #6c757d;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        
        .claim-detail-value {
            color: #2c3e50;
            font-size: 0.9em;
        }
        
        .score-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 600;
            color: white;
        }
        
        .score-1 {
            background: #27ae60;
        }
        
        .score-0 {
            background: #e74c3c;
        }
        
        .stats {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
            margin-bottom: 30px;
            text-align: center;
        }
        
        .stats h3 {
            margin: 0 0 15px 0;
            color: #2c3e50;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }
        
        .stat-item {
            background: white;
            padding: 15px;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .stat-number {
            font-size: 2em;
            font-weight: 600;
            color: #3498db;
        }
        
        .stat-label {
            color: #6c757d;
            font-size: 0.9em;
        }
        
        @media (max-width: 768px) {
            .claim-details {
                grid-template-columns: 1fr;
            }
            
            .stats-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>MedScore Response Analysis</h1>
            <p>Hierarchical visualization of medical responses with claim analysis</p>
        </div>
        
        <div class="content">
"""
    
    # Calculate statistics
    total_responses = len(combined_data)
    total_sentences = 0
    total_claims = 0
    total_score = 0
    
    for response_data in combined_data.values():
        medscore_data = response_data['medscore']
        claims = medscore_data.get('claims', [])
        total_claims += len(claims)
        total_score += medscore_data.get('score', 0)
        
        # Count unique sentences
        sentences = set(claim['sentence'] for claim in claims)
        total_sentences += len(sentences)
    
    avg_score = total_score / total_responses if total_responses > 0 else 0
    
    # Add statistics section
    html_content += f"""
            <div class="stats">
                <h3>Dataset Statistics</h3>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-number">{total_responses}</div>
                        <div class="stat-label">Total Responses</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">{total_sentences}</div>
                        <div class="stat-label">Total Sentences</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">{total_claims}</div>
                        <div class="stat-label">Total Claims</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">{avg_score:.2f}</div>
                        <div class="stat-label">Average Score</div>
                    </div>
                </div>
            </div>
    """
    
    # Generate content for each response
    for response_id, response_data in combined_data.items():
        askdocs = response_data['askdocs']
        medscore = response_data['medscore']
        
        # Group claims by sentence
        claims = medscore.get('claims', [])
        sentence_groups = group_claims_by_sentence(claims)
        
        # Clean response_id for use in HTML IDs
        clean_response_id = response_id.replace('-', '_').replace(' ', '_')
        
        html_content += f"""
            <div class="response-card">
                <div class="response-header" id="response-header-{clean_response_id}" onclick="toggleResponse('{clean_response_id}')">
                    <div class="response-id">ID: {response_id}</div>
                    <div class="response-question">{askdocs.get('question', 'No question available')}</div>
                </div>
                
                <div class="response-content" id="response-content-{clean_response_id}">
                    <div class="response-text">
                        {askdocs.get('response', 'No response available')}
                    </div>
        """
        
        # Add each sentence group
        sentence_index = 0
        for sentence, sentence_claims in sentence_groups.items():
            # Create a unique sentence ID
            sentence_id = f"sentence_{sentence_index}"
            sentence_index += 1
            
            html_content += f"""
                <div class="sentence-group">
                    <div class="sentence-header" id="sentence-header-{clean_response_id}-{sentence_id}" onclick="toggleSentence('{clean_response_id}', '{sentence_id}')">Sentence Analysis</div>
                    <div class="sentence-content" id="sentence-content-{clean_response_id}-{sentence_id}">
                        <div class="sentence-text">"{sentence}"</div>
                        <div class="claims-container">
            """
            
            # Add each claim in this sentence
            for claim in sentence_claims:
                score_class = "score-1" if claim.get('score', 0) == 1.0 else "score-0"
                html_content += f"""
                        <div class="claim-item">
                            <div class="claim-text">{claim.get('claim', 'No claim available')}</div>
                            <div class="claim-details">
                                <div class="claim-detail evidence">
                                    <div class="claim-detail-label">Evidence</div>
                                    <div class="claim-detail-value">{claim.get('evidence', 'No evidence available')}</div>
                                </div>
                                <div class="claim-detail raw">
                                    <div class="claim-detail-label">Raw Assessment</div>
                                    <div class="claim-detail-value">{claim.get('raw', 'No raw assessment available')}</div>
                                </div>
                                <div class="claim-detail score">
                                    <div class="claim-detail-label">Score</div>
                                    <div class="claim-detail-value">
                                        <span class="score-badge {score_class}">{claim.get('score', 0)}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                """
            
            html_content += """
                        </div>
                    </div>
                </div>
            """
        
        html_content += """
                </div>
            </div>
        """
    
    html_content += """
        </div>
    </div>
    
    <script>
        // Function to toggle response content
        function toggleResponse(responseId) {
            const header = document.getElementById('response-header-' + responseId);
            const content = document.getElementById('response-content-' + responseId);
            
            if (content.classList.contains('collapsed')) {
                content.classList.remove('collapsed');
                header.classList.remove('collapsed');
            } else {
                content.classList.add('collapsed');
                header.classList.add('collapsed');
            }
        }
        
        // Function to toggle sentence content
        function toggleSentence(responseId, sentenceId) {
            const header = document.getElementById('sentence-header-' + responseId + '-' + sentenceId);
            const content = document.getElementById('sentence-content-' + responseId + '-' + sentenceId);
            
            if (content.classList.contains('collapsed')) {
                content.classList.remove('collapsed');
                header.classList.remove('collapsed');
            } else {
                content.classList.add('collapsed');
                header.classList.add('collapsed');
            }
        }
        
        // Add click event listeners when page loads
        document.addEventListener('DOMContentLoaded', function() {
            // All responses start expanded
            console.log('Page loaded with dropdown functionality');
        });
    </script>
</body>
</html>
    """
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML visualization saved to: {output_path}")

def main():
    """Main function to execute the mapping and visualization."""
    # File paths
    askdocs_path = "/Users/that.phamvan/my_ws/master/med-score-small-llm/data/AskDocs.jsonl"
    medscore_path = "/Users/that.phamvan/my_ws/master/med-score-small-llm/MedScore_baseline_results/Provided_medscore_output.jsonl"
    output_path = "/Users/that.phamvan/my_ws/master/med-score-small-llm/medscore_visualization_dropdown.html"
    
    print("Loading AskDocs data...")
    askdocs_data = load_jsonl(askdocs_path)
    print(f"Loaded {len(askdocs_data)} records from AskDocs")
    
    print("Loading MedScore data...")
    medscore_data = load_jsonl(medscore_path)
    print(f"Loaded {len(medscore_data)} records from MedScore")
    
    print("Mapping data...")
    combined_data = map_data(askdocs_data, medscore_data)
    print(f"Mapped {len(combined_data)} records")
    
    print("Generating HTML visualization...")
    generate_html(combined_data, output_path)
    print("Visualization complete!")

if __name__ == "__main__":
    main()
