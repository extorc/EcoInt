document.addEventListener('DOMContentLoaded', async () => {
    // 1. Initialize Cytoscape
    const cy = cytoscape({
        container: document.getElementById('cy'),
        style: [
            {
                selector: 'node[group="Entity"]',
                style: {
                    'background-color': '#3b82f6',
                    'label': 'data(label)',
                    'color': '#f8fafc',
                    'text-valign': 'bottom',
                    'text-halign': 'center',
                    'text-margin-y': '6px',
                    'font-size': '12px',
                    'font-family': 'Inter',
                    'width': '30px',
                    'height': '30px',
                    'border-width': 2,
                    'border-color': '#60a5fa',
                    'transition-property': 'background-color, border-width, transform',
                    'transition-duration': '0.2s',
                    'shape': 'round-rectangle'
                }
            },
            {
                selector: 'node[group="Article"]',
                style: {
                    'background-color': '#10b981',
                    'label': '',
                    'width': '16px',
                    'height': '16px',
                    'border-width': 2,
                    'border-color': '#34d399',
                    'shape': 'hexagon'
                }
            },
            {
                selector: 'node:selected',
                style: {
                    'border-color': '#fcd34d',
                    'border-width': 4
                }
            },
            {
                selector: 'edge',
                style: {
                    'width': 2,
                    'line-color': '#475569',
                    'curve-style': 'bezier',
                    'target-arrow-shape': 'triangle',
                    'target-arrow-color': '#475569',
                    'opacity': 0.6
                }
            }
        ],
        layout: {
            name: 'cose',
            idealEdgeLength: 250,
            nodeOverlap: 50,
            refresh: 20,
            fit: true,
            padding: 50,
            randomize: true,
            componentSpacing: 250,
            nodeRepulsion: 1000000,
            edgeElasticity: 50,
            nestingFactor: 5,
            gravity: 40,
            numIter: 2500,
            initialTemp: 300,
            coolingFactor: 0.98,
            minTemp: 1.0
        }
    });

    // 2. Fetch Nodes from API
    try {
        const response = await fetch('/api/nodes');
        const data = await response.json();
        
        const cyNodes = data.nodes.map(n => ({
            data: { 
                id: n.id, 
                label: n.label, 
                group: n.group, 
                description: n.description 
            }
        }));
        
        const cyEdges = (data.edges || []).map(e => ({
            data: {
                id: e.source + '-' + e.target,
                source: e.source,
                target: e.target
            }
        }));
        
        cy.add([...cyNodes, ...cyEdges]);
        cy.layout({ 
            name: 'cose',
            idealEdgeLength: 250,
            nodeOverlap: 50,
            refresh: 20,
            fit: true,
            padding: 50,
            randomize: true,
            componentSpacing: 250,
            nodeRepulsion: 1000000,
            edgeElasticity: 50,
            nestingFactor: 5,
            gravity: 40,
            numIter: 2500,
            initialTemp: 300,
            coolingFactor: 0.98,
            minTemp: 1.0
        }).run();
    } catch (err) {
        console.error("Failed to load nodes:", err);
    }

    // 3. Handle Node Clicks
    const modal = document.getElementById('node-modal');
    const titleEl = document.getElementById('node-title');
    const typeEl = document.getElementById('node-type');
    const descEl = document.getElementById('node-desc');
    const closeBtn = document.getElementById('close-modal');
    const ingestBtn = document.getElementById('ingest-btn');
    const mergeBtn = document.getElementById('merge-btn');
    const deleteBtn = document.getElementById('delete-btn');
    const toast = document.getElementById('toast');
    
    let selectedNodeIds = [];

    cy.on('select unselect', 'node', function() {
        const selectedNodes = cy.$('node:selected');
        selectedNodeIds = selectedNodes.map(n => n.id());
        
        if (selectedNodeIds.length === 0) {
            modal.classList.add('hidden');
        } else if (selectedNodeIds.length === 1) {
            const node = selectedNodes[0];
            titleEl.textContent = node.data('label');
            typeEl.textContent = node.data('group') || 'ENTITY';
            descEl.textContent = node.data('description') || 'No description available.';
            modal.classList.remove('hidden');
            toast.classList.add('hidden');
            ingestBtn.classList.remove('hidden');
            mergeBtn.classList.add('hidden');
        } else {
            titleEl.textContent = `${selectedNodeIds.length} Nodes Selected`;
            typeEl.textContent = 'MULTIPLE';
            descEl.textContent = 'You have selected multiple nodes. Bulk actions are available.';
            modal.classList.remove('hidden');
            toast.classList.add('hidden');
            ingestBtn.classList.add('hidden'); 
            mergeBtn.classList.remove('hidden');
        }
    });

    cy.on('tap', function(evt) {
        if (evt.target === cy) {
            cy.elements().unselect();
        }
    });

    closeBtn.addEventListener('click', () => {
        modal.classList.add('hidden');
        cy.elements().unselect();
    });

    // 4. Handle Ingestion Trigger
    ingestBtn.addEventListener('click', async () => {
        if (selectedNodeIds.length !== 1) return;
        const selectedNodeId = selectedNodeIds[0];
        const node = cy.getElementById(selectedNodeId);
        const entityName = node.data('label'); 
        
        ingestBtn.disabled = true;
        ingestBtn.style.opacity = 0.5;
        ingestBtn.innerHTML = 'Starting...';
        
        try {
            const res = await fetch(`/api/ingest/${encodeURIComponent(entityName)}`, {
                method: 'POST'
            });
            const result = await res.json();
            
            toast.textContent = result.message || "Ingestion started successfully!";
            toast.style.color = "#86efac";
            toast.classList.remove('hidden');
            
            setTimeout(() => {
                toast.classList.add('hidden');
            }, 5000);
            
        } catch (err) {
            toast.textContent = "Error triggering ingestion.";
            toast.style.color = "#f87171";
            toast.classList.remove('hidden');
        } finally {
            setTimeout(() => {
                ingestBtn.disabled = false;
                ingestBtn.style.opacity = 1;
                ingestBtn.innerHTML = `
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
                    Fetch Articles
                `;
            }, 1000);
        }
    });

    // 5. Handle Delete Trigger
    deleteBtn.addEventListener('click', async () => {
        if (selectedNodeIds.length === 0) return;
        
        const confirmDelete = confirm(`Are you sure you want to permanently delete ${selectedNodeIds.length} node(s) and all their connections from the graph?`);
        if (!confirmDelete) return;

        deleteBtn.disabled = true;
        deleteBtn.style.opacity = 0.5;
        
        try {
            const res = await fetch(`/api/nodes/delete_bulk`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ nodes: selectedNodeIds })
            });
            
            if (!res.ok) throw new Error("Failed to delete nodes");
            
            const result = await res.json();
            
            toast.textContent = result.message || "Nodes deleted successfully!";
            toast.style.color = "#86efac";
            toast.classList.remove('hidden');
            
            selectedNodeIds.forEach(id => {
                cy.getElementById(id).remove();
            });
            selectedNodeIds = [];
            
            setTimeout(() => {
                modal.classList.add('hidden');
            }, 1000);
            
        } catch (err) {
            toast.textContent = "Error deleting nodes.";
            toast.style.color = "#f87171";
            toast.classList.remove('hidden');
        } finally {
            setTimeout(() => {
                deleteBtn.disabled = false;
                deleteBtn.style.opacity = 1;
            }, 1000);
        }
    });

    // 6. Handle Merge Trigger
    mergeBtn.addEventListener('click', async () => {
        if (selectedNodeIds.length < 2) return;
        
        const primaryNode = cy.getElementById(selectedNodeIds[0]);
        const defaultName = primaryNode.data('label');
        
        const mergedName = prompt("Enter the final name for the merged node:", defaultName);
        if (!mergedName) return;

        mergeBtn.disabled = true;
        mergeBtn.style.opacity = 0.5;
        
        try {
            const res = await fetch(`/api/nodes/merge_bulk`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ nodes: selectedNodeIds, merged_name: mergedName })
            });
            
            if (!res.ok) throw new Error("Failed to merge nodes");
            
            const result = await res.json();
            
            toast.textContent = result.message || "Nodes merged successfully!";
            toast.style.color = "#86efac";
            toast.classList.remove('hidden');
            
            const primaryId = selectedNodeIds[0];
            const others = selectedNodeIds.slice(1);
            
            cy.getElementById(primaryId).data('label', mergedName);
            
            others.forEach(id => {
                cy.getElementById(id).remove();
            });
            selectedNodeIds = [primaryId];
            
            setTimeout(() => {
                modal.classList.add('hidden');
            }, 1000);
            
        } catch (err) {
            toast.textContent = "Error merging nodes.";
            toast.style.color = "#f87171";
            toast.classList.remove('hidden');
        } finally {
            setTimeout(() => {
                mergeBtn.disabled = false;
                mergeBtn.style.opacity = 1;
            }, 1000);
        }
    });
});
