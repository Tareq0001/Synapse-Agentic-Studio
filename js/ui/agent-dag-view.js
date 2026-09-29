/**
 * Synapse-AI Agent DAG & Workflow Visualizer
 * Renders the multi-agent swarm architecture, animated inter-agent message pulses,
 * ReAct cognitive thoughts, and interactive task execution streams.
 */

export class AgentDAGView {
    constructor(container, swarmEngine, audioSynth) {
        this.container = container;
        this.swarm = swarmEngine;
        this.synth = audioSynth;
    }

    render() {
        if (!this.container) return;

        this.container.innerHTML = `
            <div class="agent-studio-grid">
                <!-- 1. Interactive Agent Swarm DAG Canvas & Controls -->
                <div class="card">
                    <div class="card-header">
                        <div>
                            <span class="badge badge-indigo">Autonomous Multi-Agent Swarm</span>
                            <h3>Agent Architecture & Task DAG Orchestrator</h3>
                        </div>
                        <span class="stat-badge">${this.swarm.agents.length} Specialized Agents</span>
                    </div>

                    <!-- Workflow Task Selector -->
                    <div class="task-preset-bar">
                        <span class="text-muted text-sm">Presets:</span>
                        <button class="btn btn-outline btn-sm btn-agent-task" data-task="Design an Enterprise Hybrid-Search RAG Architecture with Cross-Encoder Reranking">
                            📚 Enterprise RAG
                        </button>
                        <button class="btn btn-outline btn-sm btn-agent-task" data-task="Build an Autonomous Tabular Fraud Detection Pipeline with Isolation Forest and XGBoost">
                            🛡️ Fraud Detection Pipeline
                        </button>
                        <button class="btn btn-outline btn-sm btn-agent-task" data-task="Fine-tune a 7B LLM with LoRA on Medical Diagnosis Records with DPO Alignment">
                            🧬 Medical LLM Fine-Tuning
                        </button>
                    </div>

                    <!-- Task Input & Launch -->
                    <div class="task-input-bar">
                        <input type="text" id="agent-task-input" class="form-input font-mono" 
                            value="Design an Enterprise Hybrid-Search RAG Architecture with Cross-Encoder Reranking" 
                            placeholder="Enter goal for the autonomous swarm...">
                        <button class="btn btn-primary" id="btn-run-swarm">
                            🚀 Dispatch Agent Swarm
                        </button>
                    </div>

                    <!-- Visual Agent Nodes Canvas -->
                    <div class="canvas-wrap" style="height: 260px; margin-top: 16px;">
                        <canvas id="agent-dag-canvas" width="620" height="260"></canvas>
                    </div>

                    <!-- Active Agents Cards Row -->
                    <div class="agent-cards-row">
                        ${this.swarm.agents.map(a => `
                            <div class="agent-card ${a.status === 'ACTIVE' ? 'agent-active' : ''}" id="card-${a.id}">
                                <div class="agent-avatar">${a.avatar}</div>
                                <div class="agent-info">
                                    <strong class="agent-name font-mono">${a.name}</strong>
                                    <span class="agent-role text-muted">${a.role}</span>
                                </div>
                                <span class="badge badge-${a.status === 'ACTIVE' ? 'emerald' : 'secondary'} agent-status-badge">
                                    ${a.status}
                                </span>
                            </div>
                        `).join("")}
                    </div>
                </div>

                <!-- 2. ReAct Cognitive Execution Stream & Memory -->
                <div class="card">
                    <div class="card-header">
                        <div>
                            <span class="badge badge-cyan">ReAct Cognitive Stream</span>
                            <h3>Live Inter-Agent Execution Feed</h3>
                        </div>
                        <span class="stat-badge" id="swarm-step-counter">${this.swarm.executionLogs.length} Steps</span>
                    </div>

                    <div class="react-feed" id="react-execution-feed">
                        ${this.swarm.executionLogs.map(log => this._renderLogItem(log)).join("")}
                    </div>

                    <!-- Memory Footprint Drawer -->
                    <div class="memory-drawer">
                        <div class="memory-header font-mono">
                            <span>🧠 Long-Term Vector Memory</span>
                            <span class="text-cyan">${this.swarm.memory.longTermVectorStore.length} Vector Keys Cached</span>
                        </div>
                        <div class="memory-items font-mono text-sm">
                            ${this.swarm.memory.longTermVectorStore.map(m => `
                                <div class="memory-tag">
                                    <span class="text-emerald">[Similarity: ${m.similarity}]</span>
                                    <span>${m.topic}: ${m.text.slice(0, 48)}...</span>
                                </div>
                            `).join("")}
                        </div>
                    </div>
                </div>
            </div>
        `;

        this._setupEvents();
        this._drawDAGCanvas();
    }

    _renderLogItem(log) {
        const agent = this.swarm.agents.find(a => a.id === log.agentId) || { name: log.agentId, avatar: "🤖", color: "#6366f1" };

        let typeBadge = "";
        if (log.type === "THOUGHT") typeBadge = '<span class="badge badge-purple">THOUGHT</span>';
        else if (log.type === "ACTION" || log.type === "TOOL_CALL") typeBadge = '<span class="badge badge-cyan">TOOL CALL</span>';
        else if (log.type === "CRITIQUE") typeBadge = '<span class="badge badge-amber">CRITIQUE</span>';
        else typeBadge = '<span class="badge badge-emerald">FINAL OUTPUT</span>';

        return `
            <div class="react-log-card type-${log.type.toLowerCase()}">
                <div class="log-header">
                    <div class="log-agent font-mono">
                        <span>${agent.avatar}</span>
                        <strong>${agent.name}</strong>
                        ${typeBadge}
                    </div>
                    <span class="text-muted font-mono text-sm">${log.timestamp || 'Live'}</span>
                </div>
                <div class="log-body font-mono text-sm">
                    ${log.content || ''}
                    ${log.tool ? `<div class="tool-call-box"><code>${log.tool}(${JSON.stringify(log.params || {})})</code></div>` : ''}
                    ${log.output ? `<div class="tool-output-box text-emerald">➔ ${log.output}</div>` : ''}
                    ${log.score ? `<div class="critique-score text-amber">Quality Score: <strong>${log.score}</strong></div>` : ''}
                </div>
            </div>
        `;
    }

    _setupEvents() {
        const runBtn = this.container.querySelector("#btn-run-swarm");
        const taskInput = this.container.querySelector("#agent-task-input");

        runBtn?.addEventListener("click", () => {
            const task = taskInput.value.trim();
            if (!task) return;

            runBtn.disabled = true;
            runBtn.innerHTML = '⏳ Swarm Executing...';

            this.swarm.executeWorkflow(
                task,
                (step) => {
                    const feed = this.container.querySelector("#react-execution-feed");
                    if (feed) {
                        feed.insertAdjacentHTML("beforeend", this._renderLogItem(step));
                        feed.scrollTop = feed.scrollHeight;
                    }

                    // Audio cue
                    if (this.synth) {
                        if (step.type === "TOOL_CALL") this.synth.playToolCall();
                        else this.synth.playThought();
                    }

                    // Update UI state
                    this._updateAgentCardHighlights();
                    this._drawDAGCanvas();
                },
                () => {
                    runBtn.disabled = false;
                    runBtn.innerHTML = '🚀 Dispatch Agent Swarm';
                    this._updateAgentCardHighlights();
                    this._drawDAGCanvas();
                    if (this.synth) this.synth.playEpoch();
                }
            );
        });

        // Presets
        this.container.querySelectorAll(".btn-agent-task").forEach(btn => {
            btn.addEventListener("click", () => {
                taskInput.value = btn.getAttribute("data-task");
                if (this.synth) this.synth.playThought();
            });
        });
    }

    _updateAgentCardHighlights() {
        this.swarm.agents.forEach(a => {
            const card = this.container.querySelector(`#card-${a.id}`);
            if (card) {
                card.classList.toggle("agent-active", a.status === "ACTIVE");
                const badge = card.querySelector(".agent-status-badge");
                if (badge) {
                    badge.className = `badge badge-${a.status === 'ACTIVE' ? 'emerald' : 'secondary'} agent-status-badge`;
                    badge.innerText = a.status;
                }
            }
        });
    }

    _drawDAGCanvas() {
        const canvas = this.container.querySelector("#agent-dag-canvas");
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        const { width, height } = canvas;

        ctx.clearRect(0, 0, width, height);

        // Node Coordinates in DAG layout
        const nodes = [
            { id: "orchestrator", x: 100, y: 130, title: "Architect", color: "#6366f1" },
            { id: "researcher", x: 300, y: 65, title: "RAG Researcher", color: "#06b6d4" },
            { id: "coder", x: 300, y: 195, title: "Systems Coder", color: "#10b981" },
            { id: "critic", x: 520, y: 130, title: "Critic Evaluator", color: "#f59e0b" }
        ];

        // Draw DAG Curved Edges
        const edges = [
            { from: nodes[0], to: nodes[1] },
            { from: nodes[0], to: nodes[2] },
            { from: nodes[1], to: nodes[3] },
            { from: nodes[2], to: nodes[3] }
        ];

        ctx.lineWidth = 2;
        edges.forEach(e => {
            ctx.beginPath();
            ctx.moveTo(e.from.x, e.from.y);
            const cpX = (e.from.x + e.to.x) / 2;
            ctx.bezierCurveTo(cpX, e.from.y, cpX, e.to.y, e.to.x, e.to.y);
            ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
            ctx.stroke();
        });

        // Draw Nodes
        nodes.forEach(n => {
            const agentObj = this.swarm.agents.find(a => a.id === n.id);
            const isActive = agentObj && agentObj.status === "ACTIVE";

            // Glow if active
            if (isActive) {
                ctx.beginPath();
                ctx.arc(n.x, n.y, 34, 0, 2 * Math.PI);
                ctx.fillStyle = `${n.color}33`;
                ctx.fill();
            }

            // Outer Circle
            ctx.beginPath();
            ctx.arc(n.x, n.y, 24, 0, 2 * Math.PI);
            ctx.fillStyle = "#0f172a";
            ctx.strokeStyle = n.color;
            ctx.lineWidth = isActive ? 3.5 : 2;
            ctx.fill();
            ctx.stroke();

            // Label
            ctx.fillStyle = "#f8fafc";
            ctx.font = "bold 11px 'JetBrains Mono', monospace";
            ctx.textAlign = "center";
            ctx.fillText(n.title, n.x, n.y + 40);
        });
    }
}
