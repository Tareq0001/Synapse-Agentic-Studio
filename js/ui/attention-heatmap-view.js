/**
 * Synapse-AI Multi-Head Attention Heatmap Visualizer
 * Renders interactive QKV Attention weights across Transformer heads with token inspection.
 */

export class AttentionHeatmapView {
    constructor(container, llmEngine, audioSynth) {
        this.container = container;
        this.engine = llmEngine;
        this.synth = audioSynth;
        this.currentHead = 0;
    }

    render() {
        if (!this.container) return;

        const tokens = this.engine.tokens;
        const matrix = this.engine.attentionMatrices[this.currentHead];

        this.container.innerHTML = `
            <div class="attention-studio-grid">
                <!-- 1. Attention Heatmap Matrix -->
                <div class="card">
                    <div class="card-header">
                        <div>
                            <span class="badge badge-purple">Transformer Self-Attention</span>
                            <h3>Scaled Dot-Product Multi-Head Attention Matrix</h3>
                        </div>
                        <span class="stat-badge">Softmax(Q · Kᵀ / √dₖ) · V</span>
                    </div>

                    <!-- Head Switcher -->
                    <div class="head-tabs-bar">
                        ${this.engine.headSpecializations.map((spec, idx) => `
                            <button class="btn btn-sm ${this.currentHead === idx ? 'btn-primary' : 'btn-outline'} btn-head-tab" data-head="${idx}">
                                ${spec.split(":")[0]}
                            </button>
                        `).join("")}
                    </div>

                    <div class="head-description font-mono text-sm text-cyan" style="margin: 10px 0;">
                        ${this.engine.headSpecializations[this.currentHead]}
                    </div>

                    <!-- 8x8 Grid Table -->
                    <div class="heatmap-table-wrap">
                        <table class="attention-table font-mono">
                            <thead>
                                <tr>
                                    <th class="corner-header">Q \\ K</th>
                                    ${tokens.map(t => `<th class="token-header">${t}</th>`).join("")}
                                </tr>
                            </thead>
                            <tbody>
                                ${tokens.map((qTok, rIdx) => `
                                    <tr>
                                        <th class="token-header row-token">${qTok}</th>
                                        ${tokens.map((kTok, cIdx) => {
                                            const weight = matrix[rIdx][cIdx];
                                            const intensity = Math.min(1.0, weight * 1.8);
                                            return `
                                                <td class="attn-cell" 
                                                    data-r="${rIdx}" 
                                                    data-c="${cIdx}" 
                                                    data-q="${qTok}" 
                                                    data-k="${kTok}" 
                                                    data-w="${weight}"
                                                    style="background: rgba(99, 102, 241, ${intensity}); color: ${intensity > 0.4 ? '#ffffff' : '#94a3b8'};">
                                                    ${weight.toFixed(2)}
                                                </td>
                                            `;
                                        }).join("")}
                                    </tr>
                                `).join("")}
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- 2. Attention Weight Inspector & Mathematical Decomposition -->
                <div class="card">
                    <div class="card-header">
                        <div>
                            <span class="badge badge-cyan">Dynamic Token Probe</span>
                            <h3>Attention Weight Inspection</h3>
                        </div>
                    </div>

                    <div id="attn-probe-box" class="probe-box font-mono">
                        <div class="text-muted">Hover over any cell in the attention matrix to inspect QKV vector interactions.</div>
                    </div>

                    <!-- Mathematical Formulation Card -->
                    <div class="math-card font-mono text-sm" style="margin-top: 20px;">
                        <h4 class="text-purple">Self-Attention Mathematical Mechanics:</h4>
                        <div class="formula-line text-cyan">Attention(Q, K, V) = softmax((Q · Kᵀ) / √dₖ) · V</div>
                        <ul class="math-bullets text-muted">
                            <li><strong>Q (Query):</strong> Representation of what token <em>wᵢ</em> is seeking.</li>
                            <li><strong>K (Key):</strong> Representation of what token <em>wⱼ</em> contains.</li>
                            <li><strong>√dₖ Scaling:</strong> Prevents dot products from exploding in high dimensions (dₖ = 128), preventing vanishing gradients in Softmax.</li>
                        </ul>
                    </div>
                </div>
            </div>
        `;

        this._setupEvents();
    }

    _setupEvents() {
        // Head Switcher
        this.container.querySelectorAll(".btn-head-tab").forEach(btn => {
            btn.addEventListener("click", () => {
                this.currentHead = parseInt(btn.getAttribute("data-head"), 10);
                if (this.synth) this.synth.playThought();
                this.render();
            });
        });

        // Hover Probe
        const probeBox = this.container.querySelector("#attn-probe-box");
        this.container.querySelectorAll(".attn-cell").forEach(cell => {
            cell.addEventListener("mouseenter", () => {
                const q = cell.getAttribute("data-q");
                const k = cell.getAttribute("data-k");
                const w = parseFloat(cell.getAttribute("data-w"));

                if (probeBox) {
                    probeBox.innerHTML = `
                        <div class="probe-row"><span class="text-muted">Query Token (Source):</span> <strong class="text-cyan">"${q}"</strong></div>
                        <div class="probe-row"><span class="text-muted">Key Token (Target):</span> <strong class="text-purple">"${k}"</strong></div>
                        <div class="probe-row"><span class="text-muted">Attention Weight:</span> <strong class="text-emerald" style="font-size: 16px;">${(w * 100).toFixed(1)}% (${w.toFixed(4)})</strong></div>
                        <div class="probe-row" style="margin-top: 8px;">
                            <span class="text-muted">Interpretation:</span>
                            <span>${w > 0.35 ? 'Strong semantic/dependency binding between tokens.' : 'Weak lexical background association.'}</span>
                        </div>
                    `;
                }
            });
        });
    }
}
