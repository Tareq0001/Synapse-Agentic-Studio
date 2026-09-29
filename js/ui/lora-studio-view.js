/**
 * Synapse-AI LoRA Parameter Efficient Fine-Tuning (PEFT) Studio
 * Interactive Low-Rank matrix rank calculator, parameter savings inspector,
 * and live training run simulator (SFT, DPO, Pre-training).
 */

export class LoRAStudioView {
    constructor(container, llmEngine, audioSynth) {
        this.container = container;
        this.engine = llmEngine;
        this.synth = audioSynth;
    }

    render() {
        if (!this.container) return;

        const metrics = this.engine.computeLoRAMetrics(
            this.engine.loraConfig.dModel,
            this.engine.loraConfig.rank,
            32
        );

        this.container.innerHTML = `
            <div class="lora-studio-grid">
                <!-- 1. LoRA Matrix Decomposition & Parameter Savings -->
                <div class="card">
                    <div class="card-header">
                        <div>
                            <span class="badge badge-emerald">PEFT / LoRA Studio</span>
                            <h3>Low-Rank Matrix Decomposition (W = W₀ + B · A)</h3>
                        </div>
                        <span class="stat-badge text-emerald">${metrics.memorySavedPercentage} Memory Saved</span>
                    </div>

                    <!-- Visual Equation Layout -->
                    <div class="lora-equation-card font-mono">
                        <div class="matrix-block frozen-block">
                            <span class="matrix-title text-cyan">W₀ (Frozen)</span>
                            <div class="matrix-dims">4096 × 4096</div>
                            <span class="badge badge-secondary">16.7M Params</span>
                        </div>
                        <span class="op-sign">+</span>
                        <div class="scaling-factor text-purple">
                            <span class="factor-label">α / r</span>
                            <span class="factor-val font-bold">${metrics.scalingFactor}</span>
                        </div>
                        <div class="matrix-block lora-block">
                            <span class="matrix-title text-emerald">Matrix B</span>
                            <div class="matrix-dims">4096 × ${this.engine.loraConfig.rank}</div>
                            <span class="badge badge-emerald">Trainable</span>
                        </div>
                        <span class="op-sign">×</span>
                        <div class="matrix-block lora-block">
                            <span class="matrix-title text-emerald">Matrix A</span>
                            <div class="matrix-dims">${this.engine.loraConfig.rank} × 4096</div>
                            <span class="badge badge-emerald">Trainable</span>
                        </div>
                    </div>

                    <!-- Sliders -->
                    <div class="lora-sliders-box" style="margin-top: 20px;">
                        <div class="slider-row">
                            <div class="slider-info font-mono">
                                <span>LoRA Rank (r):</span>
                                <strong class="text-emerald" id="lbl-lora-rank">${this.engine.loraConfig.rank}</strong>
                            </div>
                            <input type="range" id="slider-lora-rank" min="4" max="64" step="4" value="${this.engine.loraConfig.rank}" class="knob-slider">
                        </div>

                        <div class="slider-row">
                            <div class="slider-info font-mono">
                                <span>LoRA Alpha (α):</span>
                                <strong class="text-cyan" id="lbl-lora-alpha">${this.engine.loraConfig.alpha}</strong>
                            </div>
                            <input type="range" id="slider-lora-alpha" min="8" max="64" step="8" value="${this.engine.loraConfig.alpha}" class="knob-slider">
                        </div>
                    </div>

                    <!-- Metrics Grid -->
                    <div class="lora-stats-grid font-mono text-sm" style="margin-top: 16px;">
                        <div class="stat-pill">
                            <span class="text-muted">Total Base Params:</span>
                            <strong>${metrics.totalBaseParamsFormatted}</strong>
                        </div>
                        <div class="stat-pill">
                            <span class="text-muted">Trainable LoRA Params:</span>
                            <strong class="text-emerald">${metrics.loraTrainableFormatted}</strong>
                        </div>
                        <div class="stat-pill">
                            <span class="text-muted">Trainable Percentage:</span>
                            <strong class="text-cyan">${metrics.trainablePercentage}</strong>
                        </div>
                    </div>
                </div>

                <!-- 2. Training Run Launcher & Live Curves -->
                <div class="card">
                    <div class="card-header">
                        <div>
                            <span class="badge badge-purple">Training Run Simulator</span>
                            <h3>Fine-Tuning Execution & Alignment</h3>
                        </div>
                        <span class="stat-badge" id="train-status-badge">READY</span>
                    </div>

                    <div class="training-launch-bar">
                        <button class="btn btn-primary btn-sm btn-train-mode" data-mode="SFT_LORA">
                            ⚡ SFT with LoRA
                        </button>
                        <button class="btn btn-secondary btn-sm btn-train-mode" data-mode="DPO_ALIGNMENT">
                            🎯 DPO Alignment
                        </button>
                        <button class="btn btn-outline btn-sm btn-train-mode" data-mode="PRETRAIN">
                            🌐 Causal Pre-Training
                        </button>
                    </div>

                    <!-- Live Loss Canvas -->
                    <div class="canvas-wrap" style="height: 180px; margin-top: 16px;">
                        <canvas id="train-loss-canvas" width="540" height="180"></canvas>
                    </div>

                    <!-- Real-time Metrics Readout -->
                    <div class="train-readouts font-mono text-sm" style="margin-top: 14px;">
                        <div class="readout-card">
                            <span class="text-muted">Current Loss:</span>
                            <strong class="text-emerald" id="val-train-loss">${this.engine.history.loss[this.engine.history.loss.length - 1]}</strong>
                        </div>
                        <div class="readout-card">
                            <span class="text-muted">Perplexity (PPL):</span>
                            <strong class="text-cyan" id="val-train-ppl">${this.engine.history.perplexity[this.engine.history.perplexity.length - 1]}</strong>
                        </div>
                        <div class="readout-card">
                            <span class="text-muted">Learning Rate:</span>
                            <strong class="text-purple" id="val-train-lr">2.1e-5</strong>
                        </div>
                        <div class="readout-card">
                            <span class="text-muted">GPU VRAM:</span>
                            <strong class="text-amber" id="val-train-vram">14.4 GB</strong>
                        </div>
                    </div>
                </div>
            </div>
        `;

        this._setupEvents();
        this._drawLossCanvas();
    }

    _setupEvents() {
        // Rank Slider
        const rankSlider = this.container.querySelector("#slider-lora-rank");
        const rankLbl = this.container.querySelector("#lbl-lora-rank");
        rankSlider?.addEventListener("input", (e) => {
            const r = parseInt(e.target.value, 10);
            this.engine.loraConfig.rank = r;
            if (rankLbl) rankLbl.innerText = r;
            if (this.synth) this.synth.playThought();
            this.render();
        });

        // Alpha Slider
        const alphaSlider = this.container.querySelector("#slider-lora-alpha");
        const alphaLbl = this.container.querySelector("#lbl-lora-alpha");
        alphaSlider?.addEventListener("input", (e) => {
            const a = parseInt(e.target.value, 10);
            this.engine.loraConfig.alpha = a;
            if (alphaLbl) alphaLbl.innerText = a;
            if (this.synth) this.synth.playThought();
            this.render();
        });

        // Training Mode Launchers
        this.container.querySelectorAll(".btn-train-mode").forEach(btn => {
            btn.addEventListener("click", () => {
                const mode = btn.getAttribute("data-mode");
                const badge = this.container.querySelector("#train-status-badge");
                if (badge) {
                    badge.innerText = `TRAINING (${mode})`;
                    badge.className = "stat-badge text-emerald";
                }

                btn.disabled = true;

                this.engine.startTraining(
                    mode,
                    (telemetry) => {
                        this._updateTelemetryReadouts(telemetry);
                        this._drawLossCanvas();
                        if (this.synth) this.synth.playThought();
                    },
                    (final) => {
                        btn.disabled = false;
                        if (badge) {
                            badge.innerText = "CONVERGED";
                            badge.className = "stat-badge text-cyan";
                        }
                        if (this.synth) this.synth.playEpoch();
                    }
                );
            });
        });
    }

    _updateTelemetryReadouts(t) {
        const lossEl = this.container.querySelector("#val-train-loss");
        const pplEl = this.container.querySelector("#val-train-ppl");
        const lrEl = this.container.querySelector("#val-train-lr");
        const vramEl = this.container.querySelector("#val-train-vram");

        if (lossEl) lossEl.innerText = t.loss;
        if (pplEl) pplEl.innerText = t.perplexity;
        if (lrEl) lrEl.innerText = t.lr;
        if (vramEl) vramEl.innerText = `${t.vram} GB`;
    }

    _drawLossCanvas() {
        const canvas = this.container.querySelector("#train-loss-canvas");
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        const { width, height } = canvas;

        ctx.clearRect(0, 0, width, height);

        const data = this.engine.history.loss;
        if (data.length < 2) return;

        const padding = { top: 15, bottom: 25, left: 30, right: 20 };
        const w = width - padding.left - padding.right;
        const h = height - padding.top - padding.bottom;

        const minVal = 0.0;
        const maxVal = Math.max(...data) * 1.15;

        // Line
        ctx.strokeStyle = "#10b981";
        ctx.lineWidth = 2.5;
        ctx.beginPath();

        data.forEach((val, i) => {
            const x = padding.left + (i / (data.length - 1)) * w;
            const y = height - padding.bottom - ((val - minVal) / (maxVal - minVal)) * h;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();

        ctx.fillStyle = "#94a3b8";
        ctx.font = "10px 'JetBrains Mono', monospace";
        ctx.fillText("Cross-Entropy Loss (Lower is Optimal)", padding.left, 12);
    }
}
