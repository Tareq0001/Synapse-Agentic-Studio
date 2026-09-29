/**
 * Synapse-AI Data Science & AutoML Arena View
 * Interactive correlation matrix, AutoML leaderboard, ROC-AUC,
 * live decision threshold slider, and real-time SHAP waterfall feature attribution.
 */

export class AutoMLArenaView {
    constructor(container, dsEngine) {
        this.container = container;
        this.engine = dsEngine;
    }

    render() {
        if (!this.container) return;

        const features = this.engine.features;
        const matrix = this.engine.correlationMatrix;
        const inf = this.engine.computeInferenceAndSHAP();

        this.container.innerHTML = `
            <div class="ds-arena-grid">
                <!-- 1. Correlation Matrix Heatmap -->
                <div class="card">
                    <div class="card-header">
                        <div>
                            <span class="badge badge-cyan">Exploratory Data Analysis</span>
                            <h3>Feature Pearson Correlation Matrix (6×6)</h3>
                        </div>
                    </div>

                    <div class="table-responsive">
                        <table class="data-table font-mono text-sm">
                            <thead>
                                <tr>
                                    <th>Feature</th>
                                    ${features.map(f => `<th>${f}</th>`).join("")}
                                </tr>
                            </thead>
                            <tbody>
                                ${features.map((f, rIdx) => `
                                    <tr>
                                        <td><strong>${f}</strong></td>
                                        ${features.map((_, cIdx) => {
                                            const r = matrix[rIdx][cIdx];
                                            const isPos = r > 0;
                                            const alpha = Math.abs(r) * 0.7;
                                            const color = isPos ? `rgba(16, 185, 129, ${alpha})` : `rgba(244, 63, 94, ${alpha})`;
                                            return `
                                                <td style="background: ${color}; text-align: center;">
                                                    ${r.toFixed(2)}
                                                </td>
                                            `;
                                        }).join("")}
                                    </tr>
                                `).join("")}
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- 2. AutoML Leaderboard -->
                <div class="card">
                    <div class="card-header">
                        <div>
                            <span class="badge badge-emerald">Model Arena</span>
                            <h3>AutoML Algorithmic Leaderboard</h3>
                        </div>
                        <span class="stat-badge">K-Fold Cross Validated</span>
                    </div>

                    <div class="table-responsive">
                        <table class="data-table font-mono text-sm">
                            <thead>
                                <tr>
                                    <th>Model Algorithm</th>
                                    <th>ROC-AUC</th>
                                    <th>F1-Score</th>
                                    <th>Precision</th>
                                    <th>Latency</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${this.engine.models.map(m => `
                                    <tr>
                                        <td><strong>${m.name}</strong> <small class="text-muted">(${m.type})</small></td>
                                        <td class="text-emerald"><strong>${m.rocAuc}</strong></td>
                                        <td>${m.f1Score}</td>
                                        <td>${m.precision}</td>
                                        <td>${m.trainingTimeSec}s</td>
                                        <td><span class="badge badge-${m.badgeColor}">${m.status}</span></td>
                                    </tr>
                                `).join("")}
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- 3. ROC-AUC Canvas Curve & Dynamic Threshold Confusion Matrix -->
                <div class="card">
                    <div class="card-header">
                        <div>
                            <span class="badge badge-purple">Evaluation Metrics</span>
                            <h3>ROC-AUC & Interactive Decision Threshold (τ)</h3>
                        </div>
                        <span class="stat-badge text-emerald">AUC: 0.984</span>
                    </div>

                    <!-- Threshold Slider -->
                    <div class="threshold-slider-bar font-mono text-sm" style="margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                            <span>Classification Threshold (τ): <strong class="text-cyan" id="lbl-thresh">${this.engine.decisionThreshold.toFixed(2)}</strong></span>
                            <span>F1-Score: <strong class="text-emerald" id="lbl-f1">${inf.metrics.f1}</strong></span>
                        </div>
                        <input type="range" id="slider-threshold" min="0.10" max="0.90" step="0.05" value="${this.engine.decisionThreshold}" class="knob-slider">
                    </div>

                    <div class="eval-split-grid">
                        <div class="canvas-wrap" style="height: 180px;">
                            <canvas id="roc-curve-canvas" width="280" height="180"></canvas>
                        </div>

                        <!-- 2x2 Confusion Matrix -->
                        <div class="confusion-matrix-box font-mono text-sm">
                            <div class="cm-header">True Positive / Negative Counts</div>
                            <div class="cm-grid">
                                <div class="cm-cell cm-tp">
                                    <span class="cm-label">True Pos (TP)</span>
                                    <strong class="text-emerald" id="cm-tp-val">${inf.confusionMatrix.tp}</strong>
                                </div>
                                <div class="cm-cell cm-fp">
                                    <span class="cm-label">False Pos (FP)</span>
                                    <strong class="text-rose" id="cm-fp-val">${inf.confusionMatrix.fp}</strong>
                                </div>
                                <div class="cm-cell cm-fn">
                                    <span class="cm-label">False Neg (FN)</span>
                                    <strong class="text-rose" id="cm-fn-val">${inf.confusionMatrix.fn}</strong>
                                </div>
                                <div class="cm-cell cm-tn">
                                    <span class="cm-label">True Neg (TN)</span>
                                    <strong class="text-cyan" id="cm-tn-val">${inf.confusionMatrix.tn}</strong>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 4. SHAP Feature Explainability Waterfall & Interactive Sliders -->
                <div class="card">
                    <div class="card-header">
                        <div>
                            <span class="badge badge-amber">Explainable AI (XAI)</span>
                            <h3>SHAP Waterfall Feature Attribution Plot</h3>
                        </div>
                        <span class="badge badge-${inf.decision === 'FLAGGED_FRAUD' ? 'rose' : 'emerald'}" id="lbl-decision-badge">
                            ${inf.decision}
                        </span>
                    </div>

                    <!-- Interactive Risk Tuner -->
                    <div class="feature-tuner-row font-mono text-sm" style="display: flex; gap: 14px; margin-bottom: 12px;">
                        <div style="flex: 1;">
                            <label class="text-muted">IP Risk Score: <strong class="text-rose" id="lbl-tune-ip">${this.engine.currentFeatureValues.ip_risk_score}</strong></label>
                            <input type="range" id="slider-tune-ip" min="0.05" max="1.0" step="0.05" value="${this.engine.currentFeatureValues.ip_risk_score}" class="knob-slider">
                        </div>
                        <div style="flex: 1;">
                            <label class="text-muted">Velocity (24h): <strong class="text-cyan" id="lbl-tune-vel">${this.engine.currentFeatureValues.velocity_24h}</strong></label>
                            <input type="range" id="slider-tune-vel" min="1" max="30" step="1" value="${this.engine.currentFeatureValues.velocity_24h}" class="knob-slider">
                        </div>
                    </div>

                    <div class="shap-summary font-mono text-sm" style="margin-bottom: 10px;">
                        <span>Base Risk E[f(x)]: <strong>${inf.baseValue}</strong></span> ➔ 
                        <span class="font-bold ${inf.predictedProb > 0.5 ? 'text-rose' : 'text-emerald'}" id="lbl-pred-prob">Predicted Fraud Prob: ${inf.predictedProb}</span>
                    </div>

                    <div class="shap-bars-list font-mono text-sm" id="shap-bars-container">
                        ${inf.shapValues.map(s => {
                            const isPositive = s.contribution > 0;
                            const barWidth = Math.min(100, Math.abs(s.contribution) * 220);
                            return `
                                <div class="shap-row">
                                    <div class="shap-feat">
                                        <strong>${s.feature}</strong> <span class="text-muted">(${s.value})</span>
                                    </div>
                                    <div class="shap-bar-track">
                                        <div class="shap-bar ${isPositive ? 'shap-pos' : 'shap-neg'}" style="width: ${barWidth}%;"></div>
                                    </div>
                                    <div class="shap-val ${isPositive ? 'text-rose' : 'text-emerald'}">
                                        ${isPositive ? '+' : ''}${s.contribution.toFixed(2)}
                                    </div>
                                </div>
                            `;
                        }).join("")}
                    </div>
                </div>
            </div>
        `;

        this._setupEvents();
        this._drawROCCanvas();
    }

    _setupEvents() {
        // Threshold Slider
        const threshSlider = this.container.querySelector("#slider-threshold");
        threshSlider?.addEventListener("input", (e) => {
            this.engine.decisionThreshold = parseFloat(e.target.value);
            this.render();
        });

        // IP Risk Tuner
        const ipSlider = this.container.querySelector("#slider-tune-ip");
        ipSlider?.addEventListener("input", (e) => {
            this.engine.currentFeatureValues.ip_risk_score = parseFloat(e.target.value);
            this.render();
        });

        // Velocity Tuner
        const velSlider = this.container.querySelector("#slider-tune-vel");
        velSlider?.addEventListener("input", (e) => {
            this.engine.currentFeatureValues.velocity_24h = parseInt(e.target.value, 10);
            this.render();
        });
    }

    _drawROCCanvas() {
        const canvas = this.container.querySelector("#roc-curve-canvas");
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        const { width, height } = canvas;

        ctx.clearRect(0, 0, width, height);
        const padding = { top: 15, bottom: 20, left: 25, right: 15 };
        const w = width - padding.left - padding.right;
        const h = height - padding.top - padding.bottom;

        ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(padding.left, height - padding.bottom);
        ctx.lineTo(width - padding.right, padding.top);
        ctx.stroke();
        ctx.setLineDash([]);

        const points = this.engine.getROCCurvePoints();
        ctx.strokeStyle = "#10b981";
        ctx.lineWidth = 2.5;
        ctx.beginPath();

        points.forEach((pt, i) => {
            const x = padding.left + pt.fpr * w;
            const y = height - padding.bottom - pt.tpr * h;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();

        ctx.fillStyle = "#94a3b8";
        ctx.font = "9px 'JetBrains Mono', monospace";
        ctx.fillText("ROC Curve (TPR vs FPR)", padding.left, 10);
    }
}
