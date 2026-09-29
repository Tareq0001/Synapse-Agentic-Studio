/**
 * Synapse-AI Main Application Controller
 * Integrates Autonomous Agent Design, LLM Training & Attention Internals,
 * and Data Science AutoML & Explainability into a unified studio.
 */

import { SynthAudio } from "./audio/synth-audio.js";
import { AgentSwarmEngine } from "./engine/agent-swarm.js";
import { LLMTrainerEngine } from "./engine/llm-trainer.js";
import { DataScienceEngine } from "./engine/data-science.js";
import { AgentDAGView } from "./ui/agent-dag-view.js";
import { AttentionHeatmapView } from "./ui/attention-heatmap-view.js";
import { LoRAStudioView } from "./ui/lora-studio-view.js";
import { AutoMLArenaView } from "./ui/automl-arena-view.js";
import { I18nManager } from "./ui/i18n.js";

class SynapseApp {
    constructor() {
        this.synth = new SynthAudio();
        this.swarmEngine = new AgentSwarmEngine();
        this.llmEngine = new LLMTrainerEngine();
        this.dsEngine = new DataScienceEngine();
        this.i18n = new I18nManager();

        this.currentTab = "agents";

        this.initDOM();
        this.initViews();
        this.setupEvents();
    }

    initDOM() {
        this.agentsContainer = document.getElementById("tab-agents-container");
        this.attentionContainer = document.getElementById("tab-attention-container");
        this.loraContainer = document.getElementById("tab-lora-container");
        this.automlContainer = document.getElementById("tab-automl-container");

        this.updateI18nLabels();
    }

    initViews() {
        this.agentView = new AgentDAGView(this.agentsContainer, this.swarmEngine, this.synth);
        this.agentView.render();

        this.attentionView = new AttentionHeatmapView(this.attentionContainer, this.llmEngine, this.synth);
        this.attentionView.render();

        this.loraView = new LoRAStudioView(this.loraContainer, this.llmEngine, this.synth);
        this.loraView.render();

        this.automlView = new AutoMLArenaView(this.automlContainer, this.dsEngine);
        this.automlView.render();
    }

    setupEvents() {
        // Tab Switching
        document.querySelectorAll(".nav-tab").forEach(tab => {
            tab.addEventListener("click", () => {
                const target = tab.getAttribute("data-tab");
                this.switchTab(target);
                if (this.synth) this.synth.playThought();
            });
        });

        // Language Switcher
        document.getElementById("btn-toggle-lang")?.addEventListener("click", () => {
            this.i18n.toggle();
            this.updateI18nLabels();
            this.agentView.render();
            this.attentionView.render();
            this.loraView.render();
            this.automlView.render();
            if (this.synth) this.synth.playThought();
        });

        // Sound Toggle
        const soundBtn = document.getElementById("btn-toggle-sound");
        soundBtn?.addEventListener("click", () => {
            const enabled = this.synth.toggle();
            soundBtn.innerText = enabled ? this.i18n.t("soundOn") : this.i18n.t("soundOff");
        });
    }

    switchTab(tabId) {
        this.currentTab = tabId;
        document.querySelectorAll(".nav-tab").forEach(t => {
            t.classList.toggle("active", t.getAttribute("data-tab") === tabId);
        });

        document.querySelectorAll(".tab-content").forEach(content => {
            content.classList.toggle("active", content.id === `tab-${tabId}`);
        });

        // Re-render target view if needed
        if (tabId === "agents") this.agentView.render();
        else if (tabId === "attention") this.attentionView.render();
        else if (tabId === "lora") this.loraView.render();
        else if (tabId === "automl") this.automlView.render();
    }

    updateI18nLabels() {
        const t = this.i18n;
        document.querySelectorAll("[data-i18n]").forEach(el => {
            const key = el.getAttribute("data-i18n");
            el.innerText = t.t(key);
        });

        const langBtn = document.getElementById("btn-toggle-lang");
        if (langBtn) langBtn.innerText = t.t("toggleLang");
    }
}

// Bootstrap
window.addEventListener("DOMContentLoaded", () => {
    window.synapseApp = new SynapseApp();
});
