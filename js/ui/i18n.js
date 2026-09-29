/**
 * Synapse-AI Bilingual Internationalization Engine
 * Full English & Arabic localization with RTL layout adaptation.
 */

export const translations = {
    en: {
        brandTitle: "SYNAPSE-AI STUDIO",
        brandSubtitle: "Autonomous Agent Swarms, LLM Fine-Tuning & Data Science Workbench",
        tabAgents: "🤖 Agent Swarm & DAG",
        tabAttention: "🔍 Multi-Head Attention QKV",
        tabLoRA: "⚡ LoRA PEFT Training",
        tabAutoML: "📊 Data Science & AutoML",
        kpiAgents: "Active Agents",
        kpiAttentionHeads: "Attention Heads",
        kpiLoRASavings: "LoRA Param Savings",
        kpiChampionAUC: "AutoML Champion AUC",
        btnRunSwarm: "🚀 Dispatch Agent Swarm",
        toggleLang: "العربية",
        soundOn: "🔊 Audio On",
        soundOff: "🔇 Audio Muted"
    },
    ar: {
        brandTitle: "استوديو سينابس (SYNAPSE-AI)",
        brandSubtitle: "منظومة تصميم الوكلاء المستقلين، تدريب وتطويع النماذج اللغوية، ومختبر علوم البيانات المتقدم",
        tabAgents: "🤖 تصميم الوكلاء المستقلين وسير العمل",
        tabAttention: "🔍 مصفوفة انتباه المحولات QKV",
        tabLoRA: "⚡ تدريب وتطويع النماذج (LoRA PEFT)",
        tabAutoML: "📊 علوم البيانات وحلبة التعلم الآلي",
        kpiAgents: "الوكلاء المستقلون",
        kpiAttentionHeads: "رؤوس الانتباه",
        kpiLoRASavings: "توفير معلمات LoRA",
        kpiChampionAUC: "دقة بطل النماذج (AUC)",
        btnRunSwarm: "🚀 تشغيل سرب الوكلاء المستقلين",
        toggleLang: "English",
        soundOn: "🔊 الصوت مفعّل",
        soundOff: "🔇 كتم الصوت"
    }
};

export class I18nManager {
    constructor() {
        this.lang = "ar";
    }

    t(key) {
        return translations[this.lang][key] || key;
    }

    toggle() {
        this.lang = this.lang === "ar" ? "en" : "ar";
        document.documentElement.lang = this.lang;
        document.documentElement.dir = this.lang === "ar" ? "rtl" : "ltr";
        return this.lang;
    }
}
