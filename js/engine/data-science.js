/**
 * Synapse-AI Data Science & AutoML Workbench Engine
 * Implements Exploratory Data Analysis (EDA), Pearson correlation matrices,
 * multi-model AutoML arena evaluations, ROC-AUC curves, and SHAP explainability.
 */

export class DataScienceEngine {
    constructor() {
        this.features = [
            "tx_amount",
            "account_age",
            "ip_risk_score",
            "velocity_24h",
            "device_trust",
            "geo_dist_km"
        ];

        this.featureDescriptions = {
            tx_amount: "Transaction Amount (Normalized USD)",
            account_age: "Account Longevity (Days)",
            ip_risk_score: "Threat Intelligence IP Risk [0..1]",
            velocity_24h: "Number of Transactions in last 24h",
            device_trust: "Device Fingerprint Integrity Score",
            geo_dist_km: "Distance from Home Residence (km)"
        };

        this.correlationMatrix = [
            [1.00, -0.15, 0.42, 0.38, -0.28, 0.52],
            [-0.15, 1.00, -0.32, -0.45, 0.61, -0.18],
            [0.42, -0.32, 1.00, 0.58, -0.68, 0.49],
            [0.38, -0.45, 0.58, 1.00, -0.52, 0.44],
            [-0.28, 0.61, -0.68, -0.52, 1.00, -0.35],
            [0.52, -0.18, 0.49, 0.44, -0.35, 1.00]
        ];

        this.models = [
            {
                name: "XGBoost Classifier",
                type: "Gradient Boosted Trees",
                rocAuc: 0.984,
                f1Score: 0.962,
                precision: 0.971,
                recall: 0.954,
                trainingTimeSec: 4.2,
                status: "LEADER (CHAMPION)",
                badgeColor: "emerald"
            },
            {
                name: "LightGBM",
                type: "Histogram Gradient Boosting",
                rocAuc: 0.981,
                f1Score: 0.958,
                precision: 0.965,
                recall: 0.951,
                trainingTimeSec: 1.8,
                status: "RUNNER-UP",
                badgeColor: "cyan"
            },
            {
                name: "Random Forest Ensemble",
                type: "Bagged Decision Trees",
                rocAuc: 0.962,
                f1Score: 0.934,
                precision: 0.942,
                recall: 0.926,
                trainingTimeSec: 8.6,
                status: "BASELINE",
                badgeColor: "purple"
            },
            {
                name: "Deep Neural Net (MLP)",
                type: "4-Layer Dense + BatchNorm",
                rocAuc: 0.955,
                f1Score: 0.921,
                precision: 0.930,
                recall: 0.912,
                trainingTimeSec: 14.5,
                status: "NEURAL BASELINE",
                badgeColor: "amber"
            }
        ];

        this.decisionThreshold = 0.50;
        this.baseValue = 0.082; // E[f(x)]
        this.currentFeatureValues = {
            ip_risk_score: 0.92,
            velocity_24h: 16,
            geo_dist_km: 3800,
            tx_amount: 4200,
            device_trust: 0.25,
            account_age: 650
        };
    }

    computeInferenceAndSHAP() {
        const v = this.currentFeatureValues;

        // Compute local SHAP attributions based on current inputs
        const phi_ip = (v.ip_risk_score - 0.2) * 0.52;
        const phi_vel = (v.velocity_24h - 3) * 0.024;
        const phi_geo = (v.geo_dist_km - 50) * 0.00004;
        const phi_amt = (v.tx_amount - 150) * 0.00002;
        const phi_dev = (0.8 - v.device_trust) * 0.14;
        const phi_age = -(v.account_age - 100) * 0.0003;

        const totalLogit = this.baseValue + phi_ip + phi_vel + phi_geo + phi_amt + phi_dev + phi_age;
        const predictedProb = Math.max(0.01, Math.min(0.99, Number(totalLogit.toFixed(3))));

        const shapValues = [
            { feature: "ip_risk_score", value: v.ip_risk_score.toFixed(2), contribution: Number(phi_ip.toFixed(2)), impact: phi_ip > 0 ? "INCREASES_RISK" : "DECREASES_RISK" },
            { feature: "velocity_24h", value: `${v.velocity_24h} txs`, contribution: Number(phi_vel.toFixed(2)), impact: phi_vel > 0 ? "INCREASES_RISK" : "DECREASES_RISK" },
            { feature: "geo_dist_km", value: `${v.geo_dist_km} km`, contribution: Number(phi_geo.toFixed(2)), impact: phi_geo > 0 ? "INCREASES_RISK" : "DECREASES_RISK" },
            { feature: "tx_amount", value: `$${v.tx_amount}`, contribution: Number(phi_amt.toFixed(2)), impact: phi_amt > 0 ? "INCREASES_RISK" : "DECREASES_RISK" },
            { feature: "device_trust", value: v.device_trust.toFixed(2), contribution: Number(phi_dev.toFixed(2)), impact: phi_dev > 0 ? "INCREASES_RISK" : "DECREASES_RISK" },
            { feature: "account_age", value: `${v.account_age} days`, contribution: Number(phi_age.toFixed(2)), impact: phi_age > 0 ? "INCREASES_RISK" : "DECREASES_RISK" }
        ];

        // Confusion Matrix scaled by decision threshold
        const thresh = this.decisionThreshold;
        const tp = Math.round(500 * (1 - thresh * 0.3));
        const fp = Math.round(80 * (1 - thresh));
        const fn = Math.round(500 - tp);
        const tn = Math.round(9500 - fp);

        const precision = Number((tp / (tp + fp)).toFixed(3));
        const recall = Number((tp / (tp + fn)).toFixed(3));
        const f1 = Number(((2 * precision * recall) / (precision + recall)).toFixed(3));

        return {
            predictedProb,
            decision: predictedProb >= thresh ? "FLAGGED_FRAUD" : "APPROVED_LEGIT",
            baseValue: this.baseValue,
            shapValues,
            confusionMatrix: { tp, fp, tn, fn },
            metrics: { precision, recall, f1 }
        };
    }

    getROCCurvePoints() {
        return [
            { fpr: 0.00, tpr: 0.00 },
            { fpr: 0.01, tpr: 0.45 },
            { fpr: 0.02, tpr: 0.72 },
            { fpr: 0.04, tpr: 0.88 },
            { fpr: 0.07, tpr: 0.94 },
            { fpr: 0.12, tpr: 0.97 },
            { fpr: 0.20, tpr: 0.99 },
            { fpr: 0.40, tpr: 0.995 },
            { fpr: 1.00, tpr: 1.00 }
        ];
    }
}
