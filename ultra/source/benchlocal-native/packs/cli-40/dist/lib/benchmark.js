"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.SCENARIOS = void 0;
exports.getScenarioCards = getScenarioCards;
exports.scoreModelResults = scoreModelResults;
const scenario_data_json_1 = __importDefault(require("../verification/scenario-data.json"));
const CATEGORY_ORDER = ["A", "B", "C", "D", "E", "F", "G", "H"];
const CATEGORY_INFO = {
    A: { label: "A — Text Processing", weight: 10 },
    B: { label: "B — Filesystem Operations", weight: 10 },
    C: { label: "C — Pipelines & Composition", weight: 15 },
    D: { label: "D — Archives, Compression, Encoding", weight: 10 },
    E: { label: "E — Diagnose and Fix", weight: 20 },
    F: { label: "F — Investigation", weight: 15 },
    G: { label: "G — Restraint and Safety", weight: 10 },
    H: { label: "H — Error Recovery", weight: 10 }
};
exports.SCENARIOS = scenario_data_json_1.default;
function getScenarioCards() {
    return [...exports.SCENARIOS];
}
function average(values) {
    if (values.length === 0) {
        return 0;
    }
    return values.reduce((sum, value) => sum + value, 0) / values.length;
}
function buildSummary(totalScore) {
    if (totalScore >= 90) {
        return "CLI-40 performance is excellent and consistently operational.";
    }
    if (totalScore >= 75) {
        return "CLI-40 performance is good, with a few misses in subtle or recovery-heavy scenarios.";
    }
    if (totalScore >= 60) {
        return "CLI-40 performance is adequate, but reliability drops on harder CLI tasks.";
    }
    if (totalScore >= 40) {
        return "CLI-40 performance is weak and still needs close supervision.";
    }
    return "CLI-40 performance is poor and fails most benchmark conditions.";
}
function scoreModelResults(results) {
    const scenarioMap = new Map(exports.SCENARIOS.map((scenario) => [scenario.id, scenario]));
    const categoryScores = new Map();
    for (const result of results) {
        const scenario = scenarioMap.get(result.scenarioId);
        if (!scenario) {
            continue;
        }
        const scores = categoryScores.get(scenario.categoryId) ?? [];
        scores.push(result.score);
        categoryScores.set(scenario.categoryId, scores);
    }
    const categories = CATEGORY_ORDER
        .filter((categoryId) => categoryScores.has(categoryId))
        .map((categoryId) => {
        const info = CATEGORY_INFO[categoryId];
        const score = Math.round(average(categoryScores.get(categoryId) ?? []));
        return {
            id: categoryId.toLowerCase(),
            label: info.label,
            score,
            weight: info.weight
        };
    });
    const weightedTotal = categories.length === 0
        ? 0
        : Math.round(categories.reduce((sum, category) => sum + category.score * (category.weight ?? 0), 0) /
            categories.reduce((sum, category) => sum + (category.weight ?? 0), 0));
    return {
        totalScore: weightedTotal,
        categories,
        summary: buildSummary(weightedTotal)
    };
}
