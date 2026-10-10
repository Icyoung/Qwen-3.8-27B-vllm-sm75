"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.manifest = void 0;
const sdk_1 = require("../../../../adapter.cjs");
const benchmark_1 = require("../lib/benchmark");
const orchestrator_1 = require("../lib/orchestrator");
const manifest = (0, sdk_1.loadBenchPackManifest)(__dirname);
exports.manifest = manifest;
function toModelConfig(input, baseUrl, apiKey) {
    return {
        id: input.model.id,
        label: input.model.label,
        provider: input.model.provider,
        model: input.model.model,
        baseUrl,
        apiKey
    };
}
function toScenarioResult(result) {
    return {
        scenarioId: result.scenarioId,
        status: result.status,
        score: result.points * 50,
        summary: result.summary,
        note: result.note,
        rawLog: result.rawLog,
        points: result.points
    };
}
exports.default = (0, sdk_1.defineBenchPack)({
    manifest,
    async listScenarios() {
        return benchmark_1.SCENARIOS.map((scenario) => ({
            id: scenario.id,
            title: scenario.title,
            category: scenario.category,
            description: scenario.description,
            promptText: scenario.userMessage,
            detailCards: [
                {
                    title: "What this tests",
                    content: scenario.description
                },
                {
                    title: "Success case",
                    content: benchmark_1.SCENARIO_DISPLAY_DETAILS[scenario.id]?.successCase ?? "See benchmark definition."
                },
                {
                    title: "Failure case",
                    content: benchmark_1.SCENARIO_DISPLAY_DETAILS[scenario.id]?.failureCase ?? "See benchmark definition."
                }
            ]
        }));
    },
    async prepare(context) {
        const helpers = (0, sdk_1.createHostHelpers)(context);
        return {
            async runScenario(input, emit) {
                const scenario = helpers.getScenarioById(benchmark_1.SCENARIOS, input.scenario.id);
                const provider = helpers.getRequiredProvider(input.model.provider, {
                    enabledOnly: true
                });
                const result = await (0, orchestrator_1.runScenarioForModel)(toModelConfig(input, provider.baseUrl, helpers.getSecretValue(input.model.provider)), scenario, emit, {
                    ...helpers.resolveGenerationRequest(input.generation),
                    signal: input.abortSignal
                });
                return toScenarioResult(result);
            },
            async dispose() { }
        };
    },
    scoreModelResults(results) {
        const summary = (0, benchmark_1.scoreModelResults)(results.map((result) => ({
            scenarioId: result.scenarioId,
            status: result.status,
            points: "points" in result && typeof result.points === "number" ? result.points : result.score === undefined ? 0 : result.score >= 85 ? 2 : result.score >= 60 ? 1 : 0,
            summary: result.summary,
            note: result.note,
            rawLog: result.rawLog ?? ""
        })));
        return {
            totalScore: summary.finalScore,
            categories: summary.categoryScores.map((category) => ({
                id: category.category,
                label: category.label,
                score: category.percent,
                weight: 20
            })),
            summary: summary.rating
        };
    }
});
