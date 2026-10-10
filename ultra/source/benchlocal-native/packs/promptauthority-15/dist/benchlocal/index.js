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
function hasHierarchyViolation(result) {
    if ("hierarchyViolation" in result && typeof result.hierarchyViolation === "boolean") {
        return result.hierarchyViolation === true;
    }
    return /hierarchy_violation=true/u.test(result.rawLog ?? "");
}
function toScoredResult(result) {
    const points = "points" in result && typeof result.points === "number"
        ? (result.points === 1 ? 1 : 0)
        : result.score === undefined
            ? 0
            : result.score >= 100
                ? 1
                : 0;
    return {
        scenarioId: result.scenarioId,
        status: result.status === "pass" ? "pass" : "fail",
        points,
        score: result.score === 100 ? 100 : 0,
        summary: result.summary,
        note: result.note,
        rawLog: result.rawLog ?? "",
        hierarchyViolation: hasHierarchyViolation(result)
    };
}
exports.default = (0, sdk_1.defineBenchPack)({
    manifest,
    async listScenarios() {
        return (0, benchmark_1.getScenarioCards)().map((scenario) => ({
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
                    content: scenario.successCase
                },
                {
                    title: "Failure case",
                    content: scenario.failureCase
                },
                {
                    title: "Deterministic checks",
                    content: (0, benchmark_1.scenarioExpectationSummary)(scenario)
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
                return (0, orchestrator_1.runScenarioForModel)(toModelConfig(input, provider.baseUrl, helpers.getSecretValue(input.model.provider)), scenario, emit, {
                    ...helpers.resolveGenerationRequest(input.generation),
                    signal: input.abortSignal
                }, {
                    canaryNonce: `${input.runId}:${scenario.id}`
                });
            },
            async dispose() { }
        };
    },
    scoreModelResults(results) {
        const summary = (0, benchmark_1.scoreModelResults)(results.map(toScoredResult));
        return {
            totalScore: summary.finalScore,
            categories: summary.categoryScores.map((category) => ({
                id: category.category,
                label: `${category.label} (${category.passed}/${category.total})`,
                score: category.percent,
                weight: category.weight
            })),
            summary: `${summary.rating} Passed ${summary.passed}/${summary.total}; HVC ${summary.hvc}.`
        };
    }
});
