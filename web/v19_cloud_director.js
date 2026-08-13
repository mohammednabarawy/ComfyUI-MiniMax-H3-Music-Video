import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const PROVIDERS = {
    auto: {
        primary: "gemini",
        model: "gemini-3.6-flash",
        max_output_tokens: 32768,
        temperature: 0.5,
        reasoning_effort: "low",
        timeout_sec: 300,
    },
    gemini: {
        primary: "gemini",
        model: "gemini-3.6-flash",
        max_output_tokens: 32768,
        temperature: 0.5,
        reasoning_effort: "low",
        timeout_sec: 300,
    },
    nvidia: {
        primary: "nvidia",
        model: "z-ai/glm-5.2",
        max_output_tokens: 16384,
        temperature: 0.6,
        reasoning_effort: "low",
        timeout_sec: 300,
    },
    zen: {
        primary: "zen",
        model: "nemotron-3-ultra-free",
        max_output_tokens: 16384,
        temperature: 0.6,
        reasoning_effort: "low",
        timeout_sec: 360,
    },
    custom_openai: {
        primary: "custom_openai",
        model: "",
        max_output_tokens: 12000,
        temperature: 0.5,
        reasoning_effort: "none",
        timeout_sec: 300,
    },
};

function widget(node, name) {
    return node.widgets?.find((item) => item.name === name);
}

function setValue(node, name, value) {
    const item = widget(node, name);
    if (item) item.value = value;
}

function makeModelCombo(node) {
    const item = widget(node, "model");
    if (!item) return;
    item.type = "combo";
    item.options = item.options || {};
    const current = String(item.value || "").trim();
    item.options.values = current ? [current] : [PROVIDERS.auto.model];
}

async function refreshModels(node, selectDefault = false) {
    const provider = String(widget(node, "provider")?.value || "auto");
    const model = widget(node, "model");
    const button = node.__v19RefreshButton;
    if (!model || !button || node.__v19Refreshing) return;
    node.__v19Refreshing = true;
    button.name = `Fetching ${PROVIDERS[provider]?.primary || provider} models...`;
    node.setDirtyCanvas?.(true, true);
    try {
        const response = await api.fetchApi("/minimax_music_video/v19/models", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                provider,
                custom_base_url: String(widget(node, "custom_base_url")?.value || ""),
                credential_file: String(widget(node, "credential_file")?.value || ""),
                api_key: String(widget(node, "api_key")?.value || ""),
            }),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
        const models = Array.isArray(payload.models) ? payload.models : [];
        if (!models.length) throw new Error("Provider returned no models");
        model.options.values = models;
        if (selectDefault || !models.includes(model.value)) {
            model.value = models.includes(payload.default_model) ? payload.default_model : models[0];
        }
        button.name = `Refresh Models — ${payload.provider}: ${models.length} found`;
    } catch (error) {
        const fallback = PROVIDERS[provider]?.model;
        if (fallback) {
            model.options.values = [fallback];
            if (!model.value) model.value = fallback;
        }
        button.name = `Refresh Models — ${String(error.message || error).slice(0, 55)}`;
    } finally {
        node.__v19Refreshing = false;
        node.setDirtyCanvas?.(true, true);
    }
}

function applyProvider(node, provider) {
    const preset = PROVIDERS[provider] || PROVIDERS.auto;
    setValue(node, "use_provider_preset", true);
    setValue(node, "planning_scope", "whole_song");
    setValue(node, "fallback_chain", "gemini,nvidia,zen");
    setValue(node, "max_output_tokens", preset.max_output_tokens);
    setValue(node, "temperature", preset.temperature);
    setValue(node, "reasoning_effort", preset.reasoning_effort);
    setValue(node, "timeout_sec", preset.timeout_sec);
    if (provider !== "custom_openai") {
        setValue(node, "model", preset.model);
        setValue(node, "custom_base_url", "");
        setValue(node, "credential_file", "");
    }
    refreshModels(node, true);
}

app.registerExtension({
    name: "MiniMaxMusicVideo.V19CloudDirector",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MiniMaxDirectorCloud") return;

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            makeModelCombo(this);
            const provider = widget(this, "provider");
            if (provider) {
                const originalCallback = provider.callback;
                provider.callback = (value) => {
                    originalCallback?.call(provider, value);
                    applyProvider(this, String(value || "auto"));
                };
            }
            this.__v19RefreshButton = this.addWidget(
                "button",
                "Refresh Models from selected provider",
                null,
                () => refreshModels(this, false),
                { serialize: false },
            );
            this.setSize?.([Math.max(this.size?.[0] || 0, 620), Math.max(this.size?.[1] || 0, 690)]);
            setTimeout(() => refreshModels(this, false), 500);
            return result;
        };

        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalConfigure?.apply(this, arguments);
            makeModelCombo(this);
            setTimeout(() => refreshModels(this, false), 500);
            return result;
        };
    },
});
