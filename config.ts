// Make sure to replace the values with your actual API key and model

// USING ANTHROPIC CLAUDE SONNET 4 is strongly recommended for best results

export const config = {
  aiSdk: {
    // The base URL for the AI SDK, leave blank for e.g. openai
    baseUrl: "https://openrouter.ai/api/v1",

    // Your API key for provider, if using Ollama enter "ollama" here
    apiKey: "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",

    // The model to use, e.g., "gpt-4", "gpt-3.5-turbo", or "ollama/llama2"
    model: "anthropic/claude-sonnet-4",
  },
} as const;
