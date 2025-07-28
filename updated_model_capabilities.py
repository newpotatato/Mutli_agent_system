# Автоматически обновленные способности моделей
# Сгенерировано на основе реальных тестовых данных

MODEL_CAPABILITIES = {
    "GPT-4": ModelCapabilities(
        math=0.568,
        code=0.500, 
        text=0.618,
        analysis=0.477,
        creative=0.853,
        explanation=0.535,
        planning=0.618,
        research=0.500,
        optimization=0.617,
        technical_specs=TechnicalSpecs(
            avg_response_time=2.0,  # Обновится из реальных данных
            cost_per_1k_tokens=0.02,
            reliability_score=0.95,
            context_window=32000,
            max_output_tokens=4000
        )
    ),
    "Claude-3.5-Sonnet": ModelCapabilities(
        math=0.523,
        code=0.500, 
        text=0.853,
        analysis=0.469,
        creative=0.618,
        explanation=0.605,
        planning=0.657,
        research=0.500,
        optimization=0.852,
        technical_specs=TechnicalSpecs(
            avg_response_time=2.0,  # Обновится из реальных данных
            cost_per_1k_tokens=0.02,
            reliability_score=0.95,
            context_window=32000,
            max_output_tokens=4000
        )
    ),
    "GPT-3.5-Turbo": ModelCapabilities(
        math=0.530,
        code=0.500, 
        text=0.657,
        analysis=0.430,
        creative=0.657,
        explanation=0.547,
        planning=0.854,
        research=0.500,
        optimization=0.656,
        technical_specs=TechnicalSpecs(
            avg_response_time=2.0,  # Обновится из реальных данных
            cost_per_1k_tokens=0.02,
            reliability_score=0.95,
            context_window=32000,
            max_output_tokens=4000
        )
    )
}