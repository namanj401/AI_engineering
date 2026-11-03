from contextlib import asynccontextmanager
import uuid
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine


# 1. The Schema
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100


# 2. The Lifespan Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    engine_args = AsyncEngineArgs(
        model="mlx-community/Llama-3.2-1B-Instruct-4bit",
        max_model_len=2048,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    app.state.engine = engine
    yield


# 3. The App & Route
app = FastAPI(lifespan=lifespan)


@app.post("/generate")
async def generate(request: GenerateRequest):
    request_id = uuid.uuid4().hex
    sampling_params = SamplingParams(max_tokens=request.max_tokens)

    async def stream_tokens():
        results_generator = app.state.engine.generate(
            request.prompt, sampling_params, request_id
        )
        previous_text = ""
        async for request_output in results_generator:
            current_text = request_output.outputs[0].text
            new_chunk = current_text[len(previous_text):]
            yield f"data: {new_chunk}\n\n"
            previous_text = current_text

    return StreamingResponse(stream_tokens(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
