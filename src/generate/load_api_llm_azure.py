from generate.openai_chat_batch import AsyncOpenAIClient, OpenAIClient

def load_api_chat_completion(model_name, async_=False, *args, **kargs):
    if async_:
        return AsyncOpenAIClient(model=model_name)
    else:
        return OpenAIClient(model=model_name)
    