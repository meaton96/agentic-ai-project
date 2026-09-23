from openai import OpenAI

client = OpenAI(
    base_url="https://api.genai.gccis.rit.edu/v1",
    api_key="sk-ritgenai-nxsvse-ai-agentic-sandbox-project-me3870-e41f44c3318ae31899a51f6f783c9da9"
)

models = client.models.list()
for model in models.data:
    print(model.id)