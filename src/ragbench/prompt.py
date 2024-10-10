from pydantic import BaseModel, Field
from ragas.prompt.pydantic_prompt import PydanticPrompt


class RAGInput(BaseModel):
    query: str = Field(description="User query")
    context: str = Field(description="Context to search for answers")


class RAGOutput(BaseModel):
    response: str = Field(description="Response to the user query")


class RAGPrompt(PydanticPrompt[RAGInput, RAGOutput]):
    instructions = "Answer the user query based on the context"
    input_model = RAGInput
    output_model = RAGOutput
    examples = [
        (
            RAGInput(
                query="What is the capital of France?",
                context="Paris is the capital of France.",
            ),
            RAGOutput(response="Paris"),
        )
    ]
