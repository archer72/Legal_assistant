from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser


judgment_summarizer_prompt = PromptTemplate.from_template(
    """
You are an expert summarizer for Supreme Court judgments. Read the judgment below and produce:
- A 2-3 sentence summary of the facts.
- The holding (ratio decidendi) in clear language.
- Key legal principles stated by the Court.
- Any orders / directions given.

Judgment text:
{judgment_text}

Provide the summary in bullet points and keep it concise.
"""
)


def build_judgment_summarizer_chain(llm):
    return RunnableSequence(
        steps=[
            judgment_summarizer_prompt,
            llm,
            StrOutputParser()
        ]
    )
