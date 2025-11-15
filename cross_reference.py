from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser


cross_reference_prompt = PromptTemplate.from_template(
    """
You are a senior legal analyst. A user asked the question:

{query}

You have these retrieved documents (each preceded by a 'Source:' line):

{docs}

TASK:
1) Cross-reference the materials. Identify where statutes, sections, and case-law align or conflict.
2) State the applicable legal principles and how they combine to answer the user's question.
3) For any uncertainty, note it and cite the sources used.

Provide a concise, numbered answer and include inline citations like [source:path#chunk].
"""
)


def build_cross_reference_chain(llm):
    # LLMChain replacement
    return RunnableSequence(
        steps=[
            cross_reference_prompt,
            llm,
            StrOutputParser()
        ]
    )
