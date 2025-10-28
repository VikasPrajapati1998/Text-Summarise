from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template="""
Summarize the research paper titled "{paper_input}" with the following specifications:

Explanation Style: {style_input}
Explanation Length: {length_input}

1. Mathematical Details:
   - Include relevant mathematical equations if present in the paper.
   - Explain mathematical concepts with simple, intuitive code snippets.
2. Analogies:
   - Use relatable analogies to simplify complex ideas.

If certain information is not available in the paper, respond with: 
"Insufficient information available" instead of guessing.

Ensure the summary is clear, accurate, and aligned with the specified style and length.
""",
    input_variables=["paper_input", "style_input", "length_input"], 
    validate_template=True
)

template.save('template.json')
