from ai4gcnpy.chains import LABEL_PROMPT, AUTHORSHIP_PROMPT


auther_content = """
A. Smith (NASA/GSFC), B. Johnson (MIT), and C. Lee (Caltech) report on behalf of the Swift Team.
"""
author_messages = LABEL_PROMPT.invoke({"numbered_paragraphs": auther_content}).to_messages()
for msg in author_messages:
    role = "System" if msg.type == "system" else "Human"
    print(f"[{role}]:\n{msg.content}")