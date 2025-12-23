from .core import _run_graphrag
import streamlit as st


st.set_page_config(page_title="GraphRAG Query Interface", layout="wide")
st.title("GraphRAG Knowledge Graph Query")

with st.form("query_form"):
    query_text = st.text_area("User Query", placeholder="Enter your natural language question...", height=100)
    
    col1, col2 = st.columns(2)
    with col1:
        model = st.text_input("Model", value="deepseek-chat")
        model_provider = st.text_input("Model Provider", value="deepseek")
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        max_tokens = st.number_input("Max Tokens", min_value=1, max_value=4096, value=512)
        reasoning = st.checkbox("Reasoning Mode", value=False)

    with col2:
        url = st.text_input("Neo4j URL", value="bolt://localhost:7687")
        username = st.text_input("Username", value="neo4j")
        password = st.text_input("Password", type="password")
        database = st.text_input("Database", value="neo4j")

    submitted = st.form_submit_button("Run Query")

if submitted:
    if not query_text.strip():
        st.error("Please enter a query.")
    else:
        try:
            with st.spinner("Processing query..."):
                response = _run_graphrag(
                    query_text=query_text.strip(),
                    model=model,
                    model_provider=model_provider,
                    temperature=temperature if temperature != 0.7 else None,  # 可选：只传非默认值
                    max_tokens=int(max_tokens) if max_tokens != 512 else None,
                    reasoning=reasoning or None,
                    url=url or None,
                    username=username or None,
                    password=password or None,
                    database=database,
                )

            # 1. User Query
            st.subheader("👤 User Query")
            st.write(f"> {response.get('query')}")

            # 2. Generated Cypher
            cypher = response.get("cypher_statement")
            if cypher:
                st.subheader("🧩 Generated Cypher")
                st.code(cypher, language="cypher")

            # 3. Final Answer
            answer = response.get("answer")
            if answer:
                st.subheader("✅ Final Answer")
                st.markdown(answer)

            # 4. Retrieved Evidence
            retrieved_chunks = response.get("retrieved_chunks")
            if retrieved_chunks:
                st.subheader("📚 Retrieved Evidence")
                for i, rec in enumerate(retrieved_chunks, 1):
                    with st.expander(f"Record {i}"):
                        for k, v in rec.items():
                            st.write(f"**{k}**: `{v}`")

        except Exception as e:
            st.error(f"❌ Query failed: {str(e)}")
            st.exception(e)  # 可选：开发时显示完整 traceback