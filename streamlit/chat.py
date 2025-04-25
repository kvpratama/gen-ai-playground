import streamlit as st
import time
import re
import rag

def is_valid_url(url):
    # Simple regex for URL validation
    return re.match(r'^https?://[^\s/$.?#].[^\s]*$', url) is not None

with st.form("my_form"):
   st.write("Enter a URL to chat with:")
   url = st.text_input(
        "Enter a URL to chat with:",
        placeholder="https://streamlit.io/",
        key="placeholder",
    )
   submitted = st.form_submit_button('Submit URL')

print(url)


# Reset chat history if a new URL is submitted
if submitted:
    if not url or not is_valid_url(url):
        st.error("Please enter a valid URL (starting with http:// or https://).")
    else:
        st.session_state.messages = [{"role": "assistant", "content": f"What question do you have for {url}?"}]


if url and is_valid_url(url):
    # Initialize chat history
    rag_assistant = rag.RAG(url)

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        print(prompt)
        assistant_response = rag_assistant.query(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Simulate stream of response with milliseconds delay
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
