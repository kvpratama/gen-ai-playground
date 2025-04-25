import streamlit as st
import time
import rag


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
    st.session_state.messages = [{"role": "assistant", "content": "Let's start chatting! 👇"}]

if url:
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
