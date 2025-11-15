import streamlit as st
import requests
import os

API_URL = os.getenv('API_URL', 'http://localhost:8000')

st.title('Indian Law Query Assistant — Demo')
query = st.text_area('Enter your legal question', height=150)
conv_id = st.text_input('Conversation ID (leave blank to create new)')

if st.button('Ask'):
    if not query.strip():
        st.warning('Please enter a question')
    else:
        with st.spinner('Fetching answer...'):
            payload = {'query': query, 'conversation_id': conv_id or None, 'top_k': 6}
            try:
                resp = requests.post(f'{API_URL}/query', json=payload, timeout=120)
            except Exception as e:
                st.error(f'Failed to reach API: {e}')
            else:
                if resp.status_code == 200:
                    data = resp.json()
                    st.subheader('Answer (RAG)')
                    st.write(data.get('answer'))
                    st.subheader('Cross-referenced reasoning')
                    st.write(data.get('cross_reference'))
                    if data.get('summary'):
                        st.subheader('Summary (judgment)')
                        st.write(data.get('summary'))
                    st.subheader('Sources (top)')
                    for s in data['sources']:
                        st.write(f"- {s['source']} (chunk {s['chunk']})")
                    st.write('Conversation ID:', data.get('conversation_id'))
                else:
                    st.error(f'API error: {resp.status_code} - {resp.text}')
